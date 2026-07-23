//! 세션·프로젝트 환경설정(선호) 로드/저장/병합 — 세션별 `session-prefs/` 와
//! 프로젝트 settings 파일에서 model/effort 선호를 읽고, 우선순위로 병합해
//! effort 를 해석한다. `LiveCli` 가 시작 시 소비하는 순수 IO/파싱 책임.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use zo_cli::tui::modals::Effort;

pub(super) const SESSION_PREFERENCES_DIR: &str = "session-prefs";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SessionPreferences {
    pub(super) model: Option<String>,
    pub(super) effort: Option<String>,
    pub(super) effort_budget: Option<u32>,
    pub(super) model_handoff_memory: Option<String>,
}

pub(super) fn preferences_path(session_path: &Path) -> PathBuf {
    let stem = session_path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("session");
    let file_name = format!("{stem}.json");
    let Some(parent) = session_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return PathBuf::from(SESSION_PREFERENCES_DIR).join(file_name);
    };
    let preferences_dir = if parent
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name == "sessions")
    {
        parent
            .parent()
            .unwrap_or(parent)
            .join(SESSION_PREFERENCES_DIR)
    } else {
        parent.join(SESSION_PREFERENCES_DIR)
    };
    preferences_dir.join(file_name)
}

pub(super) fn load_session_preferences(session_path: &Path) -> SessionPreferences {
    let path = preferences_path(session_path);
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<SessionPreferences>(&text).ok())
        .unwrap_or_default()
}

pub(super) fn save_session_preferences(
    session_path: &Path,
    preferences: &SessionPreferences,
) -> io::Result<()> {
    let path = preferences_path(session_path);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_json::to_string_pretty(preferences).map_err(io::Error::other)?;
    crate::write_atomic(&path, rendered.as_bytes())
}

pub(super) fn project_settings_path(cwd: &Path) -> PathBuf {
    cwd.join(".zo").join("settings.local.json")
}

pub(super) fn load_project_preferences(
    cwd: &Path,
) -> Result<SessionPreferences, Box<dyn std::error::Error>> {
    let shared = load_settings_preferences(&cwd.join(".zo").join("settings.json"))?;
    let local = load_settings_preferences(&project_settings_path(cwd))?;
    let project = merge_preferences(local, shared);
    let cli = runtime::ConfigLoader::cli_settings_file()
        .as_deref()
        .map(load_settings_preferences)
        .transpose()?
        .unwrap_or_default();
    Ok(merge_preferences(cli, project))
}

pub(super) fn load_settings_preferences(path: &Path) -> io::Result<SessionPreferences> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(SessionPreferences::default());
        }
        Err(error) => return Err(error),
    };
    if contents.trim().is_empty() {
        return Ok(SessionPreferences::default());
    }
    let value: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let object = value.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} must contain a JSON object", path.display()),
        )
    })?;
    Ok(preferences_from_settings_object(object))
}

pub(super) fn preferences_from_settings_object(
    object: &serde_json::Map<String, serde_json::Value>,
) -> SessionPreferences {
    SessionPreferences {
        model: object
            .get("model")
            .and_then(serde_json::Value::as_str)
            .map(crate::cli_args::resolve_model_alias),
        effort: object
            .get("reasoningEffort")
            .and_then(serde_json::Value::as_str)
            .and_then(Effort::from_token)
            .map(|effort| effort.canonical().to_string()),
        effort_budget: object
            .get("thinkingBudget")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|budget| *budget > 0),
        model_handoff_memory: None,
    }
}

pub(super) fn merge_preferences(
    session: SessionPreferences,
    project: SessionPreferences,
) -> SessionPreferences {
    let has_session_effort = session.effort.is_some() || session.effort_budget.is_some();
    let (effort, effort_budget) = if has_session_effort {
        (session.effort, session.effort_budget)
    } else {
        (project.effort, project.effort_budget)
    };
    SessionPreferences {
        model: session.model.or(project.model),
        effort,
        effort_budget,
        model_handoff_memory: session.model_handoff_memory,
    }
}

pub(super) fn has_effort_preference(preferences: &SessionPreferences) -> bool {
    preferences.effort.is_some() || preferences.effort_budget.is_some()
}

pub(super) fn save_project_preferences(
    cwd: &Path,
    preferences: &SessionPreferences,
) -> io::Result<()> {
    let path = project_settings_path(cwd);
    super::smart_settings::update_settings_file(&path, |document| {
        if let Some(model) = preferences.model.as_deref() {
            document.insert(
                "model".to_string(),
                serde_json::Value::String(model.to_string()),
            );
        }

        if let Some(effort) = preferences.effort.as_deref() {
            document.insert(
                "reasoningEffort".to_string(),
                serde_json::Value::String(effort.to_string()),
            );
            document.remove("thinkingBudget");
        } else if let Some(budget) = preferences.effort_budget {
            document.remove("reasoningEffort");
            document.insert("thinkingBudget".to_string(), serde_json::json!(budget));
        } else {
            document.remove("reasoningEffort");
            document.remove("thinkingBudget");
        }
        Ok(())
    })
}

/// The project's deliberately-pinned reasoning effort (`reasoningEffort` /
/// `thinkingBudget` in `.zo/settings*.json`), or `None` when the project
/// pins no effort. Composes the existing loader + resolver so there is one
/// source of truth for "what effort did the project ask for"; a load error
/// degrades to `None` (defensive — the effort selection must never fail the
/// run). The headless `-p` path consults this so a pinned project effort is
/// honored instead of being overwritten by the prompt-shape default (Gap D).
///
/// Returns only a *named* effort tier: a `thinkingBudget` that does not map to
/// a preset budget resolves to `None`, because the headless `-p` effort model
/// carries a named tier (`set_effort` derives the budget from it), not an
/// arbitrary raw budget. Canonical preset budgets resolve to their tier.
pub(crate) fn project_effort_preference(cwd: &Path) -> Option<Effort> {
    let preferences = load_project_preferences(cwd).ok()?;
    if !has_effort_preference(&preferences) {
        return None;
    }
    effort_from_preferences(&preferences).0
}

pub(super) fn effort_from_preferences(
    preferences: &SessionPreferences,
) -> (Option<Effort>, Option<u32>) {
    let explicit_effort = preferences.effort.as_deref().and_then(Effort::from_token);
    let budget = if let Some(effort) = explicit_effort {
        (effort.budget() > 0).then_some(effort.budget())
    } else {
        preferences.effort_budget.filter(|budget| *budget > 0)
    };
    let effort = explicit_effort.or_else(|| Effort::from_budget(budget));
    (effort, budget)
}

#[cfg(test)]
mod tests {
    use super::{SessionPreferences, load_project_preferences, project_effort_preference};
    use zo_cli::tui::modals::Effort;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A scratch cwd under the OS temp dir, removed on drop. Avoids adding a
    /// `tempfile` dev-dependency for this one test (the crate does not carry it).
    struct ScratchCwd {
        path: PathBuf,
    }

    impl ScratchCwd {
        fn new() -> Self {
            let unique = format!(
                "zo-prefs-test-{}-{}",
                std::process::id(),
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).expect("create scratch cwd");
            Self { path }
        }

        fn write_settings_local(&self, body: &str) {
            let dir = self.path.join(".zo");
            fs::create_dir_all(&dir).expect("create .zo dir");
            fs::write(dir.join("settings.local.json"), body).expect("write settings.local.json");
        }
    }

    impl Drop for ScratchCwd {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct CliOverrideReset;

    impl Drop for CliOverrideReset {
        fn drop(&mut self) {
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides::default());
        }
    }

    #[test]
    fn cli_settings_override_startup_model_and_effort_preferences() {
        let _guard = crate::test_env_lock();
        let _reset = CliOverrideReset;
        let scratch = ScratchCwd::new();
        scratch.write_settings_local(
            r#"{ "model": "claude-opus-4-8", "reasoningEffort": "low" }"#,
        );
        let cli_settings = scratch.path.join("bench-settings.json");
        fs::write(
            &cli_settings,
            r#"{ "model": "gpt-5.6-sol[fast]", "reasoningEffort": "xhigh" }"#,
        )
        .expect("write CLI settings");
        runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides {
            settings_file: Some(cli_settings),
            strict_mcp_config: false,
        });

        let preferences =
            load_project_preferences(&scratch.path).expect("load startup preferences");

        assert_eq!(preferences.model.as_deref(), Some("gpt-5.6-sol[fast]"));
        assert_eq!(preferences.effort.as_deref(), Some("xhigh"));
    }

    #[test]
    fn project_effort_preference_is_none_without_settings() {
        // Gap D: an empty project (no settings file) pins no effort, so the
        // headless path falls through to its prompt-shape default unchanged.
        let _guard = crate::test_env_lock();
        let scratch = ScratchCwd::new();
        assert_eq!(project_effort_preference(&scratch.path), None);
    }

    #[test]
    fn project_effort_preference_reads_pinned_reasoning_effort() {
        // A deliberately pinned `reasoningEffort` is surfaced so a headless `-p`
        // run honors it instead of overwriting it with the prompt-shape default.
        // "ultracode" is the pre-rename (P9) persisted token and MUST keep
        // resolving to `Effort::Smart` — an existing settings.local.json must
        // not silently stop pinning its saved preset after the rename.
        let _guard = crate::test_env_lock();
        let scratch = ScratchCwd::new();
        scratch.write_settings_local(r#"{ "reasoningEffort": "ultracode" }"#);
        assert_eq!(
            project_effort_preference(&scratch.path),
            Some(Effort::Smart)
        );
    }

    #[test]
    fn project_effort_preference_reads_smart_and_ultra() {
        // The current canonical token (`smart`) and the new static `ultra`
        // level both round-trip through persisted settings.
        let _guard = crate::test_env_lock();
        let scratch = ScratchCwd::new();
        scratch.write_settings_local(r#"{ "reasoningEffort": "smart" }"#);
        assert_eq!(
            project_effort_preference(&scratch.path),
            Some(Effort::Smart)
        );

        let scratch = ScratchCwd::new();
        scratch.write_settings_local(r#"{ "reasoningEffort": "ultra" }"#);
        assert_eq!(
            project_effort_preference(&scratch.path),
            Some(Effort::Ultra)
        );
    }

    #[test]
    fn project_effort_preference_resolves_thinking_budget_pin() {
        // A `thinkingBudget` pin (no `reasoningEffort`) still resolves to the
        // matching named effort, mirroring `effort_from_preferences`.
        let _guard = crate::test_env_lock();
        let scratch = ScratchCwd::new();
        scratch.write_settings_local(&format!(
            r#"{{ "thinkingBudget": {} }}"#,
            Effort::High.budget()
        ));
        assert_eq!(project_effort_preference(&scratch.path), Some(Effort::High));
    }

    #[test]
    fn project_effort_preference_ignores_a_preferences_struct_without_effort() {
        // Guard: the "no effort pinned" path is keyed on `has_effort_preference`,
        // not on a defaulted struct silently reading as `Off`.
        let prefs = SessionPreferences::default();
        assert!(prefs.effort.is_none() && prefs.effort_budget.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn sidecar_replace_failure_preserves_previous_preferences() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = ScratchCwd::new();
        let sessions = scratch.path.join("sessions");
        fs::create_dir_all(&sessions).expect("create sessions directory");
        let session_path = sessions.join("session.jsonl");
        let path = super::preferences_path(&session_path);
        let preferences_dir = path.parent().expect("preferences directory");
        fs::create_dir_all(preferences_dir).expect("create preferences directory");
        let before = br#"{"model":"old"}"#;
        fs::write(&path, before).expect("seed preferences");
        fs::set_permissions(preferences_dir, fs::Permissions::from_mode(0o555))
            .expect("make preferences directory read-only");

        let probe = preferences_dir.join("probe");
        if fs::write(&probe, b"probe").is_ok() {
            let _ = fs::remove_file(probe);
            fs::set_permissions(preferences_dir, fs::Permissions::from_mode(0o755))
                .expect("restore preferences directory permissions");
            return;
        }

        let result = super::save_session_preferences(
            &session_path,
            &SessionPreferences {
                model: Some("new".to_string()),
                ..SessionPreferences::default()
            },
        );

        fs::set_permissions(preferences_dir, fs::Permissions::from_mode(0o755))
            .expect("restore preferences directory permissions");
        let after = fs::read(&path).expect("read preferences after failed save");
        assert!(result.is_err(), "allocating the sibling temp must fail");
        assert_eq!(after, before, "failed save must preserve prior preferences");
    }

    #[cfg(unix)]
    #[test]
    fn project_settings_replace_failure_preserves_previous_bytes() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = ScratchCwd::new();
        let settings_dir = scratch.path.join(".zo");
        let path = settings_dir.join("settings.local.json");
        let before = br#"{"custom":true,"model":"old"}"#;
        scratch.write_settings_local(std::str::from_utf8(before).expect("settings are utf-8"));
        fs::set_permissions(&settings_dir, fs::Permissions::from_mode(0o555))
            .expect("make settings directory read-only");

        let probe = settings_dir.join("probe");
        if fs::write(&probe, b"probe").is_ok() {
            let _ = fs::remove_file(probe);
            fs::set_permissions(&settings_dir, fs::Permissions::from_mode(0o755))
                .expect("restore settings directory permissions");
            return;
        }

        let result = super::save_project_preferences(
            &scratch.path,
            &SessionPreferences {
                model: Some("new".to_string()),
                ..SessionPreferences::default()
            },
        );

        fs::set_permissions(&settings_dir, fs::Permissions::from_mode(0o755))
            .expect("restore settings directory permissions");
        let after = fs::read(&path).expect("read settings after failed save");
        assert!(result.is_err(), "settings lock or sibling temp allocation must fail");
        assert_eq!(after, before, "failed save must preserve prior settings");
    }
}
