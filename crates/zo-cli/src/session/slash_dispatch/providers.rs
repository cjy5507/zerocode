//! `/providers` — the register / edit / delete manager for model providers.
//!
//! The modal in [`crate::tui::modals::provider_manager`] owns presentation and
//! nothing else; every read and every write goes through here, so the global
//! settings file stays the single source of truth and one code path decides
//! what "the same provider" means.

use std::fmt::Write as _;

use zo_cli::tui::modals::{
    CustomProviderPrefill, ProviderKeyState, ProviderManagerAction,
    ProviderManagerRow, ProviderOrigin,
};

use super::auth::{ConnectReport, live_refresh_note};
use super::context::DispatchCtx;
use super::output::CommandOutput;
use super::provider_store::{self, StoredProvider};

/// Open the manager, listing what is connected and registered right now.
///
/// This is the one surface for the whole question — `/connect`, `/login`, and
/// `/logout` with no argument all land here rather than each owning a partial
/// view of it.
pub(super) fn providers(ctx: &mut DispatchCtx<'_>) -> CommandOutput {
    match manager_rows() {
        Ok(rows) => {
            ctx.app.open_provider_manager_modal(
                super::auth::account_rows(),
                rows,
                settings_path_display(),
            );
            CommandOutput::Quiet
        }
        Err(error) => CommandOutput::error(format!("Providers: failed to read settings: {error}")),
    }
}

/// Plain-text listing for the headless REPL, which has no modal to open.
/// Mirrors the manager's tree so the two surfaces read the same.
pub(crate) fn providers_text_report() -> String {
    let path = settings_path_display();
    let rows = match manager_rows() {
        Ok(rows) => rows,
        Err(error) => return format!("Providers: failed to read settings: {error}"),
    };
    let mut out = String::from("Accounts");
    for account in super::auth::account_rows() {
        let state = if account.connected {
            "connected"
        } else {
            "not connected"
        };
        let _ = write!(out, "\n  {} — {state} · {}", account.label, account.detail);
    }
    if rows.is_empty() {
        let _ = write!(
            out,
            "\n\nRegistered providers — none.\n  global · {path}\n  Add one with /connect <preset> or /connect https://host/v1"
        );
        return out;
    }
    let _ = write!(
        out,
        "\n\nRegistered providers ({})\n  global · {path}",
        rows.len()
    );
    for row in rows {
        let scope = match row.origin {
            ProviderOrigin::GlobalSettings => "",
            ProviderOrigin::EnvOverride => "  (env, read-only)",
        };
        let key = match row.key_state {
            ProviderKeyState::Stored => "key saved",
            ProviderKeyState::FromEnv => "key from env",
            ProviderKeyState::Keyless => "keyless",
            ProviderKeyState::Missing => "KEY MISSING",
        };
        let _ = write!(out, "\n\n  {} — {key}{scope}", row.name);
        if !row.base_url.is_empty() {
            let _ = write!(out, "\n    {}", row.base_url);
        }
        if let Some(env) = &row.auth_env {
            let _ = write!(out, "\n    auth_env: {env}");
        }
        for model in &row.models {
            let _ = write!(out, "\n      {model}");
        }
    }
    out.push_str("\n\n  Manage them interactively with /providers in the TUI.");
    out
}

/// Path shown in the manager header so the user can see the registration is
/// machine-wide rather than tied to this repository.
pub(crate) fn settings_path_display() -> String {
    provider_store::global_settings_path().display().to_string()
}

/// Everything the manager lists: the editable global registrations first, then
/// any provider that only exists because of an operator `ZO_CUSTOM_PROVIDERS`
/// export — shown so the list matches what `/model` offers, but read-only,
/// because zo does not own that variable.
pub(crate) fn manager_rows() -> std::io::Result<Vec<ProviderManagerRow>> {
    let path = provider_store::global_settings_path();
    let stored = provider_store::list_providers(&path)?;
    let mut rows: Vec<ProviderManagerRow> = stored
        .iter()
        .map(|provider| ProviderManagerRow {
            name: provider.name.clone(),
            base_url: provider.base_url.clone(),
            auth_env: provider.auth_env.clone(),
            models: provider.models.clone(),
            key_state: key_state(provider),
            origin: ProviderOrigin::GlobalSettings,
            key_shared: provider.auth_env.as_deref().is_some_and(|env| {
                provider_store::auth_env_shared_with_other_provider(&path, &provider.name, env)
                    .unwrap_or(false)
            }),
        })
        .collect();

    for (name, models) in api::custom_provider_catalog() {
        if rows
            .iter()
            .any(|row| row.name.eq_ignore_ascii_case(name))
        {
            continue;
        }
        rows.push(ProviderManagerRow {
            name: name.to_string(),
            base_url: String::new(),
            auth_env: None,
            models,
            key_state: ProviderKeyState::Keyless,
            origin: ProviderOrigin::EnvOverride,
            key_shared: false,
        });
    }
    Ok(rows)
}

/// Where a provider's credential currently comes from — the difference between
/// "registered and usable" and "registered but every request would 401".
fn key_state(provider: &StoredProvider) -> ProviderKeyState {
    let Some(env) = provider.auth_env.as_deref().filter(|_| provider.requires_auth) else {
        return ProviderKeyState::Keyless;
    };
    if std::env::var(env)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return ProviderKeyState::FromEnv;
    }
    if api::load_openai_compat_api_key(env)
        .ok()
        .flatten()
        .is_some()
    {
        return ProviderKeyState::Stored;
    }
    ProviderKeyState::Missing
}

/// Outcome of one manager action: a report line plus whether the list changed.
pub(crate) struct ManagerOutcome {
    pub(crate) report: ConnectReport,
    /// `false` for actions the host turns into another modal (add / edit), so
    /// it knows not to re-list behind the form it just opened.
    pub(crate) refresh_list: bool,
}

/// Apply a manager action. Only `DeleteProvider`, `DeleteModel`, and
/// `Rediscover` reach here; `Add` and `Edit` open the wizard instead and are
/// handled by the host before this is called.
pub(crate) fn apply(action: &ProviderManagerAction) -> Option<ManagerOutcome> {
    match action {
        // Handled by the host: these open another surface rather than writing.
        ProviderManagerAction::Add
        | ProviderManagerAction::Edit { .. }
        | ProviderManagerAction::ConnectAccount { .. } => None,
        ProviderManagerAction::DisconnectAccount { id } => {
            Some(super::auth::disconnect_account(id))
        }
        ProviderManagerAction::DeleteProvider { name, delete_key } => {
            Some(delete_provider(name, *delete_key))
        }
        ProviderManagerAction::DeleteModel { name, model } => Some(delete_model(name, model)),
        ProviderManagerAction::Rediscover { name } => Some(rediscover(name)),
    }
}

fn delete_provider(name: &str, delete_key: bool) -> ManagerOutcome {
    let path = provider_store::global_settings_path();
    let removed = match provider_store::remove_provider(&path, name) {
        Ok(Some(removed)) => removed,
        Ok(None) => {
            return ManagerOutcome {
                report: ConnectReport::Warn(format!("{name}: not registered — nothing to delete.")),
                refresh_list: true,
            };
        }
        Err(error) => {
            return ManagerOutcome {
                report: ConnectReport::Error(format!("{name}: failed to update settings: {error}")),
                refresh_list: true,
            };
        }
    };

    // Take the key only when asked *and* only when no surviving provider still
    // authenticates with it — `auth_env` is a credential name, not a provider
    // id, so a shared one belongs to more than this entry.
    let mut key_note = String::new();
    if delete_key {
        if let Some(env) = removed.auth_env.as_deref() {
            let shared = provider_store::auth_env_shared_with_other_provider(&path, &removed.name, env)
                .unwrap_or(true);
            if shared {
                key_note = format!("\n  Kept {env}: another provider still uses it.");
            } else {
                key_note = match api::delete_openai_compat_api_key(env) {
                    Ok(true) => format!("\n  Deleted the stored {env}."),
                    Ok(false) => format!("\n  No stored {env} to delete."),
                    Err(error) => format!("\n  Failed to delete {env}: {error}"),
                };
            }
        }
    }

    let refresh_note = live_refresh_note();
    ManagerOutcome {
        report: ConnectReport::Info(format!(
            "{}: removed {} model(s) from {} and {refresh_note}.{key_note}",
            removed.name,
            removed.models.len(),
            path.display(),
        )),
        refresh_list: true,
    }
}

fn delete_model(name: &str, model: &str) -> ManagerOutcome {
    let path = provider_store::global_settings_path();
    match provider_store::remove_model(&path, name, model) {
        Ok(true) => {
            let refresh_note = live_refresh_note();
            ManagerOutcome {
                report: ConnectReport::Info(format!(
                    "{name}: removed model {model} and {refresh_note}."
                )),
                refresh_list: true,
            }
        }
        Ok(false) => ManagerOutcome {
            report: ConnectReport::Warn(format!("{name}: {model} was not registered.")),
            refresh_list: true,
        },
        Err(error) => ManagerOutcome {
            report: ConnectReport::Error(format!("{name}: failed to update settings: {error}")),
            refresh_list: true,
        },
    }
}

/// Re-probe the endpoint's `/models` and union whatever it advertises into the
/// entry, so a provider that gained models upstream picks them up without the
/// user retyping ids.
fn rediscover(name: &str) -> ManagerOutcome {
    let path = provider_store::global_settings_path();
    let stored = match provider_store::list_providers(&path) {
        Ok(providers) => providers
            .into_iter()
            .find(|provider| provider.name.eq_ignore_ascii_case(name)),
        Err(error) => {
            return ManagerOutcome {
                report: ConnectReport::Error(format!("{name}: failed to read settings: {error}")),
                refresh_list: true,
            };
        }
    };
    let Some(stored) = stored else {
        return ManagerOutcome {
            report: ConnectReport::Warn(format!("{name}: not registered.")),
            refresh_list: true,
        };
    };

    let discovered = match stored.auth_env.as_deref().and_then(stored_or_env_key) {
        Some(key) => api::sync_bridge::run_blocking(api::discover_models_with_bearer(
            &stored.base_url,
            &key,
        ))
        .unwrap_or_default(),
        None => api::sync_bridge::run_blocking(api::discover_models(&stored.base_url)),
    };
    if discovered.is_empty() {
        return ManagerOutcome {
            report: ConnectReport::Warn(format!(
                "{}: no models advertised at {}. Its registered models were kept.",
                stored.name, stored.base_url
            )),
            refresh_list: true,
        };
    }

    let before = stored.models.len();
    let draft = provider_store::ProviderDraft::connect(
        &stored.name,
        &stored.base_url,
        stored.auth_env.as_deref(),
        &discovered,
    );
    if let Err(error) = provider_store::upsert_provider(&path, &draft) {
        return ManagerOutcome {
            report: ConnectReport::Error(format!(
                "{}: failed to update settings: {error}",
                stored.name
            )),
            refresh_list: true,
        };
    }
    let after = provider_store::list_providers(&path)
        .ok()
        .and_then(|providers| {
            providers
                .into_iter()
                .find(|provider| provider.name == stored.name)
        })
        .map_or(before, |provider| provider.models.len());
    let refresh_note = live_refresh_note();
    ManagerOutcome {
        report: ConnectReport::Info(format!(
            "{}: {} model(s) advertised, {} new and {refresh_note}.",
            stored.name,
            discovered.len(),
            after.saturating_sub(before),
        )),
        refresh_list: true,
    }
}

fn stored_or_env_key(env: &str) -> Option<String> {
    std::env::var(env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| api::load_openai_compat_api_key(env).ok().flatten())
}

/// Field values for the edit form: what is stored, plus whether a credential
/// for it already resolves so the form need not demand the secret again.
pub(crate) fn edit_prefill(name: &str) -> Option<CustomProviderPrefill> {
    let path = provider_store::global_settings_path();
    let stored = provider_store::list_providers(&path)
        .ok()?
        .into_iter()
        .find(|provider| provider.name.eq_ignore_ascii_case(name))?;
    // The same resolution `key_state` and the runtime use: an exported variable
    // authenticates just as well as a stored key. Consulting only the store
    // reported an env-authenticated provider as keyless, and the form then
    // refused to save an unrelated edit until the secret was pasted again.
    let has_existing_key = stored
        .auth_env
        .as_deref()
        .and_then(stored_or_env_key)
        .is_some();
    Some(CustomProviderPrefill {
        name: stored.name,
        base_url: stored.base_url,
        auth_env: stored.auth_env,
        models: stored.models,
        context_window: stored.token_limits.context_window,
        max_output_tokens: stored.token_limits.max_output_tokens,
        include_usage: stored.include_usage,
        supports_reasoning_effort: stored.supports_reasoning_effort,
        has_existing_key,
    })
}

#[cfg(test)]
mod tests {
    use super::edit_prefill;

    struct EnvVarGuard {
        key: &'static str,
        old: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let old = std::env::var(key).ok();
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, old }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.old.as_deref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// Register one keyed provider in an isolated config home and read back the
    /// edit form's view of it, with `ZO_ENV_ONLY_API_KEY` exported or not.
    fn prefill_with_exported_key(tag: &str, exported: Option<&str>) -> bool {
        let home = std::env::temp_dir().join(format!(
            "zo-edit-prefill-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).expect("config home");
        let home_str = home.to_str().expect("utf8 config home").to_string();

        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", None);
        let _home_guard = EnvVarGuard::set("HOME", Some(&home_str));
        // Exported only — nothing is ever written to the credential store.
        let _key = EnvVarGuard::set("ZO_ENV_ONLY_API_KEY", exported);

        std::fs::write(
            home.join("settings.json"),
            r#"{"providers":[{"name":"env-only","base_url":"https://env.example/v1","auth_env":"ZO_ENV_ONLY_API_KEY","requires_auth":true,"models":["m1"]}]}"#,
        )
        .expect("settings written");

        let prefill = edit_prefill("env-only").expect("the provider is registered");
        let _ = std::fs::remove_dir_all(&home);
        prefill.has_existing_key
    }

    /// A provider authenticated from the process environment must open the edit
    /// form with its credential already satisfied.
    ///
    /// `key_state` counts an exported variable as `FromEnv` and the runtime
    /// authenticates from it, so consulting only the credential store here made
    /// the form demand the secret again — changing one model id became
    /// impossible without the key on hand.
    #[test]
    fn an_exported_key_counts_as_the_providers_existing_credential() {
        let _lock = crate::test_env_lock();
        assert!(prefill_with_exported_key("exported", Some("sk-from-env")));
    }

    /// With the variable unset and nothing stored, there is genuinely no key —
    /// the form must still ask for one rather than silently saving a provider
    /// that cannot authenticate.
    #[test]
    fn a_provider_with_no_credential_anywhere_still_reports_none() {
        let _lock = crate::test_env_lock();
        assert!(!prefill_with_exported_key("absent", None));
    }
}
