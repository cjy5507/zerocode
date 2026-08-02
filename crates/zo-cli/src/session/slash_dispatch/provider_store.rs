//! Read/merge/write access to the user-global `providers` array in
//! `settings.json`.
//!
//! Every provider mutation in the product — `/connect`'s presets, the custom
//! endpoint wizard, and the `/providers` manager modal — funnels through this
//! module so one file owns the on-disk shape and one set of merge rules.
//!
//! ## Why merge instead of replace
//!
//! A provider entry is keyed by its `name`, and the credential it authenticates
//! with is keyed by that entry's `auth_env`. Two `/connect` runs against the
//! same name therefore describe *the same account*, not two accounts: their
//! model lists must union rather than clobber, and the second run must not
//! erase an `auth_env`, a token-limit override, or any hand-authored field
//! (`headers`, `supports_vision`, `user_agent`, …) the first run left behind.
//! [`upsert_provider`] preserves all three, and keeps the entry at its original
//! index so the manager list does not reshuffle under the user.
//!
//! ## Scope
//!
//! Writes always land in the user-global config home (`~/.zo/settings.json` or
//! `ZO_CONFIG_HOME`), never in a worktree — see [`global_settings_path`].

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Optional per-provider token-limit overrides written alongside the models.
///
/// OpenAI-compatible `/models` responses do not advertise token limits, so a
/// custom endpoint needs a durable place to record them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ProviderTokenLimits {
    pub(crate) context_window: Option<u64>,
    pub(crate) max_output_tokens: Option<u64>,
}

/// How an upsert treats the model ids already stored under this provider name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelWrite {
    /// `/connect`: add the discovered/curated ids to whatever is already
    /// registered. Reconnecting a provider never loses models the user has.
    Union,
    /// The manager's edit form: the submitted list is authoritative, so a model
    /// the user removed from the field actually disappears.
    Replace,
}

/// One provider entry to write.
#[derive(Clone, Debug)]
pub(crate) struct ProviderDraft<'a> {
    pub(crate) name: &'a str,
    pub(crate) base_url: &'a str,
    /// `None` means "do not change the stored credential binding" under
    /// [`ModelWrite::Union`], and "keyless" under [`ModelWrite::Replace`].
    pub(crate) auth_env: Option<&'a str>,
    pub(crate) models: &'a [String],
    pub(crate) token_limits: ProviderTokenLimits,
    /// `None` keeps whatever the entry already declared.
    pub(crate) include_usage: Option<bool>,
    /// `None` keeps whatever the entry already declared.
    pub(crate) supports_reasoning_effort: Option<bool>,
    pub(crate) model_write: ModelWrite,
}

impl<'a> ProviderDraft<'a> {
    /// A `/connect`-shaped draft: union models, keep existing credentials.
    pub(crate) const fn connect(
        name: &'a str,
        base_url: &'a str,
        auth_env: Option<&'a str>,
        models: &'a [String],
    ) -> Self {
        Self {
            name,
            base_url,
            auth_env,
            models,
            token_limits: ProviderTokenLimits {
                context_window: None,
                max_output_tokens: None,
            },
            include_usage: None,
            supports_reasoning_effort: None,
            model_write: ModelWrite::Union,
        }
    }

    #[must_use]
    pub(crate) const fn with_token_limits(mut self, limits: ProviderTokenLimits) -> Self {
        self.token_limits = limits;
        self
    }

    #[must_use]
    pub(crate) const fn with_include_usage(mut self, include_usage: Option<bool>) -> Self {
        self.include_usage = include_usage;
        self
    }

    #[must_use]
    pub(crate) const fn with_reasoning_effort(mut self, supported: Option<bool>) -> Self {
        self.supports_reasoning_effort = supported;
        self
    }

    #[must_use]
    pub(crate) const fn with_model_write(mut self, model_write: ModelWrite) -> Self {
        self.model_write = model_write;
        self
    }
}

/// A provider entry as it is stored today, projected for the manager UI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StoredProvider {
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) auth_env: Option<String>,
    pub(crate) models: Vec<String>,
    pub(crate) requires_auth: bool,
    pub(crate) token_limits: ProviderTokenLimits,
    pub(crate) include_usage: Option<bool>,
    pub(crate) supports_reasoning_effort: Option<bool>,
}

/// What a delete actually removed, so the caller can report it accurately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RemovedProvider {
    pub(crate) name: String,
    pub(crate) models: Vec<String>,
    pub(crate) auth_env: Option<String>,
}

/// Env var names are also credential-store keys, so they must stay shell-legal.
pub(crate) fn valid_auth_env_name(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

/// The single write location for provider registrations: the primary global
/// config home. Deliberately *not* the project `.zo/` directory — a provider
/// registered in one repo must be visible from every other one, and project
/// settings drop their `providers` array at load time unless explicitly
/// opted in (`ConfigLoader`'s untrusted-workspace gate).
pub(crate) fn global_settings_path() -> PathBuf {
    runtime::ConfigLoader::default_for(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    )
    .config_home()
    .join("settings.json")
}

fn invalid_auth_env_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "auth_env must match [A-Za-z_][A-Za-z0-9_]*",
    )
}

/// Read the settings document at `path` as a JSON object, treating a missing or
/// blank file as an empty document.
fn read_document(path: &Path) -> io::Result<Map<String, Value>> {
    match fs::read_to_string(path) {
        Ok(contents) if contents.trim().is_empty() => Ok(Map::new()),
        Ok(contents) => serde_json::from_str::<Value>(&contents)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "settings.json must contain a JSON object",
                )
            }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Map::new()),
        Err(error) => Err(error),
    }
}

fn write_document(path: &Path, document: &Map<String, Value>) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_json::to_string_pretty(document).map_err(io::Error::other)?;
    runtime::replace_file_atomic(path, format!("{rendered}\n").as_bytes())
}

/// Run `mutate` over the `providers` array under the shared `settings.json`
/// lock, writing the document back only when the closure asks for it.
///
/// The model-catalog overlay writes the same file, so an unlocked
/// read-modify-write here would drop whichever section lost the race. Reading
/// *inside* the lock is what makes the merge in [`upsert_provider`] correct
/// under concurrent sessions, not just correct in isolation.
fn with_providers<T>(
    path: &Path,
    mutate: impl FnOnce(&mut Vec<Value>) -> io::Result<(T, bool)>,
) -> io::Result<T> {
    let _lock = runtime::SettingsFileLock::acquire(path)?;
    let mut document = read_document(path)?;
    let mut providers = providers_array(&document);
    let (outcome, changed) = mutate(&mut providers)?;
    if changed {
        document.insert("providers".to_string(), Value::Array(providers));
        write_document(path, &document)?;
    }
    Ok(outcome)
}

fn providers_array(document: &Map<String, Value>) -> Vec<Value> {
    document
        .get("providers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn entry_name(entry: &Value) -> Option<&str> {
    entry.get("name").and_then(Value::as_str)
}

/// Index of the entry `name` addresses. An exact match always wins; a
/// case-insensitive match is the fallback so `/connect DeepSeek` lands on the
/// existing `deepseek` account instead of registering a second copy that would
/// shadow it at routing time (model lookup is case-insensitive).
fn find_entry(providers: &[Value], name: &str) -> Option<usize> {
    providers
        .iter()
        .position(|entry| entry_name(entry) == Some(name))
        .or_else(|| {
            providers.iter().position(|entry| {
                entry_name(entry).is_some_and(|existing| existing.eq_ignore_ascii_case(name))
            })
        })
}

fn string_list(entry: &Value, key: &str) -> Vec<String> {
    entry
        .get(key)
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Union `additions` into `existing`, preserving first-seen order and comparing
/// case-insensitively — the same comparison the router uses to match a
/// `--model` value to a registered id, so a case variant is a duplicate here.
fn union_models(existing: &[String], additions: &[String]) -> Vec<String> {
    let mut merged: Vec<String> = Vec::with_capacity(existing.len() + additions.len());
    for model in existing.iter().chain(additions.iter()) {
        let model = model.trim();
        if model.is_empty() {
            continue;
        }
        if merged
            .iter()
            .any(|kept| kept.eq_ignore_ascii_case(model))
        {
            continue;
        }
        merged.push(model.to_string());
    }
    merged
}

fn dedupe_models(models: &[String]) -> Vec<String> {
    union_models(&[], models)
}

/// Merge `draft` into the `providers` array of the settings document at `path`.
///
/// The entry keeps its original position and every field this function does not
/// own (`headers`, `supports_vision`, `user_agent`, `client_fingerprint`, …),
/// so re-running `/connect` is non-destructive to hand-authored config.
pub(crate) fn upsert_provider(path: &Path, draft: &ProviderDraft<'_>) -> io::Result<()> {
    if draft.auth_env.is_some_and(|env| !valid_auth_env_name(env)) {
        return Err(invalid_auth_env_error());
    }
    with_providers(path, |providers| {
        merge_draft(providers, draft);
        Ok(((), true))
    })
}

fn merge_draft(providers: &mut Vec<Value>, draft: &ProviderDraft<'_>) {
    let existing = find_entry(providers, draft.name).map(|index| (index, providers[index].clone()));
    let mut entry = existing
        .as_ref()
        .and_then(|(_, value)| value.as_object().cloned())
        .unwrap_or_default();

    // The stored name is authoritative once an entry exists: re-registering
    // `DeepSeek` over `deepseek` must not rename the account out from under any
    // other settings that reference it.
    let name = existing
        .as_ref()
        .and_then(|(_, value)| entry_name(value))
        .unwrap_or(draft.name)
        .to_string();
    entry.insert("name".to_string(), Value::String(name));
    entry.insert(
        "base_url".to_string(),
        Value::String(draft.base_url.to_string()),
    );

    let previous_auth_env = entry
        .get("auth_env")
        .and_then(Value::as_str)
        .map(str::to_string);
    let auth_env = match (draft.auth_env, draft.model_write) {
        (Some(env), _) => Some(env.to_string()),
        // A `/connect` run that carries no env (a keyless local preset, or a
        // re-probe) must not silently unbind a key the user already stored.
        (None, ModelWrite::Union) => previous_auth_env,
        // An explicit edit that clears the field means keyless.
        (None, ModelWrite::Replace) => None,
    };
    match &auth_env {
        Some(env) => {
            entry.insert("auth_env".to_string(), Value::String(env.clone()));
        }
        None => {
            entry.remove("auth_env");
        }
    }
    entry.insert("requires_auth".to_string(), Value::Bool(auth_env.is_some()));

    let models = match draft.model_write {
        ModelWrite::Union => {
            let stored = existing
                .as_ref()
                .map(|(_, value)| string_list(value, "models"))
                .unwrap_or_default();
            union_models(&stored, draft.models)
        }
        ModelWrite::Replace => dedupe_models(draft.models),
    };
    entry.insert(
        "models".to_string(),
        Value::Array(models.into_iter().map(Value::String).collect()),
    );

    apply_token_limit(
        &mut entry,
        "context_window",
        draft.token_limits.context_window,
        draft.model_write,
    );
    apply_token_limit(
        &mut entry,
        "max_output_tokens",
        draft.token_limits.max_output_tokens,
        draft.model_write,
    );
    // `None` is "no opinion": leave whatever the entry already declared.
    if let Some(include_usage) = draft.include_usage {
        entry.insert("include_usage".to_string(), Value::Bool(include_usage));
    }
    if let Some(supported) = draft.supports_reasoning_effort {
        entry.insert(
            "supports_reasoning_effort".to_string(),
            Value::Bool(supported),
        );
    }

    match existing {
        Some((index, _)) => providers[index] = Value::Object(entry),
        None => providers.push(Value::Object(entry)),
    }
}

/// A `0` or absent override means "unset"; under `Union` an absent value keeps
/// whatever was stored, under `Replace` it clears the field.
fn apply_token_limit(
    entry: &mut Map<String, Value>,
    key: &str,
    value: Option<u64>,
    model_write: ModelWrite,
) {
    match value.filter(|&value| value > 0) {
        Some(value) => {
            entry.insert(key.to_string(), Value::Number(serde_json::Number::from(value)));
        }
        None => {
            if matches!(model_write, ModelWrite::Replace) {
                entry.remove(key);
            }
        }
    }
}

/// Delete the whole entry `name` addresses. `Ok(None)` means nothing matched.
pub(crate) fn remove_provider(path: &Path, name: &str) -> io::Result<Option<RemovedProvider>> {
    with_providers(path, |providers| {
        let Some(index) = find_entry(providers, name) else {
            return Ok((None, false));
        };
        let removed = providers.remove(index);
        Ok((
            Some(RemovedProvider {
                name: entry_name(&removed).unwrap_or(name).to_string(),
                models: string_list(&removed, "models"),
                auth_env: removed
                    .get("auth_env")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }),
            true,
        ))
    })
}

/// Drop one model id from a provider, leaving the entry (and its key) in place.
/// `Ok(false)` means the provider or the model was not registered.
pub(crate) fn remove_model(path: &Path, name: &str, model: &str) -> io::Result<bool> {
    with_providers(path, |providers| {
        let Some(index) = find_entry(providers, name) else {
            return Ok((false, false));
        };
        let stored = string_list(&providers[index], "models");
        let kept: Vec<String> = stored
            .iter()
            .filter(|candidate| !candidate.eq_ignore_ascii_case(model.trim()))
            .cloned()
            .collect();
        if kept.len() == stored.len() {
            return Ok((false, false));
        }
        let Some(entry) = providers[index].as_object_mut() else {
            return Ok((false, false));
        };
        entry.insert(
            "models".to_string(),
            Value::Array(kept.into_iter().map(Value::String).collect()),
        );
        Ok((true, true))
    })
}

/// Every provider registered in the global settings file, in stored order.
pub(crate) fn list_providers(path: &Path) -> io::Result<Vec<StoredProvider>> {
    let document = read_document(path)?;
    Ok(providers_array(&document)
        .iter()
        .filter_map(|entry| {
            let name = entry_name(entry)?.trim();
            if name.is_empty() {
                return None;
            }
            let auth_env = entry
                .get("auth_env")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(StoredProvider {
                name: name.to_string(),
                base_url: entry
                    .get("base_url")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                requires_auth: entry
                    .get("requires_auth")
                    .and_then(Value::as_bool)
                    .unwrap_or(auth_env.is_some()),
                auth_env,
                models: string_list(entry, "models"),
                token_limits: ProviderTokenLimits {
                    context_window: entry.get("context_window").and_then(Value::as_u64),
                    max_output_tokens: entry.get("max_output_tokens").and_then(Value::as_u64),
                },
                include_usage: entry.get("include_usage").and_then(Value::as_bool),
                supports_reasoning_effort: entry
                    .get("supports_reasoning_effort")
                    .and_then(Value::as_bool),
            })
        })
        .collect())
}

/// Whether any *other* registered provider still authenticates with `env`.
/// Deleting a shared credential would break those, so the caller offers the
/// "also delete the stored key" choice only when this is `false`.
pub(crate) fn auth_env_shared_with_other_provider(
    path: &Path,
    name: &str,
    env: &str,
) -> io::Result<bool> {
    Ok(list_providers(path)?.iter().any(|provider| {
        !provider.name.eq_ignore_ascii_case(name)
            && provider
                .auth_env
                .as_deref()
                .is_some_and(|candidate| candidate == env)
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        ModelWrite, ProviderDraft, ProviderTokenLimits, auth_env_shared_with_other_provider,
        list_providers, remove_model, remove_provider, upsert_provider,
    };
    use serde_json::Value;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zo-provider-store-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join("settings.json")
    }

    fn read(path: &std::path::Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json")
    }

    fn models(path: &std::path::Path, name: &str) -> Vec<String> {
        list_providers(path)
            .expect("list")
            .into_iter()
            .find(|provider| provider.name == name)
            .expect("provider")
            .models
    }

    #[test]
    fn reconnecting_the_same_name_unions_models_instead_of_replacing() {
        let path = temp_path("union");
        upsert_provider(
            &path,
            &ProviderDraft::connect(
                "deepseek",
                "https://api.deepseek.com",
                Some("DEEPSEEK_API_KEY"),
                &["deepseek-chat".to_string()],
            ),
        )
        .expect("first connect");
        upsert_provider(
            &path,
            &ProviderDraft::connect(
                "deepseek",
                "https://api.deepseek.com",
                Some("DEEPSEEK_API_KEY"),
                &["deepseek-reasoner".to_string(), "deepseek-chat".to_string()],
            ),
        )
        .expect("second connect");

        assert_eq!(
            models(&path, "deepseek"),
            vec![
                "deepseek-chat".to_string(),
                "deepseek-reasoner".to_string()
            ],
            "same provider name shares one key, so its models accumulate"
        );
        assert_eq!(
            read(&path)["providers"].as_array().expect("array").len(),
            1
        );
    }

    #[test]
    fn union_keeps_entry_position_and_hand_authored_fields() {
        let path = temp_path("preserve");
        std::fs::write(
            &path,
            r#"{
              "providers": [
                {"name":"deepseek","base_url":"https://api.deepseek.com","models":["deepseek-chat"]},
                {"name":"my-vllm","base_url":"http://10.0.0.5:8000/v1","auth_env":"ZO_MY_VLLM_API_KEY",
                 "models":["qwen3-32b"],"supports_vision":true,
                 "headers":{"X-Gateway":"edge"},"context_window":128000}
              ]
            }"#,
        )
        .expect("seed");

        upsert_provider(
            &path,
            &ProviderDraft::connect(
                "my-vllm",
                "http://10.0.0.5:8000/v1",
                None,
                &["qwen3-8b".to_string()],
            ),
        )
        .expect("reconnect");

        let value = read(&path);
        let entry = &value["providers"][1];
        assert_eq!(entry["name"], "my-vllm", "entry keeps its original index");
        assert_eq!(
            entry["auth_env"], "ZO_MY_VLLM_API_KEY",
            "a keyless reconnect must not unbind the stored credential"
        );
        assert_eq!(entry["requires_auth"], true);
        assert_eq!(entry["supports_vision"], true, "hand-authored field survives");
        assert_eq!(entry["headers"]["X-Gateway"], "edge");
        assert_eq!(entry["context_window"], 128_000);
        assert_eq!(entry["models"][0], "qwen3-32b");
        assert_eq!(entry["models"][1], "qwen3-8b");
    }

    #[test]
    fn case_variant_name_merges_into_the_existing_account() {
        let path = temp_path("case");
        upsert_provider(
            &path,
            &ProviderDraft::connect("deepseek", "https://api.deepseek.com", None, &[
                "deepseek-chat".to_string(),
            ]),
        )
        .expect("first");
        upsert_provider(
            &path,
            &ProviderDraft::connect("DeepSeek", "https://api.deepseek.com", None, &[
                "DeepSeek-Chat".to_string(),
                "deepseek-reasoner".to_string(),
            ]),
        )
        .expect("second");

        let value = read(&path);
        let providers = value["providers"].as_array().expect("array");
        assert_eq!(providers.len(), 1, "a case variant is the same account");
        assert_eq!(providers[0]["name"], "deepseek", "stored casing wins");
        assert_eq!(
            models(&path, "deepseek"),
            vec![
                "deepseek-chat".to_string(),
                "deepseek-reasoner".to_string()
            ],
            "a case-variant model id is a duplicate, not a new model"
        );
    }

    #[test]
    fn replace_mode_lets_an_edit_drop_models_and_unbind_the_key() {
        let path = temp_path("replace");
        upsert_provider(
            &path,
            &ProviderDraft::connect("local", "http://localhost:8000/v1", Some("ZO_LOCAL_API_KEY"), &[
                "a".to_string(),
                "b".to_string(),
            ]),
        )
        .expect("seed");
        upsert_provider(
            &path,
            &ProviderDraft::connect("local", "http://localhost:8000/v1", None, &["b".to_string()])
                .with_model_write(ModelWrite::Replace),
        )
        .expect("edit");

        let value = read(&path);
        let entry = &value["providers"][0];
        assert_eq!(entry["models"].as_array().expect("models").len(), 1);
        assert_eq!(entry["models"][0], "b");
        assert!(entry.get("auth_env").is_none(), "an explicit edit can go keyless");
        assert_eq!(entry["requires_auth"], false);
    }

    #[test]
    fn removing_a_provider_reports_what_it_dropped_and_leaves_siblings() {
        let path = temp_path("remove");
        upsert_provider(
            &path,
            &ProviderDraft::connect("deepseek", "https://api.deepseek.com", Some("DEEPSEEK_API_KEY"), &[
                "deepseek-chat".to_string(),
            ]),
        )
        .expect("a");
        upsert_provider(
            &path,
            &ProviderDraft::connect("ollama", "http://localhost:11434/v1", None, &[
                "llama3.1".to_string(),
            ]),
        )
        .expect("b");

        let removed = remove_provider(&path, "DEEPSEEK")
            .expect("remove")
            .expect("matched case-insensitively");
        assert_eq!(removed.name, "deepseek");
        assert_eq!(removed.models, vec!["deepseek-chat".to_string()]);
        assert_eq!(removed.auth_env.as_deref(), Some("DEEPSEEK_API_KEY"));

        let remaining = list_providers(&path).expect("list");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].name, "ollama");
        assert!(remove_provider(&path, "deepseek").expect("second remove").is_none());
    }

    #[test]
    fn removing_one_model_keeps_the_provider_and_its_key() {
        let path = temp_path("remove-model");
        upsert_provider(
            &path,
            &ProviderDraft::connect("kimi", "https://api.moonshot.ai/v1", Some("MOONSHOT_API_KEY"), &[
                "kimi-k2".to_string(),
                "kimi-k1".to_string(),
            ]),
        )
        .expect("seed");

        assert!(remove_model(&path, "kimi", "KIMI-K1").expect("remove"));
        assert!(!remove_model(&path, "kimi", "absent").expect("no-op"));

        let providers = list_providers(&path).expect("list");
        assert_eq!(providers[0].models, vec!["kimi-k2".to_string()]);
        assert_eq!(providers[0].auth_env.as_deref(), Some("MOONSHOT_API_KEY"));
    }

    #[test]
    fn shared_auth_env_is_detected_so_a_delete_cannot_orphan_a_sibling() {
        let path = temp_path("shared-env");
        for name in ["gateway-a", "gateway-b"] {
            upsert_provider(
                &path,
                &ProviderDraft::connect(name, "https://gw.example/v1", Some("ZO_GW_API_KEY"), &[
                    "m".to_string(),
                ]),
            )
            .expect("seed");
        }
        assert!(
            auth_env_shared_with_other_provider(&path, "gateway-a", "ZO_GW_API_KEY")
                .expect("check")
        );
        remove_provider(&path, "gateway-b").expect("remove");
        assert!(
            !auth_env_shared_with_other_provider(&path, "gateway-a", "ZO_GW_API_KEY")
                .expect("check")
        );
    }

    #[test]
    fn invalid_auth_env_is_rejected_before_touching_the_file() {
        let path = temp_path("invalid-env");
        let error = upsert_provider(
            &path,
            &ProviderDraft::connect("bad", "https://example.com/v1", Some("FOO=bar"), &[]),
        )
        .expect_err("rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!path.exists(), "a rejected draft must not create the file");
    }

    #[test]
    fn reasoning_effort_capability_is_written_and_read_back() {
        let path = temp_path("reasoning-effort");
        upsert_provider(
            &path,
            &ProviderDraft::connect("agent", "https://agentrouter.org/v1", Some("ZO_AGENT_API_KEY"), &[
                "kimi-k3".to_string(),
            ])
            .with_reasoning_effort(Some(true)),
        )
        .expect("seed");

        assert_eq!(read(&path)["providers"][0]["supports_reasoning_effort"], true);
        let stored = list_providers(&path).expect("list");
        assert_eq!(stored[0].supports_reasoning_effort, Some(true));

        // A reconnect that declares nothing must not silently drop the
        // capability — same contract the token limits have.
        upsert_provider(
            &path,
            &ProviderDraft::connect("agent", "https://agentrouter.org/v1", Some("ZO_AGENT_API_KEY"), &[]),
        )
        .expect("reconnect");
        assert_eq!(read(&path)["providers"][0]["supports_reasoning_effort"], true);

        // Turning it back off is a real write, not a no-op.
        upsert_provider(
            &path,
            &ProviderDraft::connect("agent", "https://agentrouter.org/v1", Some("ZO_AGENT_API_KEY"), &[])
                .with_reasoning_effort(Some(false)),
        )
        .expect("disable");
        assert_eq!(
            read(&path)["providers"][0]["supports_reasoning_effort"],
            false
        );
    }

    #[test]
    fn token_limits_survive_a_reconnect_that_declares_none() {
        let path = temp_path("limits");
        upsert_provider(
            &path,
            &ProviderDraft::connect("x", "https://api.x.ai/v1", Some("XAI_API_KEY"), &[
                "grok-4.5".to_string(),
            ])
            .with_token_limits(ProviderTokenLimits {
                context_window: Some(256_000),
                max_output_tokens: Some(32_000),
            })
            .with_include_usage(Some(false)),
        )
        .expect("seed");
        upsert_provider(
            &path,
            &ProviderDraft::connect("x", "https://api.x.ai/v1", Some("XAI_API_KEY"), &[]),
        )
        .expect("reconnect");

        let value = read(&path);
        let entry = &value["providers"][0];
        assert_eq!(entry["context_window"], 256_000);
        assert_eq!(entry["max_output_tokens"], 32_000);
        assert_eq!(entry["include_usage"], false);
    }
}
