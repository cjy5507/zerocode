//! User-managed overlay for the built-in OAuth model catalog.
//!
//! Model preferences live in global `settings.json`, never in credential storage.
//! This module owns parsing, merging, validation, and locked atomic persistence so
//! TUI widgets and provider clients do not perform ad-hoc JSON access.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use api::{AuthRoute, ProviderKind};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const SETTINGS_KEY: &str = "modelCatalog";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CatalogProvider {
    Anthropic,
    Openai,
    Google,
}

impl CatalogProvider {
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Anthropic => "claude",
            Self::Openai => "openai",
            Self::Google => "google",
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::Openai => "OpenAI",
            Self::Google => "Google",
        }
    }

    #[must_use]
    pub const fn kind(self) -> ProviderKind {
        match self {
            Self::Anthropic => ProviderKind::Anthropic,
            Self::Openai => ProviderKind::OpenAi,
            Self::Google => ProviderKind::Google,
        }
    }

    #[must_use]
    pub fn from_key(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Some(Self::Anthropic),
            "openai" | "chatgpt" | "codex" => Some(Self::Openai),
            "google" | "gemini" => Some(Self::Google),
            _ => None,
        }
    }
}

fn canonical_model_id(provider: CatalogProvider, id: &str) -> String {
    api::provider_catalog()
        .iter()
        .find(|entry| {
            entry.provider == provider.kind()
                && (entry.alias.eq_ignore_ascii_case(id)
                    || entry.canonical_model_id.eq_ignore_ascii_case(id))
        })
        .map_or_else(|| id.trim().to_ascii_lowercase(), |entry| {
            entry.canonical_model_id.to_ascii_lowercase()
        })
}

fn same_model_id(provider: CatalogProvider, left: &str, right: &str) -> bool {
    canonical_model_id(provider, left) == canonical_model_id(provider, right)
}

fn same_row_id(left: &str, right: &str) -> bool {
    left.trim().eq_ignore_ascii_case(right.trim())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogRow {
    pub provider: CatalogProvider,
    pub id: String,
    pub display_name: String,
    pub auth_route: AuthRoute,
    pub builtin: bool,
    pub hidden: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Overlay {
    #[serde(default)]
    models: Vec<UserModel>,
    #[serde(default)]
    hidden: Vec<ModelKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UserModel {
    provider: CatalogProvider,
    id: String,
    display_name: String,
    #[serde(default = "legacy_user_auth_route")]
    auth_route: AuthRoute,
}

const fn legacy_user_auth_route() -> AuthRoute {
    AuthRoute::OAuth
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ModelKey {
    provider: CatalogProvider,
    id: String,
}

#[derive(Debug, Clone)]
pub struct ModelCatalog {
    overlay: Overlay,
    path: PathBuf,
}

impl ModelCatalog {
    pub fn load() -> io::Result<Self> {
        Self::load_from(global_settings_path())
    }

    pub fn load_from(path: PathBuf) -> io::Result<Self> {
        let root = read_settings(&path)?;
        let overlay = root
            .get(SETTINGS_KEY)
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .unwrap_or_default();
        Ok(Self { overlay, path })
    }

    #[must_use]
    pub fn rows(&self, connected: &[CatalogProvider], include_hidden: bool) -> Vec<CatalogRow> {
        let mut rows = Vec::new();
        for &(provider, id, display_name) in builtin_rows() {
            if !connected.contains(&provider) {
                continue;
            }
            let hidden = self.is_hidden(provider, id);
            if !hidden || include_hidden {
                rows.push(CatalogRow {
                    provider,
                    id: id.to_string(),
                    display_name: display_name.to_string(),
                    auth_route: AuthRoute::Auto,
                    builtin: true,
                    hidden,
                });
            }
        }
        for model in &self.overlay.models {
            if !connected.contains(&model.provider) {
                continue;
            }
            let hidden_builtin = builtin_rows().iter().any(|(provider, id, _)| {
                *provider == model.provider
                    && same_row_id(id, &model.id)
                    && self.is_hidden(model.provider, id)
            });
            if let Some(row) = rows.iter_mut().find(|row| {
                row.provider == model.provider && same_row_id(&row.id, &model.id)
            }) {
                row.display_name.clone_from(&model.display_name);
                row.auth_route = model.auth_route;
            } else if !hidden_builtin {
                rows.push(CatalogRow {
                    provider: model.provider,
                    id: model.id.clone(),
                    display_name: model.display_name.clone(),
                    auth_route: model.auth_route,
                    builtin: false,
                    hidden: false,
                });
            }
        }
        rows
    }

    #[must_use]
    pub fn builtin_hidden(&self, provider: ProviderKind, id: &str) -> bool {
        let aliases = builtin_rows()
            .iter()
            .filter(|(catalog_provider, alias, _)| {
                catalog_provider.kind() == provider
                    && canonical_model_id(*catalog_provider, alias)
                        == canonical_model_id(*catalog_provider, id)
            })
            .map(|(catalog_provider, alias, _)| (*catalog_provider, *alias))
            .collect::<Vec<_>>();
        !aliases.is_empty()
            && aliases
                .iter()
                .all(|(catalog_provider, alias)| self.is_hidden(*catalog_provider, alias))
    }

    #[must_use]
    pub fn provider_for_model(&self, id: &str) -> Option<ProviderKind> {
        let id = id.trim();
        if let Some((provider, model_id)) = id.split_once('/') {
            let provider = CatalogProvider::from_key(provider)?;
            return (!model_id.trim().is_empty()).then(|| provider.kind());
        }

        let providers = self
            .rows(
                &[
                    CatalogProvider::Anthropic,
                    CatalogProvider::Openai,
                    CatalogProvider::Google,
                ],
                false,
            )
            .into_iter()
            .filter(|row| same_row_id(&row.id, id))
            .map(|row| row.provider)
            .collect::<std::collections::HashSet<_>>();
        if providers.len() != 1 {
            return None;
        }
        providers.into_iter().next().map(CatalogProvider::kind)
    }

    #[must_use]
    pub fn auth_route_for_model(&self, id: &str) -> Option<AuthRoute> {
        let id = id.trim();
        let rows = self.rows(
            &[
                CatalogProvider::Anthropic,
                CatalogProvider::Openai,
                CatalogProvider::Google,
            ],
            false,
        );
        if let Some((provider, model_id)) = id.split_once('/') {
            let provider = CatalogProvider::from_key(provider)?;
            return rows
                .iter()
                .find(|row| row.provider == provider && same_row_id(&row.id, model_id))
                .map(|row| row.auth_route);
        }
        let mut matches = rows.iter().filter(|row| same_row_id(&row.id, id));
        let route = matches.next()?.auth_route;
        matches.next().is_none().then_some(route)
    }

    #[must_use]
    pub fn selection_token(&self, provider: CatalogProvider, id: &str) -> String {
        let collision = self
            .rows(
                &[
                    CatalogProvider::Anthropic,
                    CatalogProvider::Openai,
                    CatalogProvider::Google,
                ],
                false,
            )
            .iter()
            .any(|row| row.provider != provider && same_row_id(&row.id, id));
        if collision {
            format!("{}/{}", provider.key(), id.trim())
        } else {
            id.trim().to_string()
        }
    }

    pub fn add(&mut self, provider: CatalogProvider, id: &str, display_name: &str) -> Result<(), String> {
        self.add_with_auth_route(provider, id, display_name, AuthRoute::OAuth)
    }

    pub fn add_with_auth_route(
        &mut self,
        provider: CatalogProvider,
        id: &str,
        display_name: &str,
        auth_route: AuthRoute,
    ) -> Result<(), String> {
        validate_fields(id, display_name)?;
        if self
            .rows(&[provider], true)
            .iter()
            .any(|row| same_model_id(provider, &row.id, id))
        {
            return Err("A model with this provider and ID already exists".to_string());
        }
        self.overlay.models.push(UserModel {
            provider,
            id: id.trim().to_string(),
            display_name: display_name.trim().to_string(),
            auth_route,
        });
        self.persist().map_err(|error| error.to_string())
    }

    pub fn edit(
        &mut self,
        original: &CatalogRow,
        provider: CatalogProvider,
        id: &str,
        display_name: &str,
    ) -> Result<(), String> {
        self.edit_with_auth_route(
            original,
            provider,
            id,
            display_name,
            original.auth_route,
        )
    }

    pub fn edit_with_auth_route(
        &mut self,
        original: &CatalogRow,
        provider: CatalogProvider,
        id: &str,
        display_name: &str,
        auth_route: AuthRoute,
    ) -> Result<(), String> {
        validate_fields(id, display_name)?;
        let duplicate = self
            .rows(
                &[
                    CatalogProvider::Anthropic,
                    CatalogProvider::Openai,
                    CatalogProvider::Google,
                ],
                true,
            )
            .iter()
            .any(|row| {
                let is_original = row.provider == original.provider
                    && same_row_id(&row.id, &original.id);
                !is_original
                    && row.provider == provider
                    && same_model_id(provider, &row.id, id)
            });
        if duplicate {
            return Err("A model with this provider and ID already exists".to_string());
        }
        if original.builtin {
            if original.provider != provider || !same_row_id(&original.id, id) {
                self.hide_builtin(original.provider, &original.id);
            }
            self.upsert_user(provider, id, display_name, auth_route);
        } else if let Some(model) = self.overlay.models.iter_mut().find(|model| {
            model.provider == original.provider && same_row_id(&model.id, &original.id)
        }) {
            model.provider = provider;
            model.id = id.trim().to_string();
            model.display_name = display_name.trim().to_string();
            model.auth_route = auth_route;
        }
        self.persist().map_err(|error| error.to_string())
    }

    pub fn delete_or_hide(&mut self, row: &CatalogRow) -> Result<(), String> {
        if row.builtin {
            self.hide_builtin(row.provider, &row.id);
        } else {
            self.overlay.models.retain(|model| {
                model.provider != row.provider || !same_row_id(&model.id, &row.id)
            });
        }
        self.persist().map_err(|error| error.to_string())
    }

    pub fn restore(&mut self, row: &CatalogRow) -> Result<(), String> {
        self.overlay.hidden.retain(|key| {
            key.provider != row.provider || !same_row_id(&key.id, &row.id)
        });
        self.persist().map_err(|error| error.to_string())
    }

    fn upsert_user(
        &mut self,
        provider: CatalogProvider,
        id: &str,
        display_name: &str,
        auth_route: AuthRoute,
    ) {
        if let Some(model) = self.overlay.models.iter_mut().find(|model| {
            model.provider == provider && same_row_id(&model.id, id)
        }) {
            model.display_name = display_name.trim().to_string();
            model.auth_route = auth_route;
        } else {
            self.overlay.models.push(UserModel {
                provider,
                id: id.trim().to_string(),
                display_name: display_name.trim().to_string(),
                auth_route,
            });
        }
    }

    fn hide_builtin(&mut self, provider: CatalogProvider, id: &str) {
        if !self.is_hidden(provider, id) {
            self.overlay.hidden.push(ModelKey {
                provider,
                id: id.trim().to_string(),
            });
        }
    }

    fn is_hidden(&self, provider: CatalogProvider, id: &str) -> bool {
        self.overlay
            .hidden
            .iter()
            .any(|key| key.provider == provider && same_row_id(&key.id, id))
    }

    fn persist(&self) -> io::Result<()> {
        let _lock = SettingsFileLock::acquire(&self.path)?;
        let mut root = read_settings(&self.path)?;
        root.insert(
            SETTINGS_KEY.to_string(),
            serde_json::to_value(&self.overlay).map_err(io::Error::other)?,
        );
        let rendered = serde_json::to_string_pretty(&Value::Object(root)).map_err(io::Error::other)?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        crate::file_ops::replace_file_atomic(&self.path, format!("{rendered}\n").as_bytes())
    }
}

fn validate_fields(id: &str, display_name: &str) -> Result<(), String> {
    let id = id.trim();
    if id.is_empty() {
        return Err("Model ID cannot be empty".to_string());
    }
    if id.chars().any(char::is_control) {
        return Err("Model ID cannot contain control characters".to_string());
    }
    if display_name.trim().is_empty() {
        return Err("Display name cannot be empty".to_string());
    }
    if display_name.chars().any(char::is_control) {
        return Err("Display name cannot contain control characters".to_string());
    }
    Ok(())
}

fn global_settings_path() -> PathBuf {
    crate::default_config_home().join("settings.json")
}

fn read_settings(path: &Path) -> io::Result<Map<String, Value>> {
    match fs::read_to_string(path) {
        Ok(text) if text.trim().is_empty() => Ok(Map::new()),
        Ok(text) => serde_json::from_str::<Value>(&text)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .as_object()
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "settings.json must contain an object")),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Map::new()),
        Err(error) => Err(error),
    }
}

struct SettingsFileLock(PathBuf);

impl SettingsFileLock {
    fn acquire(settings_path: &Path) -> io::Result<Self> {
        if let Some(parent) = settings_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let path = settings_path.with_extension("json.lock");
        for _ in 0..200 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    writeln!(file, "{}", std::process::id())?;
                    return Ok(Self(path));
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => thread::sleep(Duration::from_millis(10)),
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(io::ErrorKind::WouldBlock, format!("timed out waiting for {}", path.display())))
    }
}

impl Drop for SettingsFileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[must_use]
pub fn builtin_rows() -> &'static [(CatalogProvider, &'static str, &'static str)] {
    &[
        (CatalogProvider::Anthropic, "fable", "Fable 5"),
        (CatalogProvider::Anthropic, "opus", "Opus 4.8"),
        (CatalogProvider::Anthropic, "opus[1m]", "Opus 4.8 1M"),
        (CatalogProvider::Anthropic, "sonnet", "Sonnet 5"),
        (CatalogProvider::Anthropic, "haiku", "Haiku 4.5"),
        (CatalogProvider::Openai, "gpt-5.6-sol", "GPT-5.6-Sol"),
        (CatalogProvider::Openai, "gpt-5.6-terra", "GPT-5.6-Terra"),
        (CatalogProvider::Openai, "gpt-5.6-luna", "GPT-5.6-Luna"),
        (CatalogProvider::Openai, "gpt-5.3-codex-spark", "GPT-5.3-Codex-Spark"),
        (CatalogProvider::Google, "gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
        (CatalogProvider::Google, "gemini-3.6-flash", "Gemini 3.6 Flash"),
        (CatalogProvider::Google, "gemini-3.5-flash", "Gemini 3.5 Flash"),
        (CatalogProvider::Google, "gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("zo-model-catalog-{name}-{}-{}.json", std::process::id(), std::thread::current().name().unwrap_or("test")))
    }

    #[test]
    fn overlay_add_edit_hide_restore_round_trips_without_touching_other_settings() {
        let path = path("roundtrip");
        let _ = fs::remove_file(&path);
        fs::write(&path, r#"{"theme":"dark"}"#).unwrap();
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        catalog.add(CatalogProvider::Google, "gemini-4.0-flash", "Gemini 4.0 Flash").unwrap();
        let user = catalog.rows(&[CatalogProvider::Google], false).into_iter().find(|row| row.id == "gemini-4.0-flash").unwrap();
        catalog.edit(&user, CatalogProvider::Google, "gemini-4.0-flash", "Future Flash").unwrap();
        catalog.delete_or_hide(&CatalogRow { provider: CatalogProvider::Google, id: "gemini-3.5-flash".into(), display_name: "Gemini 3.5 Flash".into(), auth_route: AuthRoute::Auto, builtin: true, hidden: false }).unwrap();
        let loaded = ModelCatalog::load_from(path.clone()).unwrap();
        assert!(!loaded.rows(&[CatalogProvider::Google], false).iter().any(|row| row.id == "gemini-3.5-flash"));
        assert_eq!(loaded.provider_for_model("gemini-4.0-flash"), Some(ProviderKind::Google));
        let hidden = loaded.rows(&[CatalogProvider::Google], true).into_iter().find(|row| row.id == "gemini-3.5-flash").unwrap();
        let mut loaded = loaded;
        loaded.restore(&hidden).unwrap();
        assert!(loaded.rows(&[CatalogProvider::Google], false).iter().any(|row| row.id == "gemini-3.5-flash"));
        let root: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(root["theme"], "dark");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn auth_route_migrates_legacy_rows_and_survives_builtin_hide_restore() {
        let path = path("auth-route");
        let _ = fs::remove_file(&path);
        fs::write(
            &path,
            r#"{"modelCatalog":{"models":[{"provider":"google","id":"gemini-legacy-flash","displayName":"Legacy Flash"}]}}"#,
        )
        .unwrap();

        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let legacy = catalog
            .rows(&[CatalogProvider::Google], false)
            .into_iter()
            .find(|row| row.id == "gemini-legacy-flash")
            .unwrap();
        assert_eq!(legacy.auth_route, AuthRoute::OAuth);
        let builtin = catalog
            .rows(&[CatalogProvider::Google], false)
            .into_iter()
            .find(|row| row.id == "gemini-3.5-flash")
            .unwrap();
        assert_eq!(builtin.auth_route, AuthRoute::Auto);

        catalog
            .edit_with_auth_route(
                &builtin,
                builtin.provider,
                &builtin.id,
                &builtin.display_name,
                AuthRoute::ApiKey,
            )
            .unwrap();
        let overridden = catalog
            .rows(&[CatalogProvider::Google], false)
            .into_iter()
            .find(|row| row.id == "gemini-3.5-flash")
            .unwrap();
        assert_eq!(overridden.auth_route, AuthRoute::ApiKey);
        catalog.delete_or_hide(&overridden).unwrap();
        let hidden = catalog
            .rows(&[CatalogProvider::Google], true)
            .into_iter()
            .find(|row| row.id == "gemini-3.5-flash")
            .unwrap();
        assert!(hidden.hidden);
        assert_eq!(hidden.auth_route, AuthRoute::ApiKey);
        catalog.restore(&hidden).unwrap();

        let loaded = ModelCatalog::load_from(path.clone()).unwrap();
        let restored = loaded
            .rows(&[CatalogProvider::Google], false)
            .into_iter()
            .find(|row| row.id == "gemini-3.5-flash")
            .unwrap();
        assert_eq!(restored.auth_route, AuthRoute::ApiKey);
        let root: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(root["modelCatalog"]["models"][1]["authRoute"], "api-key");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn promoted_builtin_preserves_existing_oauth_overlay() {
        let path = path("promoted-builtin");
        let _ = fs::remove_file(&path);
        fs::write(
            &path,
            r#"{"modelCatalog":{"models":[{"provider":"google","id":"gemini-3.6-flash","displayName":"Gemini 3.6 Flash","authRoute":"oauth"}],"hidden":[{"provider":"google","id":"gemini-3.5-flash"}]}}"#,
        )
        .unwrap();

        let catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let visible = catalog.rows(&[CatalogProvider::Google], false);
        let promoted = visible
            .iter()
            .find(|row| row.id == "gemini-3.6-flash")
            .unwrap();

        assert!(promoted.builtin);
        assert_eq!(promoted.auth_route, AuthRoute::OAuth);
        assert!(!visible.iter().any(|row| row.id == "gemini-3.5-flash"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn editing_builtin_to_future_id_persists_replacement_and_reversible_tombstone() {
        let path = path("builtin-edit");
        let _ = fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let builtin = catalog
            .rows(&[CatalogProvider::Google], false)
            .into_iter()
            .find(|row| row.id == "gemini-3.5-flash")
            .unwrap();

        catalog
            .edit(&builtin, CatalogProvider::Google, "gemini-4.0-flash", "Gemini 4.0 Flash")
            .unwrap();

        let loaded = ModelCatalog::load_from(path.clone()).unwrap();
        let visible = loaded.rows(&[CatalogProvider::Google], false);
        assert!(!visible.iter().any(|row| row.id == "gemini-3.5-flash"));
        assert!(visible.iter().any(|row| !row.builtin && row.id == "gemini-4.0-flash"));
        let hidden = loaded
            .rows(&[CatalogProvider::Google], true)
            .into_iter()
            .find(|row| row.id == "gemini-3.5-flash")
            .unwrap();
        assert!(hidden.hidden);
        let mut loaded = loaded;
        loaded.restore(&hidden).unwrap();
        assert!(loaded.rows(&[CatalogProvider::Google], false).iter().any(|row| row.id == "gemini-3.5-flash"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn same_id_builtin_override_can_be_hidden_and_is_excluded_from_smart() {
        let path = path("override-hide");
        let _ = fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let builtin = catalog.rows(&[CatalogProvider::Google], false).into_iter()
            .find(|row| row.id == "gemini-3.5-flash").unwrap();
        catalog.edit(&builtin, CatalogProvider::Google, "gemini-3.5-flash", "Preferred Flash").unwrap();
        let overridden = catalog.rows(&[CatalogProvider::Google], false).into_iter()
            .find(|row| row.id == "gemini-3.5-flash").unwrap();
        assert_eq!(overridden.display_name, "Preferred Flash");
        catalog.delete_or_hide(&overridden).unwrap();
        assert!(!catalog.rows(&[CatalogProvider::Google], false).iter().any(|row| row.id == "gemini-3.5-flash"));
        assert!(catalog.builtin_hidden(ProviderKind::Google, "gemini-3.5-flash"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn validation_rejects_empty_control_and_duplicate_ids() {
        let path = path("validation");
        let _ = fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        assert!(catalog.add(CatalogProvider::Google, "", "Empty").is_err());
        assert!(catalog.add(CatalogProvider::Google, "bad\nid", "Bad").is_err());
        assert!(catalog.add(CatalogProvider::Google, "gemini-3.5-flash", "Duplicate").is_err());
        assert!(catalog.add(CatalogProvider::Google, "gemini-flash", "Alias duplicate").is_err());
        catalog
            .add(CatalogProvider::Openai, "gemini-3.5-flash", "Provider-qualified")
            .unwrap();
        catalog
            .add(CatalogProvider::Google, "shared-future-id", "Google Shared")
            .unwrap();
        catalog
            .add(CatalogProvider::Openai, "shared-future-id", "OpenAI Shared")
            .unwrap();
        assert_eq!(
            catalog.selection_token(CatalogProvider::Google, "shared-future-id"),
            "google/shared-future-id"
        );
        assert_eq!(
            catalog.provider_for_model("google/shared-future-id"),
            Some(ProviderKind::Google)
        );
        assert_eq!(
            catalog.provider_for_model("openai/shared-future-id"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(
            catalog.selection_token(CatalogProvider::Google, "gemini-3.5-flash"),
            "google/gemini-3.5-flash"
        );
        assert_eq!(
            catalog.selection_token(CatalogProvider::Openai, "gemini-3.5-flash"),
            "openai/gemini-3.5-flash"
        );
        assert_eq!(catalog.provider_for_model("gemini-3.5-flash"), None);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn hiding_one_anthropic_alias_keeps_the_other_alias_and_smart_model_visible() {
        let path = path("alias-hide");
        let _ = fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let one_million = catalog
            .rows(&[CatalogProvider::Anthropic], false)
            .into_iter()
            .find(|row| row.id == "opus[1m]")
            .unwrap();

        catalog.delete_or_hide(&one_million).unwrap();

        let visible = catalog.rows(&[CatalogProvider::Anthropic], false);
        assert!(visible.iter().any(|row| row.id == "opus"));
        assert!(!visible.iter().any(|row| row.id == "opus[1m]"));
        assert!(!catalog.builtin_hidden(ProviderKind::Anthropic, "claude-opus-4-8"));

        let regular = visible.into_iter().find(|row| row.id == "opus").unwrap();
        catalog.delete_or_hide(&regular).unwrap();
        assert!(catalog.builtin_hidden(ProviderKind::Anthropic, "claude-opus-4-8"));

        let hidden = catalog
            .rows(&[CatalogProvider::Anthropic], true)
            .into_iter()
            .find(|row| row.id == "opus[1m]")
            .unwrap();
        catalog.restore(&hidden).unwrap();
        assert!(!catalog.builtin_hidden(ProviderKind::Anthropic, "claude-opus-4-8"));
        let _ = fs::remove_file(path);
    }
}
