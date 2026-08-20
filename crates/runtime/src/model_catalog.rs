//! User-managed overlay for the built-in OAuth model catalog.
//!
//! Model preferences live in global `settings.json`, never in credential storage.
//! This module owns parsing, merging, validation, and locked atomic persistence so
//! TUI widgets and provider clients do not perform ad-hoc JSON access.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::file_ops::SettingsFileLock;

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
    /// Alias → canonical rows in the `api` catalog's own shape, carried opaque
    /// for the same reason [`UserModel::wire`] is: that crate owns the schema
    /// and parses it. This is how an operator repoints a short alias
    /// (`gemini-flash`, `google-latest`) at a model the binary predates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    aliases: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UserModel {
    provider: CatalogProvider,
    id: String,
    display_name: String,
    #[serde(default = "legacy_user_auth_route")]
    auth_route: AuthRoute,
    /// The id(s) the provider actually serves this model under, when they
    /// differ from `id` — either a bare string or `{"low":…,"medium":…,"high":…}`
    /// for a provider that bakes the reasoning tier into the id (Gemini does).
    ///
    /// Held as an opaque value on purpose: the schema belongs to the `api`
    /// crate's model catalog, which is where it is parsed. Restating it here
    /// would make one shape two definitions that could drift. Absent means the
    /// selection id IS the wire id, which is what every model needed until
    /// Google started shipping tiered ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    wire: Option<Value>,
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

    /// The overlay's model-catalog declarations — served ids and alias rows —
    /// in the `api` catalog's shape, for the process bridge that carries them
    /// into that crate.
    ///
    /// `None` when nothing is declared, so the caller can tell "nothing to
    /// publish" from "publish an empty catalog" — the two differ under an
    /// operator-set override, which must not be clobbered by an empty mirror.
    #[must_use]
    pub fn catalog_overlay_json(&self) -> Option<String> {
        let models = self
            .overlay
            .models
            .iter()
            .filter_map(|model| {
                let wire = model.wire.as_ref()?;
                let id = model.id.trim();
                if id.is_empty() || wire.is_null() {
                    return None;
                }
                Some(serde_json::json!({
                    "provider": model.provider.key(),
                    "ids": [id],
                    "wire": wire,
                }))
            })
            .collect::<Vec<_>>();
        if models.is_empty() && self.overlay.aliases.is_empty() {
            return None;
        }
        serde_json::to_string(&serde_json::json!({
            "models": models,
            "aliases": self.overlay.aliases,
        }))
        .ok()
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
        if let Some(existing) = self
            .rows(&[provider], true)
            .into_iter()
            .find(|row| same_model_id(provider, &row.id, id))
        {
            return Err(duplicate_error(&existing));
        }
        self.overlay.models.push(UserModel {
            provider,
            id: id.trim().to_string(),
            display_name: display_name.trim().to_string(),
            auth_route,
            wire: None,
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
            .into_iter()
            .find(|row| {
                let is_original = row.provider == original.provider
                    && same_row_id(&row.id, &original.id);
                !is_original
                    && row.provider == provider
                    && same_model_id(provider, &row.id, id)
            });
        if let Some(existing) = duplicate {
            return Err(duplicate_error(&existing));
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
                wire: None,
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

/// The inline form error for an ID that another row already resolves to.
///
/// Names the offending row instead of a bare "already exists": the collision is
/// on the *canonical* id, so an alias row (`opus` → `claude-opus-5`) blocks its
/// own canonical id and the bare message read as a dead end. A hidden row is
/// invisible in the list, so say so — restoring it is the fix, not renaming.
fn duplicate_error(existing: &CatalogRow) -> String {
    let hint = if existing.hidden {
        " — hidden, press r to restore"
    } else {
        " — edit that row instead"
    };
    format!(
        "Already provided by \"{}\" (id `{}`){hint}",
        existing.display_name, existing.id
    )
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

#[must_use]
pub fn builtin_rows() -> &'static [(CatalogProvider, &'static str, &'static str)] {
    &[
        (CatalogProvider::Anthropic, "fable", "Fable 5"),
        // One row per distinct model. The `opus[1m]` alias still resolves (see
        // `MODEL_REGISTRY`) but is not listed: Opus 5's 1M window is its default
        // AND its maximum, so a second row was the same model twice.
        (CatalogProvider::Anthropic, "opus", "Opus 5"),
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

/// Human-readable family label for `model`, derived from [`builtin_rows`]
/// instead of a hardcoded release name.
///
/// Matching is on the resolved wire id, so a short alias (`opus`), a label
/// variant (`opus[1m]`), a full id (`claude-opus-5`), and a `provider/model`
/// ref all land on the same row — and bumping a catalog row to the next
/// release moves every label with it.
///
/// Returns `None` for a model the built-in catalog does not carry (a custom
/// provider's id, a user-added row). Callers then show the raw id, which is
/// honest, rather than inventing a family name for it.
#[must_use]
pub fn model_family_label(model: &str) -> Option<String> {
    let wire = api::wire_model_id(model);
    let wire = wire.trim();
    if wire.is_empty() {
        return None;
    }
    builtin_rows()
        .iter()
        .find(|(_, id, _)| api::wire_model_id(id).eq_ignore_ascii_case(wire))
        .map(|(provider, _, display_name)| match provider {
            // Anthropic rows carry the bare family name ("Opus 5"); every other
            // provider's display name already names its family ("GPT-5.6-Sol").
            CatalogProvider::Anthropic => format!("Claude {display_name}"),
            CatalogProvider::Openai | CatalogProvider::Google => (*display_name).to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A settings-declared model carries the ids its provider serves it under,
    /// and those survive a round trip through the file — this is what lets a
    /// model the binary predates be reachable without a rebuild.
    #[test]
    fn a_declared_wire_mapping_round_trips_and_publishes() {
        let path = path("wire-round-trip");
        let _ = fs::remove_file(&path);
        fs::write(
            &path,
            r#"{"modelCatalog":{"models":[
                {"provider":"google","id":"gemini-3.7-flash","displayName":"Gemini 3.7 Flash",
                 "wire":{"low":"gemini-3.7-flash-low","high":"gemini-3.7-flash-high"}},
                {"provider":"openai","id":"gpt-5.7","displayName":"GPT-5.7"}
            ]}}"#,
        )
        .unwrap();

        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let published: Value =
            serde_json::from_str(&catalog.catalog_overlay_json().expect("declared")).unwrap();
        let models = published["models"].as_array().unwrap();
        assert_eq!(models.len(), 1, "only declaring models are published");
        assert_eq!(models[0]["ids"][0], "gemini-3.7-flash");
        assert_eq!(models[0]["wire"]["high"], "gemini-3.7-flash-high");

        // Renaming through the picker must not drop the declaration: the row
        // would keep its name and quietly go back to being served under its own
        // id, which is the silent-substitution failure this whole path exists
        // to remove.
        let row = catalog
            .rows(&[CatalogProvider::Google], false)
            .into_iter()
            .find(|row| row.id == "gemini-3.7-flash")
            .expect("row");
        catalog
            .edit(&row, CatalogProvider::Google, "gemini-3.7-flash", "Flash 3.7")
            .unwrap();
        let reloaded = ModelCatalog::load_from(path.clone()).unwrap();
        let published: Value =
            serde_json::from_str(&reloaded.catalog_overlay_json().expect("still declared")).unwrap();
        assert_eq!(
            published["models"][0]["wire"]["low"],
            "gemini-3.7-flash-low",
            "an edit preserves the served ids"
        );

        let _ = fs::remove_file(&path);
    }

    /// Nothing declared is not the same as an empty catalog — the bridge needs
    /// to tell them apart so it never clobbers an operator's own export.
    #[test]
    fn no_declaration_publishes_nothing() {
        let path = path("wire-none");
        let _ = fs::remove_file(&path);
        fs::write(
            &path,
            r#"{"modelCatalog":{"models":[{"provider":"openai","id":"gpt-5.7","displayName":"GPT-5.7"}]}}"#,
        )
        .unwrap();
        let catalog = ModelCatalog::load_from(path.clone()).unwrap();
        assert!(catalog.catalog_overlay_json().is_none());
        let _ = fs::remove_file(&path);
    }

    /// Alias rows travel through the overlay too, so repointing a short name is
    /// a settings edit rather than a rebuild.
    #[test]
    fn declared_aliases_reach_the_overlay_payload() {
        let path = path("alias-overlay");
        let _ = fs::remove_file(&path);
        fs::write(
            &path,
            r#"{"modelCatalog":{"aliases":[{"alias":"google-latest","canonical":"gemini-3.7-flash","provider":"google"}]}}"#,
        )
        .unwrap();
        let catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let published: Value =
            serde_json::from_str(&catalog.catalog_overlay_json().expect("declared")).unwrap();
        assert_eq!(published["aliases"][0]["canonical"], "gemini-3.7-flash");
        assert!(published["models"].as_array().unwrap().is_empty());
        let _ = fs::remove_file(&path);
    }

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

    /// Hiding the `opus` alias row hides the canonical for the smart router, and
    /// restoring it brings it back. Anthropic ships one row per distinct model,
    /// so the alias row IS the canonical's only representation — the former
    /// `opus`/`opus[1m]` pair was the same model listed twice.
    #[test]
    fn hiding_the_opus_alias_hides_its_canonical_and_restore_brings_it_back() {
        let path = path("alias-hide");
        let _ = fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let rows = catalog.rows(&[CatalogProvider::Anthropic], false);
        assert!(
            !rows.iter().any(|row| row.id == "opus[1m]"),
            "the 1M label alias must not be a duplicate catalog row"
        );
        assert!(!catalog.builtin_hidden(ProviderKind::Anthropic, "claude-opus-5"));

        let opus = rows.into_iter().find(|row| row.id == "opus").unwrap();
        catalog.delete_or_hide(&opus).unwrap();

        let visible = catalog.rows(&[CatalogProvider::Anthropic], false);
        assert!(!visible.iter().any(|row| row.id == "opus"));
        assert!(catalog.builtin_hidden(ProviderKind::Anthropic, "claude-opus-5"));
        // A sibling canonical is unaffected by hiding Opus.
        assert!(visible.iter().any(|row| row.id == "fable"));

        let hidden = catalog
            .rows(&[CatalogProvider::Anthropic], true)
            .into_iter()
            .find(|row| row.id == "opus")
            .unwrap();
        catalog.restore(&hidden).unwrap();
        assert!(!catalog.builtin_hidden(ProviderKind::Anthropic, "claude-opus-5"));
        let _ = fs::remove_file(path);
    }

    /// The duplicate-ID error names the row that already covers the id, so the
    /// user can act on it instead of hitting a bare "already exists" dead end.
    #[test]
    fn duplicate_id_error_names_the_conflicting_row() {
        let path = path("dup-msg");
        let _ = fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        // `claude-opus-5` is what the built-in `opus` row resolves to.
        let error = catalog
            .add(CatalogProvider::Anthropic, "claude-opus-5", "Opus 5")
            .expect_err("canonical of an existing alias row must collide");
        assert!(error.contains("Opus 5"), "{error}");
        assert!(error.contains("opus"), "{error}");
        assert!(error.contains("edit that row"), "{error}");

        // Hidden rows are invisible in the picker, so the error must say so.
        let opus = catalog
            .rows(&[CatalogProvider::Anthropic], false)
            .into_iter()
            .find(|row| row.id == "opus")
            .unwrap();
        catalog.delete_or_hide(&opus).unwrap();
        let error = catalog
            .add(CatalogProvider::Anthropic, "opus", "My Opus")
            .expect_err("hidden built-in still owns the id");
        assert!(error.contains("hidden"), "{error}");

        // A genuinely new id still lands.
        catalog
            .add(CatalogProvider::Anthropic, "claude-opus-4-6", "Opus 4.6")
            .expect("unregistered older Opus is not a duplicate");
        let _ = fs::remove_file(path);
    }
}
