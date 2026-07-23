//! `PluginManager` — discovery + install/update/uninstall + state I/O.
//!
//! Owns the [`PluginManagerConfig`] and the cached [`PluginRegistry`]
//! (and the "installed-from-disk" snapshot used during sync). Every
//! observable change to `~/.zo/plugins/installed.json` or the
//! enabled-flag in `settings.json` flows through one of this type's
//! methods so the caches stay in sync.
//!
//! The manager composes helpers from the rest of the crate:
//! [`builtin::load_plugin_definition`] resolves a discovered root,
//! [`install::*`] handles parsing + materialising + copying sources,
//! [`manifest_io::plugin_manifest_path`] locates `plugin.json`, and
//! [`path_validators::*`] (re-exported via the registered plugin) keeps
//! command paths valid at activation time.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::builtin::load_plugin_definition;
use super::install::{
    copy_dir_all, discover_plugin_dirs, ensure_object, git_head_commit, hash_plugin_tree,
    materialize_source, parse_install_source, resolve_local_source, update_settings_json,
    write_atomic,
};
use super::manifest::{BUNDLED_MARKETPLACE, EXTERNAL_MARKETPLACE};
use super::manifest_io::plugin_manifest_path;
use super::registry::PluginDiscovery;
use super::util::{describe_install_source, plugin_id, sanitize_plugin_id, unix_time_ms};
use super::{
    builtin_plugins, load_plugin_from_directory, InstalledPluginRecord, InstalledPluginRegistry,
    Plugin, PluginDefinition, PluginError, PluginHooks, PluginInstallSource, PluginKind,
    PluginLoadFailure, PluginManagerConfig, PluginManifest, PluginMetadata, PluginRegistry,
    PluginRegistryReport, PluginSummary, PluginTool, RegisteredPlugin,
};

pub(crate) const SETTINGS_FILE_NAME: &str = "settings.json";
pub(crate) const REGISTRY_FILE_NAME: &str = "installed.json";

#[derive(Debug, Clone)]
pub struct PluginManager {
    config: PluginManagerConfig,
    cache: Arc<Mutex<PluginManagerCache>>,
}

#[derive(Debug, Default)]
struct PluginManagerCache {
    registry: Option<PluginRegistry>,
    installed_registry: Option<PluginRegistry>,
}

impl PartialEq for PluginManager {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
    }
}

impl Eq for PluginManager {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallOutcome {
    pub plugin_id: String,
    pub version: String,
    pub install_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateOutcome {
    pub plugin_id: String,
    pub old_version: String,
    pub new_version: String,
    pub install_path: PathBuf,
}

impl PluginManager {
    #[must_use]
    pub fn new(config: PluginManagerConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(PluginManagerCache::default())),
        }
    }

    #[must_use]
    pub fn bundled_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bundled")
    }

    #[must_use]
    pub fn install_root(&self) -> PathBuf {
        self.config
            .install_root
            .clone()
            .unwrap_or_else(|| self.config.config_home.join("plugins").join("installed"))
    }

    #[must_use]
    pub fn registry_path(&self) -> PathBuf {
        self.config.registry_path.clone().unwrap_or_else(|| {
            self.config
                .config_home
                .join("plugins")
                .join(REGISTRY_FILE_NAME)
        })
    }

    #[must_use]
    pub fn settings_path(&self) -> PathBuf {
        self.config.config_home.join(SETTINGS_FILE_NAME)
    }

    pub fn plugin_registry(&self) -> Result<PluginRegistry, PluginError> {
        if let Some(registry) = self.cached_registry() {
            return Ok(registry);
        }

        let registry = self.plugin_registry_report()?.into_registry()?;
        self.store_cached_registry(registry.clone());
        Ok(registry)
    }

    pub fn plugin_registry_report(&self) -> Result<PluginRegistryReport, PluginError> {
        if let Some(registry) = self.cached_registry() {
            return Ok(PluginRegistryReport::new(registry, Vec::new()));
        }

        self.sync_bundled_plugins()?;

        let mut discovery = PluginDiscovery::default();
        discovery.plugins.extend(builtin_plugins());

        let installed = self.discover_installed_plugins_with_failures()?;
        discovery.extend(installed);

        let external =
            self.discover_external_directory_plugins_with_failures(&discovery.plugins)?;
        discovery.extend(external);

        let report = self.build_registry_report(discovery);
        if !report.has_failures() {
            self.store_cached_registry(report.registry().clone());
        }
        Ok(report)
    }

    pub fn list_plugins(&self) -> Result<Vec<PluginSummary>, PluginError> {
        Ok(self.plugin_registry()?.summaries())
    }

    pub fn list_installed_plugins(&self) -> Result<Vec<PluginSummary>, PluginError> {
        Ok(self.installed_plugin_registry()?.summaries())
    }

    pub fn discover_plugins(&self) -> Result<Vec<PluginDefinition>, PluginError> {
        Ok(self
            .plugin_registry()?
            .plugins
            .into_iter()
            .map(|plugin| plugin.definition)
            .collect())
    }

    pub fn aggregated_hooks(&self) -> Result<PluginHooks, PluginError> {
        self.plugin_registry()?.aggregated_hooks()
    }

    pub fn aggregated_tools(&self) -> Result<Vec<PluginTool>, PluginError> {
        self.plugin_registry()?.aggregated_tools()
    }

    pub fn validate_plugin_source(&self, source: &str) -> Result<PluginManifest, PluginError> {
        let path = resolve_local_source(source)?;
        load_plugin_from_directory(&path)
    }

    pub fn install(&mut self, source: &str) -> Result<InstallOutcome, PluginError> {
        let install_source = parse_install_source(source)?;
        let temp_root = self.install_root().join(".tmp");
        let staged_source = materialize_source(&install_source, &temp_root)?;
        let is_git_source = matches!(install_source, PluginInstallSource::GitUrl { .. });
        let manifest = load_plugin_from_directory(&staged_source)?;
        // Record the exact commit checked out (provenance) before the staged
        // git tree is cleaned up.
        let resolved_commit = is_git_source
            .then(|| git_head_commit(&staged_source))
            .flatten();

        let plugin_id = plugin_id(&manifest.name, EXTERNAL_MARKETPLACE);
        let install_path = self.install_root().join(sanitize_plugin_id(&plugin_id));
        if install_path.exists() {
            fs::remove_dir_all(&install_path)?;
        }
        copy_dir_all(&staged_source, &install_path)?;
        if is_git_source {
            let _ = fs::remove_dir_all(&staged_source);
        }
        // Integrity baseline over the materialised copy, checked on every load.
        let content_sha256 = Some(hash_plugin_tree(&install_path)?);

        let now = unix_time_ms();
        let record = InstalledPluginRecord {
            kind: PluginKind::External,
            id: plugin_id.clone(),
            name: manifest.name,
            version: manifest.version.clone(),
            description: manifest.description,
            install_path: install_path.clone(),
            source: install_source,
            installed_at_unix_ms: now,
            updated_at_unix_ms: now,
            resolved_commit,
            content_sha256,
        };

        let mut registry = self.load_registry()?;
        registry.plugins.insert(plugin_id.clone(), record);
        self.store_registry(&registry)?;
        self.write_enabled_state(&plugin_id, Some(true))?;
        self.config.enabled_plugins.insert(plugin_id.clone(), true);
        self.invalidate_cache();

        Ok(InstallOutcome {
            plugin_id,
            version: manifest.version,
            install_path,
        })
    }

    pub fn enable(&mut self, plugin_id: &str) -> Result<(), PluginError> {
        self.ensure_known_plugin(plugin_id)?;
        self.write_enabled_state(plugin_id, Some(true))?;
        self.config
            .enabled_plugins
            .insert(plugin_id.to_string(), true);
        self.invalidate_cache();
        Ok(())
    }

    pub fn disable(&mut self, plugin_id: &str) -> Result<(), PluginError> {
        self.ensure_known_plugin(plugin_id)?;
        self.write_enabled_state(plugin_id, Some(false))?;
        self.config
            .enabled_plugins
            .insert(plugin_id.to_string(), false);
        self.invalidate_cache();
        Ok(())
    }

    pub fn uninstall(&mut self, plugin_id: &str) -> Result<(), PluginError> {
        let mut registry = self.load_registry()?;
        let record = registry.plugins.remove(plugin_id).ok_or_else(|| {
            PluginError::NotFound(format!("plugin `{plugin_id}` is not installed"))
        })?;
        if record.kind == PluginKind::Bundled {
            registry.plugins.insert(plugin_id.to_string(), record);
            return Err(PluginError::CommandFailed(format!(
                "plugin `{plugin_id}` is bundled and managed automatically; disable it instead"
            )));
        }
        if record.install_path.exists() {
            fs::remove_dir_all(&record.install_path)?;
        }
        self.store_registry(&registry)?;
        self.write_enabled_state(plugin_id, None)?;
        self.config.enabled_plugins.remove(plugin_id);
        self.invalidate_cache();
        Ok(())
    }

    pub fn update(&mut self, plugin_id: &str) -> Result<UpdateOutcome, PluginError> {
        let mut registry = self.load_registry()?;
        let record = registry.plugins.get(plugin_id).cloned().ok_or_else(|| {
            PluginError::NotFound(format!("plugin `{plugin_id}` is not installed"))
        })?;

        let temp_root = self.install_root().join(".tmp");
        let staged_source = materialize_source(&record.source, &temp_root)?;
        let is_git_source = matches!(record.source, PluginInstallSource::GitUrl { .. });
        let manifest = load_plugin_from_directory(&staged_source)?;
        let resolved_commit = is_git_source
            .then(|| git_head_commit(&staged_source))
            .flatten();

        if record.install_path.exists() {
            fs::remove_dir_all(&record.install_path)?;
        }
        copy_dir_all(&staged_source, &record.install_path)?;
        if is_git_source {
            let _ = fs::remove_dir_all(&staged_source);
        }
        let content_sha256 = Some(hash_plugin_tree(&record.install_path)?);

        let updated_record = InstalledPluginRecord {
            version: manifest.version.clone(),
            description: manifest.description,
            updated_at_unix_ms: unix_time_ms(),
            resolved_commit,
            content_sha256,
            ..record.clone()
        };
        registry
            .plugins
            .insert(plugin_id.to_string(), updated_record);
        self.store_registry(&registry)?;
        self.invalidate_cache();

        Ok(UpdateOutcome {
            plugin_id: plugin_id.to_string(),
            old_version: record.version,
            new_version: manifest.version,
            install_path: record.install_path,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn discover_installed_plugins_with_failures(&self) -> Result<PluginDiscovery, PluginError> {
        let mut registry = self.load_registry()?;
        let mut discovery = PluginDiscovery::default();
        let mut seen_ids = BTreeSet::<String>::new();
        let mut seen_paths = BTreeSet::<PathBuf>::new();
        let mut stale_registry_ids = Vec::new();

        let primary_install_root = self.install_root();
        for install_path in discover_plugin_dirs(&primary_install_root)? {
            let matched_record = registry
                .plugins
                .values()
                .find(|record| record.install_path == install_path);
            let kind = matched_record.map_or(PluginKind::External, |record| record.kind);
            let source = matched_record.map_or_else(
                || install_path.display().to_string(),
                |record| describe_install_source(&record.source),
            );
            if let Err(error) = validate_installed_plugin_path(&primary_install_root, &install_path)
            {
                if let Some(record) = matched_record {
                    // A registered higher-priority plugin reserves its ID when
                    // validation fails, so a lower-root copy cannot take over.
                    seen_ids.insert(record.id.clone());
                }
                seen_paths.insert(install_path.clone());
                discovery.push_failure(PluginLoadFailure::new(install_path, kind, source, error));
                continue;
            }
            // Supply-chain guard: reject a tampered on-disk copy before loading.
            if let Some(record) = matched_record {
                if let Err(error) = verify_plugin_integrity(record, &install_path) {
                    // A registered higher-priority plugin reserves its ID even
                    // when its contents fail integrity verification.
                    seen_ids.insert(record.id.clone());
                    // Mark the path seen so the registry-fallback loop below
                    // does not re-load the very tree we just rejected.
                    seen_paths.insert(install_path.clone());
                    discovery.push_failure(PluginLoadFailure::new(
                        install_path,
                        kind,
                        source,
                        error,
                    ));
                    continue;
                }
            }
            match load_plugin_definition(&install_path, kind, source.clone(), kind.marketplace()) {
                Ok(plugin) => {
                    if seen_ids.insert(plugin.metadata().id.clone()) {
                        seen_paths.insert(install_path);
                        discovery.push_plugin(plugin);
                    }
                }
                Err(error) => {
                    seen_paths.insert(install_path.clone());
                    discovery.push_failure(PluginLoadFailure::new(
                        install_path,
                        kind,
                        source,
                        error,
                    ));
                }
            }
        }

        self.discover_secondary_installed_plugins(
            &mut discovery,
            &mut seen_ids,
            &mut seen_paths,
        )?;

        for record in registry.plugins.values() {
            if seen_paths.contains(&record.install_path) {
                continue;
            }
            if !record.install_path.exists() || plugin_manifest_path(&record.install_path).is_err()
            {
                stale_registry_ids.push(record.id.clone());
                continue;
            }
            let source = describe_install_source(&record.source);
            if let Err(error) = validate_installed_plugin_path(&primary_install_root, &record.install_path)
            {
                seen_ids.insert(record.id.clone());
                discovery.push_failure(PluginLoadFailure::new(
                    record.install_path.clone(),
                    record.kind,
                    source,
                    error,
                ));
                continue;
            }
            if let Err(error) = verify_plugin_integrity(record, &record.install_path) {
                seen_ids.insert(record.id.clone());
                discovery.push_failure(PluginLoadFailure::new(
                    record.install_path.clone(),
                    record.kind,
                    source,
                    error,
                ));
                continue;
            }
            match load_plugin_definition(
                &record.install_path,
                record.kind,
                source.clone(),
                record.kind.marketplace(),
            ) {
                Ok(plugin) => {
                    if seen_ids.insert(plugin.metadata().id.clone()) {
                        seen_paths.insert(record.install_path.clone());
                        discovery.push_plugin(plugin);
                    }
                }
                Err(error) => {
                    discovery.push_failure(PluginLoadFailure::new(
                        record.install_path.clone(),
                        record.kind,
                        source,
                        error,
                    ));
                }
            }
        }

        if !stale_registry_ids.is_empty() {
            for plugin_id in stale_registry_ids {
                registry.plugins.remove(&plugin_id);
            }
            self.store_registry(&registry)?;
        }

        Ok(discovery)
    }

    #[allow(clippy::too_many_lines)]
    fn discover_secondary_installed_plugins(
        &self,
        discovery: &mut PluginDiscovery,
        seen_ids: &mut BTreeSet<String>,
        seen_paths: &mut BTreeSet<PathBuf>,
    ) -> Result<(), PluginError> {
        // The primary install root is scanned first, so these sets preserve its
        // precedence. Secondary canonical roots are discovery-only, but their
        // installed registries remain the authority for provenance and digests.
        for secondary_root in &self.config.discovery_install_roots {
            let registry = load_registry_at(&secondary_registry_path(secondary_root))?;
            for install_path in discover_plugin_dirs(secondary_root)? {
                if seen_paths.contains(&install_path) {
                    continue;
                }
                let matched_record = registry
                    .plugins
                    .values()
                    .find(|record| record.install_path == install_path);
                let kind = matched_record.map_or(PluginKind::External, |record| record.kind);
                let source = matched_record.map_or_else(
                    || install_path.display().to_string(),
                    |record| describe_install_source(&record.source),
                );
                if let Err(error) = validate_installed_plugin_path(secondary_root, &install_path) {
                    if let Some(record) = matched_record {
                        seen_ids.insert(record.id.clone());
                    }
                    seen_paths.insert(install_path.clone());
                    discovery.push_failure(PluginLoadFailure::new(install_path, kind, source, error));
                    continue;
                }
                if let Some(record) = matched_record {
                    if let Err(error) = verify_plugin_integrity(record, &install_path) {
                        seen_ids.insert(record.id.clone());
                        seen_paths.insert(install_path.clone());
                        discovery.push_failure(PluginLoadFailure::new(
                            install_path,
                            kind,
                            source,
                            error,
                        ));
                        continue;
                    }
                }
                match load_plugin_definition(&install_path, kind, source.clone(), kind.marketplace()) {
                    Ok(plugin) => {
                        if seen_ids.insert(plugin.metadata().id.clone()) {
                            seen_paths.insert(install_path);
                            discovery.push_plugin(plugin);
                        }
                    }
                    Err(error) => {
                        if let Some(record) = matched_record {
                            seen_ids.insert(record.id.clone());
                        }
                        seen_paths.insert(install_path.clone());
                        discovery.push_failure(PluginLoadFailure::new(
                            install_path,
                            kind,
                            source,
                            error,
                        ));
                    }
                }
            }

            for record in registry.plugins.values() {
                if seen_paths.contains(&record.install_path)
                    || !record.install_path.exists()
                    || plugin_manifest_path(&record.install_path).is_err()
                {
                    continue;
                }
                let source = describe_install_source(&record.source);
                if let Err(error) = validate_installed_plugin_path(secondary_root, &record.install_path)
                {
                    seen_ids.insert(record.id.clone());
                    discovery.push_failure(PluginLoadFailure::new(
                        record.install_path.clone(),
                        record.kind,
                        source,
                        error,
                    ));
                    continue;
                }
                if let Err(error) = verify_plugin_integrity(record, &record.install_path) {
                    seen_ids.insert(record.id.clone());
                    discovery.push_failure(PluginLoadFailure::new(
                        record.install_path.clone(),
                        record.kind,
                        source,
                        error,
                    ));
                    continue;
                }
                match load_plugin_definition(
                    &record.install_path,
                    record.kind,
                    source.clone(),
                    record.kind.marketplace(),
                ) {
                    Ok(plugin) => {
                        if seen_ids.insert(plugin.metadata().id.clone()) {
                            seen_paths.insert(record.install_path.clone());
                            discovery.push_plugin(plugin);
                        }
                    }
                    Err(error) => {
                        seen_ids.insert(record.id.clone());
                        discovery.push_failure(PluginLoadFailure::new(
                            record.install_path.clone(),
                            record.kind,
                            source,
                            error,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn discover_external_directory_plugins_with_failures(
        &self,
        existing_plugins: &[PluginDefinition],
    ) -> Result<PluginDiscovery, PluginError> {
        let mut discovery = PluginDiscovery::default();

        for directory in &self.config.external_dirs {
            for root in discover_plugin_dirs(directory)? {
                let source = root.display().to_string();
                match load_plugin_definition(
                    &root,
                    PluginKind::External,
                    source.clone(),
                    EXTERNAL_MARKETPLACE,
                ) {
                    Ok(plugin) => {
                        if existing_plugins
                            .iter()
                            .chain(discovery.plugins.iter())
                            .all(|existing| existing.metadata().id != plugin.metadata().id)
                        {
                            discovery.push_plugin(plugin);
                        }
                    }
                    Err(error) => {
                        discovery.push_failure(PluginLoadFailure::new(
                            root,
                            PluginKind::External,
                            source,
                            error,
                        ));
                    }
                }
            }
        }

        Ok(discovery)
    }

    pub fn installed_plugin_registry_report(&self) -> Result<PluginRegistryReport, PluginError> {
        if let Some(registry) = self.cached_installed_registry() {
            return Ok(PluginRegistryReport::new(registry, Vec::new()));
        }

        self.sync_bundled_plugins()?;
        let report = self.build_registry_report(self.discover_installed_plugins_with_failures()?);
        if !report.has_failures() {
            self.store_cached_installed_registry(report.registry().clone());
        }
        Ok(report)
    }

    fn sync_bundled_plugins(&self) -> Result<(), PluginError> {
        let bundled_root = self
            .config
            .bundled_root
            .clone()
            .unwrap_or_else(Self::bundled_root);
        let bundled_plugins = discover_plugin_dirs(&bundled_root)?;
        let mut registry = self.load_registry()?;
        let mut changed = false;
        let install_root = self.install_root();
        let mut active_bundled_ids = BTreeSet::new();

        for source_root in bundled_plugins {
            let manifest = load_plugin_from_directory(&source_root)?;
            let plugin_id = plugin_id(&manifest.name, BUNDLED_MARKETPLACE);
            active_bundled_ids.insert(plugin_id.clone());
            let install_path = install_root.join(sanitize_plugin_id(&plugin_id));
            let now = unix_time_ms();
            let existing_record = registry.plugins.get(&plugin_id);
            let installed_copy_is_valid =
                install_path.exists() && load_plugin_from_directory(&install_path).is_ok();
            let needs_sync = existing_record.is_none_or(|record| {
                record.kind != PluginKind::Bundled
                    || record.version != manifest.version
                    || record.name != manifest.name
                    || record.description != manifest.description
                    || record.install_path != install_path
                    || !record.install_path.exists()
                    || !installed_copy_is_valid
            });

            if !needs_sync {
                continue;
            }

            if install_path.exists() {
                fs::remove_dir_all(&install_path)?;
            }
            copy_dir_all(&source_root, &install_path)?;

            let installed_at_unix_ms =
                existing_record.map_or(now, |record| record.installed_at_unix_ms);
            registry.plugins.insert(
                plugin_id.clone(),
                InstalledPluginRecord {
                    kind: PluginKind::Bundled,
                    id: plugin_id,
                    name: manifest.name,
                    version: manifest.version,
                    description: manifest.description,
                    install_path,
                    source: PluginInstallSource::LocalPath { path: source_root },
                    installed_at_unix_ms,
                    updated_at_unix_ms: now,
                    // Bundled plugins ship with the binary; their integrity is
                    // governed by the binary, not by a per-install digest.
                    resolved_commit: None,
                    content_sha256: None,
                },
            );
            changed = true;
        }

        let stale_bundled_ids = registry
            .plugins
            .iter()
            .filter_map(|(plugin_id, record)| {
                (record.kind == PluginKind::Bundled && !active_bundled_ids.contains(plugin_id))
                    .then_some(plugin_id.clone())
            })
            .collect::<Vec<_>>();

        for plugin_id in stale_bundled_ids {
            if let Some(record) = registry.plugins.remove(&plugin_id) {
                if record.install_path.exists() {
                    fs::remove_dir_all(&record.install_path)?;
                }
                changed = true;
            }
        }

        if changed {
            self.store_registry(&registry)?;
        }

        Ok(())
    }

    fn is_enabled(&self, metadata: &PluginMetadata) -> bool {
        self.config
            .enabled_plugins
            .get(&metadata.id)
            .copied()
            .unwrap_or(match metadata.kind {
                PluginKind::External => false,
                PluginKind::Builtin | PluginKind::Bundled => metadata.default_enabled,
            })
    }

    fn ensure_known_plugin(&self, plugin_id: &str) -> Result<(), PluginError> {
        if self.plugin_registry()?.contains(plugin_id) {
            Ok(())
        } else {
            Err(PluginError::NotFound(format!(
                "plugin `{plugin_id}` is not installed or discoverable"
            )))
        }
    }

    pub(crate) fn load_registry(&self) -> Result<InstalledPluginRegistry, PluginError> {
        load_registry_at(&self.registry_path())
    }

    pub(crate) fn store_registry(
        &self,
        registry: &InstalledPluginRegistry,
    ) -> Result<(), PluginError> {
        let path = self.registry_path();
        write_atomic(&path, serde_json::to_string_pretty(registry)?.as_bytes())?;
        Ok(())
    }

    pub(crate) fn write_enabled_state(
        &self,
        plugin_id: &str,
        enabled: Option<bool>,
    ) -> Result<(), PluginError> {
        update_settings_json(&self.settings_path(), |root| {
            let enabled_plugins = ensure_object(root, "enabledPlugins");
            match enabled {
                Some(value) => {
                    enabled_plugins.insert(plugin_id.to_string(), Value::Bool(value));
                }
                None => {
                    enabled_plugins.remove(plugin_id);
                }
            }
        })
    }

    fn installed_plugin_registry(&self) -> Result<PluginRegistry, PluginError> {
        if let Some(registry) = self.cached_installed_registry() {
            return Ok(registry);
        }

        let registry = self.installed_plugin_registry_report()?.into_registry()?;
        self.store_cached_installed_registry(registry.clone());
        Ok(registry)
    }

    fn build_registry_report(&self, discovery: PluginDiscovery) -> PluginRegistryReport {
        PluginRegistryReport::new(
            PluginRegistry::new(
                discovery
                    .plugins
                    .into_iter()
                    .map(|plugin| {
                        let enabled = self.is_enabled(plugin.metadata());
                        RegisteredPlugin::new(plugin, enabled)
                    })
                    .collect(),
            ),
            discovery.failures,
        )
    }

    fn cached_registry(&self) -> Option<PluginRegistry> {
        self.cache
            .lock()
            // Poison policy: recover — the cache is two independent
            // Option<PluginRegistry> memo fields (single-value writes); a
            // poisoned holder leaves at worst a stale/None entry that the
            // next load recomputes. (plugins deliberately has no api dep,
            // so the shared lock_recovered helper is out of reach here.)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .registry
            .clone()
    }

    fn store_cached_registry(&self, registry: PluginRegistry) {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .registry = Some(registry);
    }

    fn cached_installed_registry(&self) -> Option<PluginRegistry> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .installed_registry
            .clone()
    }

    fn store_cached_installed_registry(&self, registry: PluginRegistry) {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .installed_registry = Some(registry);
    }

    fn invalidate_cache(&self) {
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.registry = None;
        cache.installed_registry = None;
    }
}

fn load_registry_at(path: &Path) -> Result<InstalledPluginRegistry, PluginError> {
    match fs::read_to_string(path) {
        Ok(contents) if contents.trim().is_empty() => Ok(InstalledPluginRegistry::default()),
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(InstalledPluginRegistry::default())
        }
        Err(error) => Err(PluginError::Io(error)),
    }
}

fn secondary_registry_path(install_root: &Path) -> PathBuf {
    install_root
        .parent()
        .map_or_else(|| install_root.join(REGISTRY_FILE_NAME), |parent| {
            parent.join(REGISTRY_FILE_NAME)
        })
}

fn validate_installed_plugin_path(install_root: &Path, install_path: &Path) -> Result<(), PluginError> {
    let metadata = fs::symlink_metadata(install_path)?;
    if metadata.file_type().is_symlink() {
        return Err(PluginError::InvalidManifest(format!(
            "installed plugin directory `{}` must not be a symlink",
            install_path.display(),
        )));
    }

    let canonical_root = install_root.canonicalize().map_err(|error| {
        PluginError::InvalidManifest(format!(
            "configured install root `{}` could not be resolved for containment check: {error}",
            install_root.display(),
        ))
    })?;
    let canonical_path = install_path.canonicalize().map_err(|error| {
        PluginError::InvalidManifest(format!(
            "installed plugin directory `{}` could not be resolved for containment check: {error}",
            install_path.display(),
        ))
    })?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(PluginError::InvalidManifest(format!(
            "installed plugin directory `{}` resolves outside configured install root `{}`",
            install_path.display(),
            install_root.display(),
        )));
    }
    Ok(())
}

/// Verify that an installed plugin's on-disk tree still matches the SHA-256
/// recorded at install time. Records without a stored digest (installed before
/// supply-chain checks existed) pass unconditionally.
fn verify_plugin_integrity(
    record: &InstalledPluginRecord,
    install_path: &Path,
) -> Result<(), PluginError> {
    let Some(expected) = record.content_sha256.as_deref() else {
        return Ok(());
    };
    let actual = hash_plugin_tree(install_path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(PluginError::IntegrityMismatch(format!(
            "plugin `{}` failed its integrity check: contents at {} no longer match the digest recorded at install time",
            record.id,
            install_path.display(),
        )))
    }
}
