//! Filesystem-backed OAuth credential persistence.
//!
//! This module provides load/save/clear operations for OAuth token sets,
//! as well as PKCE helper functions needed by the OAuth authorization flow.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use core_types::paths::restrict_permissions_owner_only;
use core_types::{OAuthTokenSet, OpenAiOAuthTokens};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use core_types::PkceCodePair;

const OAUTH_KEY: &str = "oauth";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredOAuthCredentials {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_at: Option<u64>,
    #[serde(default)]
    scopes: Vec<String>,
}

impl From<OAuthTokenSet> for StoredOAuthCredentials {
    fn from(value: OAuthTokenSet) -> Self {
        Self {
            access_token: value.access_token,
            refresh_token: value.refresh_token,
            expires_at: value.expires_at,
            scopes: value.scopes,
        }
    }
}

impl From<&OAuthTokenSet> for StoredOAuthCredentials {
    fn from(value: &OAuthTokenSet) -> Self {
        Self {
            access_token: value.access_token.clone(),
            refresh_token: value.refresh_token.clone(),
            expires_at: value.expires_at,
            scopes: value.scopes.clone(),
        }
    }
}

impl From<StoredOAuthCredentials> for OAuthTokenSet {
    fn from(value: StoredOAuthCredentials) -> Self {
        Self {
            access_token: value.access_token,
            refresh_token: value.refresh_token,
            expires_at: value.expires_at,
            scopes: value.scopes,
        }
    }
}

// --- Generic credential-key helpers ---
//
// These compose [`read_credentials_root`]/[`update_credentials_root`] with the
// `StoredOAuthCredentials` wire shape, and are the shared building blocks for
// the Anthropic and Gemini entries below as well as the runtime's per-server
// MCP token storage (which layers its own nested layout on top).

/// Decode a stored credential JSON value into an [`OAuthTokenSet`].
pub fn token_set_from_value(value: &Value) -> io::Result<OAuthTokenSet> {
    StoredOAuthCredentials::deserialize(value)
        .map(Into::into)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Encode an [`OAuthTokenSet`] into its stored JSON representation.
pub fn token_set_to_value(token_set: &OAuthTokenSet) -> io::Result<Value> {
    serde_json::to_value(StoredOAuthCredentials::from(token_set))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Load the [`OAuthTokenSet`] stored under `key` at the top level of
/// `credentials.json`, treating a missing or `null` entry as `None`.
pub fn load_token_set(key: &str) -> io::Result<Option<OAuthTokenSet>> {
    let root = read_credentials_root(&credentials_path()?)?;
    match root.get(key) {
        Some(entry) if !entry.is_null() => token_set_from_value(entry).map(Some),
        _ => Ok(None),
    }
}

/// Persist `token_set` under `key`, leaving every other entry untouched.
pub fn save_token_set(key: &str, token_set: &OAuthTokenSet) -> io::Result<()> {
    update_credentials_root(&credentials_path()?, |root| {
        root.insert(key.to_owned(), token_set_to_value(token_set)?);
        Ok(())
    })
}

/// Remove the entry stored under `key`, leaving every other entry untouched.
pub fn clear_token_key(key: &str) -> io::Result<()> {
    update_credentials_root(&credentials_path()?, |root| {
        root.remove(key);
        Ok(())
    })
}

pub fn generate_pkce_pair() -> io::Result<PkceCodePair> {
    let verifier = generate_random_token(32)?;
    Ok(PkceCodePair {
        challenge: code_challenge_s256(&verifier),
        verifier,
        challenge_method: core_types::PkceChallengeMethod::S256,
    })
}

pub fn generate_state() -> io::Result<String> {
    generate_random_token(32)
}

#[must_use]
pub fn code_challenge_s256(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64url_encode(&digest)
}

#[must_use]
pub fn loopback_redirect_uri(port: u16) -> String {
    format!("http://127.0.0.1:{port}/callback")
}

pub fn credentials_path() -> io::Result<PathBuf> {
    Ok(credentials_home_dir().join("credentials.json"))
}

pub fn load_oauth_credentials() -> io::Result<Option<OAuthTokenSet>> {
    load_token_set(OAUTH_KEY)
}

pub fn save_oauth_credentials(token_set: &OAuthTokenSet) -> io::Result<()> {
    save_token_set(OAUTH_KEY, token_set)
}

pub fn clear_oauth_credentials() -> io::Result<()> {
    clear_token_key(OAUTH_KEY)
}

const OPENAI_OAUTH_KEY: &str = "openai_oauth";

/// On-disk representation of [`OpenAiOAuthTokens`] (camelCase), stored under its
/// own key so the Anthropic and ChatGPT credentials never collide. Mirrors the
/// `runtime::oauth` copy the CLI writes; this api-side reader is what the
/// provider router and token refresh consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredOpenAiOAuth {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_at: Option<u64>,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default)]
    scopes: Vec<String>,
}

impl From<&OpenAiOAuthTokens> for StoredOpenAiOAuth {
    fn from(value: &OpenAiOAuthTokens) -> Self {
        Self {
            access_token: value.access_token.clone(),
            refresh_token: value.refresh_token.clone(),
            expires_at: value.expires_at,
            account_id: value.account_id.clone(),
            scopes: value.scopes.clone(),
        }
    }
}

impl From<StoredOpenAiOAuth> for OpenAiOAuthTokens {
    fn from(value: StoredOpenAiOAuth) -> Self {
        Self {
            access_token: value.access_token,
            refresh_token: value.refresh_token,
            expires_at: value.expires_at,
            account_id: value.account_id,
            scopes: value.scopes,
        }
    }
}

/// Load saved ChatGPT OAuth tokens for the provider router.
pub fn load_openai_oauth() -> io::Result<Option<OpenAiOAuthTokens>> {
    let path = credentials_path()?;
    let root = read_credentials_root(&path)?;
    let Some(entry) = root.get(OPENAI_OAUTH_KEY) else {
        return Ok(None);
    };
    if entry.is_null() {
        return Ok(None);
    }
    let stored = StoredOpenAiOAuth::deserialize(entry)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(Some(stored.into()))
}

/// Persist refreshed ChatGPT OAuth tokens (after a token refresh).
pub fn save_openai_oauth(tokens: &OpenAiOAuthTokens) -> io::Result<()> {
    let path = credentials_path()?;
    update_credentials_root(&path, |root| {
        root.insert(
            OPENAI_OAUTH_KEY.to_owned(),
            serde_json::to_value(StoredOpenAiOAuth::from(tokens))
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        );
        Ok(())
    })
}

/// Remove saved ChatGPT OAuth tokens.
pub fn clear_openai_oauth() -> io::Result<()> {
    let path = credentials_path()?;
    update_credentials_root(&path, |root| {
        root.remove(OPENAI_OAUTH_KEY);
        Ok(())
    })
}

const OPENAI_COMPAT_API_KEYS_KEY: &str = "openai_compat_api_keys";

/// Load a saved API key for an OpenAI-compatible adapter env var (for example
/// `DEEPSEEK_API_KEY`). Environment variables still take precedence at request
/// time; this store is the durable fallback populated by TUI `/connect`.
pub fn load_openai_compat_api_key(env_key: &str) -> io::Result<Option<String>> {
    let path = credentials_path()?;
    let root = read_credentials_root(&path)?;
    let Some(entry) = root.get(OPENAI_COMPAT_API_KEYS_KEY).and_then(Value::as_object) else {
        return Ok(None);
    };
    Ok(entry
        .get(env_key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string))
}

/// Persist an OpenAI-compatible adapter API key under its configured env var
/// name, leaving OAuth credentials and other adapter keys untouched.
pub fn save_openai_compat_api_key(env_key: &str, api_key: &str) -> io::Result<()> {
    let trimmed_env = env_key.trim();
    let trimmed_key = api_key.trim();
    if trimmed_env.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "adapter API-key env var name cannot be empty",
        ));
    }
    if trimmed_key.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "adapter API key cannot be empty",
        ));
    }
    let path = credentials_path()?;
    update_credentials_root(&path, |root| {
        let entry = root
            .entry(OPENAI_COMPAT_API_KEYS_KEY.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        let Some(keys) = entry.as_object_mut() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "openai-compatible API key store must be a JSON object",
            ));
        };
        keys.insert(trimmed_env.to_string(), Value::String(trimmed_key.to_string()));
        Ok(())
    })
}

const GOOGLE_CODE_ASSIST_OAUTH_KEY: &str = "google_code_assist_oauth";

/// Load saved Gemini Code Assist OAuth tokens, if `/login google` has run.
pub fn load_google_code_assist_oauth() -> io::Result<Option<OAuthTokenSet>> {
    load_token_set(GOOGLE_CODE_ASSIST_OAUTH_KEY)
}

/// Persist Gemini Code Assist OAuth tokens.
pub fn save_google_code_assist_oauth(tokens: &OAuthTokenSet) -> io::Result<()> {
    save_token_set(GOOGLE_CODE_ASSIST_OAUTH_KEY, tokens)
}

/// Remove saved Gemini Code Assist OAuth tokens.
pub fn clear_google_code_assist_oauth() -> io::Result<()> {
    clear_token_key(GOOGLE_CODE_ASSIST_OAUTH_KEY)
}

/// A saved-OAuth provider whose presence `doctor` reports without reading token
/// values. Each maps to the top-level credentials key written by its login.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavedOAuthProvider {
    /// Anthropic Claude OAuth (`oauth`).
    Anthropic,
    /// ChatGPT / OpenAI OAuth (`openai_oauth`).
    OpenAi,
    /// Gemini Code Assist OAuth (`google_code_assist_oauth`).
    GoogleCodeAssist,
}

impl SavedOAuthProvider {
    const fn key(self) -> &'static str {
        match self {
            Self::Anthropic => OAUTH_KEY,
            Self::OpenAi => OPENAI_OAUTH_KEY,
            Self::GoogleCodeAssist => GOOGLE_CODE_ASSIST_OAUTH_KEY,
        }
    }
}

/// Whether a saved OAuth credential is present for `provider`, established
/// without following a symlink at the `credentials.json` leaf and without
/// exposing any token value. This is a secret-safe *presence* probe for
/// `doctor`: it never parses, prints, refreshes, or mints credentials.
///
/// The credentials file is read once through a no-follow secure reader
/// (`read_file`), so a `credentials.json` replaced by a symlink is rejected
/// rather than followed — closing the preflight-lstat-then-following-read race
/// that the ordinary `fs::read_to_string`-backed loaders would leave open. A
/// missing or empty file, or a `null`/absent entry, reports absent.
pub fn saved_oauth_present(
    provider: SavedOAuthProvider,
    read_file: &dyn Fn(&Path) -> io::Result<Option<String>>,
) -> io::Result<bool> {
    let path = credentials_path()?;
    let Some(contents) = read_file(&path)? else {
        return Ok(false);
    };
    if contents.trim().is_empty() {
        return Ok(false);
    }
    let root: Map<String, Value> = serde_json::from_str::<Value>(&contents)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "credentials file must contain a JSON object",
            )
        })?;
    Ok(saved_oauth_entry_present(&root, provider))
}

/// Whether a saved OAuth credential is present for `provider` across the full
/// effective credential view — the `ZO_CONFIG_HOME → ZO_HOME → ~/.zo → ~/.forge`
/// chain that the normal provider loaders read — established without following a
/// symlink at any `credentials.json` and without exposing any token value.
///
/// Each root's `credentials.json` is read once through the caller-supplied
/// no-follow `read_file` (so a symlinked or unsafe leaf at any root surfaces as
/// an error rather than being followed), then layered highest-priority-first
/// with the same atomic-replacement and `null`-tombstone semantics as
/// [`read_credentials_root`]. A provider whose highest-priority present entry is
/// a `null` tombstone reports absent, matching a logout. This is the effective
/// counterpart of [`saved_oauth_present`], which only inspects the primary root.
pub fn saved_oauth_present_effective(
    provider: SavedOAuthProvider,
    read_file: &dyn Fn(&Path) -> io::Result<Option<String>>,
) -> io::Result<bool> {
    // Layer low-to-high so the highest-priority root wins, mirroring
    // `read_lower_roots` + primary. `null` acts as a tombstone via `apply_root`.
    let mut effective = Map::new();
    for root_dir in credential_roots().into_iter().rev() {
        let path = root_dir.join("credentials.json");
        let Some(contents) = read_file(&path)? else {
            continue;
        };
        if contents.trim().is_empty() {
            continue;
        }
        let overlay: Map<String, Value> = serde_json::from_str::<Value>(&contents)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "credentials root is not a JSON object",
                )
            })?;
        effective = apply_root(effective, &overlay);
    }
    // `apply_root` already drops `null` tombstones, so a present entry here is a
    // live credential.
    Ok(effective
        .get(provider.key())
        .is_some_and(|entry| !entry.is_null()))
}

/// Whether `root` carries a non-null saved-OAuth entry for `provider`.
fn saved_oauth_entry_present(root: &Map<String, Value>, provider: SavedOAuthProvider) -> bool {
    root.get(provider.key())
        .is_some_and(|entry| !entry.is_null())
}

fn generate_random_token(bytes: usize) -> io::Result<String> {
    let mut buffer = vec![0_u8; bytes];
    File::open("/dev/urandom")?.read_exact(&mut buffer)?;
    Ok(base64url_encode(&buffer))
}

fn credential_roots() -> Vec<PathBuf> {
    let mut roots = core_types::paths::zo_global_config_roots();
    if roots.is_empty() {
        roots.push(core_types::paths::default_config_home());
    }
    roots
}

fn credentials_home_dir() -> PathBuf {
    credential_roots()
        .into_iter()
        .next()
        .unwrap_or_else(core_types::paths::default_config_home)
}

/// Top-level credential keys whose value is a *collection map* of independent
/// per-entry credentials: the MCP server token map (`mcp_oauth`, owned by
/// `runtime::oauth`) and the adapter API-key map ([`OPENAI_COMPAT_API_KEYS_KEY`]).
/// Only these keys merge per entry across roots — each entry is atomic, and a
/// `null` entry value is a per-entry tombstone. Every other top-level entry is
/// an atomic credential object: the highest root wins the whole object and its
/// fields are never combined across roots.
const MCP_OAUTH_COLLECTION_KEY: &str = "mcp_oauth";

fn is_collection_map_key(key: &str) -> bool {
    key == OPENAI_COMPAT_API_KEYS_KEY || key == MCP_OAUTH_COLLECTION_KEY
}

/// Atomically read, mutate, and write back the credential root object under an
/// advisory file lock. Exposed so dependent crates (e.g. the runtime's MCP
/// token storage) can compose their own credential layouts on the same file.
///
/// Writes go only to the primary root. The closure operates on the merged
/// *effective* view (primary over the lower `ZO_HOME`/`HOME`/.zo roots); the
/// resulting primary file is reconciled against those lower roots so a value
/// the closure removed is written back as a `null` tombstone whenever a lower
/// root still carries it. That keeps logout durable even though the lower copy
/// is never touched.
pub fn update_credentials_root(
    path: &Path,
    update: impl FnOnce(&mut Map<String, Value>) -> io::Result<()>,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
        restrict_permissions_owner_only(parent)?;
    }
    let lock_path = path.with_extension("json.lock");
    let _lock = CredentialFileLock::acquire(&lock_path)?;

    let primary = core_types::paths::default_config_home();
    let is_primary_root =
        path.file_name().is_some() && path.parent() == Some(primary.as_path());
    if !is_primary_root {
        // Files outside the primary config home have no lower-root layering, so
        // a plain single-file read-modify-write is the whole story.
        let mut root = read_one_credentials_root(path)?;
        update(&mut root)?;
        return write_credentials_root(path, &root);
    }

    let file_name = path.file_name().expect("primary root checked above");
    let lower = read_lower_roots(file_name)?;
    let primary_before = read_one_credentials_root(path)?;
    let mut effective = apply_root(lower.clone(), &primary_before);
    update(&mut effective)?;
    let primary_after = reconcile_primary(&lower, &effective);
    write_credentials_root(path, &primary_after)
}

/// Read the *effective* credential view for `path`: the primary root layered
/// over the lower canonical roots. Non-collection entries are atomic (the
/// highest root wins the whole object); the known collection maps merge per
/// entry. `null` values act as tombstones and are dropped from the result, so
/// callers see a cleaned view with no deleted keys or entries.
pub fn read_credentials_root(path: &Path) -> io::Result<Map<String, Value>> {
    let primary = core_types::paths::default_config_home();
    let Some(file_name) = path.file_name() else {
        return read_one_credentials_root(path);
    };
    if path.parent() != Some(primary.as_path()) {
        return read_one_credentials_root(path);
    }

    let lower = read_lower_roots(file_name)?;
    let primary_root = read_one_credentials_root(path)?;
    Ok(apply_root(lower, &primary_root))
}

/// Merge the lower canonical roots (every root except the primary) low-to-high
/// into a single effective view. Returns an empty map when only the primary
/// root exists.
fn read_lower_roots(file_name: &std::ffi::OsStr) -> io::Result<Map<String, Value>> {
    let mut roots = credential_roots();
    if roots.is_empty() {
        return Ok(Map::new());
    }
    let lower = roots.split_off(1);
    let mut acc = Map::new();
    for root in lower.into_iter().rev() {
        let candidate = root.join(file_name);
        acc = apply_root(acc, &read_one_credentials_root(&candidate)?);
    }
    Ok(acc)
}

/// Layer `overlay` onto `base`, producing the effective view. Non-collection
/// keys replace atomically (fields are never combined); collection-map keys
/// merge per entry. `null` in `overlay` deletes: a top-level `null` drops the
/// whole entry, a per-entry `null` inside a collection map drops that entry.
/// The returned map never contains `null` tombstones.
fn apply_root(mut base: Map<String, Value>, overlay: &Map<String, Value>) -> Map<String, Value> {
    for (key, value) in overlay {
        if is_collection_map_key(key) {
            apply_collection_overlay(&mut base, key, value);
        } else if value.is_null() {
            base.remove(key);
        } else {
            base.insert(key.clone(), value.clone());
        }
    }
    base
}

/// Apply a collection-map overlay entry-by-entry, honoring per-entry `null`
/// tombstones and a whole-map `null` tombstone.
fn apply_collection_overlay(base: &mut Map<String, Value>, key: &str, overlay_value: &Value) {
    if overlay_value.is_null() {
        base.remove(key);
        return;
    }
    let Some(overlay_map) = overlay_value.as_object() else {
        // Malformed overlay (not an object): fall back to atomic replacement.
        base.insert(key.to_string(), overlay_value.clone());
        return;
    };
    let mut merged = base
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (entry_key, entry_value) in overlay_map {
        if entry_value.is_null() {
            merged.remove(entry_key);
        } else {
            merged.insert(entry_key.clone(), entry_value.clone());
        }
    }
    if merged.is_empty() {
        base.remove(key);
    } else {
        base.insert(key.to_string(), Value::Object(merged));
    }
}

/// Compute the primary-root contents so that layering the lower roots under it
/// reproduces `effective`. Present values are written through (migrating
/// lower-root-only credentials up); values that `effective` dropped but a lower
/// root still carries are written as `null` tombstones so the deletion sticks.
fn reconcile_primary(
    lower: &Map<String, Value>,
    effective: &Map<String, Value>,
) -> Map<String, Value> {
    let keys: BTreeSet<&String> = effective.keys().chain(lower.keys()).collect();
    let mut out = Map::new();
    for key in keys {
        let eff = effective.get(key).filter(|value| !value.is_null());
        let low = lower.get(key).filter(|value| !value.is_null());
        if is_collection_map_key(key) {
            let eff_map = eff.and_then(Value::as_object);
            let low_map = low.and_then(Value::as_object);
            let mut sub = Map::new();
            if let Some(entries) = eff_map {
                for (entry_key, entry_value) in entries {
                    if !entry_value.is_null() {
                        sub.insert(entry_key.clone(), entry_value.clone());
                    }
                }
            }
            if let Some(entries) = low_map {
                for entry_key in entries.keys() {
                    let still_present = eff_map
                        .and_then(|map| map.get(entry_key))
                        .is_some_and(|value| !value.is_null());
                    if !still_present {
                        sub.entry(entry_key.clone()).or_insert(Value::Null);
                    }
                }
            }
            if !sub.is_empty() {
                out.insert(key.clone(), Value::Object(sub));
            }
        } else {
            match (eff, low) {
                (Some(value), _) => {
                    out.insert(key.clone(), value.clone());
                }
                (None, Some(_)) => {
                    out.insert(key.clone(), Value::Null);
                }
                (None, None) => {}
            }
        }
    }
    out
}

fn read_one_credentials_root(path: &Path) -> io::Result<Map<String, Value>> {
    match fs::read_to_string(path) {
        Ok(contents) => {
            if contents.trim().is_empty() {
                return Ok(Map::new());
            }
            serde_json::from_str::<Value>(&contents)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
                .as_object()
                .cloned()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "credentials file must contain a JSON object",
                    )
                })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Map::new()),
        Err(error) => Err(error),
    }
}

static CREDENTIAL_TEMP_SEQUENCE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn write_credentials_root(path: &Path, root: &Map<String, Value>) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
        restrict_permissions_owner_only(parent)?;
    }
    let mut rendered = serde_json::to_string_pretty(root)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    rendered.push('\n');

    let temp_path = write_credentials_candidate(path, rendered.as_bytes())?;
    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|directory| directory.sync_all());
    }
    Ok(())
}

/// Write and sync one owner-only candidate before it becomes visible at the
/// public credential path. `create_new` plus a process/counter suffix prevents
/// concurrent writers or a planted fixed temp symlink from sharing the inode.
fn write_credentials_candidate(path: &Path, contents: &[u8]) -> io::Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "credentials path must have a file name",
        )
    })?;
    for _ in 0..64 {
        let sequence = CREDENTIAL_TEMP_SEQUENCE
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp_path = path.with_file_name(format!(
            ".{}.tmp.{}.{sequence}",
            file_name.to_string_lossy(),
            std::process::id(),
        ));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = match options.open(&temp_path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let write = std::io::Write::write_all(&mut file, contents)
            .and_then(|()| file.sync_all());
        drop(file);
        if let Err(error) = write {
            let _ = fs::remove_file(&temp_path);
            return Err(error);
        }
        return Ok(temp_path);
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "unable to allocate a unique credentials temp file",
    ))
}

/// Cross-process advisory lock held by the open file descriptor. The lock file
/// stays in place permanently: dropping the descriptor releases the OS lock
/// without an unlink/recreate race or a crash-stale existence lock.
struct CredentialFileLock {
    _file: File,
}

impl CredentialFileLock {
    fn acquire(lock_path: &Path) -> io::Result<Self> {
        let file = open_credentials_lock(lock_path)?;
        for attempt in 0..50 {
            match file.try_lock() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(error) => {
                    let error: io::Error = error.into();
                    if error.kind() != io::ErrorKind::WouldBlock {
                        return Err(error);
                    }
                    if attempt < 49 {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "failed to acquire credentials file lock",
        ))
    }
}

fn open_credentials_lock(lock_path: &Path) -> io::Result<File> {
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options
            .mode(0o600)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    }
    let file = options.open(lock_path)?;
    if !file.metadata()?.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("credentials lock path is not a regular file: {}", lock_path.display()),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    restrict_permissions_owner_only(lock_path)?;
    Ok(file)
}

fn base64url_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut output = String::with_capacity((bytes.len() * 4).div_ceil(3));
    let mut index = 0;
    while index + 3 <= bytes.len() {
        let block = (u32::from(bytes[index]) << 16)
            | (u32::from(bytes[index + 1]) << 8)
            | u32::from(bytes[index + 2]);
        output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
        output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
        output.push(TABLE[((block >> 6) & 0x3F) as usize] as char);
        output.push(TABLE[(block & 0x3F) as usize] as char);
        index += 3;
    }
    match bytes.len().saturating_sub(index) {
        1 => {
            let block = u32::from(bytes[index]) << 16;
            output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
        }
        2 => {
            let block = (u32::from(bytes[index]) << 16) | (u32::from(bytes[index + 1]) << 8);
            output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 6) & 0x3F) as usize] as char);
        }
        _ => {}
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        clear_google_code_assist_oauth, clear_oauth_credentials, clear_openai_oauth,
        credentials_path, load_google_code_assist_oauth, load_oauth_credentials,
        load_openai_compat_api_key, load_openai_oauth, read_one_credentials_root,
        save_oauth_credentials, save_openai_compat_api_key, save_openai_oauth,
        update_credentials_root, write_credentials_root, CredentialFileLock, OAUTH_KEY,
    };
    use core_types::{OAuthTokenSet, OpenAiOAuthTokens};
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn unique_config_home() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zo-oauth-store-tests-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn no_home_oauth_round_trip_uses_private_fallback() {
        let _env_lock = crate::test_env_lock();
        let prior = [
            (
                core_types::paths::ZO_CONFIG_HOME_ENV,
                std::env::var_os(core_types::paths::ZO_CONFIG_HOME_ENV),
            ),
            (
                core_types::paths::ZO_HOME_ENV,
                std::env::var_os(core_types::paths::ZO_HOME_ENV),
            ),
            ("HOME", std::env::var_os("HOME")),
        ];
        for (key, _) in &prior {
            std::env::remove_var(key);
        }

        let path = credentials_path().expect("secure fallback credentials path");
        let token_set = OAuthTokenSet {
            access_token: "no-home-access".into(),
            refresh_token: Some("no-home-refresh".into()),
            expires_at: Some(9876),
            scopes: vec!["user:inference".into()],
        };
        save_oauth_credentials(&token_set).expect("fallback credentials should save");
        assert_eq!(
            load_oauth_credentials().expect("fallback credentials should load"),
            Some(token_set)
        );
        clear_oauth_credentials().expect("fallback credentials should clear");

        for (key, value) in prior {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }

        assert!(path.is_absolute());
        assert_eq!(
            path.parent().and_then(|parent| parent.file_name()),
            Some(std::ffi::OsStr::new(".zo"))
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn oauth_credentials_round_trip_and_clear() {
        let _env_lock = crate::test_env_lock();
        let config_home = unique_config_home();
        std::env::set_var("ZO_CONFIG_HOME", &config_home);

        let token_set = OAuthTokenSet {
            access_token: "access-token".into(),
            refresh_token: Some("refresh-token".into()),
            expires_at: Some(1234),
            scopes: vec!["org:read".into(), "user:write".into()],
        };

        save_oauth_credentials(&token_set).expect("credentials should save");
        assert_eq!(
            load_oauth_credentials().expect("credentials should load"),
            Some(token_set)
        );

        clear_oauth_credentials().expect("credentials should clear");
        assert_eq!(
            load_oauth_credentials().expect("cleared credentials should load"),
            None
        );

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn save_preserves_unrelated_entries_and_clear_is_scoped() {
        let _env_lock = crate::test_env_lock();
        let config_home = unique_config_home();
        let lower_home = config_home.with_extension("lower");
        let user_home = config_home.with_extension("home");
        let _roots = RootEnvGuard::set(&config_home, &lower_home, &user_home);
        let path = credentials_path().expect("credentials path");
        fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
        fs::write(&path, "{\"other\":\"value\"}\n").expect("seed credentials");

        let token_set = OAuthTokenSet {
            access_token: "access-token".into(),
            refresh_token: Some("refresh-token".into()),
            expires_at: Some(123),
            scopes: vec!["scope:a".into()],
        };
        save_oauth_credentials(&token_set).expect("save credentials");
        let saved = fs::read_to_string(&path).expect("read saved");
        assert!(saved.contains("\"other\": \"value\""));
        assert!(saved.contains("\"oauth\""));

        clear_oauth_credentials().expect("clear credentials");
        let cleared = fs::read_to_string(&path).expect("read cleared");
        assert!(cleared.contains("\"other\": \"value\""));
        assert!(!cleared.contains("\"oauth\""));

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn openai_oauth_round_trip_isolated_from_anthropic() {
        let _env_lock = crate::test_env_lock();
        let config_home = unique_config_home();
        std::env::set_var("ZO_CONFIG_HOME", &config_home);

        let anthropic = OAuthTokenSet {
            access_token: "anthropic-at".into(),
            refresh_token: Some("anthropic-rt".into()),
            expires_at: Some(111),
            scopes: vec!["user:inference".into()],
        };
        save_oauth_credentials(&anthropic).expect("anthropic credentials should save");

        let openai = OpenAiOAuthTokens {
            access_token: "openai-at".into(),
            refresh_token: Some("openai-rt".into()),
            expires_at: Some(222),
            account_id: Some("acc_123".into()),
            scopes: vec!["openid".into()],
        };
        save_openai_oauth(&openai).expect("openai credentials should save");

        assert_eq!(
            load_openai_oauth().expect("openai credentials should load"),
            Some(openai)
        );
        assert_eq!(
            load_oauth_credentials().expect("anthropic credentials should load"),
            Some(anthropic)
        );

        clear_openai_oauth().expect("openai credentials should clear");
        assert_eq!(
            load_openai_oauth().expect("cleared openai credentials should load"),
            None
        );
        assert!(load_oauth_credentials()
            .expect("anthropic credentials should survive openai clear")
            .is_some());

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn openai_compat_api_key_round_trip_preserves_oauth_entries() {
        let _env_lock = crate::test_env_lock();
        let config_home = unique_config_home();
        std::env::set_var("ZO_CONFIG_HOME", &config_home);

        let anthropic = OAuthTokenSet {
            access_token: "anthropic-at".into(),
            refresh_token: Some("anthropic-rt".into()),
            expires_at: Some(111),
            scopes: vec!["user:inference".into()],
        };
        save_oauth_credentials(&anthropic).expect("anthropic credentials should save");

        save_openai_compat_api_key("DEEPSEEK_API_KEY", "sk-deepseek")
            .expect("adapter API key should save");
        assert_eq!(
            load_openai_compat_api_key("DEEPSEEK_API_KEY").expect("adapter key should load"),
            Some("sk-deepseek".to_string())
        );
        assert_eq!(
            load_oauth_credentials().expect("anthropic credentials should survive"),
            Some(anthropic)
        );

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn load_oauth_credentials_treats_null_entry_as_missing() {
        let _env_lock = crate::test_env_lock();
        let config_home = unique_config_home();
        let credentials_file = config_home.join("credentials.json");
        fs::create_dir_all(&config_home).expect("config home should exist");
        fs::write(&credentials_file, format!("{{\"{OAUTH_KEY}\":null}}\n"))
            .expect("credentials file should write");
        std::env::set_var("ZO_CONFIG_HOME", &config_home);

        assert_eq!(
            load_oauth_credentials().expect("null oauth entry should load"),
            None
        );
        assert_eq!(
            credentials_path().expect("config-home override should drive the credentials path"),
            credentials_file
        );

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = fs::remove_dir_all(config_home);
    }

    fn unique_root(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zo-oauth-store-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[cfg(unix)]
    #[test]
    fn credential_write_does_not_follow_fixed_temp_symlink() {
        let dir = unique_root("temp-symlink");
        fs::create_dir_all(&dir).expect("credential dir");
        let path = dir.join("credentials.json");
        let temp_path = path.with_extension("json.tmp");
        let victim = dir.join("victim.txt");
        fs::write(&victim, "untouched\n").expect("victim");
        std::os::unix::fs::symlink(&victim, &temp_path).expect("temp symlink");

        let mut root = serde_json::Map::new();
        root.insert(
            OAUTH_KEY.to_string(),
            serde_json::json!({ "accessToken": "must-not-leak" }),
        );
        write_credentials_root(&path, &root).expect("credential write");

        assert_eq!(
            fs::read_to_string(&victim).expect("victim after write"),
            "untouched\n",
            "a planted fixed-temp symlink must never receive credential bytes",
        );
        assert!(
            fs::symlink_metadata(&path)
                .expect("published credentials")
                .file_type()
                .is_file(),
            "the published credential path must be a regular file, not the planted symlink",
        );
        assert_eq!(
            read_one_credentials_root(&path)
                .expect("published root")
                .get(OAUTH_KEY)
                .and_then(|value| value.get("accessToken"))
                .and_then(Value::as_str),
            Some("must-not-leak"),
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn live_credentials_lock_is_not_stolen_when_mtime_is_old() {
        let dir = unique_root("live-old-lock");
        fs::create_dir_all(&dir).expect("credential dir");
        let lock_path = dir.join("credentials.json.lock");
        let held = CredentialFileLock::acquire(&lock_path).expect("first lock");
        fs::OpenOptions::new()
            .write(true)
            .open(&lock_path)
            .expect("open lock for mtime")
            .set_modified(SystemTime::now() - Duration::from_secs(20))
            .expect("backdate live lock");

        let Err(error) = CredentialFileLock::acquire(&lock_path) else {
            panic!("a live holder must not lose its lock because its mtime is old");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);

        drop(held);
        assert!(lock_path.exists(), "the reusable lock inode must remain after release");
        let reacquired = CredentialFileLock::acquire(&lock_path).expect("reacquire persistent lock");
        drop(reacquired);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn concurrent_credential_updates_preserve_disjoint_keys() {
        use std::sync::{Arc, Barrier};

        let dir = unique_root("concurrent-updates");
        fs::create_dir_all(&dir).expect("credential dir");
        let path = Arc::new(dir.join("credentials.json"));
        let barrier = Arc::new(Barrier::new(2));
        let workers = [("provider-a", "secret-a"), ("provider-b", "secret-b")]
            .into_iter()
            .map(|(key, value)| {
                let path = Arc::clone(&path);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    update_credentials_root(path.as_path(), |root| {
                        root.insert(key.to_string(), Value::String(value.to_string()));
                        Ok(())
                    })
                })
            })
            .collect::<Vec<_>>();

        for worker in workers {
            worker
                .join()
                .expect("credential writer thread")
                .expect("credential update");
        }
        let root = read_one_credentials_root(path.as_path()).expect("parse credentials");
        assert_eq!(root.get("provider-a").and_then(Value::as_str), Some("secret-a"));
        assert_eq!(root.get("provider-b").and_then(Value::as_str), Some("secret-b"));
        let _ = fs::remove_dir_all(dir);
    }

    /// Pins the canonical root environment for a test and restores the prior
    /// values on drop.
    struct RootEnvGuard {
        prior: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl RootEnvGuard {
        fn set(primary: &std::path::Path, lower: &std::path::Path, home: &std::path::Path) -> Self {
            let prior = ["ZO_CONFIG_HOME", "ZO_HOME", "HOME"]
                .into_iter()
                .map(|key| (key, std::env::var_os(key)))
                .collect();
            std::env::set_var("ZO_CONFIG_HOME", primary);
            std::env::set_var("ZO_HOME", lower);
            std::env::set_var("HOME", home);
            Self { prior }
        }

        fn home_only(home: &std::path::Path) -> Self {
            let prior = ["ZO_CONFIG_HOME", "ZO_HOME", "HOME"]
                .into_iter()
                .map(|key| (key, std::env::var_os(key)))
                .collect();
            std::env::remove_var("ZO_CONFIG_HOME");
            std::env::remove_var("ZO_HOME");
            std::env::set_var("HOME", home);
            Self { prior }
        }
    }

    impl Drop for RootEnvGuard {
        fn drop(&mut self) {
            for (key, value) in &self.prior {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    #[test]
    fn legacy_forge_credentials_load_but_save_writes_primary_zo_only() {
        let _env_lock = crate::test_env_lock();
        let home = unique_root("legacy-home");
        let legacy = home.join(".forge");
        let primary = home.join(".zo");
        let _roots = RootEnvGuard::home_only(&home);
        fs::create_dir_all(&legacy).expect("legacy credentials dir");
        let legacy_contents = r#"{
  "oauth": {
    "accessToken": "legacy-access",
    "refreshToken": "legacy-refresh",
    "expiresAt": 123,
    "scopes": ["user:inference"]
  }
}
"#;
        fs::write(legacy.join("credentials.json"), legacy_contents)
            .expect("seed legacy credentials");

        let loaded = load_oauth_credentials()
            .expect("load legacy credentials")
            .expect("legacy oauth present");
        assert_eq!(loaded.access_token, "legacy-access");
        assert_eq!(loaded.refresh_token.as_deref(), Some("legacy-refresh"));

        let replacement = OAuthTokenSet {
            access_token: "zo-access".into(),
            refresh_token: Some("zo-refresh".into()),
            expires_at: Some(456),
            scopes: vec!["user:inference".into()],
        };
        save_oauth_credentials(&replacement).expect("save primary credentials");

        assert_eq!(
            credentials_path().expect("primary credentials path"),
            primary.join("credentials.json")
        );
        assert!(primary.join("credentials.json").exists());
        assert_eq!(
            fs::read_to_string(legacy.join("credentials.json"))
                .expect("legacy credentials after save"),
            legacy_contents,
            "legacy credentials must remain read-only"
        );
        assert_eq!(
            load_oauth_credentials().expect("reload primary credentials"),
            Some(replacement)
        );

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn lower_root_logout_stays_cleared_with_primary_tombstones() {
        let _env_lock = crate::test_env_lock();
        let primary = unique_root("primary-a");
        let lower = unique_root("lower-a");
        let home = unique_root("home-a");
        let _roots = RootEnvGuard::set(&primary, &lower, &home);

        fs::create_dir_all(&lower).expect("lower root dir");
        fs::write(
            lower.join("credentials.json"),
            r#"{
  "oauth": {"accessToken":"low-anthropic","refreshToken":"low-anthropic-rt","expiresAt":111,"scopes":["user:inference"]},
  "openai_oauth": {"accessToken":"low-openai","refreshToken":"low-openai-rt","expiresAt":222,"scopes":["openid"]},
  "google_code_assist_oauth": {"accessToken":"low-google","refreshToken":"low-google-rt","expiresAt":333,"scopes":["cloud"]}
}
"#,
        )
        .expect("seed lower credentials");

        // The merged effective view exposes every lower-root credential.
        assert!(load_oauth_credentials().expect("load anthropic").is_some());
        assert!(load_openai_oauth().expect("load openai").is_some());
        assert!(load_google_code_assist_oauth().expect("load google").is_some());

        // Primary-only logout.
        clear_oauth_credentials().expect("clear anthropic");
        clear_openai_oauth().expect("clear openai");
        clear_google_code_assist_oauth().expect("clear google");

        // A fresh read re-merges the untouched lower root; the deletions stick.
        assert!(load_oauth_credentials().expect("reload anthropic").is_none());
        assert!(load_openai_oauth().expect("reload openai").is_none());
        assert!(load_google_code_assist_oauth().expect("reload google").is_none());

        // The primary file records `null` tombstones (not absence), which is
        // what suppresses the still-present lower-root copies.
        let primary_json: Value = serde_json::from_str(
            &fs::read_to_string(primary.join("credentials.json")).expect("primary credentials"),
        )
        .expect("primary json");
        for key in ["oauth", "openai_oauth", "google_code_assist_oauth"] {
            assert_eq!(primary_json.get(key), Some(&Value::Null), "{key} tombstone");
        }

        let _ = fs::remove_dir_all(&primary);
        let _ = fs::remove_dir_all(&lower);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn lower_root_adapter_key_merges_and_tombstones_per_entry() {
        let _env_lock = crate::test_env_lock();
        let primary = unique_root("primary-b");
        let lower = unique_root("lower-b");
        let home = unique_root("home-b");
        let _roots = RootEnvGuard::set(&primary, &lower, &home);

        fs::create_dir_all(&lower).expect("lower dir");
        fs::write(
            lower.join("credentials.json"),
            r#"{"openai_compat_api_keys":{"DEEPSEEK_API_KEY":"sk-low-deepseek","OPENAI_API_KEY":"sk-low-openai"}}
"#,
        )
        .expect("seed lower adapter keys");
        fs::create_dir_all(&primary).expect("primary dir");
        fs::write(
            primary.join("credentials.json"),
            r#"{"openai_compat_api_keys":{"DEEPSEEK_API_KEY":"sk-primary-deepseek","OPENAI_API_KEY":null}}
"#,
        )
        .expect("seed primary adapter keys");

        // The collection map merges per entry: primary overrides one entry and
        // tombstones another, while the untouched lower entry survives.
        assert_eq!(
            load_openai_compat_api_key("DEEPSEEK_API_KEY").expect("deepseek"),
            Some("sk-primary-deepseek".to_string())
        );
        assert_eq!(
            load_openai_compat_api_key("OPENAI_API_KEY").expect("openai tombstoned"),
            None
        );

        // A subsequent primary-only save must not resurrect the tombstoned
        // entry when it rewrites the primary map.
        save_openai_compat_api_key("ANTHROPIC_API_KEY", "sk-new").expect("save new key");
        assert_eq!(
            load_openai_compat_api_key("OPENAI_API_KEY").expect("still tombstoned"),
            None
        );
        assert_eq!(
            load_openai_compat_api_key("DEEPSEEK_API_KEY").expect("deepseek after save"),
            Some("sk-primary-deepseek".to_string())
        );

        let _ = fs::remove_dir_all(&primary);
        let _ = fs::remove_dir_all(&lower);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn mixed_token_fields_are_atomic_across_roots() {
        let _env_lock = crate::test_env_lock();
        let primary = unique_root("primary-c");
        let lower = unique_root("lower-c");
        let home = unique_root("home-c");
        let _roots = RootEnvGuard::set(&primary, &lower, &home);

        fs::create_dir_all(&lower).expect("lower dir");
        fs::write(
            lower.join("credentials.json"),
            r#"{"oauth":{"accessToken":"low-at","refreshToken":"low-rt","expiresAt":1,"scopes":["low"]}}
"#,
        )
        .expect("seed lower oauth");
        fs::create_dir_all(&primary).expect("primary dir");
        fs::write(
            primary.join("credentials.json"),
            r#"{"oauth":{"accessToken":"primary-at","scopes":["primary"]}}
"#,
        )
        .expect("seed primary oauth");

        let loaded = load_oauth_credentials()
            .expect("load merged oauth")
            .expect("oauth present");
        // The primary object wins wholesale; lower-root fields are never
        // combined into it, so the missing refresh token stays missing.
        assert_eq!(loaded.access_token, "primary-at");
        assert_eq!(loaded.refresh_token, None);
        assert_eq!(loaded.expires_at, None);
        assert_eq!(loaded.scopes, vec!["primary".to_string()]);

        let _ = fs::remove_dir_all(&primary);
        let _ = fs::remove_dir_all(&lower);
        let _ = fs::remove_dir_all(&home);
    }
}
