//! Claude Code keychain session credentials — read, evaluate, refresh, write back.
//!
//! Mirrors the Claude Code CLI's own OAuth mechanism. The macOS keychain item
//! `Claude Code-credentials` holds `{"claudeAiOauth": {accessToken, refreshToken,
//! expiresAt (Unix ms), scopes, …}}`. When the access token expires, Claude Code
//! refreshes it against the shared token endpoint (`client_id` `9d1c250a…`) and
//! writes the new token set back to the keychain. Zo previously stopped at
//! "expired → fall back", which stranded every session on a scope-less fallback
//! token whenever the desktop app wasn't around to refresh — the recurring
//! "keychain token expired / lacks user:inference" warnings. This module
//! completes the parity: expired + refresh token present → refresh → write back
//! → use. The write-back keeps the keychain the single source of truth shared
//! with Claude Code (required if the server rotates refresh tokens: without it,
//! consuming the keychain's refresh token would strand Claude Code itself).
//!
//! Living in the `api` crate (not the CLI) so the sub-agent provider path uses
//! the *same* resolution chain as the interactive client instead of skipping
//! the keychain.

use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use core_types::{OAuthConfig, OAuthRefreshRequest};
use serde_json::Value;

use super::{AnthropicClient, AuthSource, OAuthTokenSet, read_base_url};
use crate::error::ApiError;

/// Keychain service name Claude Code stores its OAuth bundle under.
const KEYCHAIN_SERVICE: &str = "Claude Code-credentials";

/// Treat a token expiring within this window as already expired and refresh it
/// proactively, instead of letting the request race the boundary and 401.
/// Milliseconds because the keychain's `expiresAt` is Unix ms; mirrors the
/// 60-second `OAUTH_EXPIRY_BUFFER_SECS` used for zo-saved tokens.
const KEYCHAIN_EXPIRY_BUFFER_MS: u64 = 60_000;

/// After a refresh attempt fails (network down, refresh token revoked), don't
/// re-attempt the network round-trip for this long. The read path runs at every
/// turn boundary while auth is in fallback, so an unguarded failure would
/// hammer the token endpoint once per turn.
const REFRESH_FAILURE_COOLDOWN: Duration = Duration::from_secs(60);

/// Kill switch: set `ZO_DISABLE_KEYCHAIN=1` to skip the Claude Code keychain
/// entirely (also keeps unit tests hermetic on developer machines where the
/// real keychain item exists).
const DISABLE_KEYCHAIN_ENV: &str = "ZO_DISABLE_KEYCHAIN";

/// The official Claude Code subscription OAuth application. `platform.claude.com`
/// is the developer/console flow, which mints tokens the server refuses to grant
/// `user:inference` on — every `/v1/messages` then 403s `OAuth token does not
/// meet scope requirement`. The subscription flow authorizes on `claude.ai` and
/// exchanges/refreshes on `console.anthropic.com`; both share this client id,
/// which is also the client id the keychain's refresh token was minted for.
#[must_use]
pub fn claude_code_oauth_config() -> OAuthConfig {
    OAuthConfig {
        client_id: String::from("9d1c250a-e61b-44d9-88ed-5944d1962f5e"),
        authorize_url: String::from("https://claude.ai/oauth/authorize"),
        token_url: String::from("https://console.anthropic.com/v1/oauth/token"),
        callback_port: None,
        manual_redirect_url: None,
        scopes: vec![
            String::from("user:profile"),
            String::from("user:inference"),
            String::from("user:sessions:claude_code"),
            String::from("org:create_api_key"),
            String::from("user:mcp_servers"),
            String::from("user:file_upload"),
        ],
        client_secret: None,
    }
}

/// A usable Claude Code keychain session: the bearer plus its expiry so the
/// caller can schedule a proactive re-read before the next lapse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeychainSession {
    pub access_token: String,
    /// Unix milliseconds, when the blob records one.
    pub expires_at_ms: Option<u64>,
}

/// Result of inspecting a Claude Code keychain credential blob.
#[derive(Debug, PartialEq, Eq)]
enum KeychainOutcome {
    /// Usable, unexpired session token carrying `user:inference`.
    Usable(String),
    /// Token past (or within the buffer of) its `expiresAt`.
    Expired,
    /// Token present and unexpired but its `scopes` list omits
    /// `user:inference`, so `/v1/messages` would 403 — unusable for inference.
    MissingInferenceScope,
    /// No usable `claudeAiOauth.accessToken` in the blob.
    Absent,
}

/// Pure evaluation of a parsed keychain JSON blob against the current time
/// (Unix milliseconds). Split out from the `security` shell-out so the expiry
/// and scope rules are unit-testable without touching the real keychain.
fn evaluate_keychain_credentials(creds: &Value, now_ms: u64) -> KeychainOutcome {
    let Some(oauth) = creds.get("claudeAiOauth") else {
        return KeychainOutcome::Absent;
    };
    let Some(token) = oauth.get("accessToken").and_then(Value::as_str) else {
        return KeychainOutcome::Absent;
    };
    if token.is_empty() {
        return KeychainOutcome::Absent;
    }
    if let Some(expires_at) = oauth.get("expiresAt").and_then(Value::as_u64) {
        if now_ms.saturating_add(KEYCHAIN_EXPIRY_BUFFER_MS) > expires_at {
            return KeychainOutcome::Expired;
        }
    }
    // When a `scopes` list is present, require `user:inference`; an absent list
    // is treated permissively (older blobs predate the field).
    if let Some(scopes) = oauth.get("scopes").and_then(Value::as_array) {
        let has_inference = scopes
            .iter()
            .filter_map(Value::as_str)
            .any(|scope| scope == "user:inference");
        if !has_inference {
            return KeychainOutcome::MissingInferenceScope;
        }
    }
    KeychainOutcome::Usable(token.to_string())
}

/// Read the Claude Code keychain session, refreshing it first when expired —
/// the same lifecycle Claude Code itself runs. Returns `None` when the keychain
/// has no usable bundle and refresh is impossible/failed (callers then fall
/// back to zo-managed auth).
#[must_use]
pub fn read_claude_code_keychain_session() -> Option<KeychainSession> {
    if std::env::var_os(DISABLE_KEYCHAIN_ENV).is_some() {
        return None;
    }
    let blob = read_keychain_blob()?;
    let now_ms = now_unix_millis();
    match evaluate_keychain_credentials(&blob, now_ms) {
        KeychainOutcome::Usable(access_token) => {
            eprintln!("\x1b[2mUsing Claude Code session credentials.\x1b[0m");
            let expires_at_ms = blob
                .get("claudeAiOauth")
                .and_then(|oauth| oauth.get("expiresAt"))
                .and_then(Value::as_u64);
            Some(KeychainSession {
                access_token,
                expires_at_ms,
            })
        }
        KeychainOutcome::Expired => {
            let refreshed = refresh_expired_keychain_blob(&blob);
            match &refreshed {
                Some(_) => {
                    eprintln!("\x1b[2mRefreshed Claude Code session credentials.\x1b[0m");
                }
                None => eprintln!(
                    "\x1b[33mClaude Code keychain token expired and could not be refreshed, falling back to Zo auth.\x1b[0m"
                ),
            }
            refreshed
        }
        KeychainOutcome::MissingInferenceScope => {
            eprintln!(
                "\x1b[33mClaude Code keychain token lacks user:inference scope, falling back to Zo auth.\x1b[0m"
            );
            None
        }
        KeychainOutcome::Absent => None,
    }
}

/// Token-only convenience over [`read_claude_code_keychain_session`].
#[must_use]
pub fn read_claude_code_keychain_token() -> Option<String> {
    read_claude_code_keychain_session().map(|session| session.access_token)
}

/// Refresh an expired keychain blob via its `refreshToken`, persist the result
/// (keychain write-back + zo credential mirror), and return the fresh
/// session. `None` when the blob has no refresh token, a recent attempt already
/// failed (cool-down), or the token endpoint rejects the refresh.
fn refresh_expired_keychain_blob(blob: &Value) -> Option<KeychainSession> {
    let oauth = blob.get("claudeAiOauth")?;
    let refresh_token = oauth
        .get("refreshToken")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty())?
        .to_string();

    if refresh_failure_cooldown_active() {
        return None;
    }

    // Re-request the original grant's scopes; an empty/absent list falls back
    // to the standard subscription scopes (the server still bounds the result
    // by the original grant).
    let scopes: Vec<String> = oauth
        .get("scopes")
        .and_then(Value::as_array)
        .map(|scopes| {
            scopes
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let config = claude_code_oauth_config();
    let request = OAuthRefreshRequest::from_config(
        &config,
        refresh_token.clone(),
        (!scopes.is_empty()).then_some(scopes),
    );

    let refreshed = match refresh_token_set_on_own_thread(&config, &request) {
        Ok(refreshed) => refreshed,
        Err(error) => {
            mark_refresh_failure();
            eprintln!("\x1b[33mClaude Code OAuth refresh failed: {error}\x1b[0m");
            return None;
        }
    };

    // The endpoint may rotate the refresh token; keep the old one only when no
    // replacement arrives.
    let resolved_refresh_token = refreshed
        .refresh_token
        .clone()
        .unwrap_or_else(|| refresh_token.clone());
    let rotated = resolved_refresh_token != refresh_token;

    let updated_blob = updated_keychain_blob(blob, &refreshed, &resolved_refresh_token);
    let wrote_back =
        keychain_account().is_some_and(|account| write_keychain_blob(&account, &updated_blob));
    if !wrote_back && rotated {
        // Claude Code still holds the now-invalidated refresh token; it will
        // ask the user to log in again next time it runs. Zo stays healthy
        // via the credential mirror below.
        eprintln!(
            "\x1b[33mwarning: refreshed Claude Code OAuth token could not be written back to the keychain; Claude Code may require a re-login.\x1b[0m"
        );
    }

    // Mirror into zo's own credential store so the fresh token set survives
    // a refused keychain write (and upgrades any stale scope-less `zo login`
    // token in passing — the mirror carries `user:inference`).
    let _ = crate::oauth_store::save_oauth_credentials(&core_types::OAuthTokenSet {
        access_token: refreshed.access_token.clone(),
        refresh_token: Some(resolved_refresh_token),
        expires_at: refreshed.expires_at,
        scopes: refreshed.scopes.clone(),
    });

    Some(KeychainSession {
        access_token: refreshed.access_token,
        expires_at_ms: refreshed.expires_at.map(|secs| secs.saturating_mul(1000)),
    })
}

/// Hard bounds on the refresh round-trip. Credential resolution can run on a
/// startup/turn-boundary path; a blackholed network (offline, sandboxed test
/// runner) must bound the wait instead of hanging that path forever — the
/// shared HTTP pool deliberately carries no overall timeout for streaming, so
/// the refresh uses its own client.
const REFRESH_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REFRESH_TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

/// Run the token-endpoint refresh on a dedicated OS thread with its own
/// single-threaded runtime. The read path is synchronous but gets called from
/// every flavor of context — plain startup code, `spawn_blocking` workers, and
/// agent threads already inside `Handle::block_on` — and a nested
/// `Runtime::new().block_on` panics in the last case. A scoped thread has no
/// ambient tokio context, so this is safe everywhere; the cost (one short-lived
/// thread per ~8-hourly refresh) is negligible.
pub(super) fn refresh_token_set_on_own_thread(
    config: &OAuthConfig,
    request: &OAuthRefreshRequest,
) -> Result<OAuthTokenSet, ApiError> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(ApiError::from)?;
                let http = reqwest::Client::builder()
                    .connect_timeout(REFRESH_CONNECT_TIMEOUT)
                    .timeout(REFRESH_TOTAL_TIMEOUT)
                    .build()
                    .map_err(ApiError::from)?;
                let client = AnthropicClient::from_auth(AuthSource::None)
                    .with_base_url(read_base_url())
                    .with_http_client(http);
                runtime.block_on(client.refresh_oauth_token(config, request))
            })
            .join()
            .map_err(|_| ApiError::Auth("keychain OAuth refresh thread panicked".to_string()))?
    })
}

/// Pure blob update: replace the OAuth fields the refresh changed, preserve
/// everything else (`subscriptionType`, unknown future fields) so the write-back
/// never strips data Claude Code relies on. `expiresAt` is converted from the
/// token set's Unix seconds to the blob's Unix milliseconds.
fn updated_keychain_blob(blob: &Value, refreshed: &OAuthTokenSet, refresh_token: &str) -> Value {
    let mut updated = blob.clone();
    if let Some(oauth) = updated
        .get_mut("claudeAiOauth")
        .and_then(Value::as_object_mut)
    {
        oauth.insert(
            "accessToken".to_string(),
            Value::String(refreshed.access_token.clone()),
        );
        oauth.insert(
            "refreshToken".to_string(),
            Value::String(refresh_token.to_string()),
        );
        if let Some(expires_at) = refreshed.expires_at {
            oauth.insert(
                "expiresAt".to_string(),
                Value::from(expires_at.saturating_mul(1000)),
            );
        }
        if !refreshed.scopes.is_empty() {
            oauth.insert("scopes".to_string(), Value::from(refreshed.scopes.clone()));
        }
    }
    updated
}

fn read_keychain_blob() -> Option<Value> {
    let output = Command::new("security")
        .args(["find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    serde_json::from_str(raw.trim()).ok()
}

/// The keychain account the credential item is stored under, needed to address
/// the write-back. Parsed from the item's attribute listing (`-w` prints only
/// the secret).
fn keychain_account() -> Option<String> {
    let output = Command::new("security")
        .args(["find-generic-password", "-s", KEYCHAIN_SERVICE])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_keychain_account(&String::from_utf8_lossy(&output.stdout))
}

/// Extract the account from `security find-generic-password` attribute output:
/// a line of the form `    "acct"<blob>="joe"`. Hex-encoded (non-UTF-8)
/// accounts are not handled — the caller skips the write-back rather than
/// guessing.
fn parse_keychain_account(attributes: &str) -> Option<String> {
    let line = attributes
        .lines()
        .find(|line| line.trim_start().starts_with("\"acct\""))?;
    let (_, value) = line.split_once("=\"")?;
    let account = value.strip_suffix('"')?;
    (!account.is_empty()).then(|| account.to_string())
}

/// Best-effort keychain write-back (`-U` updates the existing item in place).
/// The secret travels via argv, which is briefly visible to same-user processes
/// — the same trust boundary as the existing `-w` read (any same-user process
/// could read the item directly), so this adds no new exposure. Uses the same
/// `security` binary the read path uses, so an item ACL that admits the read
/// admits the write without a new GUI prompt.
fn write_keychain_blob(account: &str, blob: &Value) -> bool {
    Command::new("security")
        .args([
            "add-generic-password",
            "-U",
            "-a",
            account,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
            &blob.to_string(),
        ])
        .output()
        .is_ok_and(|output| output.status.success())
}

static LAST_REFRESH_FAILURE: Mutex<Option<Instant>> = Mutex::new(None);

fn refresh_failure_cooldown_active() -> bool {
    LAST_REFRESH_FAILURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some_and(|at| at.elapsed() < REFRESH_FAILURE_COOLDOWN)
}

fn mark_refresh_failure() {
    *LAST_REFRESH_FAILURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| u64::try_from(elapsed.as_millis()).ok())
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        KEYCHAIN_EXPIRY_BUFFER_MS, KeychainOutcome, REFRESH_CONNECT_TIMEOUT, REFRESH_TOTAL_TIMEOUT,
        claude_code_oauth_config, evaluate_keychain_credentials, parse_keychain_account,
        updated_keychain_blob,
    };
    use crate::providers::anthropic::OAuthTokenSet;
    use std::time::Duration;

    /// A `now` far enough from the fixture expiries that the proactive buffer
    /// does not flip outcomes unintentionally.
    const NOW_MS: u64 = 1_000_000_000;
    const FUTURE_MS: u64 = NOW_MS + KEYCHAIN_EXPIRY_BUFFER_MS + 1;

    #[test]
    fn oauth_refresh_timeout_is_bounded_control_plane_timeout() {
        // Generation streams deliberately have no total request timeout, but
        // OAuth refresh is a short control-plane POST and must stay bounded so
        // startup/turn-boundary credential resolution cannot hang indefinitely.
        assert_eq!(REFRESH_CONNECT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(REFRESH_TOTAL_TIMEOUT, Duration::from_secs(30));
        assert!(REFRESH_CONNECT_TIMEOUT < REFRESH_TOTAL_TIMEOUT);
    }

    #[test]
    fn keychain_usable_with_unexpired_inference_scope() {
        let creds = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-abc",
                "expiresAt": FUTURE_MS,
                "scopes": ["user:profile", "user:inference"],
            }
        });
        assert_eq!(
            evaluate_keychain_credentials(&creds, NOW_MS),
            KeychainOutcome::Usable("sk-ant-oat01-abc".to_string())
        );
    }

    #[test]
    fn keychain_expired_past_expiry() {
        let creds = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-abc",
                "expiresAt": NOW_MS - 1,
                "scopes": ["user:inference"],
            }
        });
        assert_eq!(
            evaluate_keychain_credentials(&creds, NOW_MS),
            KeychainOutcome::Expired
        );
    }

    #[test]
    fn keychain_expiring_within_buffer_counts_as_expired() {
        // Proactive refresh: a token lapsing in under the buffer must refresh
        // now instead of racing the boundary and 401ing mid-turn.
        let creds = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-abc",
                "expiresAt": NOW_MS + KEYCHAIN_EXPIRY_BUFFER_MS - 1,
                "scopes": ["user:inference"],
            }
        });
        assert_eq!(
            evaluate_keychain_credentials(&creds, NOW_MS),
            KeychainOutcome::Expired
        );
    }

    #[test]
    fn keychain_rejected_without_inference_scope() {
        // The exact failure behind the 403: a token whose scopes omit
        // `user:inference` must not be handed to the inference path.
        let creds = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-abc",
                "expiresAt": FUTURE_MS,
                "scopes": ["user:profile", "org:create_api_key"],
            }
        });
        assert_eq!(
            evaluate_keychain_credentials(&creds, NOW_MS),
            KeychainOutcome::MissingInferenceScope
        );
    }

    #[test]
    fn keychain_absent_without_token() {
        assert_eq!(
            evaluate_keychain_credentials(&serde_json::json!({}), NOW_MS),
            KeychainOutcome::Absent
        );
        let empty = serde_json::json!({ "claudeAiOauth": { "accessToken": "" } });
        assert_eq!(
            evaluate_keychain_credentials(&empty, NOW_MS),
            KeychainOutcome::Absent
        );
    }

    #[test]
    fn keychain_permissive_when_scopes_field_absent() {
        // Older keychain blobs predate the scopes field; don't lock those out.
        let creds = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-abc",
                "expiresAt": FUTURE_MS,
            }
        });
        assert_eq!(
            evaluate_keychain_credentials(&creds, NOW_MS),
            KeychainOutcome::Usable("sk-ant-oat01-abc".to_string())
        );
    }

    #[test]
    fn keychain_without_expiry_field_is_not_expired() {
        let creds = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-abc",
                "scopes": ["user:inference"],
            }
        });
        assert_eq!(
            evaluate_keychain_credentials(&creds, u64::MAX),
            KeychainOutcome::Usable("sk-ant-oat01-abc".to_string())
        );
    }

    #[test]
    fn updated_blob_replaces_oauth_fields_and_preserves_siblings() {
        let blob = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
                "expiresAt": 1_111_u64,
                "scopes": ["user:inference"],
                "subscriptionType": "max",
            },
            "otherTopLevel": true,
        });
        let refreshed = OAuthTokenSet {
            access_token: "new-access".to_string(),
            refresh_token: Some("new-refresh".to_string()),
            expires_at: Some(2_000),
            scopes: vec!["user:inference".to_string(), "user:profile".to_string()],
        };
        let updated = updated_keychain_blob(&blob, &refreshed, "new-refresh");
        let oauth = updated.get("claudeAiOauth").expect("oauth object");
        assert_eq!(oauth["accessToken"], "new-access");
        assert_eq!(oauth["refreshToken"], "new-refresh");
        // Unix seconds from the token endpoint → Unix milliseconds in the blob.
        assert_eq!(oauth["expiresAt"], 2_000_000_u64);
        assert_eq!(
            oauth["scopes"],
            serde_json::json!(["user:inference", "user:profile"])
        );
        // Fields the refresh does not own survive untouched.
        assert_eq!(oauth["subscriptionType"], "max");
        assert_eq!(updated["otherTopLevel"], true);
    }

    #[test]
    fn updated_blob_keeps_old_expiry_and_scopes_when_response_omits_them() {
        let blob = serde_json::json!({
            "claudeAiOauth": {
                "accessToken": "old-access",
                "refreshToken": "old-refresh",
                "expiresAt": 1_111_u64,
                "scopes": ["user:inference"],
            }
        });
        let refreshed = OAuthTokenSet {
            access_token: "new-access".to_string(),
            refresh_token: None,
            expires_at: None,
            scopes: Vec::new(),
        };
        // No rotation: caller passes the old refresh token through.
        let updated = updated_keychain_blob(&blob, &refreshed, "old-refresh");
        let oauth = updated.get("claudeAiOauth").expect("oauth object");
        assert_eq!(oauth["accessToken"], "new-access");
        assert_eq!(oauth["refreshToken"], "old-refresh");
        assert_eq!(oauth["expiresAt"], 1_111_u64);
        assert_eq!(oauth["scopes"], serde_json::json!(["user:inference"]));
    }

    #[test]
    fn parses_account_from_security_attribute_listing() {
        let attributes = concat!(
            "keychain: \"/Users/joe/Library/Keychains/login.keychain-db\"\n",
            "version: 512\n",
            "class: \"genp\"\n",
            "attributes:\n",
            "    0x00000007 <blob>=\"Claude Code-credentials\"\n",
            "    \"acct\"<blob>=\"joe\"\n",
            "    \"svce\"<blob>=\"Claude Code-credentials\"\n",
        );
        assert_eq!(parse_keychain_account(attributes), Some("joe".to_string()));
    }

    #[test]
    fn account_parse_rejects_missing_or_unquoted_forms() {
        assert_eq!(parse_keychain_account(""), None);
        // Hex-encoded (non-UTF-8) account: skip the write-back, don't guess.
        assert_eq!(
            parse_keychain_account("    \"acct\"<blob>=0x6A6F65\n"),
            None
        );
        assert_eq!(parse_keychain_account("    \"acct\"<blob>=\"\"\n"), None);
    }

    #[test]
    fn subscription_oauth_config_targets_the_claude_ai_flow() {
        // Regression guard for the 403 class of bugs: the config must stay on
        // the subscription flow (claude.ai authorize, console token endpoint)
        // and keep requesting `user:inference`.
        let config = claude_code_oauth_config();
        assert_eq!(config.client_id, "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
        assert!(config.authorize_url.starts_with("https://claude.ai/"));
        assert!(
            config
                .token_url
                .starts_with("https://console.anthropic.com/")
        );
        assert!(config.scopes.iter().any(|scope| scope == "user:inference"));
    }
}
