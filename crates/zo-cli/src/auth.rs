use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, TcpListener};
use std::process::Command;
use std::time::{Duration, Instant};

use api::{AnthropicClient, AuthSource};
use runtime::{
    clear_oauth_credentials, clear_openai_oauth, generate_pkce_pair, generate_state,
    loopback_redirect_uri, parse_oauth_callback_request_target, save_oauth_credentials,
    save_openai_oauth, OAuthAuthorizationRequest, OAuthConfig, OAuthTokenExchangeRequest,
};

use crate::DEFAULT_OAUTH_CALLBACK_PORT;

const OAUTH_CALLBACK_TIMEOUT: Duration = Duration::from_secs(180);
const OAUTH_CALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(10);
const OAUTH_CALLBACK_ACCEPT_POLL: Duration = Duration::from_millis(50);

/// The Claude Code subscription OAuth application — single definition lives in
/// the api crate ([`api::claude_code_oauth_config`]) so the login flow, the
/// interactive client's refresh path, and sub-agent auth resolution can never
/// drift apart on client id / endpoints / scopes again.
pub(crate) fn default_oauth_config() -> OAuthConfig {
    api::claude_code_oauth_config()
}

pub(crate) fn run_login_provider(provider: &str) -> Result<(), Box<dyn std::error::Error>> {
    match provider {
        "claude" | "anthropic" => run_login_claude(),
        "openai" | "gpt" | "codex" => run_login_openai_oauth(),
        "google" | "gemini" => run_login_google(),
        "google-adc" | "gemini-adc" => run_login_google_adc(),
        "xai" | "grok" => {
            run_login_xai();
            Ok(())
        }
        _ => Err(format!(
            "Unknown provider: {provider}. Use: claude, openai, google, google-adc, xai"
        )
        .into()),
    }
}

/// Drive an async OAuth call to completion whether or not a tokio runtime is
/// already active — the CLI is entered from both sync (`zo login`) and async
/// (in-session `/login`) contexts.
fn block_on_oauth<F, T>(future: F) -> Result<T, Box<dyn std::error::Error>>
where
    F: std::future::Future<Output = Result<T, api::ApiError>>,
{
    api::sync_bridge::run_blocking(future).map_err(Into::into)
}

/// Wait for the loopback callback and return the authorization `code`, after
/// confirming the returned CSRF `state` matches what we sent.
fn await_validated_callback(
    listeners: OAuthCallbackListeners,
    expected_state: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let callback = listeners.wait()?;
    if let Some(error) = callback.error {
        let description = callback
            .error_description
            .unwrap_or_else(|| "authorization failed".to_string());
        return Err(io::Error::other(format!("{error}: {description}")).into());
    }
    let code = callback.code.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "callback did not include code")
    })?;
    let returned_state = callback.state.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "callback did not include state")
    })?;
    if returned_state != expected_state {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "oauth state mismatch").into());
    }
    Ok(code)
}

fn run_login_claude() -> Result<(), Box<dyn std::error::Error>> {
    let oauth = default_oauth_config();
    let callback_port = oauth.callback_port.unwrap_or(DEFAULT_OAUTH_CALLBACK_PORT);
    let redirect_uri = oauth
        .manual_redirect_url
        .clone()
        .unwrap_or_else(|| format!("http://localhost:{callback_port}/callback"));
    let pkce = generate_pkce_pair()?;
    let state = generate_state()?;
    let authorize_url =
        OAuthAuthorizationRequest::from_config(&oauth, redirect_uri.clone(), state.clone(), &pkce)
            .build_url();

    let callback = OAuthCallbackListeners::bind(callback_port)?;

    if !crate::tui_active() {
        println!("Opening browser to sign in to Claude...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback, &state)?;
    let client = AnthropicClient::from_auth(AuthSource::None).with_base_url(api::read_base_url());
    let exchange_request =
        OAuthTokenExchangeRequest::from_config(&oauth, code, state, pkce.verifier, redirect_uri);
    let token_set = block_on_oauth(client.exchange_oauth_code(&oauth, &exchange_request))?;

    let scope_warning = missing_inference_scope_warning(&token_set.scopes);
    save_oauth_credentials(&runtime::OAuthTokenSet {
        access_token: token_set.access_token,
        refresh_token: token_set.refresh_token,
        expires_at: token_set.expires_at,
        scopes: token_set.scopes,
    })?;
    if let Some(warning) = scope_warning {
        eprintln!("{warning}");
    }
    if !crate::tui_active() {
        println!("Zo OAuth login complete (claude).");
    }
    Ok(())
}

/// OAuth-first guard: a `zo login` token without the `user:inference`
/// scope 403s on every `/v1/messages` — historically discovered only at the
/// first turn. Surface it at login time instead. Empty scope lists stay
/// silent: the server simply did not report scopes, and warning there would
/// be a false alarm on every login.
fn missing_inference_scope_warning(scopes: &[String]) -> Option<String> {
    if scopes.is_empty() || scopes.iter().any(|scope| scope == "user:inference") {
        return None;
    }
    Some(format!(
        "warning: this OAuth token lacks the `user:inference` scope (got: {}) — API calls \
         will 403. Zo will prefer the Claude Code keychain when available; re-run \
         `zo login` if inference access was expected.",
        scopes.join(" ")
    ))
}

/// ChatGPT (OpenAI) OAuth sign-in. The access token is sent straight to the
/// ChatGPT backend, so — unlike the other providers — there is no API key to
/// export; the token bundle (carrying its `account_id`) is persisted for the
/// backend client to consume.
fn run_login_openai_oauth() -> Result<(), Box<dyn std::error::Error>> {
    let config = api::openai_oauth_config();
    let callback_port = config
        .callback_port
        .unwrap_or(api::OPENAI_OAUTH_CALLBACK_PORT);
    let redirect_uri = format!("http://localhost:{callback_port}/auth/callback");
    let pkce = generate_pkce_pair()?;
    let state = generate_state()?;
    let authorize_url = api::openai_authorize_url(&config, &redirect_uri, state.clone(), &pkce);

    let callback = OAuthCallbackListeners::bind(callback_port)?;

    if !crate::tui_active() {
        println!("Opening browser to sign in to ChatGPT (OpenAI)...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback, &state)?;
    let tokens = block_on_oauth(api::exchange_openai_code(
        &code,
        &pkce.verifier,
        &redirect_uri,
    ))?;
    if tokens.account_id.is_none() {
        eprintln!("warning: no ChatGPT account_id in token — backend calls may be rejected.");
    }
    save_openai_oauth(&tokens)?;
    if !crate::tui_active() {
        println!(
            "ChatGPT OAuth login complete. Use /model {} to chat with your subscription.",
            api::OPENAI_LATEST_MODEL_ALIAS
        );
    }
    Ok(())
}

fn run_login_google() -> Result<(), Box<dyn std::error::Error>> {
    let config = api::google_code_assist_oauth_config()?;
    let callback_port = config.callback_port.unwrap_or(DEFAULT_OAUTH_CALLBACK_PORT);
    let redirect_uri = api::google_code_assist_redirect_uri(callback_port);
    let pkce = generate_pkce_pair()?;
    let state = generate_state()?;
    let authorize_url =
        api::google_code_assist_authorize_url(&config, &redirect_uri, state.clone(), &pkce);

    let callback = OAuthCallbackListeners::bind(callback_port)?;

    if !crate::tui_active() {
        println!("Opening browser to sign in to Google Gemini...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback, &state)?;
    let tokens = block_on_oauth(api::exchange_google_code_assist_code(
        &code,
        &pkce.verifier,
        &redirect_uri,
    ))?;
    api::save_google_code_assist_oauth(&tokens)?;

    match block_on_oauth(api::google_code_assist_setup_saved_user()) {
        Ok(project) if !crate::tui_active() => {
            if let Some(project) = project {
                println!(
                    "Google Gemini OAuth login complete (project: {project}). Use /model {} to switch.",
                    api::GOOGLE_LATEST_MODEL_ALIAS
                );
            } else {
                println!(
                    "Google Gemini OAuth login complete. Use /model {} to switch.",
                    api::GOOGLE_LATEST_MODEL_ALIAS
                );
            }
        }
        Ok(_) => {}
        Err(error) => {
            eprintln!(
                "warning: Google Gemini OAuth token was saved, but Code Assist setup check failed: {error}"
            );
            if !crate::tui_active() {
                println!(
                    "Google Gemini OAuth login complete. Use /model {} to switch.",
                    api::GOOGLE_LATEST_MODEL_ALIAS
                );
            }
        }
    }
    Ok(())
}

fn run_login_google_adc() -> Result<(), Box<dyn std::error::Error>> {
    let scopes = api::google_gemini_oauth_scopes_csv();
    let mut command = Command::new("gcloud");
    command.args(["auth", "application-default", "login", "--scopes", &scopes]);
    if let Some(client_id_file) = google_oauth_client_id_file() {
        command.arg(format!("--client-id-file={client_id_file}"));
    }

    if !crate::tui_active() {
        println!(
            "Opening browser to sign in to Google Gemini via Application Default Credentials..."
        );
        println!(
            "If gcloud is not installed, Zo will use {}=/path/to/client_secret.json for built-in OAuth.",
            api::GOOGLE_OAUTH_CLIENT_ID_FILE_ENV
        );
    }

    match command.status() {
        Ok(status) if status.success() => {
            let _token = block_on_oauth(api::google_gemini_access_token())?;
            if !crate::tui_active() {
                println!(
                    "Google Gemini OAuth login complete. Use /model {} to switch.",
                    api::GOOGLE_LATEST_MODEL_ALIAS
                );
            }
            Ok(())
        }
        Ok(status) if google_oauth_client_id_file().is_some() => {
            if !crate::tui_active() {
                eprintln!(
                    "gcloud auth application-default login failed with status {status}; falling back to Zo-managed Google OAuth."
                );
            }
            run_login_google_builtin_oauth()
        }
        Ok(status) => Err(format!(
            "gcloud auth application-default login failed with status {status}. \
             For OAuth without gcloud, set {}=/path/to/client_secret.json and retry `/login google`.",
            api::GOOGLE_OAUTH_CLIENT_ID_FILE_ENV
        )
        .into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => run_login_google_builtin_oauth(),
        Err(error) => Err(io::Error::new(
            error.kind(),
            format!("failed to run `gcloud auth application-default login`: {error}"),
        )
        .into()),
    }
}

fn google_oauth_client_id_file() -> Option<String> {
    std::env::var(api::GOOGLE_OAUTH_CLIENT_ID_FILE_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn run_login_google_builtin_oauth() -> Result<(), Box<dyn std::error::Error>> {
    let Some(client_id_file) = google_oauth_client_id_file() else {
        return Err(format!(
            "gcloud is not installed, and built-in Google OAuth needs a desktop OAuth client file. \
             Create/download a Google OAuth client JSON and retry with {}=/path/to/client_secret.json. \
             This keeps Gemini on OAuth/ADC; GOOGLE_API_KEY is not required.",
            api::GOOGLE_OAUTH_CLIENT_ID_FILE_ENV
        )
        .into());
    };

    let client = api::load_google_oauth_client_config(&client_id_file)?;
    let redirect_uri = loopback_redirect_uri(DEFAULT_OAUTH_CALLBACK_PORT);
    let pkce = generate_pkce_pair()?;
    let state = generate_state()?;
    let authorize_url = api::google_oauth_authorize_url(&client, &redirect_uri, &state, &pkce);

    let callback = OAuthCallbackListeners::bind(DEFAULT_OAUTH_CALLBACK_PORT)?;

    if !crate::tui_active() {
        println!("Opening browser to sign in to Google Gemini (Zo-managed OAuth)...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback, &state)?;
    let saved = block_on_oauth(api::exchange_google_oauth_code_and_save_adc(
        &client,
        &code,
        &pkce.verifier,
        &redirect_uri,
    ))?;
    if !crate::tui_active() {
        println!(
            "Google Gemini OAuth login complete. Saved ADC credentials to {}. Use /model {} to switch.",
            saved.path.display(),
            api::GOOGLE_LATEST_MODEL_ALIAS
        );
    }
    Ok(())
}

fn run_login_xai() {
    println!("xAI Grok: Set XAI_API_KEY manually:");
    println!("  export XAI_API_KEY=xai-...");
    println!("  Then use /model grok");
    println!("\nGet a key at: https://console.x.ai");
}

pub(crate) fn run_logout() -> Result<(), Box<dyn std::error::Error>> {
    let delegated = delegate_claude_auth(&["auth", "logout"]);

    clear_oauth_credentials()?;
    clear_openai_oauth()?;
    api::clear_google_code_assist_oauth()?;
    if let Err(error) = delegated {
        eprintln!(
            "zo: Claude CLI logout failed; cleared Zo OAuth credentials anyway: {error}"
        );
    }
    println!(
        "Zo OAuth credentials cleared (Claude, ChatGPT, and Google Gemini). Google ADC credentials are still managed by gcloud."
    );
    Ok(())
}

fn delegate_claude_auth(args: &[&str]) -> Result<bool, Box<dyn std::error::Error>> {
    let Ok(status) = Command::new("claude").args(args).status() else {
        return Ok(false);
    };
    if status.success() {
        return Ok(true);
    }
    Err(format!("claude {} failed with status {status}", args.join(" ")).into())
}

pub(crate) use runtime::open_browser;

#[cfg(unix)]
fn loopback_family_unavailable(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::AddrNotAvailable
        || error.raw_os_error() == Some(nix::libc::EAFNOSUPPORT)
}

#[cfg(not(unix))]
fn loopback_family_unavailable(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::AddrNotAvailable
}

fn callback_bind_error(port: u16, address: IpAddr, error: &io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("oauth callback port {port} could not be opened on {address}: {error}"),
    )
}

/// Loopback addresses the callback is opened on, in the order they are tried.
///
/// Named rather than inlined in the loop because the order is observable: the
/// all-unavailable error reports whichever address came last.
const LOOPBACK_CALLBACK_ADDRESSES: [IpAddr; 2] = [
    IpAddr::V6(Ipv6Addr::LOCALHOST),
    IpAddr::V4(Ipv4Addr::LOCALHOST),
];

/// Decide the outcome of opening the callback port on every loopback address.
///
/// `bind_one` is injected so the decisions here — which failures abort the whole
/// call, and which merely skip a family — are reachable from tests without a
/// host that genuinely lacks an address family or holds one family's port but
/// not the other's. It takes `port` instead of capturing it, so the port that
/// gets bound can never drift from the port named in the error text.
fn open_loopback_listeners<F>(port: u16, mut bind_one: F) -> io::Result<Vec<TcpListener>>
where
    F: FnMut(IpAddr, u16) -> io::Result<TcpListener>,
{
    let mut listeners = Vec::new();
    let mut unavailable: Option<(IpAddr, io::Error)> = None;
    for address in LOOPBACK_CALLBACK_ADDRESSES {
        match bind_one(address, port) {
            Ok(listener) => {
                listener.set_nonblocking(true)?;
                listeners.push(listener);
            }
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!(
                        "oauth callback port {port} is already in use on {address} — \
                         close the other sign-in holding it and retry"
                    ),
                ));
            }
            Err(error) if loopback_family_unavailable(&error) => {
                unavailable = Some((address, error));
            }
            Err(error) => return Err(callback_bind_error(port, address, &error)),
        }
    }
    match unavailable {
        Some((address, error)) if listeners.is_empty() => {
            Err(callback_bind_error(port, address, &error))
        }
        _ => Ok(listeners),
    }
}

/// Loopback sockets that are already accepting before the browser is launched.
///
/// The socket has to exist before [`open_browser`], not after: a provider the
/// user is still signed in to redirects with no interaction at all, and a
/// listener opened afterwards misses that callback outright. Binding first
/// leaves the redirect queued in the socket backlog until it is read.
pub(crate) struct OAuthCallbackListeners {
    listeners: Vec<TcpListener>,
}

impl OAuthCallbackListeners {
    /// Open the callback port on every loopback stack this host serves.
    ///
    /// The redirect URIs registered with the providers name the `localhost`
    /// host, which resolves to `::1` ahead of `127.0.0.1` on macOS, so
    /// listening on IPv4 alone left the browser's first connection refused and
    /// the authorization code arrived only if it retried on the other stack.
    /// Each address is bound on its own — never the wildcard — so the code
    /// stays off the network.
    ///
    /// A bind that fails for any reason other than the host lacking that
    /// address family fails the whole call. `localhost` may resolve to exactly
    /// the family that could not be opened, so carrying on with the other one
    /// looks like success while the browser hands the authorization code to
    /// whoever owns the port — or to nobody, until the full timeout expires.
    /// [`loopback_family_unavailable`] is the only tolerated case.
    pub(crate) fn bind(port: u16) -> io::Result<Self> {
        let listeners =
            open_loopback_listeners(port, |address, port| TcpListener::bind((address, port)))?;
        Ok(Self { listeners })
    }

    /// Read the redirect off whichever stack the browser reached.
    pub(crate) fn wait(self) -> Result<runtime::OAuthCallbackParams, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut stream = 'accept: loop {
            for listener in &self.listeners {
                match listener.accept() {
                    Ok((stream, _)) => break 'accept stream,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => return Err(error.into()),
                }
            }
            if started.elapsed() >= OAUTH_CALLBACK_TIMEOUT {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "oauth callback timed out after {}s",
                        OAUTH_CALLBACK_TIMEOUT.as_secs()
                    ),
                )
                .into());
            }
            std::thread::sleep(OAUTH_CALLBACK_ACCEPT_POLL);
        };
        // The listeners poll for `accept`, and on BSD-derived hosts (macOS) the
        // accepted socket inherits their `O_NONBLOCK`. Left that way, the read
        // below returns `WouldBlock` the moment the request bytes have not landed
        // yet and the mapping turns that into a bogus "timed out after 10s" —
        // instantly, without ever waiting. Blocking mode is what makes
        // `set_read_timeout` the real deadline.
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(OAUTH_CALLBACK_READ_TIMEOUT))?;
        let mut buffer = [0_u8; 4096];
        let bytes_read = stream.read(&mut buffer).map_err(|error| {
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "oauth callback request read timed out after {}s",
                        OAUTH_CALLBACK_READ_TIMEOUT.as_secs()
                    ),
                )
            } else {
                error
            }
        })?;
        let request = String::from_utf8_lossy(&buffer[..bytes_read]);
        let request_line = request.lines().next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing callback request line")
        })?;
        let target = request_line.split_whitespace().nth(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "missing callback request target",
            )
        })?;
        let callback = parse_oauth_callback_request_target(target)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let body = if callback.error.is_some() {
            "Zo OAuth login failed. You can close this window."
        } else {
            "Zo OAuth login succeeded. You can close this window."
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/plain; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes())?;
        Ok(callback)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, TcpListener, TcpStream};

    use super::{
        OAuthCallbackListeners, loopback_family_unavailable, missing_inference_scope_warning,
        open_loopback_listeners,
    };

    /// A host serving neither loopback family has to say which port and address
    /// it could not open. Returning the bare OS error left the operator with
    /// "address not available" and nothing to act on.
    #[test]
    fn all_unavailable_callback_bind_errors_report_port_and_address() {
        let error = open_loopback_listeners(4545, |_address, _port| {
            Err(std::io::Error::new(
                std::io::ErrorKind::AddrNotAvailable,
                "address family unavailable",
            ))
        })
        .expect_err("a host with no loopback family cannot serve the callback");

        assert_eq!(error.kind(), std::io::ErrorKind::AddrNotAvailable);
        assert!(
            error.to_string().contains("4545"),
            "the error must name the callback port: {error}"
        );
        assert!(
            error.to_string().contains("127.0.0.1"),
            "the error must name the address it failed on: {error}"
        );
    }

    /// A host that genuinely serves only one loopback family must still log in
    /// on that family — the tolerated skip is the whole reason `EAFNOSUPPORT`
    /// is not fatal.
    #[test]
    fn an_unavailable_family_still_binds_the_other() {
        let _guard = socket_test_lock();
        let listeners = open_loopback_listeners(4545, |address, _port| match address {
            IpAddr::V6(_) => Err(std::io::Error::new(
                std::io::ErrorKind::AddrNotAvailable,
                "no ipv6 loopback on this host",
            )),
            IpAddr::V4(_) => TcpListener::bind(("127.0.0.1", 0)),
        })
        .expect("the surviving family must still be opened");

        assert_eq!(listeners.len(), 1);
    }

    /// Any failure that is *not* a missing address family must abort the whole
    /// bind even when the other family opened. Carrying on left `localhost`
    /// free to resolve to the family nothing was listening on, and the browser
    /// then delivered the authorization code nowhere.
    #[test]
    fn a_non_tolerated_failure_fails_even_when_the_other_family_binds() {
        let _guard = socket_test_lock();
        let error = open_loopback_listeners(4545, |address, _port| match address {
            IpAddr::V6(_) => TcpListener::bind(("::1", 0)),
            IpAddr::V4(_) => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "permission denied",
            )),
        })
        .expect_err("a denied bind must not be masked by the other family");

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(
            error.to_string().contains("4545"),
            "the error must name the callback port: {error}"
        );
        assert!(
            error.to_string().contains("127.0.0.1"),
            "the error must name the address it failed on: {error}"
        );
    }

    #[test]
    fn only_a_missing_address_family_is_tolerated_when_binding() {
        assert!(loopback_family_unavailable(&std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            "address family unavailable",
        )));
        assert!(!loopback_family_unavailable(&std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        )));
        assert!(!loopback_family_unavailable(&std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            "address in use",
        )));

        #[cfg(unix)]
        {
            assert!(loopback_family_unavailable(&std::io::Error::from_raw_os_error(
                nix::libc::EAFNOSUPPORT,
            )));
            assert!(!loopback_family_unavailable(&std::io::Error::from_raw_os_error(
                nix::libc::EACCES,
            )));
        }
    }

    /// Serialize the tests that hand out or occupy a loopback port.
    ///
    /// [`free_port`] has to release the probe before the code under test can
    /// bind it, and these tests run concurrently: two of them could be handed
    /// the same number, and the one that lost the race then connected to the
    /// other's listener. The receiving test accepted that dataless connection
    /// and timed out reading a redirect that was never sent.
    fn socket_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Claim a port, then release it, so a bind under test starts from a known
    /// free number on both stacks. The returned guard must be held for as long
    /// as the port is in use.
    fn free_port() -> (u16, std::sync::MutexGuard<'static, ()>) {
        let guard = socket_test_lock();
        let probe = TcpListener::bind(("127.0.0.1", 0)).expect("probe for a free port");
        let port = probe.local_addr().expect("probe address").port();
        drop(probe);
        (port, guard)
    }

    /// The registered redirect URIs name the `localhost` host, which resolves
    /// to `::1` ahead of `127.0.0.1` on macOS. Every loopback stack the host
    /// can serve has to accept the callback, or the authorization code is lost
    /// on whichever one the browser tries first.
    #[test]
    fn callback_listener_accepts_on_every_loopback_stack() {
        let (port, _guard) = free_port();
        let _listeners = OAuthCallbackListeners::bind(port).expect("bind callback listeners");

        for address in [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
        ] {
            // A stack this host cannot serve at all is not the bug under test.
            if TcpListener::bind((address, 0)).is_err() {
                continue;
            }
            assert!(
                TcpStream::connect((address, port)).is_ok(),
                "callback listener refused a connection on {address}"
            );
        }
    }

    /// A provider the user is still signed in to redirects with no interaction,
    /// so the callback can land before anything waits for it. The socket is
    /// bound before the browser opens precisely so that redirect survives in
    /// the backlog instead of being refused.
    #[test]
    fn a_callback_arriving_before_the_wait_is_still_read() {
        let (port, _guard) = free_port();
        let listeners = OAuthCallbackListeners::bind(port).expect("bind callback listeners");

        // Nothing is accepting yet — this is the window the browser used to win.
        let mut early = TcpStream::connect(("127.0.0.1", port)).expect("early redirect connects");
        early
            .write_all(
                b"GET /callback?code=early-code&state=early-state HTTP/1.1\r\nhost: localhost\r\n\r\n",
            )
            .expect("early redirect sends");

        let callback = listeners.wait().expect("the queued redirect is read");
        assert_eq!(callback.code.as_deref(), Some("early-code"));
        assert_eq!(callback.state.as_deref(), Some("early-state"));
    }

    /// The bytes of a redirect need not be present at `accept` time — the
    /// browser connects and writes as two steps. `wait` polls `accept`, so the
    /// listener is non-blocking, and on BSD-derived hosts the accepted socket
    /// inherits that flag: the first read then returns `WouldBlock`, which the
    /// error mapping dresses up as "timed out after 10s" without waiting at all.
    /// Holding the request back until after `accept` is what makes that
    /// misreport reproducible instead of racing on loopback speed.
    #[test]
    fn a_redirect_whose_bytes_arrive_after_accept_is_still_read() {
        let _guard = socket_test_lock();
        // Bind the ephemeral port directly and keep it: no probe-and-release
        // window, so this test cannot collide with another for a port.
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind a callback listener");
        listener
            .set_nonblocking(true)
            .expect("listeners poll for accept, exactly as bind leaves them");
        let address = listener.local_addr().expect("listener address");
        let listeners = OAuthCallbackListeners {
            listeners: vec![listener],
        };

        // Connected, so `accept` succeeds on the first poll — but silent, so the
        // read that follows has nothing to return yet.
        let mut redirect = TcpStream::connect(address).expect("browser connects");
        let writer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            redirect
                .write_all(
                    b"GET /callback?code=late-code&state=late-state HTTP/1.1\r\nhost: localhost\r\n\r\n",
                )
                .expect("browser sends the redirect");
        });

        let callback = listeners
            .wait()
            .expect("a redirect that lands after accept must still be read");
        writer.join().expect("writer thread");

        assert_eq!(callback.code.as_deref(), Some("late-code"));
        assert_eq!(callback.state.as_deref(), Some("late-state"));
    }

    /// A port already held on one stack must fail the bind outright. Falling
    /// back to the free family looks like success but leaves `localhost` free
    /// to resolve to the occupied one, handing the authorization code to
    /// whatever process owns it — and it also hid the plain "port taken" error
    /// the IPv4-only listener used to report immediately.
    #[test]
    fn an_occupied_stack_fails_the_bind_instead_of_using_the_other_one() {
        let (port, _guard) = free_port();
        let squatter = TcpListener::bind((IpAddr::V4(Ipv4Addr::LOCALHOST), port))
            .expect("hold the IPv4 callback port");

        let error = OAuthCallbackListeners::bind(port)
            .err()
            .expect("an occupied callback port must not bind");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        assert!(
            error.to_string().contains(&port.to_string()),
            "the error must name the contended port: {error}"
        );

        drop(squatter);
    }

    /// inference scope 부재만 경고하고, 보유·미보고(빈 목록)는 침묵한다.
    #[test]
    fn warns_only_when_scopes_reported_without_inference() {
        let scopes = |list: &[&str]| list.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(missing_inference_scope_warning(&scopes(&["user:profile"]))
            .is_some_and(|warning| warning.contains("user:inference")));
        assert!(
            missing_inference_scope_warning(&scopes(&["user:inference", "user:profile"])).is_none()
        );
        assert!(missing_inference_scope_warning(&[]).is_none());
    }
}
