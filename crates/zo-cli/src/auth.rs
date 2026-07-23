use std::io::{self, Read, Write};
use std::net::TcpListener;
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
    callback_port: u16,
    expected_state: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let callback = wait_for_oauth_callback(callback_port)?;
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

    if !crate::tui_active() {
        println!("Opening browser to sign in to Claude...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback_port, &state)?;
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

    if !crate::tui_active() {
        println!("Opening browser to sign in to ChatGPT (OpenAI)...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback_port, &state)?;
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
            "ChatGPT OAuth login complete. Use /model gpt-5.5 to chat with your subscription."
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

    if !crate::tui_active() {
        println!("Opening browser to sign in to Google Gemini...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(callback_port, &state)?;
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
                    "Google Gemini OAuth login complete (project: {project}). Use /model gemini-3.5-flash to switch."
                );
            } else {
                println!(
                    "Google Gemini OAuth login complete. Use /model gemini-3.5-flash to switch."
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
                    "Google Gemini OAuth login complete. Use /model gemini-3.5-flash to switch."
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
                println!("Google Gemini OAuth login complete. Use /model gemini-3.5-flash to switch.");
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

    if !crate::tui_active() {
        println!("Opening browser to sign in to Google Gemini (Zo-managed OAuth)...");
        println!("If the browser didn't open, visit:\n{authorize_url}");
    }
    if let Err(error) = open_browser(&authorize_url) {
        eprintln!("warning: failed to open browser automatically: {error}");
    }

    let code = await_validated_callback(DEFAULT_OAUTH_CALLBACK_PORT, &state)?;
    let saved = block_on_oauth(api::exchange_google_oauth_code_and_save_adc(
        &client,
        &code,
        &pkce.verifier,
        &redirect_uri,
    ))?;
    if !crate::tui_active() {
        println!(
            "Google Gemini OAuth login complete. Saved ADC credentials to {}. Use /model gemini-3.5-flash to switch.",
            saved.path.display()
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

pub(crate) fn wait_for_oauth_callback(
    port: u16,
) -> Result<runtime::OAuthCallbackParams, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    listener.set_nonblocking(true)?;
    let started = Instant::now();
    let (mut stream, _) = loop {
        match listener.accept() {
            Ok(pair) => break pair,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
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
            }
            Err(error) => return Err(error.into()),
        }
    };
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

#[cfg(test)]
mod tests {
    use super::missing_inference_scope_warning;

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
