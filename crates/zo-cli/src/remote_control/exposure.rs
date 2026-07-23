use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub(crate) const DEFAULT_REMOTE_PORT: u16 = 8_788;
const LAST_REMOTE_PORT: u16 = 8_797;
const AUTO_SERVE_ENV: &str = "ZO_REMOTE_AUTO_SERVE";
const DEV_MODE_ENV: &str = "ZO_REMOTE_DEV";
const ONBOARDING_TIMING: OnboardingTiming = OnboardingTiming {
    silent_window: Duration::from_millis(500),
    poll_interval: Duration::from_secs(3),
    wait_timeout: Duration::from_secs(5 * 60),
};

#[derive(Clone, Copy)]
struct OnboardingTiming {
    silent_window: Duration,
    poll_interval: Duration,
    wait_timeout: Duration,
}

#[derive(Debug, Clone)]
pub(crate) struct Exposure {
    pub(crate) bind_addr: SocketAddr,
    pub(crate) host: String,
    pub(crate) mount_path: String,
    pub(crate) origin: String,
}

impl Exposure {
    pub(crate) fn url(&self) -> String {
        format!("{}{}/", self.origin, self.mount_path)
    }

    fn serve_path(&self) -> &str {
        if self.mount_path.is_empty() {
            "/"
        } else {
            &self.mount_path
        }
    }

    fn target(&self) -> String {
        format!("http://{}", self.bind_addr)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExposureProgress {
    SettingUp,
    ApprovalNeeded { url: String, browser_opened: bool },
}

impl ExposureProgress {
    pub(crate) fn message(&self) -> String {
        match self {
            Self::SettingUp => {
                "Zo Remote\n  Status            Setting up Tailscale Serve…".to_string()
            }
            Self::ApprovalNeeded {
                url,
                browser_opened: true,
            } => format!(
                "Zo Remote\n  Approval needed — opened your browser: {url} (waiting for approval…)"
            ),
            Self::ApprovalNeeded {
                url,
                browser_opened: false,
            } => format!(
                "Zo Remote\n  Approval needed — open this URL: {url} (waiting for approval…)"
            ),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ExposureError {
    #[error("failed to run `{command}`: {message}")]
    Command { command: String, message: String },
    #[error("Tailscale is not running")]
    NotRunning,
    #[error("Tailscale did not report this device's DNS name")]
    MissingDnsName,
    #[error("Tailscale Funnel/public exposure is enabled; Zo Remote requires tailnet-only Serve")]
    PublicExposure,
    #[error("Tailscale Serve does not map https://{host}{mount_path}/ to {target}\nConfigure it explicitly, then retry: {command}")]
    MissingMapping {
        host: String,
        mount_path: String,
        target: String,
        command: String,
    },
    #[error("Tailscale Serve already maps https://{host}{mount_path}/ to {actual}; Zo Remote will not overwrite it with {target}")]
    ConflictingMapping {
        host: String,
        mount_path: String,
        target: String,
        actual: String,
    },
    #[error("ZO_REMOTE_PORT pins Zo Remote to {port}, but {detail}")]
    PinnedPortBusy { port: u16, detail: String },
    #[error("no free Zo Remote loopback port is available in {first}..={last}")]
    NoAvailablePort { first: u16, last: u16 },
    #[error("could not bind Zo Remote to 127.0.0.1:{port}: {message}")]
    Bind { port: u16, message: String },
    #[error("Tailscale Serve approval timed out. Approve it at {url}, then retry /remote; Zo Remote will resume automatically.")]
    ApprovalTimeout { url: String },
    #[error("could not parse Tailscale {document} JSON: {message}")]
    InvalidJson {
        document: &'static str,
        message: String,
    },
}

fn configured_port() -> Option<u16> {
    std::env::var("ZO_REMOTE_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port >= 1_024)
}

pub(crate) async fn bind_listener() -> Result<TcpListener, ExposureError> {
    bind_listener_in_range(configured_port(), DEFAULT_REMOTE_PORT, LAST_REMOTE_PORT).await
}

async fn bind_listener_in_range(
    pinned: Option<u16>,
    first: u16,
    last: u16,
) -> Result<TcpListener, ExposureError> {
    if let Some(port) = pinned {
        return match TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await {
            Ok(listener) => Ok(listener),
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
                let detail = port_owner(port).await.map_or_else(
                    || "the port is already in use".to_string(),
                    |owner| format!("the port is already in use by {owner}"),
                );
                Err(ExposureError::PinnedPortBusy { port, detail })
            }
            Err(error) => Err(ExposureError::Bind {
                port,
                message: error.to_string(),
            }),
        };
    }

    for port in first..=last {
        match TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await {
            Ok(listener) => return Ok(listener),
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {}
            Err(error) => {
                return Err(ExposureError::Bind {
                    port,
                    message: error.to_string(),
                });
            }
        }
    }
    Err(ExposureError::NoAvailablePort { first, last })
}

async fn port_owner(port: u16) -> Option<String> {
    let filter = format!("-iTCP:{port}");
    let output = Command::new("lsof")
        .args(["-nP", &filter, "-sTCP:LISTEN", "-Fpc"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let output = String::from_utf8_lossy(&output.stdout);
    let pid = output.lines().find_map(|line| line.strip_prefix('p'))?;
    let command = output.lines().find_map(|line| line.strip_prefix('c'))?;
    Some(format!("{command} (PID {pid})"))
}

pub(crate) async fn discover(port: u16) -> Result<Exposure, ExposureError> {
    discover_with_executables(port, &tailscale_executables()).await
}

pub(crate) async fn discover_or_configure(
    port: u16,
    progress: impl FnMut(ExposureProgress),
) -> Result<Exposure, ExposureError> {
    if std::env::var(DEV_MODE_ENV).as_deref() == Ok("1") {
        return Ok(Exposure {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
            host: format!("localhost:{port}"),
            mount_path: String::new(),
            origin: format!("http://localhost:{port}"),
        });
    }
    let executables = tailscale_executables();
    let initial = discover(port).await;
    finish_discovery(
        initial,
        port,
        &executables,
        progress,
        open_approval_url,
        auto_serve_enabled(),
        ONBOARDING_TIMING,
    )
    .await
}

pub(crate) async fn remove_serve_mapping(exposure: &Exposure) -> Result<(), ExposureError> {
    if exposure.origin.starts_with("http://localhost:") {
        return Ok(());
    }
    let executables = tailscale_executables();
    remove_serve_mapping_with_executables(exposure, &executables).await
}

async fn remove_serve_mapping_with_executables(
    exposure: &Exposure,
    executables: &[PathBuf],
) -> Result<(), ExposureError> {
    let serve = run_json_with_executables(
        &["serve", "status", "--json"],
        "serve status",
        executables,
    )
    .await?;
    let path = exposure.serve_path();
    let target = exposure.target();
    let Some(handler) = mapping_handler(&serve, &exposure.host, path) else {
        return Ok(());
    };
    let actual = handler.get("Proxy").and_then(Value::as_str);
    if actual != Some(target.as_str()) {
        return Err(ExposureError::ConflictingMapping {
            host: exposure.host.clone(),
            mount_path: exposure.mount_path.clone(),
            target,
            actual: actual.map_or_else(|| handler.to_string(), str::to_string),
        });
    }
    run_with_executables(
        &["serve", "--https=443", "--set-path", path, "off"],
        executables,
    )
    .await
}

async fn discover_with_executables(
    port: u16,
    executables: &[PathBuf],
) -> Result<Exposure, ExposureError> {
    let status = run_json_with_executables(&["status", "--json"], "status", executables).await?;
    let serve = run_json_with_executables(
        &["serve", "status", "--json"],
        "serve status",
        executables,
    )
    .await?;
    match validate(&status, &serve, port) {
        Err(ExposureError::MissingMapping {
            host,
            mount_path,
            target,
            ..
        }) if !mount_path.is_empty() => {
            let root_target = mapping_target(&serve, &host, "/");
            if let Some(root_target) = root_target {
                if stale_zo_mapping(root_target).await {
                    return Err(missing_mapping(host, String::new(), target));
                }
            }
            Err(missing_mapping(host, mount_path, target))
        }
        Err(ExposureError::ConflictingMapping {
            host,
            mount_path,
            target,
            actual,
        }) => {
            if stale_zo_mapping(&actual).await {
                return Err(missing_mapping(host, mount_path, target));
            }
            if !mount_path.is_empty() {
                if let Some(root_target) = mapping_target(&serve, &host, "/") {
                    if stale_zo_mapping(root_target).await {
                        return Err(missing_mapping(host, String::new(), target));
                    }
                }
            }
            Err(ExposureError::ConflictingMapping {
                host,
                mount_path,
                target,
                actual,
            })
        }
        result => result,
    }
}

fn tailscale_executables() -> Vec<PathBuf> {
    #[cfg(not(target_os = "macos"))]
    {
        vec![PathBuf::from("tailscale")]
    }
    #[cfg(target_os = "macos")]
    {
        let mut executables = vec![PathBuf::from("tailscale")];
        executables.push(PathBuf::from(
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
        ));
        if let Some(home) = std::env::var_os("HOME") {
            executables.push(
                PathBuf::from(home)
                    .join("Applications/Tailscale.app/Contents/MacOS/Tailscale"),
            );
        }
        executables
    }
}

async fn run_json_with_executables(
    args: &[&str],
    source: &'static str,
    executables: &[PathBuf],
) -> Result<Value, ExposureError> {
    let command = format!("tailscale {}", args.join(" "));
    let mut last_not_found = None;
    for executable in executables {
        let output = match Command::new(executable).args(args).output().await {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                last_not_found = Some(error);
                continue;
            }
            Err(error) => {
                return Err(ExposureError::Command {
                    command,
                    message: error.to_string(),
                });
            }
        };
        if !output.status.success() {
            return Err(ExposureError::Command {
                command,
                message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }
        return serde_json::from_slice(&output.stdout).map_err(|error| {
            ExposureError::InvalidJson {
                document: source,
                message: error.to_string(),
            }
        });
    }

    Err(ExposureError::Command {
        command,
        message: last_not_found.map_or_else(
            || "no Tailscale executable candidates configured".to_string(),
            |error| error.to_string(),
        ),
    })
}

async fn run_with_executables(
    args: &[&str],
    executables: &[PathBuf],
) -> Result<(), ExposureError> {
    let command = format!("tailscale {}", args.join(" "));
    let mut last_not_found = None;
    for executable in executables {
        let output = match Command::new(executable).args(args).output().await {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                last_not_found = Some(error);
                continue;
            }
            Err(error) => {
                return Err(ExposureError::Command {
                    command,
                    message: error.to_string(),
                });
            }
        };
        if output.status.success() {
            return Ok(());
        }
        let message = if output.stderr.is_empty() {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        } else {
            String::from_utf8_lossy(&output.stderr).trim().to_string()
        };
        return Err(ExposureError::Command { command, message });
    }
    Err(ExposureError::Command {
        command,
        message: last_not_found.map_or_else(
            || "no Tailscale executable candidates configured".to_string(),
            |error| error.to_string(),
        ),
    })
}

async fn finish_discovery<F, O>(
    initial: Result<Exposure, ExposureError>,
    port: u16,
    executables: &[PathBuf],
    mut progress: F,
    open_browser: O,
    auto_serve: bool,
    timing: OnboardingTiming,
) -> Result<Exposure, ExposureError>
where
    F: FnMut(ExposureProgress),
    O: Fn(&str) -> bool,
{
    let mount_path = match initial {
        Ok(exposure) => return Ok(exposure),
        Err(error) if !matches!(error, ExposureError::MissingMapping { .. }) => {
            return Err(error);
        }
        Err(error) if !auto_serve => return Err(error),
        Err(ExposureError::MissingMapping { mount_path, .. }) => mount_path,
        Err(error) => return Err(error),
    };

    configure_serve(
        port,
        &mount_path,
        executables,
        &mut progress,
        open_browser,
        timing,
    )
    .await
}

async fn configure_serve(
    port: u16,
    mount_path: &str,
    executables: &[PathBuf],
    progress: &mut impl FnMut(ExposureProgress),
    open_browser: impl Fn(&str) -> bool,
    timing: OnboardingTiming,
) -> Result<Exposure, ExposureError> {
    let target = format!("http://127.0.0.1:{port}");
    let (mut child, command) = spawn_serve(executables, mount_path, &target)?;
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let mut stdout_task = pump_output(
        child
            .stdout
            .take()
            .expect("Tailscale Serve stdout must be piped"),
        output_tx.clone(),
    );
    let mut stderr_task = pump_output(
        child
            .stderr
            .take()
            .expect("Tailscale Serve stderr must be piped"),
        output_tx,
    );
    let mut output = String::new();
    let mut setup_reported = false;
    let mut approval_url = None;
    let setup_delay = tokio::time::sleep(timing.silent_window);
    tokio::pin!(setup_delay);
    let approval_timeout = tokio::time::sleep(timing.wait_timeout);
    tokio::pin!(approval_timeout);
    let poll = tokio::time::sleep(timing.poll_interval);
    tokio::pin!(poll);

    loop {
        tokio::select! {
            status = child.wait() => {
                let status = match status {
                    Ok(status) => status,
                    Err(error) => {
                        terminate(&mut child).await;
                        return Err(ExposureError::Command {
                            command: command.clone(),
                            message: error.to_string(),
                        });
                    }
                };
                collect_output(
                    &mut output,
                    &mut output_rx,
                    &mut stdout_task,
                    &mut stderr_task,
                )
                .await;
                if status.success() {
                    return discover_with_executables(port, executables).await;
                }
                return Err(ExposureError::Command {
                    command,
                    message: command_failure_message(&output, status.code()),
                });
            }
            Some(chunk) = output_rx.recv() => {
                output.push_str(&chunk);
                if report_approval_if_ready(
                    &output,
                    &mut setup_reported,
                    &mut approval_url,
                    progress,
                    &open_browser,
                ) {
                    poll.as_mut().reset(tokio::time::Instant::now() + timing.poll_interval);
                }
            }
            () = &mut setup_delay, if !setup_reported => {
                progress(ExposureProgress::SettingUp);
                setup_reported = true;
            }
            () = &mut poll, if approval_url.is_some() => {
                match discover_with_executables(port, executables).await {
                    Ok(exposure) => {
                        tokio::spawn(async move {
                            let _ = child.wait().await;
                        });
                        return Ok(exposure);
                    }
                    Err(ExposureError::MissingMapping { .. }) => {
                        poll.as_mut().reset(tokio::time::Instant::now() + timing.poll_interval);
                    }
                    Err(error) => {
                        terminate(&mut child).await;
                        return Err(error);
                    }
                }
            }
            () = &mut approval_timeout => {
                terminate(&mut child).await;
                return Err(timeout_error(approval_url, command, &output));
            }
        }
    }
}

fn timeout_error(approval_url: Option<String>, command: String, output: &str) -> ExposureError {
    approval_url.map_or_else(
        || ExposureError::Command {
            command,
            message: command_failure_message(output, None),
        },
        |url| ExposureError::ApprovalTimeout { url },
    )
}

fn report_approval_if_ready<F, O>(
    output: &str,
    setup_reported: &mut bool,
    approval_url: &mut Option<String>,
    progress: &mut F,
    open_browser: &O,
) -> bool
where
    F: FnMut(ExposureProgress),
    O: Fn(&str) -> bool,
{
    if approval_url.is_some() || !is_approval_prompt(output) {
        return false;
    }
    let Some(url) = extract_approval_url(output) else {
        return false;
    };
    if !*setup_reported {
        progress(ExposureProgress::SettingUp);
        *setup_reported = true;
    }
    let browser_opened = open_browser(&url);
    progress(ExposureProgress::ApprovalNeeded {
        url: url.clone(),
        browser_opened,
    });
    *approval_url = Some(url);
    true
}

fn spawn_serve(
    executables: &[PathBuf],
    mount_path: &str,
    target: &str,
) -> Result<(tokio::process::Child, String), ExposureError> {
    let args = serve_args(mount_path, target);
    let command = format!("tailscale {}", args.join(" "));
    let mut last_not_found = None;
    for executable in executables {
        let mut candidate = Command::new(executable);
        candidate
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        match candidate.spawn() {
            Ok(child) => return Ok((child, command)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                last_not_found = Some(error);
            }
            Err(error) => {
                return Err(ExposureError::Command {
                    command,
                    message: error.to_string(),
                });
            }
        }
    }

    Err(ExposureError::Command {
        command,
        message: last_not_found.map_or_else(
            || "no Tailscale executable candidates configured".to_string(),
            |error| error.to_string(),
        ),
    })
}

fn pump_output<R>(mut reader: R, output: mpsc::UnboundedSender<String>) -> JoinHandle<()>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut buffer = [0_u8; 1_024];
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) | Err(_) => return,
                Ok(read) => {
                    if output
                        .send(String::from_utf8_lossy(&buffer[..read]).into_owned())
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    })
}

async fn collect_output(
    output: &mut String,
    output_rx: &mut mpsc::UnboundedReceiver<String>,
    stdout_task: &mut JoinHandle<()>,
    stderr_task: &mut JoinHandle<()>,
) {
    let _ = tokio::join!(stdout_task, stderr_task);
    while let Ok(chunk) = output_rx.try_recv() {
        output.push_str(&chunk);
    }
}

async fn terminate(child: &mut tokio::process::Child) {
    if child.try_wait().ok().flatten().is_none() {
        let _ = child.kill().await;
    }
    let _ = child.wait().await;
}

fn command_failure_message(output: &str, status_code: Option<i32>) -> String {
    let output = output.trim();
    if output.is_empty() {
        status_code.map_or_else(
            || "command did not complete".to_string(),
            |code| format!("command exited with status {code}"),
        )
    } else {
        output.to_string()
    }
}

fn auto_serve_enabled() -> bool {
    std::env::var(AUTO_SERVE_ENV).map_or(true, |value| value != "0")
}

fn is_approval_prompt(output: &str) -> bool {
    output.contains("Serve is not enabled on your tailnet")
        || output.contains("To enable, visit:")
}

fn extract_approval_url(output: &str) -> Option<String> {
    const PREFIX: &str = "https://login.tailscale.com/";
    let start = output.find(PREFIX)?;
    let url = output[start..]
        .split(|character: char| character.is_ascii_whitespace() || matches!(character, '\"' | '\''))
        .next()?
        .trim_end_matches([',', '.', ';', ')', ']']);
    (!url.is_empty()).then(|| url.to_string())
}

#[cfg(target_os = "macos")]
fn open_approval_url(url: &str) -> bool {
    match Command::new("open").arg(url).spawn() {
        Ok(mut child) => {
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            true
        }
        Err(_) => false,
    }
}

#[cfg(not(target_os = "macos"))]
fn open_approval_url(_url: &str) -> bool {
    false
}

fn serve_args<'a>(mount_path: &'a str, target: &'a str) -> Vec<&'a str> {
    if mount_path.is_empty() {
        vec!["serve", "--bg", "--https=443", target]
    } else {
        vec![
            "serve",
            "--bg",
            "--https=443",
            "--set-path",
            mount_path,
            target,
        ]
    }
}

fn missing_mapping(host: String, mount_path: String, target: String) -> ExposureError {
    let command = format!("tailscale {}", serve_args(&mount_path, &target).join(" "));
    ExposureError::MissingMapping {
        host,
        mount_path,
        target,
        command,
    }
}

fn mapping_handler<'a>(serve: &'a Value, host: &str, path: &str) -> Option<&'a Value> {
    let authority = format!("{host}:443");
    serve
        .get("Web")
        .and_then(Value::as_object)
        .and_then(|web| web.get(&authority))
        .and_then(|site| site.get("Handlers"))
        .and_then(Value::as_object)
        .and_then(|handlers| handlers.get(path))
}

fn mapping_target<'a>(serve: &'a Value, host: &str, path: &str) -> Option<&'a str> {
    mapping_handler(serve, host, path)
        .and_then(|handler| handler.get("Proxy"))
        .and_then(Value::as_str)
}

fn zo_loopback_port(target: &str) -> Option<u16> {
    let port = target.strip_prefix("http://127.0.0.1:")?.parse().ok()?;
    (DEFAULT_REMOTE_PORT..=LAST_REMOTE_PORT)
        .contains(&port)
        .then_some(port)
}

async fn stale_zo_mapping(target: &str) -> bool {
    let Some(port) = zo_loopback_port(target) else {
        return false;
    };
    matches!(
        tokio::time::timeout(
            Duration::from_millis(250),
            TcpStream::connect((Ipv4Addr::LOCALHOST, port)),
        )
        .await,
        Ok(Err(error)) if error.kind() == std::io::ErrorKind::ConnectionRefused
    )
}

pub(crate) fn validate(
    status: &Value,
    serve: &Value,
    port: u16,
) -> Result<Exposure, ExposureError> {
    if status.get("BackendState").and_then(Value::as_str) != Some("Running") {
        return Err(ExposureError::NotRunning);
    }
    let host = status
        .pointer("/Self/DNSName")
        .and_then(Value::as_str)
        .map(|value| value.trim_end_matches('.'))
        .filter(|value| !value.is_empty())
        .ok_or(ExposureError::MissingDnsName)?
        .to_ascii_lowercase();
    if has_public_exposure(serve) {
        return Err(ExposureError::PublicExposure);
    }

    let target = format!("http://127.0.0.1:{port}");
    if mapping_target(serve, &host, "/") == Some(target.as_str()) {
        return Ok(Exposure {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
            origin: format!("https://{host}"),
            host,
            mount_path: String::new(),
        });
    }

    let mount_path = format!("/s{port}");
    let path_handler = mapping_handler(serve, &host, &mount_path);
    let path_target = path_handler
        .and_then(|handler| handler.get("Proxy"))
        .and_then(Value::as_str);
    if path_target == Some(target.as_str()) {
        return Ok(Exposure {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
            origin: format!("https://{host}"),
            host,
            mount_path,
        });
    }

    if mapping_handler(serve, &host, "/").is_none() {
        return Err(missing_mapping(host, String::new(), target));
    }
    if let Some(handler) = path_handler {
        return Err(ExposureError::ConflictingMapping {
            host,
            mount_path,
            target,
            actual: path_target.map_or_else(|| handler.to_string(), str::to_string),
        });
    }

    Err(missing_mapping(host, mount_path, target))
}

fn has_public_exposure(value: &Value) -> bool {
    match value {
        Value::Object(entries) => entries.iter().any(|(key, value)| {
            (key.to_ascii_lowercase().contains("funnel") && is_enabled(value))
                || has_public_exposure(value)
        }),
        Value::Array(values) => values.iter().any(has_public_exposure),
        _ => false,
    }
}

fn is_enabled(value: &Value) -> bool {
    match value {
        Value::Bool(enabled) => *enabled,
        Value::String(value) => !value.is_empty() && value != "false",
        Value::Array(values) => values.iter().any(is_enabled),
        Value::Object(entries) => entries.values().any(is_enabled),
        Value::Number(value) => value.as_u64().is_some_and(|value| value != 0),
        Value::Null => false,
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use std::path::{Path, PathBuf};
    #[cfg(unix)]
    use std::sync::Arc;
    #[cfg(unix)]
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(unix)]
    use std::time::Duration;

    use serde_json::json;

    #[cfg(unix)]
    use super::{
        Exposure, ExposureProgress, OnboardingTiming, bind_listener_in_range,
        discover_with_executables, finish_discovery, remove_serve_mapping_with_executables,
        run_json_with_executables,
    };
    #[cfg(target_os = "macos")]
    use super::tailscale_executables;
    use super::{ExposureError, extract_approval_url, validate};

    #[cfg(unix)]
    static ZO_RANGE_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[cfg(unix)]
    struct TestDir {
        path: PathBuf,
    }

    #[cfg(unix)]
    impl TestDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "zo-tailscale-{label}-{}-{}",
                std::process::id(),
                rand::random::<u64>()
            ));
            std::fs::create_dir_all(&path).expect("create test directory");
            Self { path }
        }
    }

    #[cfg(unix)]
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[cfg(unix)]
    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(unix)]
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    #[cfg(unix)]
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[cfg(unix)]
    fn write_executable(path: &Path, script: &str) {
        std::fs::write(path, script).expect("write fake Tailscale executable");
        let mut permissions = std::fs::metadata(path)
            .expect("read fake Tailscale metadata")
            .permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(path, permissions)
            .expect("make fake Tailscale executable");
    }

    #[cfg(unix)]
    async fn run_onboarding<F, O>(
        executable: &Path,
        port: u16,
        auto_serve: bool,
        progress: F,
        open_browser: O,
    ) -> Result<super::Exposure, ExposureError>
    where
        F: FnMut(ExposureProgress),
        O: Fn(&str) -> bool,
    {
        let executables = [executable.to_path_buf()];
        let initial = discover_with_executables(port, &executables).await;
        finish_discovery(
            initial,
            port,
            &executables,
            progress,
            open_browser,
            auto_serve,
            OnboardingTiming {
                silent_window: Duration::from_millis(200),
                poll_interval: Duration::from_millis(20),
                wait_timeout: Duration::from_secs(2),
            },
        )
        .await
    }

    #[test]
    fn extracts_serve_approval_url_from_cli_output() {
        let output = "Serve is not enabled on your tailnet.\n\
To enable, visit:\n\
https://login.tailscale.com/f/serve?node=node-name-abc123\n\
\nWaiting for approval...\n";

        assert_eq!(
            extract_approval_url(output).as_deref(),
            Some("https://login.tailscale.com/f/serve?node=node-name-abc123")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn searches_the_macos_app_bundle_after_path() {
        let executables = tailscale_executables();
        assert_eq!(executables.first().and_then(|path| path.to_str()), Some("tailscale"));
        assert!(executables.iter().any(|path| {
            path == std::path::Path::new(
                "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            )
        }));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn falls_back_when_the_first_executable_is_not_found() {
        let test_dir = std::env::temp_dir().join(format!(
            "zo-tailscale-fallback-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&test_dir).expect("create test directory");
        let fake_tailscale = test_dir.join("tailscale");
        std::fs::write(
            &fake_tailscale,
            "#!/bin/sh\nprintf '%s\\n' '{\"BackendState\":\"Running\"}'\n",
        )
        .expect("write fake tailscale");
        let mut permissions = std::fs::metadata(&fake_tailscale)
            .expect("read fake tailscale metadata")
            .permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&fake_tailscale, permissions)
            .expect("make fake tailscale executable");

        let result = run_json_with_executables(
            &["status", "--json"],
            "status",
            &[test_dir.join("missing"), fake_tailscale],
        )
        .await;
        std::fs::remove_dir_all(&test_dir).expect("remove test directory");

        assert_eq!(
            result.expect("fallback executable should return JSON"),
            json!({ "BackendState": "Running" })
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn port_scan_skips_a_busy_port_and_reserves_the_next_one() {
        // `next` 를 놓은 순간부터 스캔까지의 틈에 병렬 테스트/외부 프로세스가
        // `first + 1` 을 채갈 수 있다(TOCTOU). 실패 시 새 페어로 재시도해
        // 흔들리지 않게 한다 — 검증 대상은 "busy 스킵 후 다음 포트 예약"이지
        // 특정 포트 번호가 아니다.
        let mut last_error = None;
        for attempt_base in [20_000_u16, 25_000, 30_000, 35_000, 40_000] {
            let Some((busy, first)) = (attempt_base..attempt_base + 5_000).find_map(|first| {
                let busy = std::net::TcpListener::bind(("127.0.0.1", first)).ok()?;
                let next = std::net::TcpListener::bind(("127.0.0.1", first + 1)).ok()?;
                drop(next);
                Some((busy, first))
            }) else {
                continue;
            };

            match bind_listener_in_range(None, first, first + 1).await {
                Ok(selected) => {
                    assert_eq!(
                        selected.local_addr().expect("selected address").port(),
                        first + 1
                    );
                    drop(busy);
                    return;
                }
                Err(error) => {
                    drop(busy);
                    last_error = Some(error);
                }
            }
        }
        panic!("scan never selected the next free port: {last_error:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pinned_busy_port_reports_the_exact_pin() {
        let busy = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind busy port");
        let port = busy.local_addr().expect("busy address").port();

        let error = bind_listener_in_range(Some(port), 8_788, 8_797)
            .await
            .expect_err("pinned busy port must fail");
        let message = error.to_string();

        assert!(matches!(error, ExposureError::PinnedPortBusy { .. }));
        assert!(message.contains("ZO_REMOTE_PORT"));
        assert!(message.contains(&port.to_string()));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn absent_mapping_is_configured_without_opening_a_browser() {
        let test_dir = TestDir::new("automatic");
        let executable = test_dir.path.join("tailscale");
        let configured = test_dir.path.join("configured");
        let invocations = test_dir.path.join("invocations");
        let script = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  printf '%s\n' '{"BackendState":"Running","Self":{"DNSName":"laptop.example.ts.net."}}'
elif [ "$2" = "status" ]; then
  if [ -f '__CONFIGURED__' ]; then
    printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8788"}}}}}'
  else
    printf '%s\n' '{}'
  fi
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
  : > '__CONFIGURED__'
fi
"#
        .replace("__CONFIGURED__", &configured.display().to_string())
        .replace("__INVOCATIONS__", &invocations.display().to_string());
        write_executable(&executable, &script);
        let browser_opens = Arc::new(AtomicUsize::new(0));
        let browser_spy = Arc::clone(&browser_opens);
        let mut progress = Vec::new();

        let exposure = run_onboarding(
            &executable,
            8_788,
            true,
            |update| progress.push(update),
            move |_| {
                browser_spy.fetch_add(1, Ordering::SeqCst);
                true
            },
        )
        .await
        .expect("missing mapping should be configured");

        assert_eq!(exposure.origin, "https://laptop.example.ts.net");
        assert_eq!(browser_opens.load(Ordering::SeqCst), 0);
        assert!(progress.is_empty());
        let invocations = std::fs::read_to_string(invocations).expect("read serve invocation");
        assert_eq!(
            invocations.trim(),
            "serve --bg --https=443 http://127.0.0.1:8788"
        );
        assert!(!invocations.to_ascii_lowercase().contains("funnel"));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn approval_wait_polls_until_serve_is_enabled() {
        let test_dir = TestDir::new("approval");
        let executable = test_dir.path.join("tailscale");
        let approved = test_dir.path.join("approved");
        let polls = test_dir.path.join("polls");
        let invocations = test_dir.path.join("invocations");
        let script = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  printf '%s\n' '{"BackendState":"Running","Self":{"DNSName":"laptop.example.ts.net."}}'
elif [ "$2" = "status" ]; then
  count=0
  if [ -f '__POLLS__' ]; then
    count=$(cat '__POLLS__')
  fi
  count=$((count + 1))
  printf '%s' "$count" > '__POLLS__'
  if [ "$count" -ge 3 ]; then
    : > '__APPROVED__'
    printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8788"}}}}}'
  else
    printf '%s\n' '{}'
  fi
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
  printf '%s\n' 'Serve is not enabled on your tailnet. To enable, visit:' >&2
  printf '%s\n' 'https://login.tailscale.com/f/serve?node=laptop-abc123' >&2
  while [ ! -f '__APPROVED__' ]; do
    sleep 0.01
  done
fi
"#
        .replace("__POLLS__", &polls.display().to_string())
        .replace("__APPROVED__", &approved.display().to_string())
        .replace("__INVOCATIONS__", &invocations.display().to_string());
        write_executable(&executable, &script);
        let browser_opens = Arc::new(AtomicUsize::new(0));
        let browser_spy = Arc::clone(&browser_opens);
        let mut progress = Vec::new();

        let exposure = run_onboarding(
            &executable,
            8_788,
            true,
            |update| progress.push(update),
            move |url| {
                assert_eq!(
                    url,
                    "https://login.tailscale.com/f/serve?node=laptop-abc123"
                );
                browser_spy.fetch_add(1, Ordering::SeqCst);
                true
            },
        )
        .await
        .expect("approval should be detected by polling");

        assert_eq!(exposure.origin, "https://laptop.example.ts.net");
        assert_eq!(browser_opens.load(Ordering::SeqCst), 1);
        assert!(matches!(progress.first(), Some(ExposureProgress::SettingUp)));
        assert!(matches!(
            progress.get(1),
            Some(ExposureProgress::ApprovalNeeded {
                url,
                browser_opened: true,
            }) if url == "https://login.tailscale.com/f/serve?node=laptop-abc123"
        ));
        assert!(std::fs::read_to_string(polls)
            .expect("read poll count")
            .parse::<usize>()
            .expect("numeric poll count")
            >= 3);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn conflicting_path_mapping_is_never_overwritten() {
        // 루트 매핑이 실기기 포트 상태에 좌우되면 안 된다: stale_zo_mapping 은
        // 실제 루프백에 연결해 보므로, 루트 타깃 포트를 테스트가 직접 점유해
        // "살아있는 루트 + 경로 충돌" 시나리오를 결정적으로 고정한다. 세션
        // 포트는 루트와 다른 빈 범위 포트를 골라야 한다 — 하드코딩(8789)은
        // 실기기 게이트웨이가 8788 을 점유한 순간 루트 리스너가 8789 로
        // 밀려나 "루트 == 우리 타깃"(Ok) 으로 시나리오가 무너진다.
        let _range_lock = ZO_RANGE_TEST_LOCK.lock().await;
        let test_dir = TestDir::new("conflict");
        let executable = test_dir.path.join("tailscale");
        let invocations = test_dir.path.join("invocations");
        let (live_root, live_port) = (super::DEFAULT_REMOTE_PORT..=super::LAST_REMOTE_PORT)
            .find_map(|port| {
                std::net::TcpListener::bind(("127.0.0.1", port))
                    .ok()
                    .map(|listener| (listener, port))
            })
            .expect("reserve a live Zo-range root target");
        let session_port = (super::DEFAULT_REMOTE_PORT..=super::LAST_REMOTE_PORT)
            .filter(|port| *port != live_port)
            .find(|port| {
                std::net::TcpListener::bind(("127.0.0.1", *port)).is_ok()
            })
            .expect("reserve a distinct free Zo-range session port");
        let script = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  printf '%s\n' '{"BackendState":"Running","Self":{"DNSName":"laptop.example.ts.net."}}'
elif [ "$2" = "status" ]; then
  printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:__LIVE_PORT__"},"/s__SESSION_PORT__":{"Proxy":"http://127.0.0.1:9999"}}}}}'
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
fi
"#
        .replace("__INVOCATIONS__", &invocations.display().to_string())
        .replace("__LIVE_PORT__", &live_port.to_string())
        .replace("__SESSION_PORT__", &session_port.to_string());
        write_executable(&executable, &script);

        let result = run_onboarding(&executable, session_port, true, |_| {}, |_| true).await;
        drop(live_root);

        assert!(matches!(
            result,
            Err(ExposureError::ConflictingMapping { ref actual, .. })
                if actual == "http://127.0.0.1:9999"
        ));
        assert!(!invocations.exists());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn live_root_mapping_adds_a_path_mount_without_overwriting_root() {
        let _range_lock = ZO_RANGE_TEST_LOCK.lock().await;
        let test_dir = TestDir::new("path-mount");
        let executable = test_dir.path.join("tailscale");
        let configured = test_dir.path.join("configured");
        let invocations = test_dir.path.join("invocations");
        let (live_root, live_port) = (super::DEFAULT_REMOTE_PORT..=super::LAST_REMOTE_PORT)
            .find_map(|port| {
                std::net::TcpListener::bind(("127.0.0.1", port))
                    .ok()
                    .map(|listener| (listener, port))
            })
            .expect("reserve a live Zo-range root target");
        let script = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  printf '%s\n' '{"BackendState":"Running","Self":{"DNSName":"laptop.example.ts.net."}}'
elif [ "$2" = "status" ]; then
  if [ -f '__CONFIGURED__' ]; then
    printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:__LIVE_PORT__"},"/s18789":{"Proxy":"http://127.0.0.1:18789"}}}}}'
  else
    printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:__LIVE_PORT__"}}}}}'
  fi
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
  : > '__CONFIGURED__'
fi
"#
        .replace("__CONFIGURED__", &configured.display().to_string())
        .replace("__INVOCATIONS__", &invocations.display().to_string())
        .replace("__LIVE_PORT__", &live_port.to_string());
        write_executable(&executable, &script);

        let exposure = run_onboarding(&executable, 18_789, true, |_| {}, |_| true)
            .await
            .expect("live root should coexist with a path mount");

        assert_eq!(exposure.mount_path, "/s18789");
        assert_eq!(exposure.url(), "https://laptop.example.ts.net/s18789/");
        assert_eq!(
            std::fs::read_to_string(invocations)
                .expect("read path-mount invocation")
                .trim(),
            "serve --bg --https=443 --set-path /s18789 http://127.0.0.1:18789"
        );
        drop(live_root);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn stale_zo_root_mapping_is_reclaimed() {
        let _range_lock = ZO_RANGE_TEST_LOCK.lock().await;
        let test_dir = TestDir::new("stale-root");
        let executable = test_dir.path.join("tailscale");
        let configured = test_dir.path.join("configured");
        let invocations = test_dir.path.join("invocations");
        let stale_port = (super::DEFAULT_REMOTE_PORT..=super::LAST_REMOTE_PORT)
            .find(|port| std::net::TcpListener::bind(("127.0.0.1", *port)).is_ok())
            .expect("find an unused Zo-range port");
        let script = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  printf '%s\n' '{"BackendState":"Running","Self":{"DNSName":"laptop.example.ts.net."}}'
elif [ "$2" = "status" ]; then
  if [ -f '__CONFIGURED__' ]; then
    printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:18790"}}}}}'
  else
    printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:__STALE_PORT__"}}}}}'
  fi
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
  : > '__CONFIGURED__'
fi
"#
        .replace("__CONFIGURED__", &configured.display().to_string())
        .replace("__INVOCATIONS__", &invocations.display().to_string())
        .replace("__STALE_PORT__", &stale_port.to_string());
        write_executable(&executable, &script);

        let exposure = run_onboarding(&executable, 18_790, true, |_| {}, |_| true)
            .await
            .expect("dead Zo root should be reclaimed");

        assert!(exposure.mount_path.is_empty());
        assert_eq!(
            std::fs::read_to_string(invocations)
                .expect("read root replacement invocation")
                .trim(),
            "serve --bg --https=443 http://127.0.0.1:18790"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_removes_only_the_owned_path_mount() {
        let test_dir = TestDir::new("remove-path");
        let executable = test_dir.path.join("tailscale");
        let invocations = test_dir.path.join("invocations");
        let script = r#"#!/bin/sh
if [ "$2" = "status" ]; then
  printf '%s\n' '{"Web":{"laptop.example.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8788"},"/s8790":{"Proxy":"http://127.0.0.1:8790"}}}}}'
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
fi
"#
        .replace("__INVOCATIONS__", &invocations.display().to_string());
        write_executable(&executable, &script);
        let exposure = Exposure {
            bind_addr: "127.0.0.1:8790".parse().expect("loopback address"),
            host: "laptop.example.ts.net".to_string(),
            mount_path: "/s8790".to_string(),
            origin: "https://laptop.example.ts.net".to_string(),
        };

        remove_serve_mapping_with_executables(&exposure, &[executable])
            .await
            .expect("owned path mapping is removed");

        assert_eq!(
            std::fs::read_to_string(invocations)
                .expect("read mapping removal invocation")
                .trim(),
            "serve --https=443 --set-path /s8790 off"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn auto_serve_opt_out_skips_serve_invocation() {
        let auto_serve = {
            let _env_lock = crate::test_env_lock();
            let _auto_serve = EnvGuard::set("ZO_REMOTE_AUTO_SERVE", "0");
            assert!(!super::auto_serve_enabled());
            super::auto_serve_enabled()
        };
        let test_dir = TestDir::new("opt-out");
        let executable = test_dir.path.join("tailscale");
        let invocations = test_dir.path.join("invocations");
        let script = r#"#!/bin/sh
if [ "$1" = "status" ]; then
  printf '%s\n' '{"BackendState":"Running","Self":{"DNSName":"laptop.example.ts.net."}}'
elif [ "$2" = "status" ]; then
  printf '%s\n' '{}'
else
  printf '%s\n' "$*" >> '__INVOCATIONS__'
fi
"#
        .replace("__INVOCATIONS__", &invocations.display().to_string());
        write_executable(&executable, &script);

        let result = run_onboarding(&executable, 8_788, auto_serve, |_| {}, |_| true).await;

        assert!(matches!(result, Err(ExposureError::MissingMapping { .. })));
        assert!(!invocations.exists());
    }

    fn status() -> serde_json::Value {
        json!({
            "BackendState": "Running",
            "Self": { "DNSName": "laptop.example.ts.net." }
        })
    }

    #[test]
    fn accepts_exact_tailnet_https_mapping() {
        let serve = json!({
            "TCP": { "443": { "HTTPS": true } },
            "Web": {
                "laptop.example.ts.net:443": {
                    "Handlers": { "/": { "Proxy": "http://127.0.0.1:8788" } }
                }
            },
            "AllowFunnel": { "laptop.example.ts.net:443": false }
        });
        let exposure = validate(&status(), &serve, 8_788).expect("valid mapping");
        assert_eq!(exposure.origin, "https://laptop.example.ts.net");
        assert_eq!(exposure.bind_addr.to_string(), "127.0.0.1:8788");
        assert!(exposure.mount_path.is_empty());
        assert_eq!(exposure.url(), "https://laptop.example.ts.net/");
    }

    #[test]
    fn accepts_path_mount_for_the_selected_port() {
        let serve = json!({
            "TCP": { "443": { "HTTPS": true } },
            "Web": {
                "laptop.example.ts.net:443": {
                    "Handlers": {
                        "/": { "Proxy": "http://127.0.0.1:8788" },
                        "/s8790": { "Proxy": "http://127.0.0.1:8790" }
                    }
                }
            }
        });
        let exposure = validate(&status(), &serve, 8_790).expect("valid path mapping");
        assert_eq!(exposure.mount_path, "/s8790");
        assert_eq!(exposure.url(), "https://laptop.example.ts.net/s8790/");
    }

    #[test]
    fn rejects_conflicting_target() {
        let wrong = json!({
            "Web": {
                "laptop.example.ts.net:443": {
                    "Handlers": {
                        "/": { "Proxy": "http://127.0.0.1:8788" },
                        "/s8789": { "Proxy": "http://127.0.0.1:9999" }
                    }
                }
            }
        });
        assert!(matches!(
            validate(&status(), &wrong, 8_789),
            Err(ExposureError::ConflictingMapping { ref actual, .. })
                if actual == "http://127.0.0.1:9999"
        ));
    }

    #[test]
    fn funnel_rejection_is_unchanged() {
        let funnel = json!({
            "Web": {
                "laptop.example.ts.net:443": {
                    "Handlers": { "/": { "Proxy": "http://127.0.0.1:8788" } }
                }
            },
            "AllowFunnel": { "laptop.example.ts.net:443": true }
        });
        assert!(matches!(
            validate(&status(), &funnel, 8_788),
            Err(ExposureError::PublicExposure)
        ));
    }

    #[test]
    fn rejects_unknown_or_stopped_tailscale_state() {
        assert!(matches!(
            validate(&json!({}), &json!({}), 8_788),
            Err(ExposureError::NotRunning)
        ));
    }
}
