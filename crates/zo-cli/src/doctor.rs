//! `zo doctor` — a deterministic, secret-safe environment diagnosis with
//! Claude Code-style automatic *safe* recovery.
//!
//! One deep engine backs both the top-level `zo doctor` command and the
//! `/doctor` slash command. It diagnoses running-binary/PATH consistency,
//! config discovery/parsing, auth presence, MCP configuration and stdio command
//! availability, config/state directory privacy and writability, sandbox mode,
//! and Git worktree state, then renders a concise PASS/WARN/FAIL/FIXED report
//! with a final Healthy / Needs attention summary.
//!
//! Repair boundary (see [`DoctorMode::Repair`]): the only mutations are
//! creating the canonical Zo-owned global config/state directories and
//! tightening owner-only permissions on Zo-owned config/state directories and
//! existing regular settings/state files, always through the no-follow,
//! ownership-validating [`runtime::secure_fs`] helpers. Config content, auth
//! credentials, PATH, symlinks, binaries, and MCP servers are never touched.
//! [`DoctorMode::Check`] performs no filesystem mutation at all.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use runtime::{
    resolve_sandbox_status, ConfigLoader, McpServerConfig, RuntimeConfig,
    SandboxStatus,
};

/// Whether `doctor` may apply safe repairs, or is strictly read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DoctorMode {
    /// Diagnose and automatically apply only safe, reversible local repairs.
    Repair,
    /// Strictly read-only: diagnose without mutating the filesystem.
    Check,
}

impl DoctorMode {
    const fn repairs_enabled(self) -> bool {
        matches!(self, Self::Repair)
    }
}

/// Severity of a single check outcome. Ordering (`Pass < Fixed < Warn < Fail`)
/// drives the final summary: any `Fail` or `Warn` means "Needs attention".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Severity {
    /// The item is healthy.
    Pass,
    /// The item was unhealthy but a safe repair restored it (repair mode only).
    Fixed,
    /// Actionable but non-fatal: missing auth, no worktree, PATH mismatch, …
    Warn,
    /// A clear problem the user must resolve: malformed config, unsafe entry,
    /// unwritable required state, unavailable configured MCP command.
    Fail,
}

impl Severity {
    const fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fixed => "FIXED",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

/// One diagnosed item: a stable identifier, its severity, and a controlled,
/// secret-safe message. Messages are built from check names and paths only —
/// never from merged config values, environment contents, OAuth values, MCP
/// headers, or raw parser input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Finding {
    pub(crate) check: &'static str,
    pub(crate) severity: Severity,
    pub(crate) message: String,
}

impl Finding {
    fn new(check: &'static str, severity: Severity, message: impl Into<String>) -> Self {
        Self {
            check,
            severity,
            message: message.into(),
        }
    }
}

/// The full diagnosis: an ordered list of findings plus the running version
/// line. Deterministic — findings are emitted in a fixed check order.
#[derive(Debug, Clone)]
pub(crate) struct DoctorReport {
    version_line: String,
    findings: Vec<Finding>,
}

impl DoctorReport {
    /// Whether every finding passed or was fixed (no `Warn`/`Fail` remains).
    pub(crate) fn is_healthy(&self) -> bool {
        self.findings
            .iter()
            .all(|finding| matches!(finding.severity, Severity::Pass | Severity::Fixed))
    }

    /// Render the concise, deterministic report. One line per finding under a
    /// `Doctor` header, then a final Healthy / Needs attention summary.
    pub(crate) fn render(&self) -> String {
        let mut lines = vec![
            "Doctor".to_string(),
            format!("  Version          {}", self.version_line),
        ];
        for finding in &self.findings {
            lines.push(format!(
                "  {:<5} {:<20} {}",
                finding.severity.label(),
                finding.check,
                finding.message
            ));
        }
        lines.push(String::new());
        let (fails, warns) = self.counts();
        if self.is_healthy() {
            lines.push("Summary          Healthy — no action required".to_string());
        } else {
            lines.push(format!(
                "Summary          Needs attention — {fails} failing, {warns} warnings"
            ));
        }
        lines.join("\n")
    }

    fn counts(&self) -> (usize, usize) {
        let fails = self
            .findings
            .iter()
            .filter(|finding| finding.severity == Severity::Fail)
            .count();
        let warns = self
            .findings
            .iter()
            .filter(|finding| finding.severity == Severity::Warn)
            .count();
        (fails, warns)
    }
}

/// Run the doctor engine against `cwd` in the given mode. Pure with respect to
/// its inputs (env + filesystem); constructs no `LiveCli`, session registry, or
/// auth-refresh path, so `--check` provably performs no mutation.
///
/// A persistent Zo home is resolved once through the side-effect-free
/// [`runtime::zo_global_config_roots`]. When it is empty (no `ZO_CONFIG_HOME`,
/// `ZO_HOME`, or `HOME`), the home-owning checks report that no persistent home
/// is resolvable and skip every API that would otherwise fabricate a temporary
/// home via `default_config_home`, so `--check` never touches the filesystem.
pub(crate) fn run(mode: DoctorMode, cwd: &Path) -> DoctorReport {
    let mut findings = Vec::new();
    let home = resolve_persistent_home();

    check_binary_and_path(&mut findings);
    let runtime_config = check_config(&mut findings, cwd, home.as_deref(), mode);
    check_auth(&mut findings, home.as_deref());
    check_mcp(&mut findings, runtime_config.as_ref());
    check_config_home(&mut findings, mode, home.as_deref());
    check_project_state(&mut findings, mode, cwd, home.as_deref());
    check_config_files(&mut findings, mode, cwd, home.as_deref());
    check_sandbox(&mut findings, runtime_config.as_ref(), cwd);
    check_git(&mut findings, cwd);

    DoctorReport {
        version_line: crate::render_version_line(),
        findings,
    }
}

/// The persistent global Zo config home, resolved without side effects. Returns
/// `None` when no `ZO_CONFIG_HOME`, `ZO_HOME`, or `HOME` is set — the case in
/// which `default_config_home` would fabricate a temporary directory.
fn resolve_persistent_home() -> Option<PathBuf> {
    runtime::zo_global_config_roots().into_iter().next()
}

/// Running binary / install / PATH consistency. A resolvable `zo` on `PATH`
/// that matches the running executable is `PASS`; a mismatch or an unresolvable
/// `zo` is an actionable `WARN` (the running binary still works).
fn check_binary_and_path(findings: &mut Vec<Finding>) {
    let current = env::current_exe().ok();
    let path_hit = current
        .as_deref()
        .and_then(|exe| resolve_on_path("zo", exe));
    match (&current, path_hit) {
        (Some(_), Some(PathMatch::Same)) => findings.push(Finding::new(
            "binary/path",
            Severity::Pass,
            "running binary is the `zo` resolved on PATH",
        )),
        (Some(_), Some(PathMatch::Different)) => findings.push(Finding::new(
            "binary/path",
            Severity::Warn,
            "a different `zo` is first on PATH; the running binary is not the one PATH resolves",
        )),
        (Some(_), None) => findings.push(Finding::new(
            "binary/path",
            Severity::Warn,
            "no `zo` found on PATH; add the install directory to PATH to launch it by name",
        )),
        (None, _) => findings.push(Finding::new(
            "binary/path",
            Severity::Warn,
            "could not resolve the running executable path",
        )),
    }
}

enum PathMatch {
    Same,
    Different,
}

/// Resolve `name` against `PATH` (honoring `PATHEXT` on Windows) and report
/// whether the first executable match is the same file as `running`. Metadata
/// only — nothing is spawned.
fn resolve_on_path(name: &str, running: &Path) -> Option<PathMatch> {
    let path = env::var_os("PATH")?;
    let running_canonical = std::fs::canonicalize(running).ok();
    for dir in env::split_paths(&path) {
        for candidate in path_candidates(&dir, name) {
            if is_executable_file(&candidate) {
                let same = std::fs::canonicalize(&candidate)
                    .ok()
                    .zip(running_canonical.as_ref())
                    .is_some_and(|(candidate, running)| &candidate == running);
                return Some(if same {
                    PathMatch::Same
                } else {
                    PathMatch::Different
                });
            }
        }
    }
    None
}

#[cfg(windows)]
fn path_candidates(dir: &Path, name: &str) -> Vec<PathBuf> {
    let exts = env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let mut candidates = vec![dir.join(name)];
    for ext in exts.split(';').filter(|ext| !ext.is_empty()) {
        candidates.push(dir.join(format!("{name}{ext}")));
    }
    candidates
}

#[cfg(not(windows))]
fn path_candidates(dir: &Path, name: &str) -> Vec<PathBuf> {
    vec![dir.join(name)]
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::metadata(path).is_ok_and(|metadata| {
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    })
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
}

/// Config discovery and parsing. Before parsing, every discovered user-scope
/// settings path is snapshotted no-follow through [`runtime::secure_fs`]; a
/// symlink or non-regular user settings entry is a `FAIL` and the config is not
/// loaded from it. A parse failure becomes a `FAIL` finding (never a crash, and
/// the file is never rewritten); success reports how many discovered files
/// loaded. Returns the loaded config for downstream checks, or `None` when
/// loading was refused or failed.
fn check_config(
    findings: &mut Vec<Finding>,
    cwd: &Path,
    home: Option<&Path>,
    mode: DoctorMode,
) -> Option<RuntimeConfig> {
    if home.is_none() && mode == DoctorMode::Check {
        // No persistent home is resolvable. Loading would invoke
        // `default_config_home`, which fabricates a temporary directory — not
        // allowed in strictly read-only mode.
        findings.push(Finding::new(
            "config/parse",
            Severity::Warn,
            "no persistent Zo home is resolvable (set ZO_CONFIG_HOME, ZO_HOME, or HOME); config not loaded in --check",
        ));
        return None;
    }

    let loader = ConfigLoader::default_for(cwd);
    let discovered_entries = loader.discover();
    // Refuse symlinks at any component, non-regular files, foreign ownership,
    // or hard links before the loader can read a discovered settings path.
    // Missing optional settings files are healthy and need no probe.
    for entry in &discovered_entries {
        match probe_regular_file_absolute(&entry.path) {
            SafetyProbe::Missing | SafetyProbe::SafePrivate | SafetyProbe::SafeBroad => {}
            SafetyProbe::Unsafe(reason) => {
                findings.push(Finding::new("config/parse", Severity::Fail, reason));
                return None;
            }
        }
    }
    let discovered = discovered_entries.len();
    if let Ok(config) = loader.load() {
        let loaded_count = config.loaded_entries().len();
        findings.push(Finding::new(
            "config/parse",
            Severity::Pass,
            format!("{loaded_count} of {discovered} discovered config files loaded and parsed"),
        ));
        Some(config)
    } else {
        // Deliberately omit the underlying error text: a parser message can
        // echo raw settings source. Point the user at the files instead.
        findings.push(Finding::new(
            "config/parse",
            Severity::Fail,
            "a discovered settings file is malformed; fix its JSON (content left unchanged)",
        ));
        None
    }
}

/// Auth presence across every supported provider: Anthropic, OpenAI, Google,
/// and xAI. Missing auth is an actionable `WARN`, never a crash, and no
/// credential value is ever printed — only provider names and presence.
///
/// Environment API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
/// `GOOGLE_API_KEY`, `XAI_API_KEY`) are read for presence only. Saved OAuth for
/// Anthropic, OpenAI (ChatGPT), and Google Code Assist is detected through a
/// no-follow, secret-safe presence probe over the shared `credentials.json`; no
/// token is parsed, refreshed, minted, printed, or written, and no credential
/// helper or network call runs.
///
/// The credentials store is first validated no-follow: a symlinked or otherwise
/// unsafe `credentials.json` is a `FAIL` and no saved OAuth is read — this holds
/// even when an environment key is also present, so an unsafe store is never
/// silently masked by another provider's env key. With no persistent home the
/// saved-OAuth probe is skipped (it would fabricate a temporary home) and only
/// environment presence is reported.
fn check_auth(findings: &mut Vec<Finding>, home: Option<&Path>) {
    // Fixed provider labels for env credentials; presence only, never values.
    // Mirrors the provider auth layer: Anthropic accepts either `ANTHROPIC_API_KEY`
    // or `ANTHROPIC_AUTH_TOKEN`; Google accepts `GOOGLE_API_KEY` or a
    // `GOOGLE_ACCESS_TOKEN` override.
    const ENV_KEYS: &[(&str, &str)] = &[
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Anthropic", "ANTHROPIC_AUTH_TOKEN"),
        ("OpenAI", "OPENAI_API_KEY"),
        ("Google", "GOOGLE_API_KEY"),
        ("Google", api::GOOGLE_ACCESS_TOKEN_ENV),
        ("xAI", "XAI_API_KEY"),
    ];

    if !credentials_store_is_safe() {
        findings.push(Finding::new(
            "auth",
            Severity::Fail,
            "stored credentials entry is unsafe or unverifiable; credentials were not read",
        ));
        return;
    }

    let mut providers: Vec<&'static str> = Vec::new();
    for (label, key) in ENV_KEYS {
        if env::var_os(key).is_some_and(|value| !value.is_empty()) && !providers.contains(label) {
            providers.push(label);
        }
    }

    // Google Application Default Credentials: a readable ADC file is a valid
    // Google credential. Probed no-follow (never `Path::is_file`, never
    // `gcloud`, never a token mint). An unsafe ADC leaf is reported but not a
    // hard `FAIL` — the shared credentials store governs the FAIL contract.
    if adc_credentials_present_no_follow() && !providers.contains(&"Google") {
        providers.push("Google");
    }

    // Saved OAuth is only probed with a resolvable persistent home so `--check`
    // never fabricates one. The probe is no-follow across the full effective
    // credential chain and reports presence only. An unsafe/unreadable store
    // surfaces as an error here and is a `FAIL`, never silently masked by an env
    // key that happens to be set.
    if home.is_some() {
        for (label, provider) in [
            ("Anthropic OAuth", api::SavedOAuthProvider::Anthropic),
            ("OpenAI OAuth", api::SavedOAuthProvider::OpenAi),
            ("Google OAuth", api::SavedOAuthProvider::GoogleCodeAssist),
        ] {
            match saved_oauth_present_no_follow(provider) {
                Ok(true) => providers.push(label),
                Ok(false) => {}
                Err(_) => {
                    findings.push(Finding::new(
                        "auth",
                        Severity::Fail,
                        "stored credentials entry is unsafe or unverifiable; credentials were not read",
                    ));
                    return;
                }
            }
        }
    }

    if providers.is_empty() {
        findings.push(Finding::new(
            "auth",
            Severity::Warn,
            "no provider API key or OAuth credentials found; run `zo login` or set a provider API key",
        ));
    } else {
        findings.push(Finding::new(
            "auth",
            Severity::Pass,
            format!("credentials present for {}", providers.join(", ")),
        ));
    }
}

/// Whether the shared `credentials.json` under every configured config root is
/// safe to read no-follow: a symlink, non-regular, or foreign-owned entry makes
/// it unsafe. A missing file is safe (nothing to read yet). This preserves the
/// pre-existing "unsafe credentials store yields FAIL" behavior regardless of
/// which provider env keys are set.
fn credentials_store_is_safe() -> bool {
    for config_root in runtime::zo_global_config_roots() {
        let full = config_root.join("credentials.json");
        // `Ok(true)`/`Ok(false)` are both safe (present-and-safe or absent); an
        // `Err` (a symlink at any component including an intermediate ancestor,
        // a non-regular entry, or a foreign-owned entry) is the fail-closed
        // default. The whole absolute path is treated as attacker-controlled and
        // opened `O_NOFOLLOW` from `/`, never canonicalized — closing the
        // intermediate-symlink hole the earlier `open_root`-canonicalizing probe
        // left open.
        if runtime::secure_fs::is_safe_regular_file_absolute_no_follow(&full).is_err() {
            return false;
        }
    }
    true
}

/// No-follow, secret-safe presence probe for one saved-OAuth provider across the
/// full effective credential chain (`ZO_CONFIG_HOME → ZO_HOME → ~/.zo →
/// ~/.forge`). Each root's `credentials.json` is read through the absolute
/// no-follow reader, which opens every component (including intermediate
/// ancestors and the leaf) with `O_NOFOLLOW` and never canonicalizes, so a
/// symlink planted at any user-controlled path component surfaces as an error
/// rather than being followed. Returns presence only — no token value is ever
/// exposed. An unsafe/unreadable root propagates as `Err` (mapped to `FAIL` by
/// the caller), never silently discarded.
fn saved_oauth_present_no_follow(provider: api::SavedOAuthProvider) -> std::io::Result<bool> {
    api::saved_oauth_present_effective(provider, &|path| {
        runtime::secure_fs::read_regular_file_absolute_no_follow(path)
    })
}

/// Whether a Google Application Default Credentials file is present as a safe,
/// no-follow-readable regular file. The candidate is
/// `$GOOGLE_APPLICATION_CREDENTIALS` when set, else gcloud's well-known
/// `$HOME/.config/gcloud/application_default_credentials.json`. Presence is
/// established through the absolute no-follow probe (never `Path::is_file`), so
/// a symlinked or foreign-owned ADC leaf is not counted as present; `gcloud` is
/// never invoked and no token is minted or read for value.
fn adc_credentials_present_no_follow() -> bool {
    let Some(path) = adc_credentials_candidate() else {
        return false;
    };
    if !path.is_absolute() {
        return false;
    }
    runtime::secure_fs::is_safe_regular_file_absolute_no_follow(&path).unwrap_or(false)
}

/// The ADC file path doctor probes: an explicit `GOOGLE_APPLICATION_CREDENTIALS`
/// override, else gcloud's well-known path under `$HOME`. Returns `None` when
/// neither is resolvable.
fn adc_credentials_candidate() -> Option<PathBuf> {
    if let Some(explicit) = env::var_os("GOOGLE_APPLICATION_CREDENTIALS")
        .map(PathBuf::from)
        .filter(|value| !value.as_os_str().is_empty())
    {
        return Some(explicit);
    }
    let home = env::var_os("HOME").filter(|value| !value.is_empty())?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("gcloud")
            .join("application_default_credentials.json"),
    )
}

/// MCP configuration and stdio command availability. No server is spawned and
/// no network I/O occurs — a configured stdio command is only resolved on PATH
/// (or as an absolute/relative path) by metadata. No configured servers is a
/// `WARN`; an unavailable stdio command is a `FAIL`. Configured servers awaiting
/// project trust are reported deterministically as a `WARN` without resolving or
/// spawning them. Headers/URLs are never printed.
fn check_mcp(findings: &mut Vec<Finding>, config: Option<&RuntimeConfig>) {
    let Some(config) = config else {
        findings.push(Finding::new(
            "mcp",
            Severity::Warn,
            "MCP configuration was not checked because config parsing failed",
        ));
        return;
    };
    let servers = config.mcp().servers();
    let gated = config.mcp().untrusted_project_servers();
    if servers.is_empty() {
        if gated.is_empty() {
            findings.push(Finding::new(
                "mcp",
                Severity::Warn,
                "no MCP servers configured",
            ));
        } else {
            findings.push(Finding::new(
                "mcp",
                Severity::Warn,
                format!(
                    "{} project MCP server(s) awaiting trust; run `zo` and trust the workspace to enable them",
                    gated.len()
                ),
            ));
        }
        return;
    }

    let mut missing = Vec::new();
    for (name, scoped) in servers {
        if let McpServerConfig::Stdio(stdio) = &scoped.config {
            if !stdio_command_available(&stdio.command) {
                missing.push(name.clone());
            }
        }
    }

    let gated_suffix = if gated.is_empty() {
        String::new()
    } else {
        format!("; {} more awaiting workspace trust", gated.len())
    };

    if missing.is_empty() {
        findings.push(Finding::new(
            "mcp",
            if gated.is_empty() {
                Severity::Pass
            } else {
                Severity::Warn
            },
            format!(
                "{} configured server(s); all stdio commands resolve{gated_suffix}",
                servers.len()
            ),
        ));
    } else {
        findings.push(Finding::new(
            "mcp",
            Severity::Fail,
            format!(
                "{} configured server(s); stdio command not found for: {}{gated_suffix}",
                servers.len(),
                missing.join(", ")
            ),
        ));
    }
}

/// Whether an MCP stdio `command` resolves to an executable without running it.
/// An absolute or explicitly-relative path is checked directly; a bare name is
/// resolved against PATH. Metadata only.
fn stdio_command_available(command: &str) -> bool {
    if command.is_empty() {
        return false;
    }
    let path = Path::new(command);
    if path.components().count() > 1 || path.is_absolute() {
        return is_executable_file(path);
    }
    if let Some(dirs) = env::var_os("PATH") {
        for dir in env::split_paths(&dirs) {
            if path_candidates(&dir, command)
                .iter()
                .any(|candidate| is_executable_file(candidate))
            {
                return true;
            }
        }
    }
    false
}

/// Global Zo config-home directory: existence, owner-only privacy, and
/// writability. In repair mode a missing home is created privately and an
/// over-broad mode is tightened, with the postcondition re-checked before a
/// `FIXED` is reported. With no resolvable persistent home the check is an
/// actionable `WARN` and nothing is created.
///
/// The config home is created/validated through the absolute no-follow walker,
/// so a symlink planted at *any* user-controlled ancestor of the home is
/// rejected rather than followed and canonicalized.
fn check_config_home(findings: &mut Vec<Finding>, mode: DoctorMode, home: Option<&Path>) {
    let Some(home) = home else {
        findings.push(Finding::new(
            "config-home",
            Severity::Warn,
            "no persistent Zo home is resolvable; set ZO_CONFIG_HOME, ZO_HOME, or HOME",
        ));
        return;
    };
    // Only the config-home directory itself is Zo-owned; its parent must
    // already exist and is never created (an explicit `ZO_CONFIG_HOME` whose
    // non-Zo parent is absent fails rather than fabricating the parent).
    check_owned_dir_absolute(findings, "config-home", home, 1, mode, "global config home");
}

/// Per-project state directory: same existence/privacy/writability contract as
/// the config home, but nested several components deep
/// (`<config-home>/projects/<slug>/state` by default, or under `$ZO_STATE_DIR`).
///
/// The complete absolute target is created and validated through the absolute
/// no-follow walker, which opens every user-controlled component (including
/// intermediate ancestors) with `O_NOFOLLOW` and never canonicalizes. A symlink
/// at any ancestor — even one whose target already contains `projects/` — is
/// therefore rejected instead of traversed.
fn check_project_state(
    findings: &mut Vec<Finding>,
    mode: DoctorMode,
    cwd: &Path,
    home: Option<&Path>,
) {
    // The configured base must be resolvable (either an explicit `ZO_STATE_DIR`
    // or the persistent home); otherwise there is no trustworthy anchor.
    let base_ok = env::var_os("ZO_STATE_DIR").is_some_and(|dir| !dir.is_empty()) || home.is_some();
    if !base_ok {
        findings.push(Finding::new(
            "state-dir",
            Severity::Warn,
            "no persistent state root is resolvable; set ZO_CONFIG_HOME, ZO_HOME, HOME, or ZO_STATE_DIR",
        ));
        return;
    }
    let full = runtime::zo_project_state_dir(cwd);
    // Owned-suffix accounting:
    //   * With an explicit `$ZO_STATE_DIR`, the user designated that base as a
    //     Zo-owned state root, so Zo owns the base plus the
    //     `projects/<slug>/state` triad (suffix 4). Only the base's *parent*
    //     must already exist.
    //   * Otherwise the target is `<config-home>/projects/<slug>/state`; the
    //     config home is a non-Zo root validated by `check_config_home` and must
    //     already exist, so only the `projects/<slug>/state` triad is Zo-owned
    //     (suffix 3).
    let owned_suffix = if env::var_os("ZO_STATE_DIR").is_some_and(|dir| !dir.is_empty()) {
        4
    } else {
        3
    };
    check_owned_dir_absolute(
        findings,
        "state-dir",
        &full,
        owned_suffix,
        mode,
        "project state directory",
    );
}

/// Directory check for a Zo-owned config/state directory named by its complete
/// absolute path. Every user-controlled component — including intermediate
/// ancestors — is opened `O_NOFOLLOW` from a genuinely trusted `/` descriptor,
/// so a symlink planted at any ancestor is rejected rather than followed or
/// canonicalized. Creation and tightening operate on the complete Zo-owned
/// suffix, while ancestors above that boundary are opened no-follow but never
/// created or modified.
///
/// `--check` is metadata-only: it probes the complete owned suffix no-follow and
/// never creates or chmods anything.
fn check_owned_dir_absolute(
    findings: &mut Vec<Finding>,
    check: &'static str,
    dir: &Path,
    owned_suffix_len: usize,
    mode: DoctorMode,
    label: &str,
) {
    // A directory is healthy only when every component inside the Zo-owned
    // suffix is owner-owned and exactly 0o700. A private leaf under a broad
    // `projects/<slug>` chain must not short-circuit this full-suffix check.
    if runtime::secure_fs::is_owned_private_suffix_absolute(dir, owned_suffix_len)
        .unwrap_or(false)
    {
        findings.push(Finding::new(
            check,
            Severity::Pass,
            format!("{label} and its owned path are owner-only and writable"),
        ));
        return;
    }

    // Distinguish "missing" (nothing at the leaf) from "present but unsafe or
    // too-broad" using a single no-follow lstat of the leaf under its trusted
    // parent. Any symlink ancestor makes the parent walk fail, classifying as
    // `Unsafe` — the fail-closed default.
    match runtime::secure_fs::owned_dir_leaf_state_absolute(dir) {
        runtime::secure_fs::AbsoluteDirLeaf::Missing => {
            if !mode.repairs_enabled() {
                findings.push(Finding::new(
                    check,
                    Severity::Warn,
                    format!("{label} does not exist yet (run `zo doctor` to create it privately)"),
                ));
                return;
            }
            match runtime::secure_fs::ensure_private_dir_absolute(dir, owned_suffix_len) {
                // The postcondition verifies the *entire* Zo-owned suffix is
                // owner-only, not just the leaf, so a pre-existing broad suffix
                // directory (for example `projects/` at 0o777) is only reported
                // FIXED once every owned suffix dir is confirmed private.
                Ok(_)
                    if runtime::secure_fs::is_owned_private_suffix_absolute(
                        dir,
                        owned_suffix_len,
                    )
                    .unwrap_or(false) =>
                {
                    findings.push(Finding::new(
                        check,
                        Severity::Fixed,
                        format!("created missing {label} with owner-only permissions"),
                    ));
                }
                _ => {
                    findings.push(Finding::new(
                        check,
                        Severity::Fail,
                        format!("{label} is missing and could not be created safely"),
                    ));
                }
            }
        }
        runtime::secure_fs::AbsoluteDirLeaf::OwnedDirTooBroad => {
            if !mode.repairs_enabled() {
                findings.push(Finding::new(
                    check,
                    Severity::Warn,
                    format!("{label} or its owned path is not owner-only 0o700 (run `zo doctor` to restore)"),
                ));
                return;
            }
            match runtime::secure_fs::ensure_private_dir_absolute(dir, owned_suffix_len) {
                Ok(_)
                    if runtime::secure_fs::is_owned_private_suffix_absolute(
                        dir,
                        owned_suffix_len,
                    )
                    .unwrap_or(false) =>
                {
                    findings.push(Finding::new(
                        check,
                        Severity::Fixed,
                        format!("restored {label} and its owned path to owner-only 0o700 permissions"),
                    ));
                }
                _ => {
                    findings.push(Finding::new(
                        check,
                        Severity::Fail,
                        format!("{label} permissions could not be restored to owner-only safely"),
                    ));
                }
            }
        }
        runtime::secure_fs::AbsoluteDirLeaf::Unsafe => {
            findings.push(Finding::new(
                check,
                Severity::Fail,
                format!("{label} is not a current-user-owned directory (symlink or foreign entry); not repaired"),
            ));
        }
    }
}

/// Known user settings / credential / state files: each must be an
/// owner-owned, singly linked, non-symlink regular file with `0o600`
/// permissions. A symlink or non-regular entry is a `FAIL` and never followed or
/// modified; an over-broad mode is tightened in repair mode and re-checked
/// before `FIXED`. A missing file is not a fault (nothing to secure yet).
/// `--check` performs metadata-only probing.
fn check_config_files(
    findings: &mut Vec<Finding>,
    mode: DoctorMode,
    cwd: &Path,
    home: Option<&Path>,
) {
    let mut targets: Vec<(PathBuf, &'static str)> = Vec::new();
    if let Some(home) = home {
        targets.push((home.join("settings.json"), "user settings"));
        targets.push((home.join("credentials.json"), "stored credentials"));
    }
    targets.push((cwd.join(".zo/settings.local.json"), "local settings"));

    let mut any = false;
    for (full, label) in &targets {
        let probe = probe_regular_file_absolute(full);
        if matches!(probe, SafetyProbe::Missing) {
            continue;
        }
        any = true;
        check_owned_file(findings, full, probe, mode, label);
    }
    if !any {
        findings.push(Finding::new(
            "config-files",
            Severity::Pass,
            "no user settings/credential files present to secure",
        ));
    }
}

/// Validate and (in repair mode) tighten one known regular user file.
fn check_owned_file(
    findings: &mut Vec<Finding>,
    full: &Path,
    probe: SafetyProbe,
    mode: DoctorMode,
    label: &str,
) {
    match probe {
        SafetyProbe::Missing => {}
        SafetyProbe::Unsafe(reason) => {
            findings.push(Finding::new("config-files", Severity::Fail, reason));
        }
        SafetyProbe::SafePrivate => {
            findings.push(Finding::new(
                "config-files",
                Severity::Pass,
                format!("{label} is an owner-only regular file"),
            ));
        }
        SafetyProbe::SafeBroad => {
            if !mode.repairs_enabled() {
                findings.push(Finding::new(
                    "config-files",
                    Severity::Warn,
                    format!("{label} is broader than owner-only 0o600 (run `zo doctor` to tighten)"),
                ));
                return;
            }
            match runtime::secure_fs::restrict_existing_owner_only_regular_file_absolute(full) {
                Ok(())
                    if matches!(
                        probe_regular_file_absolute(full),
                        SafetyProbe::SafePrivate
                    ) =>
                {
                    findings.push(Finding::new(
                        "config-files",
                        Severity::Fixed,
                        format!("tightened {label} to owner-only 0o600"),
                    ));
                }
                _ => {
                    findings.push(Finding::new(
                        "config-files",
                        Severity::Fail,
                        format!("{label} permissions could not be tightened safely"),
                    ));
                }
            }
        }
    }
}

/// Safety classification of a candidate regular user file, established
/// no-follow.
enum SafetyProbe {
    /// The leaf or one of its parent directories does not exist yet.
    Missing,
    /// A safe, owner-owned, singly linked regular file with `0o600`.
    SafePrivate,
    /// A safe, owner-owned, singly linked regular file with broader-than-`0o600`
    /// permissions (repairable).
    SafeBroad,
    /// Not a safe regular file (symlink, non-regular, foreign-owned, or
    /// multiply linked). Carries a secret-safe reason. Never followed/modified.
    Unsafe(String),
}

/// Probe an absolute settings/credential path from a trusted root descriptor.
/// Every component is opened with `O_NOFOLLOW`; a safely absent leaf is distinct
/// from an unsafe symlink, non-regular, foreign-owned, or multiply linked file.
fn probe_regular_file_absolute(full: &Path) -> SafetyProbe {
    match runtime::secure_fs::is_safe_regular_file_absolute_no_follow(full) {
        Ok(false) => SafetyProbe::Missing,
        Err(_) => SafetyProbe::Unsafe(unsafe_file_reason(
            full,
            "is unsafe or could not be inspected without following symlinks; not modified",
        )),
        Ok(true) => match runtime::secure_fs::is_owned_private_regular_file_absolute(full) {
            Ok(true) => SafetyProbe::SafePrivate,
            Ok(false) => SafetyProbe::SafeBroad,
            Err(_) => SafetyProbe::Unsafe(unsafe_file_reason(
                full,
                "could not be revalidated safely; not modified",
            )),
        },
    }
}

/// A secret-safe reason string that names the containing directory (never file
/// contents). The leaf name is intentionally omitted so a credentials filename
/// choice cannot leak a token; the directory still locates the issue.
fn unsafe_file_reason(path: &Path, what: &str) -> String {
    let directory = path.parent().unwrap_or(path);
    format!(
        "a settings/credential file under {} {what}",
        directory.display()
    )
}

/// Sandbox posture, reported from the resolved status. An off sandbox is a valid
/// configuration (`PASS`); an unavailable status because config parsing failed
/// is an actionable `WARN`, not a pass.
fn check_sandbox(findings: &mut Vec<Finding>, config: Option<&RuntimeConfig>, cwd: &Path) {
    match config.map(|config| resolve_sandbox_status(config.sandbox(), cwd)) {
        Some(status) => findings.push(Finding::new(
            "sandbox",
            Severity::Pass,
            describe_sandbox(&status),
        )),
        None => findings.push(Finding::new(
            "sandbox",
            Severity::Warn,
            "sandbox status unavailable because config parsing failed",
        )),
    }
}

fn describe_sandbox(status: &SandboxStatus) -> String {
    format!(
        "enabled={} active={} mode={}",
        status.enabled,
        status.active,
        status.filesystem_mode.as_str()
    )
}

/// Git availability and worktree state via fixed-argument `git` invocations.
/// `git` unavailable or "not a worktree" are actionable `WARN`s, never crashes.
/// A user-controlled command string is never inherited.
fn check_git(findings: &mut Vec<Finding>, cwd: &Path) {
    match git_output(cwd, &["rev-parse", "--is-inside-work-tree"]) {
        GitProbe::Unavailable => findings.push(Finding::new(
            "git",
            Severity::Warn,
            "`git` is not available on PATH",
        )),
        GitProbe::NotWorktree => findings.push(Finding::new(
            "git",
            Severity::Warn,
            "not inside a Git worktree",
        )),
        GitProbe::Worktree => {
            findings.push(Finding::new("git", Severity::Pass, "inside a Git worktree"));
        }
    }
}

enum GitProbe {
    Unavailable,
    NotWorktree,
    Worktree,
}

fn git_output(cwd: &Path, args: &[&str]) -> GitProbe {
    match Command::new("git").args(args).current_dir(cwd).output() {
        Err(_) => GitProbe::Unavailable,
        Ok(output) if output.status.success() => GitProbe::Worktree,
        Ok(_) => GitProbe::NotWorktree,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// A private temp directory used as an isolated cwd or fake home root. Never
    /// touches the developer's real home.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos();
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "zo-doctor-{label}-{}-{nanos}-{seq}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn report_render_contains_summary(healthy: bool) -> String {
        let report = DoctorReport {
            version_line: "zo 0.0.0 (test, 2026-01-01)".to_string(),
            findings: vec![Finding::new(
                "auth",
                if healthy { Severity::Pass } else { Severity::Warn },
                "example",
            )],
        };
        report.render()
    }

    #[test]
    fn render_reports_healthy_summary_and_version() {
        let rendered = report_render_contains_summary(true);
        assert!(rendered.contains("Doctor"), "{rendered}");
        assert!(rendered.contains("Version"), "{rendered}");
        assert!(rendered.contains("zo 0.0.0 (test, 2026-01-01)"), "{rendered}");
        assert!(rendered.contains("PASS  auth"), "{rendered}");
        assert!(
            rendered.contains("Healthy — no action required"),
            "{rendered}"
        );
    }

    #[test]
    fn render_reports_needs_attention_with_counts() {
        let rendered = report_render_contains_summary(false);
        assert!(rendered.contains("WARN  auth"), "{rendered}");
        assert!(
            rendered.contains("Needs attention — 0 failing, 1 warnings"),
            "{rendered}"
        );
    }

    #[test]
    fn severity_orders_pass_below_warn_and_fail() {
        assert!(Severity::Pass < Severity::Warn);
        assert!(Severity::Fixed < Severity::Warn);
        assert!(Severity::Warn < Severity::Fail);
    }

    #[cfg(unix)]
    #[test]
    fn check_mode_does_not_create_missing_config_home() {
        let base = TempDir::new("check-nomut");
        let missing = base.path().join(".zo");
        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "config-home",
            &missing,
            /* owned suffix */ 1,
            DoctorMode::Check,
            "global config home",
        );
        assert!(!missing.exists(), "check mode must not create the directory");
        let finding = &findings[0];
        assert_eq!(finding.severity, Severity::Warn);
    }

    #[cfg(unix)]
    #[test]
    fn repair_mode_creates_missing_config_home_privately() {
        use std::os::unix::fs::PermissionsExt as _;
        // The parent exists; only the leaf is missing — exactly the shape of
        // `~/.zo` or `$ZO_CONFIG_HOME` with an existing parent. The absolute
        // no-follow walker creates the missing suffix privately.
        let base = TempDir::new("repair-create");
        let missing = base.path().join(".zo");
        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "config-home",
            &missing,
            /* owned suffix */ 1,
            DoctorMode::Repair,
            "global config home",
        );
        assert!(missing.is_dir(), "repair mode must create the directory");
        assert_eq!(
            fs::metadata(&missing).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(findings[0].severity, Severity::Fixed);
    }

    #[cfg(unix)]
    #[test]
    fn repair_mode_creates_nested_missing_state_dir() {
        use std::os::unix::fs::PermissionsExt as _;
        // Only the base exists; the whole nested `projects/<slug>/state` tail is
        // created wholesale through the absolute no-follow walker — the
        // first-run recovery the review flagged.
        let base = TempDir::new("repair-nested");
        let root = base.path().join("state-base");
        fs::create_dir(&root).unwrap();
        let full = root.join("projects").join("slug").join("state");

        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "state-dir",
            &full,
            /* owned suffix */ 3,
            DoctorMode::Repair,
            "project state directory",
        );
        assert!(full.is_dir(), "nested state dir must be created");
        assert_eq!(
            fs::metadata(&full).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(findings[0].severity, Severity::Fixed, "{findings:?}");
    }

    #[cfg(unix)]
    #[test]
    fn repair_tightens_preexisting_broad_owned_suffix_then_reports_fixed() {
        use std::os::unix::fs::PermissionsExt as _;
        // A pre-existing broad base and `projects/` (0o777) with `slug/state`
        // missing below: doctor must tighten every Zo-owned suffix dir to 0o700
        // and only then report FIXED — never claim FIXED while a suffix dir
        // remains broad. The non-Zo parent stays unchanged.
        let base = TempDir::new("repair-broad-suffix");
        let non_zo_parent = base.path().join("parent");
        fs::create_dir(&non_zo_parent).unwrap();
        fs::set_permissions(&non_zo_parent, fs::Permissions::from_mode(0o755)).unwrap();
        let root = non_zo_parent.join("state-base");
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o777)).unwrap();
        let projects = root.join("projects");
        fs::create_dir(&projects).unwrap();
        fs::set_permissions(&projects, fs::Permissions::from_mode(0o777)).unwrap();
        let full = projects.join("slug").join("state");

        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "state-dir",
            &full,
            /* owned suffix: base + projects + slug + state */ 4,
            DoctorMode::Repair,
            "project state directory",
        );
        for dir in [&root, &projects, &projects.join("slug"), &full] {
            assert_eq!(
                fs::metadata(dir).unwrap().permissions().mode() & 0o777,
                0o700,
                "owned suffix dir {dir:?} must be tightened"
            );
        }
        assert_eq!(
            fs::metadata(&non_zo_parent).unwrap().permissions().mode() & 0o777,
            0o755,
            "non-Zo parent must be untouched"
        );
        assert_eq!(findings[0].severity, Severity::Fixed, "{findings:?}");
    }

    #[cfg(unix)]
    fn assert_existing_state_suffix_repaired(label: &str, leaf_mode: u32) {
        use std::os::unix::fs::PermissionsExt as _;

        let base = TempDir::new(label);
        let non_zo_parent = base.path().join("parent");
        fs::create_dir(&non_zo_parent).unwrap();
        fs::set_permissions(&non_zo_parent, fs::Permissions::from_mode(0o755)).unwrap();
        let root = non_zo_parent.join("state-base");
        let projects = root.join("projects");
        let slug = projects.join("slug");
        let full = slug.join("state");
        fs::create_dir_all(&full).unwrap();
        for dir in [&root, &projects, &slug] {
            fs::set_permissions(dir, fs::Permissions::from_mode(0o777)).unwrap();
        }
        fs::set_permissions(&full, fs::Permissions::from_mode(leaf_mode)).unwrap();

        assert!(
            !runtime::secure_fs::is_owned_private_suffix_absolute(&full, 4).unwrap(),
            "broad ancestors must make the complete suffix unhealthy"
        );
        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "state-dir",
            &full,
            /* owned suffix: base + projects + slug + state */ 4,
            DoctorMode::Repair,
            "project state directory",
        );

        for dir in [&root, &projects, &slug, &full] {
            assert_eq!(
                fs::metadata(dir).unwrap().permissions().mode() & 0o777,
                0o700,
                "existing owned suffix dir {dir:?} must be tightened"
            );
        }
        assert_eq!(
            fs::metadata(&non_zo_parent).unwrap().permissions().mode() & 0o777,
            0o755,
            "non-Zo parent must be untouched"
        );
        assert_eq!(findings[0].severity, Severity::Fixed, "{findings:?}");
    }

    #[cfg(unix)]
    #[test]
    fn repair_tightens_broad_ancestors_under_existing_private_leaf() {
        // Regression: the old leaf-only PASS returned early because `state/`
        // was already 0o700, leaving its Zo-owned ancestors at 0o777.
        assert_existing_state_suffix_repaired("repair-private-leaf-broad-ancestors", 0o700);
    }

    #[cfg(unix)]
    #[test]
    fn repair_tightens_broad_ancestors_and_existing_broad_leaf() {
        // Regression: the old repair chmod'd only `state/`, then reported FIXED
        // while the rest of the Zo-owned suffix remained 0o777.
        assert_existing_state_suffix_repaired("repair-broad-leaf-broad-ancestors", 0o777);
    }

    #[cfg(unix)]
    #[test]
    fn repair_tightens_overly_broad_permissions() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = TempDir::new("repair-chmod");
        let dir = base.path().join(".zo");
        fs::create_dir(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o777)).unwrap();

        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "config-home",
            &dir,
            /* owned suffix */ 1,
            DoctorMode::Repair,
            "global config home",
        );
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(findings[0].severity, Severity::Fixed);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_dir_target_is_never_followed_or_modified() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = TempDir::new("symlink");
        // A real directory the symlink points at, with deliberately broad perms.
        let real = base.path().join("real-target");
        fs::create_dir(&real).unwrap();
        fs::set_permissions(&real, fs::Permissions::from_mode(0o777)).unwrap();
        // The doctor candidate is a symlink to that directory.
        let link = base.path().join(".zo");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "config-home",
            &link,
            /* owned suffix */ 1,
            DoctorMode::Repair,
            "global config home",
        );
        // The symlink must be reported and refused, and the target's broad
        // permissions must be left exactly as they were (never followed).
        assert_eq!(findings[0].severity, Severity::Fail);
        assert_eq!(
            fs::metadata(&real).unwrap().permissions().mode() & 0o777,
            0o777,
            "symlink target permissions must be untouched"
        );
    }

    #[cfg(unix)]
    #[test]
    fn intermediate_symlink_ancestor_is_never_traversed_or_repaired() {
        // Direct-unit regression for the NO-FOLLOW security defect: an
        // intermediate ancestor is a symlink whose target already contains the
        // next component (`real/projects`). The unfixed deepest-existing-ancestor
        // logic would pick `link/projects` as a trusted root and create/chmod
        // through the symlink; the absolute no-follow walker must refuse.
        use std::os::unix::fs::PermissionsExt as _;
        let base = TempDir::new("intermediate-symlink");
        let real = base.path().join("real");
        fs::create_dir(&real).unwrap();
        fs::create_dir(real.join("projects")).unwrap();
        fs::set_permissions(real.join("projects"), fs::Permissions::from_mode(0o777)).unwrap();
        let link = base.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        // Target descends through the symlinked ancestor.
        let full = link.join("projects").join("slug").join("state");
        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "state-dir",
            &full,
            /* owned suffix */ 3,
            DoctorMode::Repair,
            "project state directory",
        );
        assert_eq!(findings[0].severity, Severity::Fail, "{findings:?}");
        // Nothing created through the symlink and the target left untouched.
        assert_eq!(
            fs::read_dir(real.join("projects")).unwrap().count(),
            0,
            "no state created through the symlinked ancestor"
        );
        assert_eq!(
            fs::metadata(real.join("projects")).unwrap().permissions().mode() & 0o777,
            0o777,
            "the symlink target's permissions must be untouched"
        );
    }

    #[cfg(unix)]
    #[test]
    fn owner_restricted_directory_is_not_reported_healthy() {
        use std::os::unix::fs::PermissionsExt as _;
        // An owner-owned `0o500` directory has no group/other bits but the owner
        // cannot write it: it must not be reported PASS.
        let base = TempDir::new("restricted");
        let dir = base.path().join(".zo");
        fs::create_dir(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o500)).unwrap();

        let mut findings = Vec::new();
        check_owned_dir_absolute(
            &mut findings,
            "config-home",
            &dir,
            /* owned suffix */ 1,
            DoctorMode::Check,
            "global config home",
        );
        assert_ne!(findings[0].severity, Severity::Pass, "{findings:?}");
        // Restore writable perms so the temp dir can be cleaned up.
        let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_config_file_is_failed_and_not_followed() {
        // A `settings.json` that is a symlink to another file must be reported
        // unsafe (FAIL) and never followed for loading.
        let base = TempDir::new("cfg-symlink");
        let home = base.path().join(".zo");
        fs::create_dir(&home).unwrap();
        let target = base.path().join("real-settings.json");
        fs::write(&target, "{}").unwrap();
        std::os::unix::fs::symlink(&target, home.join("settings.json")).unwrap();

        let mut findings = Vec::new();
        let cfg = check_config(&mut findings, base.path(), Some(&home), DoctorMode::Check);
        assert!(cfg.is_none(), "unsafe config must not be loaded");
        assert_eq!(findings[0].severity, Severity::Fail);
    }

    #[cfg(unix)]
    #[test]
    fn broad_regular_settings_file_is_tightened() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = TempDir::new("cfg-file-chmod");
        let home = base.path().join(".zo");
        fs::create_dir(&home).unwrap();
        let settings = home.join("settings.json");
        fs::write(&settings, "{}").unwrap();
        fs::set_permissions(&settings, fs::Permissions::from_mode(0o644)).unwrap();
        let cwd = base.path().join("ws");
        fs::create_dir(&cwd).unwrap();

        let mut findings = Vec::new();
        check_config_files(&mut findings, DoctorMode::Repair, &cwd, Some(&home));
        assert_eq!(
            fs::metadata(&settings).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(
            findings.iter().any(|f| f.severity == Severity::Fixed),
            "{findings:?}"
        );
    }

    #[test]
    fn stdio_command_available_resolves_absolute_and_missing() {
        // A guaranteed-present executable resolves; a nonexistent one does not.
        assert!(stdio_command_available("/bin/sh") || stdio_command_available("sh"));
        assert!(!stdio_command_available(
            "zo-doctor-definitely-not-a-real-command-xyzzy"
        ));
        assert!(!stdio_command_available(""));
    }

    #[test]
    fn git_check_reports_worktree_state_without_crashing() {
        let non_repo = TempDir::new("no-git");
        let mut findings = Vec::new();
        check_git(&mut findings, non_repo.path());
        assert_eq!(findings[0].check, "git");
        assert!(matches!(
            findings[0].severity,
            Severity::Warn | Severity::Pass
        ));
    }
}
