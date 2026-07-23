use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest as _, Sha256};

use crate::{BUILD_TARGET, GIT_SHA, VERSION};

pub(crate) const OFFICIAL_REPO: &str = "cjy5507/zerocode";
const RELEASE_DOWNLOAD_BASE: &str = "https://github.com/cjy5507/zerocode/releases/download";
const LATEST_MANIFEST_URL: &str =
    "https://github.com/cjy5507/zerocode/releases/latest/download/manifest.txt";
const RELEASE_CHANNEL: Option<&str> = option_env!("ZO_RELEASE_CHANNEL");
const MANAGED_INSTALL_MARKER: &str = "managed-install";
const LAST_CHECK_FILE: &str = "update-last-check";
const SCHEDULE_LOCK_FILE: &str = "update-schedule.lock";
const INSTALL_LOCK_FILE: &str = ".zo-update.lock";
const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);
const VERIFY_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_ASSET_SIZE: u64 = 100 * 1024 * 1024;
const MAX_MANIFEST_SIZE: usize = 64 * 1024;
const SUPPORTED_TARGETS: [&str; 3] = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
];

type UpdateResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    fn parse(value: &str) -> Result<Self, String> {
        let mut parts = value.split('.');
        let major = parse_version_component(parts.next(), value)?;
        let minor = parse_version_component(parts.next(), value)?;
        let patch = parse_version_component(parts.next(), value)?;
        if parts.next().is_some() {
            return Err(format!("invalid semantic version: {value}"));
        }
        Ok(Self { major, minor, patch })
    }
}

fn parse_version_component(component: Option<&str>, version: &str) -> Result<u64, String> {
    let component = component.ok_or_else(|| format!("invalid semantic version: {version}"))?;
    if component.is_empty()
        || !component.bytes().all(|byte| byte.is_ascii_digit())
        || (component.len() > 1 && component.starts_with('0'))
    {
        return Err(format!("invalid semantic version: {version}"));
    }
    component
        .parse::<u64>()
        .map_err(|_| format!("invalid semantic version: {version}"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Asset {
    target: String,
    name: String,
    sha256: String,
    size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReleaseManifest {
    version_text: String,
    version: Version,
    base: String,
    assets: Vec<Asset>,
}

impl ReleaseManifest {
    fn parse(text: &str, test_base: Option<&str>) -> Result<Self, String> {
        if !text.ends_with('\n') {
            return Err("manifest must end with a newline".to_string());
        }
        let lines = text.lines().collect::<Vec<_>>();
        if lines.len() != 6 || lines[0] != "schema=1" {
            return Err("manifest must contain schema, version, base, and three asset rows".to_string());
        }
        let version_text = required_value(lines[1], "version")?.to_string();
        let version = Version::parse(&version_text)?;
        let base = required_value(lines[2], "base")?.to_string();
        validate_download_base(&base, &version_text, test_base)?;

        let mut assets = Vec::with_capacity(SUPPORTED_TARGETS.len());
        for (line, expected_target) in lines[3..].iter().zip(SUPPORTED_TARGETS) {
            let row = required_value(line, "asset")?;
            let fields = row.split('|').collect::<Vec<_>>();
            if fields.len() != 4 {
                return Err(format!("invalid asset row: {line}"));
            }
            let target = fields[0];
            if target != expected_target {
                return Err(format!("expected asset target {expected_target}, found {target}"));
            }
            let expected_name = asset_name(&version_text, target);
            if fields[1] != expected_name {
                return Err(format!("expected asset name {expected_name}, found {}", fields[1]));
            }
            if fields[2].len() != 64
                || !fields[2]
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(format!("invalid SHA-256 for {target}"));
            }
            let size = fields[3]
                .parse::<u64>()
                .map_err(|_| format!("invalid asset size for {target}"))?;
            if size == 0 || size > MAX_ASSET_SIZE {
                return Err(format!("asset size out of range for {target}"));
            }
            assets.push(Asset {
                target: target.to_string(),
                name: fields[1].to_string(),
                sha256: fields[2].to_string(),
                size,
            });
        }
        Ok(Self {
            version_text,
            version,
            base,
            assets,
        })
    }

    fn asset_for(&self, target: &str) -> Result<&Asset, String> {
        self.assets
            .iter()
            .find(|asset| asset.target == target)
            .ok_or_else(|| format!("no release asset for target {target}"))
    }
}

fn required_value<'a>(line: &'a str, key: &str) -> Result<&'a str, String> {
    let prefix = format!("{key}=");
    let value = line
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("expected {key}= line"))?;
    if value.is_empty() {
        return Err(format!("{key} cannot be empty"));
    }
    Ok(value)
}

fn asset_name(version: &str, target: &str) -> String {
    format!("zo-v{version}-{target}")
}

fn validate_download_base(base: &str, version: &str, test_base: Option<&str>) -> Result<(), String> {
    let official = format!("{RELEASE_DOWNLOAD_BASE}/v{version}");
    if base == official || test_base.is_some_and(|allowed| base == allowed) {
        return Ok(());
    }
    Err(format!("invalid release download base: {base}"))
}

struct DownloadSource {
    manifest_url: String,
    test_base: Option<String>,
}

impl DownloadSource {
    fn from_env() -> UpdateResult<Self> {
        let Some(base) = std::env::var_os("ZO_SELF_UPDATE_TEST_BASE") else {
            return Ok(Self {
                manifest_url: LATEST_MANIFEST_URL.to_string(),
                test_base: None,
            });
        };
        if std::env::var("ZO_SELF_UPDATE_TEST_ONLY").as_deref() != Ok("1") {
            return Err(update_error(
                "ZO_SELF_UPDATE_TEST_BASE requires ZO_SELF_UPDATE_TEST_ONLY=1",
            ));
        }
        let base = base
            .into_string()
            .map_err(|_| update_error("test update base must be valid UTF-8"))?;
        let base = base.trim_end_matches('/').to_string();
        let url = reqwest::Url::parse(&base)?;
        if url.scheme() != "http" || !url.host_str().is_some_and(is_loopback_host) {
            return Err(update_error(
                "test update base must use HTTP on localhost or a loopback address",
            ));
        }
        Ok(Self {
            manifest_url: format!("{base}/manifest.txt"),
            test_base: Some(base),
        })
    }
}

fn is_loopback_host(host: &str) -> bool {
    host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn update_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

#[derive(Debug)]
struct ManagedInstall {
    destination: PathBuf,
}

fn release_build_is_eligible(channel: Option<&str>, git_sha: Option<&str>) -> bool {
    channel == Some("stable")
        && git_sha.is_some_and(|sha| sha != "unknown" && !sha.ends_with("-dirty"))
}

fn paths_match(current: &Path, recorded: &Path) -> bool {
    fs::canonicalize(current)
        .ok()
        .zip(fs::canonicalize(recorded).ok())
        .is_some_and(|(left, right)| left == right)
}

fn managed_install_allowed(
    channel: Option<&str>,
    git_sha: Option<&str>,
    current: &Path,
    recorded: &Path,
) -> bool {
    release_build_is_eligible(channel, git_sha) && paths_match(current, recorded)
}

fn managed_install() -> UpdateResult<ManagedInstall> {
    if !release_build_is_eligible(RELEASE_CHANNEL, GIT_SHA) {
        return Err(update_error(
            "self-update is available only to official stable release builds",
        ));
    }
    let current = std::env::current_exe()?.canonicalize()?;
    let marker_path = core_types::paths::default_config_home().join(MANAGED_INSTALL_MARKER);
    let recorded = parse_managed_marker(&fs::read_to_string(&marker_path)?)?;
    if !managed_install_allowed(RELEASE_CHANNEL, GIT_SHA, &current, &recorded) {
        return Err(update_error(
            "this binary is not running from its installer-managed path",
        ));
    }
    refuse_symlink(&recorded)?;
    Ok(ManagedInstall {
        destination: recorded,
    })
}

fn parse_managed_marker(text: &str) -> UpdateResult<PathBuf> {
    let mut lines = text.lines();
    if lines.next() != Some("schema=1") {
        return Err(update_error("invalid managed-install marker schema"));
    }
    let path = lines
        .next()
        .and_then(|line| line.strip_prefix("path="))
        .filter(|path| !path.is_empty())
        .ok_or_else(|| update_error("invalid managed-install marker path"))?;
    if lines.next().is_some() || !text.ends_with('\n') {
        return Err(update_error("invalid managed-install marker"));
    }
    Ok(PathBuf::from(path))
}

fn refuse_symlink(path: &Path) -> UpdateResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(update_error(format!("refusing symlink destination: {}", path.display())))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn run_cli(check: bool) -> UpdateResult<()> {
    println!("{}", crate::render_version_report());
    let target = current_target()?;
    let source = DownloadSource::from_env()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let manifest = runtime.block_on(fetch_manifest(&source))?;
    let current = Version::parse(VERSION).map_err(update_error)?;
    if manifest.version <= current {
        println!("Update           up to date");
        return Ok(());
    }
    if check {
        println!("Update           v{} available", manifest.version_text);
        return Ok(());
    }
    let managed = managed_install()?;
    let asset = manifest.asset_for(target)?.clone();
    runtime.block_on(download_and_install(&manifest, &asset, &managed.destination))?;
    println!("Updated          v{}", manifest.version_text);
    Ok(())
}

pub(crate) fn run_background() -> UpdateResult<()> {
    let managed = managed_install()?;
    let target = current_target()?;
    let source = DownloadSource::from_env()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let manifest = runtime.block_on(fetch_manifest(&source))?;
    let current = Version::parse(VERSION).map_err(update_error)?;
    if manifest.version > current {
        let asset = manifest.asset_for(target)?.clone();
        runtime.block_on(download_and_install(&manifest, &asset, &managed.destination))?;
    }
    Ok(())
}

fn current_target() -> UpdateResult<&'static str> {
    let target = BUILD_TARGET.ok_or_else(|| update_error("build target is unavailable"))?;
    if SUPPORTED_TARGETS.contains(&target) {
        Ok(target)
    } else {
        Err(update_error(format!("unsupported update target: {target}")))
    }
}

async fn fetch_manifest(source: &DownloadSource) -> UpdateResult<ReleaseManifest> {
    let client = reqwest::Client::builder()
        .user_agent(format!("zo/{VERSION} ({OFFICIAL_REPO})"))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?;
    let mut response = client
        .get(&source.manifest_url)
        .send()
        .await?
        .error_for_status()?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len().saturating_add(chunk.len()) > MAX_MANIFEST_SIZE {
            return Err(update_error("release manifest is too large"));
        }
        bytes.extend_from_slice(&chunk);
    }
    let text = String::from_utf8(bytes)?;
    ReleaseManifest::parse(&text, source.test_base.as_deref()).map_err(update_error)
}

async fn download_and_install(
    manifest: &ReleaseManifest,
    asset: &Asset,
    destination: &Path,
) -> UpdateResult<()> {
    refuse_symlink(destination)?;
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| update_error("managed install destination has no parent directory"))?;
    let _lock = FileLock::acquire(&parent.join(INSTALL_LOCK_FILE))?;
    refuse_symlink(destination)?;
    let (temporary, mut file) = create_install_temp(parent)?;
    set_executable(&file)?;

    let client = reqwest::Client::builder()
        .user_agent(format!("zo/{VERSION} ({OFFICIAL_REPO})"))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?;
    let url = format!("{}/{}", manifest.base, asset.name);
    let mut response = client.get(url).send().await?.error_for_status()?;
    if response.content_length().is_some_and(|length| length != asset.size) {
        return Err(update_error("release asset content length does not match manifest"));
    }
    let mut hasher = Sha256::new();
    let mut size = 0_u64;
    while let Some(chunk) = response.chunk().await? {
        size = size
            .checked_add(u64::try_from(chunk.len())?)
            .ok_or_else(|| update_error("release asset size overflow"))?;
        if size > asset.size || size > MAX_ASSET_SIZE {
            return Err(update_error("release asset exceeds manifest size"));
        }
        hasher.update(&chunk);
        file.write_all(&chunk)?;
    }
    if size != asset.size {
        return Err(update_error("release asset size does not match manifest"));
    }
    let digest = format!("{:x}", hasher.finalize());
    if digest != asset.sha256 {
        return Err(update_error("release asset SHA-256 mismatch"));
    }
    file.flush()?;
    file.sync_all()?;
    drop(file);
    verify_downloaded_binary(temporary.path(), &manifest.version_text)?;
    refuse_symlink(destination)?;
    fs::rename(temporary.path(), destination)?;
    temporary.keep();
    let _ = File::open(parent).and_then(|directory| directory.sync_all());
    Ok(())
}

fn create_install_temp(parent: &Path) -> UpdateResult<(TemporaryPath, File)> {
    for attempt in 0..128_u32 {
        let path = parent.join(format!(".zo-update-{}-{attempt}", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((TemporaryPath::new(path), file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
            Err(error) => return Err(error.into()),
        }
    }
    Err(update_error("could not allocate update temporary file"))
}

#[cfg(unix)]
fn set_executable(file: &File) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    file.set_permissions(fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn set_executable(_file: &File) -> std::io::Result<()> {
    Ok(())
}

fn verify_downloaded_binary(path: &Path, version: &str) -> UpdateResult<()> {
    let mut child = Command::new(path)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let deadline = std::time::Instant::now() + VERIFY_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(update_error("downloaded binary version check timed out"));
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    if !status.success() {
        return Err(update_error("downloaded binary failed its version check"));
    }
    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_string(&mut stdout)?;
    }
    let expected = format!("Version          {version}");
    if !stdout.lines().any(|line| line.trim() == expected) {
        return Err(update_error(format!(
            "downloaded binary did not report manifest version {version}"
        )));
    }
    Ok(())
}

struct TemporaryPath {
    path: PathBuf,
    keep: std::cell::Cell<bool>,
}

impl TemporaryPath {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            keep: std::cell::Cell::new(false),
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn keep(&self) {
        self.keep.set(true);
    }
}

impl Drop for TemporaryPath {
    fn drop(&mut self) {
        if !self.keep.get() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

struct FileLock {
    path: PathBuf,
    _file: File,
}

impl FileLock {
    fn acquire(path: &Path) -> UpdateResult<Self> {
        refuse_symlink(path)?;
        let file = open_lock_file(path)?;
        file.try_lock()?;
        Ok(Self {
            path: path.to_path_buf(),
            _file: file,
        })
    }
}

#[cfg(unix)]
fn open_lock_file(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt as _;

    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(not(unix))]
fn open_lock_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).write(true).create(true).open(path)
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(crate) fn schedule_startup_check() {
    if std::env::var_os("ZO_DISABLE_AUTO_UPDATE").is_some() {
        return;
    }
    let config_home = core_types::paths::default_config_home();
    let Ok(_schedule_lock) = FileLock::acquire(&config_home.join(SCHEDULE_LOCK_FILE)) else {
        return;
    };
    let now = unix_seconds();
    let last_check_path = config_home.join(LAST_CHECK_FILE);
    if fs::read_to_string(&last_check_path)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .is_some_and(|last| now.saturating_sub(last) < CHECK_INTERVAL.as_secs())
    {
        return;
    }
    if managed_install().is_err() {
        return;
    }
    let Ok(current_exe) = std::env::current_exe() else {
        return;
    };
    if Command::new(current_exe)
        .arg("__self-update-background")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
    {
        let _ = fs::write(last_check_path, format!("{now}\n"));
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn manifest(version: &str, base: &str) -> String {
        let rows = SUPPORTED_TARGETS.map(|target| {
            format!("asset={target}|{}|{HASH}|123", asset_name(version, target))
        });
        format!(
            "schema=1\nversion={version}\nbase={base}\n{}\n{}\n{}\n",
            rows[0], rows[1], rows[2]
        )
    }

    #[test]
    fn manifest_accepts_the_strict_contract() {
        let base = "https://github.com/cjy5507/zerocode/releases/download/v1.2.3";
        let parsed = ReleaseManifest::parse(&manifest("1.2.3", base), None)
            .expect("valid manifest should parse");
        assert_eq!(parsed.version, Version { major: 1, minor: 2, patch: 3 });
    }

    #[test]
    fn manifest_rejects_invalid_versions_names_hashes_sizes_and_bases() {
        let base = "https://github.com/cjy5507/zerocode/releases/download/v1.2.3";
        for invalid in [
            manifest("01.2.3", base),
            manifest("1.2.3", base).replace("zo-v1.2.3-aarch64", "wrong-aarch64"),
            manifest("1.2.3", base).replace(HASH, &HASH.to_uppercase()),
            manifest("1.2.3", base).replace("|123\n", "|104857601\n"),
            manifest("1.2.3", "http://github.com/releases"),
        ] {
            assert!(ReleaseManifest::parse(&invalid, None).is_err(), "accepted: {invalid}");
        }
    }

    #[test]
    fn semantic_versions_compare_numerically() {
        assert!(Version::parse("1.10.0").expect("valid") > Version::parse("1.9.9").expect("valid"));
    }

    #[test]
    fn target_selection_contains_only_release_targets() {
        assert_eq!(SUPPORTED_TARGETS.len(), 3);
        assert!(SUPPORTED_TARGETS.contains(&"aarch64-apple-darwin"));
        assert!(SUPPORTED_TARGETS.contains(&"x86_64-apple-darwin"));
        assert!(SUPPORTED_TARGETS.contains(&"x86_64-unknown-linux-gnu"));
    }

    #[test]
    fn managed_guard_requires_stable_clean_build_and_matching_canonical_path() {
        let root = test_directory("managed-guard");
        let binary = root.join("zo");
        fs::write(&binary, b"binary").expect("write binary");
        assert!(managed_install_allowed(Some("stable"), Some("abc123"), &binary, &binary));
        assert!(!managed_install_allowed(None, Some("abc123"), &binary, &binary));
        assert!(!managed_install_allowed(Some("stable"), Some("abc123-dirty"), &binary, &binary));
        assert!(!managed_install_allowed(Some("stable"), Some("abc123"), &binary, &root.join("copy")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn abandoned_lock_file_is_reused_but_active_lock_is_rejected() {
        let root = test_directory("file-lock");
        let lock_path = root.join(INSTALL_LOCK_FILE);
        fs::write(&lock_path, b"abandoned").expect("write abandoned lock");
        let lock = FileLock::acquire(&lock_path).expect("reuse abandoned lock");
        assert!(FileLock::acquire(&lock_path).is_err());
        drop(lock);
        assert!(!lock_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn failed_binary_verification_leaves_destination_and_cleans_temp_and_lock() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = test_directory("atomic-failure");
        let destination = root.join("zo");
        fs::write(&destination, b"original").expect("write destination");
        let lock_path = root.join(INSTALL_LOCK_FILE);
        {
            let _lock = FileLock::acquire(&lock_path).expect("acquire lock");
            let (temporary, mut file) = create_install_temp(&root).expect("create temp");
            file.write_all(b"#!/bin/sh\necho 'zo 9.9.9'\n")
                .expect("write script");
            file.set_permissions(fs::Permissions::from_mode(0o755))
                .expect("make executable");
            file.sync_all().expect("sync script");
            drop(file);
            assert!(verify_downloaded_binary(temporary.path(), "1.2.3").is_err());
        }
        assert_eq!(fs::read(&destination).expect("read destination"), b"original");
        assert!(!lock_path.exists());
        assert!(fs::read_dir(&root)
            .expect("read root")
            .all(|entry| !entry.expect("entry").file_name().to_string_lossy().starts_with(".zo-update-")));
        let _ = fs::remove_dir_all(root);
    }

    fn test_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zo-self-update-{label}-{}-{}",
            std::process::id(),
            unix_seconds()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create test directory");
        path
    }
}
