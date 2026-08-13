use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

struct TempAssetDir(PathBuf);

impl Drop for TempAssetDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn is_node(candidate: &Path) -> bool {
    Command::new(candidate)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn find_node() -> Option<PathBuf> {
    if let Some(candidate) = std::env::var_os("NODE").map(PathBuf::from) {
        if is_node(&candidate) {
            return Some(candidate);
        }
    }

    let output = Command::new("which").arg("node").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let candidate = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    (!candidate.as_os_str().is_empty() && is_node(&candidate)).then_some(candidate)
}

fn assert_success(label: &str, output: &Output) {
    assert!(
        output.status.success(),
        "{label} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn install_icon_manifest_and_shell_references_are_complete() {
    let remote_web = Path::new(env!("CARGO_MANIFEST_DIR")).join("remote-web");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(remote_web.join("manifest.webmanifest"))
            .expect("read web app manifest"),
    )
    .expect("parse web app manifest");
    let icons = manifest["icons"].as_array().expect("manifest icons array");
    for source in [
        "icons/app-icon-1024.png",
        "icons/maskable-icon-1024.png",
    ] {
        let master = image::open(remote_web.join(source)).expect("open 1024px icon master");
        assert_eq!(
            (master.width(), master.height()),
            (1024, 1024),
            "wrong dimensions for {source}"
        );
    }
    for (src, sizes, purpose) in [
        ("icons/icon-192.png", "192x192", "any"),
        ("icons/icon-512.png", "512x512", "any"),
        ("icons/maskable-icon-192.png", "192x192", "maskable"),
        ("icons/maskable-icon-512.png", "512x512", "maskable"),
    ] {
        assert!(remote_web.join(src).is_file(), "missing install icon {src}");
        assert!(
            icons.iter().any(|icon| {
                icon["src"] == src
                    && icon["sizes"] == sizes
                    && icon["type"] == "image/png"
                    && icon["purpose"] == purpose
            }),
            "manifest is missing {src} ({sizes}, {purpose})"
        );
    }

    let index = std::fs::read_to_string(remote_web.join("index.html")).expect("read index");
    for href in [
        "./favicon.ico",
        "./icons/favicon-16.png",
        "./icons/favicon-32.png",
        "./apple-touch-icon.png",
    ] {
        assert!(index.contains(href), "index is missing {href}");
    }

    let service_worker =
        std::fs::read_to_string(remote_web.join("sw.js")).expect("read service worker");
    for asset in [
        "./icons/icon-192.png",
        "./icons/icon-512.png",
        "./icons/maskable-icon-192.png",
        "./icons/maskable-icon-512.png",
        "./icons/badge-96.png",
        "./apple-touch-icon.png",
    ] {
        assert!(
            service_worker.contains(asset),
            "service worker is missing {asset}"
        );
    }
}

#[test]
fn remote_web_javascript_parses_and_unit_tests_pass() {
    let Some(node) = find_node() else {
        eprintln!("skipping remote-web JavaScript checks: node was not found via NODE or PATH");
        return;
    };

    let remote_web = Path::new(env!("CARGO_MANIFEST_DIR")).join("remote-web");
    let temp_path = std::env::temp_dir().join(format!(
        "zo-remote-web-assets-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_path);
    std::fs::create_dir_all(&temp_path).expect("create remote-web asset temp directory");
    let _temp_dir = TempAssetDir(temp_path.clone());

    for (source, copy) in [
        ("app.js", "app.mjs"),
        ("remote-state.js", "remote-state.mjs"),
        ("sw.js", "sw.js"),
    ] {
        let copy = temp_path.join(copy);
        std::fs::copy(remote_web.join(source), &copy)
            .unwrap_or_else(|error| panic!("copy {source} for syntax check: {error}"));
        let output = Command::new(&node)
            .arg("--check")
            .arg(&copy)
            .output()
            .unwrap_or_else(|error| panic!("run node --check for {source}: {error}"));
        assert_success(&format!("node --check {source}"), &output);
    }

    let output = Command::new(&node)
        .args(["--test", "remote-state.test.mjs"])
        .current_dir(&remote_web)
        .output()
        .expect("run remote-state JavaScript tests");
    assert_success("node --test remote-state.test.mjs", &output);
}
