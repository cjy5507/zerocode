//! Integration tests for `tui::history` (Phase 3, Lane L6).

use std::fs;

use zo_cli::tui::history::{migrate_from_rustyline, History, HistoryRecord};

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!("zo-l6-history-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    base
}

#[test]
fn history_empty_load_when_file_missing() {
    let dir = tmp_dir("empty");
    let path = dir.join("history.jsonl");
    let hist = History::load(&path).expect("load ok");
    assert!(hist.is_empty());
    assert_eq!(hist.len(), 0);
    assert_eq!(hist.path(), path.as_path());
}

#[test]
fn history_jsonl_roundtrip() {
    let dir = tmp_dir("roundtrip");
    let path = dir.join("history.jsonl");
    let mut hist = History::load(&path).unwrap();
    hist.append(HistoryRecord::now("first".into(), "normal"))
        .unwrap();
    hist.append(HistoryRecord::now("second".into(), "normal"))
        .unwrap();
    drop(hist);

    let reloaded = History::load(&path).unwrap();
    assert_eq!(reloaded.len(), 2);
    let texts: Vec<String> = reloaded.entries().iter().map(|r| r.text.clone()).collect();
    assert_eq!(texts, vec!["first", "second"]);
}

#[test]
fn history_migrates_rustyline_fixture_and_deletes_legacy() {
    let dir = tmp_dir("migrate");
    let legacy = dir.join("history");
    // Vendored fixture is checked into tests/fixtures.
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("rustyline_history.txt");
    fs::copy(&fixture, &legacy).unwrap();

    let path = dir.join("history.jsonl");
    let hist = History::load(&path).unwrap();

    // Fixture has 5 non-comment entries.
    assert_eq!(hist.len(), 5);
    let texts: Vec<String> = hist.entries().iter().map(|r| r.text.clone()).collect();
    assert!(texts.iter().any(|t| t == "hello world"));
    assert!(texts
        .iter()
        .any(|t| t == "write a haiku about cats\nand dogs"));
    assert!(texts.iter().any(|t| t == r"c:\path\to\file"));

    // Legacy file is gone, jsonl file exists.
    assert!(!legacy.exists(), "legacy rustyline file should be deleted");
    assert!(path.exists(), "jsonl file should have been written");
}

#[test]
fn history_append_persists_to_disk() {
    let dir = tmp_dir("append");
    let path = dir.join("history.jsonl");
    {
        let mut hist = History::load(&path).unwrap();
        hist.append(HistoryRecord::now("alpha".into(), "normal"))
            .unwrap();
    }
    let raw = fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains("\"text\":\"alpha\""),
        "jsonl line missing: {raw}"
    );
    assert!(raw.ends_with('\n'));
}

#[test]
fn history_max_cap_evicts_oldest() {
    let dir = tmp_dir("cap");
    let path = dir.join("history.jsonl");
    let mut hist = History::load_with_max(&path, 3).unwrap();
    for tag in ["a", "b", "c", "d", "e"] {
        hist.append(HistoryRecord::now(tag.into(), "normal"))
            .unwrap();
    }
    assert_eq!(hist.len(), 3);
    let texts: Vec<String> = hist.entries().iter().map(|r| r.text.clone()).collect();
    assert_eq!(texts, vec!["c", "d", "e"]);

    // Reload from disk and confirm the cap was persisted.
    let reloaded = History::load_with_max(&path, 3).unwrap();
    let texts: Vec<String> = reloaded.entries().iter().map(|r| r.text.clone()).collect();
    assert_eq!(texts, vec!["c", "d", "e"]);
}

#[test]
fn history_search_is_reverse_substring() {
    let dir = tmp_dir("search");
    let path = dir.join("history.jsonl");
    let mut hist = History::load(&path).unwrap();
    for tag in ["deploy app", "explain repo", "deploy docs", "fix bug"] {
        hist.append(HistoryRecord::now(tag.into(), "normal"))
            .unwrap();
    }
    let hits: Vec<String> = hist
        .search("deploy")
        .into_iter()
        .map(|r| r.text.clone())
        .collect();
    assert_eq!(
        hits,
        vec!["deploy docs".to_string(), "deploy app".to_string()]
    );
}

#[test]
fn history_migrate_fn_parses_fixture_directly() {
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("rustyline_history.txt");
    let entries = migrate_from_rustyline(&fixture).unwrap();
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[0], "hello world");
    assert_eq!(entries[2], "write a haiku about cats\nand dogs");
}
