//! Single source of truth for "does this agent manifest belong to the current
//! session?" — the filter the HUD sidebar and the agents-detail pager share.
//!
//! Workspace-global agent manifests live in one `.zo/agents/` store, so a
//! machine running several zo sessions (or a benchmark harness that spawns
//! headless `zo` agents) accumulates manifests from *every* session there.
//! Without a session filter each session's HUD shows **all** of them — the
//! "every session shares the same agent activity" bug. Spawned manifests are
//! stamped with their owning session via `parentSessionId`
//! (`ToolContext::session_id` -> `AgentInput::parent_session_id` ->
//! `AgentOutput`), so the display layer can scope the list to the live session.
//!
//! This module owns only the `serde_json::Value` JSON shape: it pulls a
//! manifest's `parentSessionId` (or the legacy `snake_case` key) out of the
//! untyped value and delegates the actual match rule to the shared core
//! [`tools::parent_session_belongs`]. `tui_loop` (live HUD list) and
//! `workflow_progress` (Ctrl+O fan-out fallback + the Ctrl+G agents-viewer
//! reader) call here. The `tools` crate's `AgentOutput`-typed twin
//! (`agent_tools::agent_output_belongs_to_session`) delegates to the *same*
//! core, so the rule is defined once and the two surfaces can never drift.

use serde_json::Value;

/// Whether `manifest` belongs to the session identified by `session_id`.
///
/// - `session_id == None` (an unfiltered caller): always `true` — the caller
///   opted out of session scoping (e.g. a global/time-only view).
/// - `session_id == Some(id)` and the manifest carries a non-empty
///   `parentSessionId`: matches only when the two ids are equal — a manifest
///   stamped for a *different* session is hidden.
/// - `session_id == Some(id)` and the manifest is **unstamped** (no/empty
///   `parentSessionId`): governed by `allow_unstamped`. The live HUD passes
///   `false` so foreign/legacy/benchmark agents (which never carried a session
///   id) stop bleeding into this session's view; a time-scoped caller that
///   wants post-upgrade back-compat passes `true` to keep unstamped manifests
///   visible under its own freshness window.
///
/// Allocation-free: borrows the manifest's id as `&str` and compares in place
/// (this runs once per manifest per HUD frame).
#[must_use]
pub fn manifest_belongs_to_session(
    manifest: &Value,
    session_id: Option<&str>,
    allow_unstamped: bool,
) -> bool {
    // This module owns only the `serde_json::Value` JSON shape: pull the
    // manifest's stamp (both the camelCase and the legacy snake_case key) and
    // delegate the actual match rule to the shared core in `tools`, so the CLI
    // surfaces and the `tools` stop paths can never drift.
    let parent_session_id = manifest
        .get("parentSessionId")
        .or_else(|| manifest.get("parent_session_id"))
        .and_then(Value::as_str);
    tools::parent_session_belongs(parent_session_id, session_id, allow_unstamped)
}

#[cfg(test)]
mod tests {
    use super::manifest_belongs_to_session;
    use serde_json::json;

    #[test]
    fn matching_parent_session_is_visible() {
        let manifest = json!({ "parentSessionId": "sess-1" });
        assert!(manifest_belongs_to_session(
            &manifest,
            Some("sess-1"),
            false
        ));
        // The stricter `allow_unstamped` flag does not affect a stamped match.
        assert!(manifest_belongs_to_session(&manifest, Some("sess-1"), true));
    }

    #[test]
    fn different_parent_session_is_hidden() {
        let manifest = json!({ "parentSessionId": "sess-2" });
        assert!(!manifest_belongs_to_session(
            &manifest,
            Some("sess-1"),
            false
        ));
        // A stamped-foreign manifest is hidden even when unstamped ones are kept.
        assert!(!manifest_belongs_to_session(
            &manifest,
            Some("sess-1"),
            true
        ));
    }

    #[test]
    fn unstamped_manifest_is_hidden_when_strict() {
        // The cross-session leak fix: the live HUD (`allow_unstamped = false`)
        // hides manifests that never carried a session id.
        let manifest = json!({ "status": "running" });
        assert!(!manifest_belongs_to_session(
            &manifest,
            Some("sess-1"),
            false
        ));
        // An explicitly null id is treated the same as absent.
        let null_manifest = json!({ "parentSessionId": null });
        assert!(!manifest_belongs_to_session(
            &null_manifest,
            Some("sess-1"),
            false
        ));
        // An empty-string id is also treated as unstamped.
        let blank_manifest = json!({ "parentSessionId": "   " });
        assert!(!manifest_belongs_to_session(
            &blank_manifest,
            Some("sess-1"),
            false
        ));
    }

    #[test]
    fn unstamped_manifest_is_kept_when_back_compat_allowed() {
        let manifest = json!({ "status": "running" });
        assert!(manifest_belongs_to_session(&manifest, Some("sess-1"), true));
    }

    #[test]
    fn no_session_filter_keeps_everything() {
        let manifest = json!({ "parentSessionId": "sess-2" });
        assert!(manifest_belongs_to_session(&manifest, None, false));
        let unstamped = json!({ "status": "running" });
        assert!(manifest_belongs_to_session(&unstamped, None, false));
        // A blank caller id behaves like None.
        assert!(manifest_belongs_to_session(&manifest, Some("  "), false));
    }

    #[test]
    fn snake_case_parent_session_id_is_honored() {
        // Legacy/alternate serialization key.
        let manifest = json!({ "parent_session_id": "sess-1" });
        assert!(manifest_belongs_to_session(
            &manifest,
            Some("sess-1"),
            false
        ));
        let foreign = json!({ "parent_session_id": "sess-2" });
        assert!(!manifest_belongs_to_session(
            &foreign,
            Some("sess-1"),
            false
        ));
    }
}
