//! P2 verdict→worker attribution: fold a validator's judgement of an agent's
//! output back onto that agent's routed model, in the SAME bounded
//! route-outcome history the spawn recorder writes.
//!
//! Run-level outcomes (the spawn recorder) say "the model finished"; verdict
//! outcomes say "the work was right". Both are decisive samples for the
//! feedback scorer, so a model that reliably completes but produces work that
//! fails verification stops winning its role on run-completions alone.
//!
//! Only WELL-BOUND (verdict, worker) pairs are recorded — pairs where the
//! judged output is structurally known to belong to one agent:
//! - the repair loop's focused reverify judging the FIXER's change, and
//! - a validator emitting unusable output (a quality failure of the
//!   VALIDATOR itself).
//!
//! Anything with ambiguous provenance (a new unrelated finding, an initial
//! validator sweep over merged worker output) records nothing.

use std::path::Path;

use runtime::RouteOutcomeRecord;
use serde_json::Value;

/// Weight a verdict signal contributes to the learned-specialty scorer
/// (Phase 6 consumer of the v2 `signalWeight` field). A strict pass/fail
/// judgement — the repair loop's focused reverify, a validator's own
/// usable-output check, the deep-gate VERIFY panel, a planner-bound
/// reviewer→worker pair, or an ad-hoc standalone review — is direct evidence
/// about correctness, so it counts at full weight. A preference judgement
/// (e.g. a council/self-consistency "pick the better of N" — no caller wired
/// yet, reserved for a future source) only says "better than its peers", a
/// strictly weaker claim than "correct in isolation", so it counts at half
/// weight. Constants are simple, documented, fixed points — not fit to any
/// live data — so a future caller can rely on them without re-deriving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerdictKind {
    PassFail,
    #[allow(
        dead_code,
        reason = "reserved for a future preference-based verdict source (Phase 6 \
                  council/self-consistency winner) — no caller constructs it yet, \
                  exercised only by the attribution unit test"
    )]
    Preference,
}

impl VerdictKind {
    const fn weight(self) -> f32 {
        match self {
            Self::PassFail => 1.0,
            Self::Preference => 0.5,
        }
    }
}

/// Best-effort production entry: resolve the agent store and project cwd, then
/// attribute `passed` to the agent's manifest model. Silent on any failure —
/// attribution must never disturb the workflow that produced the verdict.
pub(crate) fn record_verdict_outcome_for_agent(agent_id: &str, passed: bool, kind: VerdictKind) {
    let Ok(store) = crate::misc_tools::agent_store_dir() else {
        return;
    };
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    record_verdict_outcome_at(&cwd, &store, agent_id, passed, kind);
}

/// Store/cwd-injected core, split out so tests can exercise the seam against
/// temp directories without touching process env or real project state.
fn record_verdict_outcome_at(
    cwd: &Path,
    store: &Path,
    agent_id: &str,
    passed: bool,
    kind: VerdictKind,
) {
    let agent_id = agent_id.trim();
    if agent_id.is_empty() {
        return;
    }
    let Ok(raw) = std::fs::read_to_string(store.join(format!("{agent_id}.json"))) else {
        return;
    };
    let Ok(manifest) = serde_json::from_str::<Value>(&raw) else {
        return;
    };
    let Some(model) = manifest_model(&manifest) else {
        // Without a resolved model there is nothing to credit or blame.
        return;
    };
    let target = manifest
        .get("subagentType")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("general-purpose");
    // Terminal-only by construction: a verdict is always a settled pass/fail
    // judgement, never `still_running` — the shared recorder-side doctrine
    // guard in `runtime::record_route_outcome` would skip-write (debug-assert
    // in dev) if this ever produced anything else.
    let status = if passed { "completed" } else { "failed" };
    let record = RouteOutcomeRecord::new(
        "subagent",
        target,
        // P3 canonicalization-at-write, same as the spawn recorder.
        crate::misc_tools::canonicalize_route_model_id(&model),
        status,
    )
    .with_signal("verdict")
    .with_role(manifest_string_field(&manifest, "routeRole"))
    .with_complexity(manifest_string_field(&manifest, "routeComplexity"))
    .with_risk(manifest_string_field(&manifest, "routeRisk"))
    .with_route_source(manifest_string_field(&manifest, "routeSource"))
    .with_requested_model(manifest_string_field(&manifest, "requestedModel"))
    .with_output_tokens(manifest_output_tokens(&manifest))
    .with_signal_weight(Some(kind.weight()));
    let _ = runtime::record_route_outcome(cwd, &record);
}

/// Best-effort total output tokens for the judged agent's whole run, summed
/// from the manifest's persisted `tokenHistory` (per-turn output-token
/// deltas — see `AgentOutput::token_history`). Cheaper and simpler than
/// threading a live token counter through the attribution seam; `0` when the
/// manifest has no history (legacy manifest, or the run recorded none).
fn manifest_output_tokens(manifest: &Value) -> u64 {
    manifest
        .get("tokenHistory")
        .and_then(Value::as_array)
        .map_or(0, |entries| entries.iter().filter_map(Value::as_u64).sum())
}

fn manifest_model(manifest: &Value) -> Option<String> {
    ["resolvedModel", "model"]
        .iter()
        .filter_map(|key| manifest.get(*key))
        .filter_map(Value::as_str)
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(ToString::to_string)
}

/// P3 v2 route-decision metadata, read straight off the same on-disk
/// manifest JSON already used for `manifest_model` — the spawn recorder's
/// `AgentOutput::route_role`/`route_complexity`/`route_risk`/`route_source`
/// fields, under their `routeRole`/`routeComplexity`/`routeRisk`/
/// `routeSource` wire names. `None` on legacy manifests, explicit models, or
/// routing-off spawns — same absence conditions as `routeReason`.
fn manifest_string_field(manifest: &Value, key: &str) -> Option<String> {
    manifest
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::{record_verdict_outcome_at, VerdictKind};
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Same seam as the CLI's `isolate_global_zo_home_for_tests` (test cfg
    /// does not cross crates): route the global Zo home at one per-process
    /// temp dir so `record_verdict_outcome_at(cwd, …)` writes its
    /// route-outcomes under temp instead of the developer's real
    /// `~/.zo/projects/`.
    fn isolate_global_zo_home() {
        use std::sync::OnceLock;
        static HOME: OnceLock<PathBuf> = OnceLock::new();
        let home = HOME.get_or_init(|| {
            let dir = std::env::temp_dir()
                .join(format!("zo-test-home-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            std::env::set_var("ZO_CONFIG_HOME", &dir);
            dir
        });
        if std::env::var_os("ZO_CONFIG_HOME").is_none_or(|value| value.is_empty()) {
            std::env::set_var("ZO_CONFIG_HOME", home);
        }
    }

    fn unique_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        isolate_global_zo_home();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "zo-verdict-attr-{tag}-{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// These tests depend on `ZO_CONFIG_HOME` staying stable for their whole
    /// body (`record_verdict_outcome_at` resolves the outcome store through it
    /// per call) — hold the crate-wide env lock so tests in other modules that
    /// legitimately swap that variable under the same lock (e.g. the spawn
    /// round-trips) cannot interleave and split our records across two homes.
    /// Without this the suite is order-dependent: green in a full run, flaky
    /// under a filtered `cargo test -p tools verdict`.
    fn locked_env() -> std::sync::MutexGuard<'static, ()> {
        crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write_manifest(store: &std::path::Path, id: &str, manifest: &serde_json::Value) {
        std::fs::write(
            store.join(format!("{id}.json")),
            serde_json::to_string(manifest).expect("manifest json"),
        )
        .expect("write manifest");
    }

    fn read_outcomes(cwd: &std::path::Path) -> Vec<runtime::RouteOutcomeRecord> {
        runtime::read_route_outcomes(cwd).unwrap_or_default()
    }

    #[test]
    fn verdict_outcome_attributes_to_the_worker_manifest_model() {
        let _env = locked_env();
        let cwd = unique_dir("cwd");
        let store = unique_dir("store");
        write_manifest(
            &store,
            "agent-1",
            &json!({
                "agentId": "agent-1",
                "subagentType": "Refactor",
                "model": "requested-model",
                "resolvedModel": "worker-model",
                "requestedModel": "requested-model",
                "tokenHistory": [100, 250]
            }),
        );

        record_verdict_outcome_at(&cwd, &store, "agent-1", true, VerdictKind::PassFail);
        record_verdict_outcome_at(&cwd, &store, "agent-1", false, VerdictKind::PassFail);

        let outcomes = read_outcomes(&cwd);
        assert_eq!(outcomes.len(), 2, "one record per verdict");
        assert!(outcomes.iter().all(|record| {
            record.route_key == "subagent:Refactor"
                && record.selected_model == "worker-model"
                && record.signal.as_deref() == Some("verdict")
                && record.requested_model.as_deref() == Some("requested-model")
                && record.output_tokens == 350
                && record.signal_weight == Some(1.0)
        }));
        assert_eq!(outcomes[0].status, "completed");
        assert_eq!(outcomes[1].status, "failed");
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(store);
    }

    #[test]
    fn verdict_outcome_weights_a_preference_signal_at_half() {
        let _env = locked_env();
        let cwd = unique_dir("pref-cwd");
        let store = unique_dir("pref-store");
        write_manifest(
            &store,
            "agent-pref",
            &json!({
                "agentId": "agent-pref",
                "subagentType": "Refactor",
                "resolvedModel": "worker-model"
            }),
        );

        record_verdict_outcome_at(&cwd, &store, "agent-pref", true, VerdictKind::Preference);

        let outcomes = read_outcomes(&cwd);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].signal_weight, Some(0.5));
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(store);
    }

    #[test]
    fn verdict_outcome_canonicalizes_model_and_stamps_v2_route_metadata() {
        let _env = locked_env();
        let cwd = unique_dir("v2-cwd");
        let store = unique_dir("v2-store");
        write_manifest(
            &store,
            "agent-2",
            &json!({
                "agentId": "agent-2",
                "subagentType": "Plan",
                "model": "claude-opus-4.8",
                "resolvedModel": "claude-opus-4.8",
                "routeRole": "analysis",
                "routeComplexity": "large",
                "routeRisk": "medium",
                "routeSource": "auto"
            }),
        );

        record_verdict_outcome_at(&cwd, &store, "agent-2", true, VerdictKind::PassFail);

        let outcomes = read_outcomes(&cwd);
        assert_eq!(outcomes.len(), 1);
        let record = &outcomes[0];
        assert_eq!(
            record.selected_model, "claude-opus-4-8",
            "write-time canonicalization must dash-normalize the dot variant"
        );
        assert_eq!(record.role.as_deref(), Some("analysis"));
        assert_eq!(record.complexity.as_deref(), Some("large"));
        assert_eq!(record.risk.as_deref(), Some("medium"));
        assert_eq!(record.route_source.as_deref(), Some("auto"));
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(store);
    }

    #[test]
    fn verdict_outcome_is_silent_without_manifest_or_model() {
        let _env = locked_env();
        let cwd = unique_dir("silent-cwd");
        let store = unique_dir("silent-store");

        // No manifest at all.
        record_verdict_outcome_at(&cwd, &store, "missing", false, VerdictKind::PassFail);
        // Manifest without any model — nothing to credit or blame.
        write_manifest(&store, "modelless", &json!({"subagentType": "Explore"}));
        record_verdict_outcome_at(&cwd, &store, "modelless", true, VerdictKind::PassFail);
        // Blank agent id.
        record_verdict_outcome_at(&cwd, &store, "  ", true, VerdictKind::PassFail);

        assert!(
            read_outcomes(&cwd).is_empty(),
            "best-effort attribution must record nothing on missing provenance"
        );
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(store);
    }
}
