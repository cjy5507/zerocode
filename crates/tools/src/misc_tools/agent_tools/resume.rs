//! `SendMessage` resume support: the serializable spawn parameters a terminal
//! agent needs to be re-spawned WITH ITS CONTEXT INTACT (the conversation
//! itself lives in the sibling `<id>.session.jsonl` transcript).
//!
//! Only what cannot be re-derived at resume time is snapshotted. The harness
//! (system prompt, tool allow-list, permission rules) is re-resolved from the
//! manifest's `subagentType` exactly like a fresh spawn — deliberately, so a
//! custom-agent definition edited between runs takes effect — and live-context
//! values (LSP registry, MCP passthrough, hook config) come from the resuming
//! session. Verdict-channel bindings (`judged_agent`) are NOT carried: a
//! follow-up Q&A turn must never re-credit a review verdict.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::spawn::AgentJob;
use super::AgentOutput;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentResumeSnapshot {
    /// Working directory (worktree isolation). Resuming in the same tree the
    /// agent worked in is load-bearing — relative paths in its context refer
    /// to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Structured-output schema: a parent that consumed `StructuredOutput`
    /// results expects the follow-up to answer the same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_effort: Option<api::EffortLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_concurrency: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub route_fallback_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_model: Option<String>,
}

impl AgentResumeSnapshot {
    pub(super) fn from_job(job: &AgentJob) -> Self {
        Self {
            cwd: job.cwd.clone(),
            schema: job.schema.clone(),
            thinking_budget_tokens: job.thinking_budget_tokens,
            route_effort: job.route_effort,
            api_concurrency: job.api_concurrency,
            route_fallback_models: job.route_fallback_models.clone(),
            parent_model: job.parent_model.clone(),
        }
    }
}

/// `<store>/<id>.session.jsonl`, derived from the manifest's own path so no
/// caller needs the store-dir env resolution (and a relocated store keeps the
/// trio of files together).
pub(super) fn transcript_path_for(manifest: &AgentOutput) -> Result<PathBuf, String> {
    super::manifest::trusted_agent_transcript_path(manifest)
}

/// `<store>/<id>.resume.json`, sibling of the manifest.
#[cfg(test)]
pub(super) fn resume_snapshot_path_for(manifest: &AgentOutput) -> PathBuf {
    PathBuf::from(&manifest.manifest_file).with_extension("resume.json")
}

/// Persist the resume snapshot at spawn time. Best-effort at the call site —
/// a failed write only costs future resumability, never the spawn itself.
pub(super) fn write_agent_resume_snapshot(job: &AgentJob) -> Result<(), String> {
    let snapshot = AgentResumeSnapshot::from_job(job);
    let json = serde_json::to_string_pretty(&snapshot).map_err(|error| error.to_string())?;
    super::manifest::write_agent_resume_snapshot_file(&job.manifest, &json)
}

/// Load the snapshot for a resume. A missing or unreadable snapshot degrades
/// to defaults (fresh-spawn behavior for every field) rather than blocking the
/// resume — the transcript, not the snapshot, is the part that cannot be
/// reconstructed.
pub(super) fn load_agent_resume_snapshot(manifest: &AgentOutput) -> AgentResumeSnapshot {
    super::manifest::read_agent_resume_snapshot_file(manifest)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc_tools::agent_tools::AgentActivityTelemetry;

    fn manifest_at(dir: &std::path::Path, id: &str) -> AgentOutput {
        AgentOutput {
            agent_id: id.to_string(),
            parent_session_id: None,
            tool_call_id: None,
            name: id.to_string(),
            label: None,
            description: "agent".to_string(),
            subagent_type: Some("general-purpose".to_string()),
            requested_model: None,
            resolved_model: None,
            route_reason: None,
            route_role: None,
            route_complexity: None,
            route_risk: None,
            route_source: None,
            model: None,
            status: "completed".to_string(),
            output_file: dir.join(format!("{id}.md")).display().to_string(),
            manifest_file: dir.join(format!("{id}.json")).display().to_string(),
            created_at: "100".to_string(),
            owner_pid: None,
            run_generation: 0,
            started_at: Some("100".to_string()),
            completed_at: Some("200".to_string()),
            completion_published_at: None,
            lane_events: Vec::new(),
            current_blocker: None,
            error: None,
            token_history: Vec::new(),
            current_tool: None,
            recent_tools: Vec::new(),
            tool_calls: 0,
            current_phase: None,
            output_tail: String::new(),
            last_activity_at: None,
            activity: AgentActivityTelemetry::default(),
        }
    }

    #[test]
    fn sibling_paths_derive_from_the_manifest_file() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-resume-sibling-paths-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let store = std::fs::canonicalize(&dir).expect("canonical store");
        let manifest = manifest_at(&store, "agent-7");
        assert_eq!(
            transcript_path_for(&manifest).expect("trusted transcript path"),
            store.join("agent-7.session.jsonl")
        );
        assert_eq!(
            resume_snapshot_path_for(&manifest),
            store.join("agent-7.resume.json")
        );
        std::fs::remove_dir_all(&store).expect("remove store");
    }

    #[test]
    fn missing_snapshot_degrades_to_defaults() {
        let manifest = manifest_at(std::path::Path::new("/tmp/definitely-missing"), "none");
        let snapshot = load_agent_resume_snapshot(&manifest);
        assert!(snapshot.cwd.is_none());
        assert!(snapshot.schema.is_none());
        assert!(snapshot.route_fallback_models.is_empty());
    }

    #[test]
    fn snapshot_roundtrips_through_disk() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-resume-snapshot-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let manifest = manifest_at(&dir, "roundtrip");

        let job = AgentJob {
            manifest: manifest.clone(),
            prompt: "original task".to_string(),
            system_prompt: Vec::new(),
            allowed_tools: std::collections::BTreeSet::new(),
            permission_rules: None,
            permission_mode: None,
            cwd: Some(PathBuf::from("/tmp/worktree-a")),
            lsp: None,
            schema: Some(serde_json::json!({"type": "object"})),
            workflow_member: false,
            time_budget: None,
            thinking_budget_tokens: Some(4096),
            route_effort: Some(api::EffortLevel::High),
            api_concurrency: Some(2),
            route_fallback_models: vec!["fallback-model".to_string()],
            mcp_passthrough: None,
            hook_config: runtime::RuntimeHookConfig::default(),
            cancel_signal: runtime::HookAbortSignal::new(),
            judged_agent: Some("worker-1".to_string()),
            parent_model: Some("claude-opus-4-8".to_string()),
            steering: runtime::SteeringQueue::default(),
            transcript_path: Some(
                transcript_path_for(&manifest).expect("trusted transcript path"),
            ),
            resume: false,
        };
        write_agent_resume_snapshot(&job).expect("write snapshot");

        let loaded = load_agent_resume_snapshot(&manifest);
        assert_eq!(loaded.cwd.as_deref(), Some(std::path::Path::new("/tmp/worktree-a")));
        assert_eq!(loaded.schema, Some(serde_json::json!({"type": "object"})));
        assert_eq!(loaded.thinking_budget_tokens, Some(4096));
        assert_eq!(loaded.route_effort, Some(api::EffortLevel::High));
        assert_eq!(loaded.api_concurrency, Some(2));
        assert_eq!(loaded.route_fallback_models, vec!["fallback-model".to_string()]);
        assert_eq!(loaded.parent_model.as_deref(), Some("claude-opus-4-8"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_snapshot_degrades_to_defaults() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-resume-corrupt-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let manifest = manifest_at(&dir, "corrupt");
        std::fs::write(resume_snapshot_path_for(&manifest), "not-json{{{")
            .expect("write corrupt snapshot");

        let snapshot = load_agent_resume_snapshot(&manifest);
        assert!(snapshot.cwd.is_none());
        assert!(snapshot.schema.is_none());
        assert!(snapshot.route_fallback_models.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn send_message_seam_settles_dead_owner_and_preserves_resume_artifacts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = std::fs::canonicalize(dir.path()).expect("canonical store");
        let mut manifest = manifest_at(&store, "send-dead-owner");
        manifest.status = "running".to_string();
        manifest.owner_pid = Some(u32::MAX - 9);
        std::fs::write(&manifest.output_file, "# Agent\n").expect("write output");
        std::fs::write(
            &manifest.manifest_file,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        let transcript = transcript_path_for(&manifest).expect("trusted transcript path");
        let transcript_bytes =
            b"{\"type\":\"session_meta\",\"session_id\":\"send-dead-owner\",\"version\":1}\n";
        std::fs::write(&transcript, transcript_bytes).expect("write transcript");
        std::fs::write(
            resume_snapshot_path_for(&manifest),
            b"{\"cwd\":\"/tmp/worktree-a\"}",
        )
        .expect("write snapshot");

        assert!(super::super::settle_dead_owner_agent_with_live(
            &manifest,
            &std::collections::HashSet::new(),
        ));

        let settled_bytes =
            std::fs::read(&manifest.manifest_file).expect("read settled manifest bytes");
        let settled: AgentOutput =
            serde_json::from_slice(&settled_bytes).expect("parse settled manifest");
        assert_eq!(settled.status, "stopped");
        assert!(
            String::from_utf8_lossy(&settled_bytes)
                .contains("orphaned: owning process exited")
        );
        assert_eq!(
            load_agent_resume_snapshot(&settled).cwd.as_deref(),
            Some(std::path::Path::new("/tmp/worktree-a"))
        );
        assert_eq!(
            std::fs::read(transcript).expect("read transcript after settle"),
            transcript_bytes
        );
    }

    #[test]
    fn send_message_seam_keeps_live_owner_manifest_byte_identical() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = std::fs::canonicalize(dir.path()).expect("canonical store");
        let mut manifest = manifest_at(&store, "send-live-owner");
        manifest.status = "running".to_string();
        let owner_pid = u32::MAX - 9;
        manifest.owner_pid = Some(owner_pid);
        std::fs::write(&manifest.output_file, "# Agent\n").expect("write output");
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
        std::fs::write(&manifest.manifest_file, &manifest_bytes).expect("write manifest");
        let live = std::collections::HashSet::from([owner_pid]);

        assert!(!super::super::settle_dead_owner_agent_with_live(
            &manifest, &live,
        ));
        assert_eq!(
            std::fs::read(&manifest.manifest_file).expect("read untouched manifest"),
            manifest_bytes
        );
    }

    /// The "running but no live handle" incident seam, through PRODUCTION
    /// reconciliation: a running manifest whose owner pid is dead settles via
    /// the orphan reap, and the settled agent keeps its resume artifacts —
    /// the snapshot still loads and the transcript path still resolves, so
    /// the orphan stays resumable instead of stranded.
    #[test]
    fn reaped_dead_owner_agent_stays_resumable() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-resume-reap-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let store = std::fs::canonicalize(&dir).expect("canonical store");
        let mut manifest = manifest_at(&store, "orphan");
        manifest.status = "running".to_string();
        // Beyond macOS/Linux pid_max — can never be a live zo process.
        manifest.owner_pid = Some(u32::MAX - 7);
        std::fs::write(&manifest.output_file, "# Agent\n").expect("write output");
        std::fs::write(
            &manifest.manifest_file,
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        let transcript = transcript_path_for(&manifest).expect("trusted transcript path");
        let transcript_body = "{\"type\":\"session_meta\",\"session_id\":\"orphan\",\"version\":1}\n";
        std::fs::write(&transcript, transcript_body).expect("write transcript");

        let job = AgentJob {
            manifest: manifest.clone(),
            prompt: "original task".to_string(),
            system_prompt: Vec::new(),
            allowed_tools: std::collections::BTreeSet::new(),
            permission_rules: None,
            permission_mode: None,
            cwd: Some(PathBuf::from("/tmp/worktree-a")),
            lsp: None,
            schema: None,
            workflow_member: false,
            time_budget: None,
            thinking_budget_tokens: None,
            route_effort: None,
            api_concurrency: None,
            route_fallback_models: Vec::new(),
            mcp_passthrough: None,
            hook_config: runtime::RuntimeHookConfig::default(),
            cancel_signal: runtime::HookAbortSignal::new(),
            judged_agent: None,
            parent_model: None,
            steering: runtime::SteeringQueue::default(),
            transcript_path: Some(
                transcript_path_for(&manifest).expect("trusted transcript path"),
            ),
            resume: false,
        };
        write_agent_resume_snapshot(&job).expect("write snapshot");

        let reaped = super::super::reap_orphaned_agents_in_store(&store);
        assert_eq!(reaped, 1, "dead-owner running manifest must settle");

        let settled: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read settled manifest"),
        )
        .expect("parse settled manifest");
        assert_eq!(settled.status, "stopped");

        let snapshot = load_agent_resume_snapshot(&settled);
        assert_eq!(
            snapshot.cwd.as_deref(),
            Some(std::path::Path::new("/tmp/worktree-a"))
        );
        // The reap must not disturb the resume inputs: the transcript survives
        // byte-identical (the rehydration source) and the generation stamp is
        // intact for the next resume to bump. Driving a live SendMessage
        // round-trip needs a host process + provider, out of unit scope — the
        // resume INPUTS proven here are what that path consumes.
        let rehydrated = std::fs::read_to_string(
            transcript_path_for(&settled).expect("settled orphan keeps a transcript path"),
        )
        .expect("read transcript after reap");
        assert_eq!(rehydrated, transcript_body);
        assert_eq!(settled.run_generation, 0, "generation stamp survives the reap");

        std::fs::remove_dir_all(&store).ok();
    }
}
