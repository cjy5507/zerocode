//! Verified-state observation: the conversation-layer half of
//! [`crate::verified_state`].
//!
//! Two seams, both on paths the sync and streaming loops already share:
//!
//! * **record** — [`ConversationRuntime::record_verified_state_from_turn`] runs
//!   at `record_turn_completed`, folding the finished turn's ordered green
//!   checks and edits into the session ledger (which write-throughs to its
//!   sidecar, so the next stage's process sees them).
//! * **observe** — [`ConversationRuntime::inject_verified_state_reminder`] runs
//!   at turn start next to the other input-triggered reminders, so the
//!   observation lands where the model PLANS the stage.
//!
//! That placement is the whole lesson of the reverted attempt (`0c39f26`),
//! which appended a "you already saved this file" note to the tail of tool
//! RESULTS and measured zero adoption in a paid run: by the time a result is
//! read the plan that will re-verify has already been made. Turn start is the
//! only moment the observation can change what the model decides to do.
//!
//! ## Why the classifier is shared, not re-derived
//!
//! `exec_green_checks` already owns "what counts as a check that ran green",
//! including the bash success landmine (`returnCodeInterpretation` absent =
//! success, `interrupted: false`, no `backgroundTaskId`). A second definition
//! here would drift, and the drift would be invisible — both surfaces report
//! the same class of fact. So the vocabulary stays in `deep_gate` and this
//! module only differs where the SEMANTICS differ: `exec_green_checks` DROPS
//! every check that predates the turn's last edit (stale evidence for a
//! verifier judging one diff), whereas the session ledger KEEPS it and reports
//! the edit delta instead — a check invalidated by a later edit is exactly the
//! fact worth telling the planner.
//!
//! ## Deferred: joining the session ledger into the VERIFY leg
//!
//! The deep-gate VERIFY prompt reports `exec_green_checks` — THIS attempt's
//! post-edit greens. Folding the session ledger's fresh greens in beside them
//! is deferred, and the reason is a correctness one rather than a size one: the
//! ledger settles at TURN END, so while a turn is in flight it does not yet
//! know about the edits that turn just made. A session check rendered into a
//! VERIFY prompt mid-turn could therefore assert "nothing edited since" about a
//! tree the EXEC leg had just changed — the verifier would be handed, as a
//! harness-owned IO fact, something that is not true. Making the join safe
//! means recording each check/edit at the tool-result seam instead of at turn
//! end (the command text is only reachable there via the `tool_use` input), and
//! that is a larger change than this one. Until then the standing observation
//! is RETIRED at the turn's first successful mutation
//! ([`ConversationRuntime::retire_verified_state_observation_on_mutation`]), so
//! a VERIFY leg either sees an observation that is still true or sees none.

use super::deep_gate::{
    bash_result_exited_zero, command_is_check_shaped, tool_result_path, truncate_on_boundary,
    EXEC_CHECK_COMMAND_BYTES,
};
use super::{
    ApiClient, ContentBlock, ConversationMessage, ConversationRuntime, ToolExecutor, TurnSummary,
};
use crate::verified_state::{VerifiedStateEvent, VERIFIED_STATE_REMINDER_PREFIX};

/// The completed turn's ledger-worthy facts, in the order the executor settled
/// them.
///
/// Order across the whole turn is what makes "edited SINCE the check" a real
/// question: results are appended to `summary.tool_results` in dispatch order,
/// so walking them once yields the interleaving directly. `tool_use` inputs
/// (which carry the command text) are joined by id from the assistant messages,
/// mirroring `exec_green_checks`.
pub(super) fn turn_verified_state_events(summary: &TurnSummary) -> Vec<VerifiedStateEvent> {
    let mut commands: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for message in &summary.assistant_messages {
        for block in &message.blocks {
            if let ContentBlock::ToolUse { id, name, input } = block {
                if name == "bash" {
                    commands.insert(id.as_str(), input.as_str());
                }
            }
        }
    }
    let mut events: Vec<VerifiedStateEvent> = Vec::new();
    for message in &summary.tool_results {
        for block in &message.blocks {
            let ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
                ..
            } = block
            else {
                continue;
            };
            if *is_error {
                continue;
            }
            if super::is_edit_or_write_tool(tool_name) {
                if let Some(path) = tool_result_path(output) {
                    events.push(VerifiedStateEvent::Edit(path));
                }
                continue;
            }
            if tool_name != "bash" || !bash_result_exited_zero(output) {
                continue;
            }
            let Some(command) = green_check_command(commands.get(tool_use_id.as_str()).copied())
            else {
                continue;
            };
            events.push(VerifiedStateEvent::GreenCheck(command));
        }
    }
    events
}

/// The check-shaped command inside a bash `tool_use` input, capped for storage.
/// `None` for anything that is not a check (`ls -la` is green every time and
/// proves nothing) or whose input did not parse.
fn green_check_command(input: Option<&str>) -> Option<String> {
    let mut command = serde_json::from_str::<serde_json::Value>(input?)
        .ok()?
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)?;
    if !command_is_check_shaped(&command) {
        return None;
    }
    truncate_on_boundary(&mut command, EXEC_CHECK_COMMAND_BYTES);
    Some(command)
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    /// Fold this turn's observed checks and edits into the session ledger.
    ///
    /// Called from `record_turn_completed`, which BOTH turn loops reach — so a
    /// green check run in a sync turn is still visible to the next streaming
    /// one. Recording is idempotent under a repeated summary (both kinds dedupe
    /// newest-wins), which matters because a deep-lane subturn and its parent
    /// can each complete.
    pub(super) fn record_verified_state_from_turn(&mut self, summary: &TurnSummary) {
        let events = turn_verified_state_events(summary);
        self.verified_state.record_turn(&events);
    }

    /// Install this turn's verified-state observation, or nothing at all.
    ///
    /// Turn-scoped like the other input-triggered reminders: cleared at turn
    /// start by `clear_turn_start_transient_reminders`, re-armed here. It rides
    /// the transient-reminder path, so it is absorbed into the transcript as a
    /// trailing `System` message at request-build time
    /// (`absorb_wire_reminders_into_session`) — an APPEND behind the new user
    /// message, never a rewrite of the cached prefix.
    ///
    /// A ledger with no green check renders `None` and installs nothing, which
    /// is the byte-neutrality guarantee: the feature costs zero until the
    /// session has actually verified something.
    pub(super) fn inject_verified_state_reminder(&mut self) {
        let root = self.trace_cwd();
        let Some(body) = self.verified_state.render_observation(root.as_deref()) else {
            // A distinct decline from the ablation one below: "this session has
            // verified nothing yet" is normal operation, while a held-out arm
            // means a block WAS available and an experiment suppressed it.
            // Conflating them is how a never-firing feature reads as idle.
            telemetry::attest_declined(
                telemetry::HarnessFeature::VerifiedStateObservation,
                "no_green_checks",
            );
            return;
        };
        // Checked after the content gate so an ablation decline counts turns
        // that would really have carried an observation.
        if telemetry::attest_ablated(telemetry::HarnessFeature::VerifiedStateObservation) {
            return;
        }
        telemetry::attest_fired(telemetry::HarnessFeature::VerifiedStateObservation);
        self.replace_transient_system_reminder_by_prefix(
            VERIFIED_STATE_REMINDER_PREFIX,
            Some(&body),
        );
    }

    /// Stop re-asserting the observation once this turn has actually mutated
    /// the workspace.
    ///
    /// The ledger settles at turn end, so mid-turn the standing block cannot
    /// know about the edit that just happened — and the reminder absorber
    /// re-appends a standing set near the tail when it drifts past its dedupe
    /// window, which would place a stale "nothing edited since" BELOW the edits
    /// that falsified it. Two places would have read that lie: the model's next
    /// iteration, and a deep-lane VERIFY leg (an internal subturn, which by
    /// design does not re-run the turn-start injection).
    ///
    /// Retiring rather than re-rendering keeps the feature's cost honest: a
    /// mid-turn refresh would append a fresh block after every mutating batch,
    /// which is exactly the automatic token burn this feature is not allowed to
    /// become. The copy persisted at turn-start position is left alone — it was
    /// true where it sits, and the model reads the turn's edits after it.
    pub(super) fn retire_verified_state_observation_on_mutation(
        &mut self,
        message: &ConversationMessage,
    ) {
        let mutated = message.blocks.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolResult { tool_name, is_error, .. }
                    if !is_error && super::is_edit_or_write_tool(tool_name)
            )
        });
        if mutated {
            self.replace_transient_system_reminder_by_prefix(VERIFIED_STATE_REMINDER_PREFIX, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{turn_verified_state_events, VERIFIED_STATE_REMINDER_PREFIX};
    use crate::conversation::{
        ApiClient, ApiRequest, AssistantEvent, ConversationRuntime, RuntimeError,
        StaticToolExecutor, TurnSummary,
    };
    use crate::permissions::{PermissionMode, PermissionPolicy};
    use crate::session::{ContentBlock, ConversationMessage, Session};
    use crate::usage::TokenUsage;
    use crate::verified_state::VerifiedStateEvent;

    /// A client that never talks to a provider: every test here drives the
    /// record/observe seams directly, so the turn loop is deliberately absent.
    struct SilentApiClient;

    impl ApiClient for SilentApiClient {
        fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
            Ok(vec![AssistantEvent::MessageStop])
        }
    }

    fn static_runtime() -> ConversationRuntime<SilentApiClient, StaticToolExecutor> {
        ConversationRuntime::new(
            Session::new(),
            SilentApiClient,
            StaticToolExecutor::new(),
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
    }

    fn turn_summary_with(
        assistant_blocks: Vec<ContentBlock>,
        tool_results: Vec<ConversationMessage>,
    ) -> TurnSummary {
        let assistant_messages = if assistant_blocks.is_empty() {
            Vec::new()
        } else {
            vec![ConversationMessage::assistant(assistant_blocks)]
        };
        TurnSummary {
            assistant_messages,
            tool_results,
            prompt_cache_events: Vec::new(),
            iterations: 1,
            usage: TokenUsage::default(),
            turn_output_tokens: 0,
            auto_compaction: None,
            microcompact: None,
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted: None,
        }
    }

    fn bash_use(id: &str, command: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({ "command": command }).to_string(),
        }
    }

    /// The classifier keeps the same bash-success landmine as
    /// `exec_green_checks` (non-zero / interrupted / backgrounded are not
    /// green, non-check shapes are not evidence) but — unlike it — does NOT
    /// drop a check that a later edit invalidated: it emits both, in order, and
    /// lets the ledger report the delta.
    #[test]
    fn events_carry_green_checks_and_edits_in_order() {
        let summary = turn_summary_with(
            vec![
                bash_use("pre", "cargo build --tests"),
                bash_use("post", "cargo test -p runtime"),
                bash_use("red", "cargo test broken"),
                bash_use("bg", "cargo test --workspace"),
                bash_use("noise", "ls -la"),
            ],
            vec![
                ConversationMessage::tool_result("pre", "bash", r#"{"stdout":"ok"}"#, false),
                ConversationMessage::tool_result(
                    "edit-1",
                    "edit_file",
                    r#"{"filePath":"/repo/src/main.rs"}"#,
                    false,
                ),
                ConversationMessage::tool_result("post", "bash", r#"{"stdout":"ok"}"#, false),
                ConversationMessage::tool_result(
                    "red",
                    "bash",
                    r#"{"stdout":"","returnCodeInterpretation":"exit_code:101"}"#,
                    false,
                ),
                ConversationMessage::tool_result(
                    "bg",
                    "bash",
                    r#"{"stdout":"started","backgroundTaskId":"t1"}"#,
                    false,
                ),
                ConversationMessage::tool_result("noise", "bash", r#"{"stdout":"x"}"#, false),
                // A failed edit changed nothing and must not invalidate anything.
                ConversationMessage::tool_result(
                    "edit-2",
                    "edit_file",
                    r#"{"filePath":"/repo/src/other.rs"}"#,
                    true,
                ),
            ],
        );

        assert_eq!(
            turn_verified_state_events(&summary),
            vec![
                VerifiedStateEvent::GreenCheck("cargo build --tests".to_string()),
                VerifiedStateEvent::Edit("/repo/src/main.rs".to_string()),
                VerifiedStateEvent::GreenCheck("cargo test -p runtime".to_string()),
            ]
        );
    }

    /// (a) Record in one turn, observe at the next turn's start — the FRESH
    /// form, wired end to end through the runtime rather than the ledger alone.
    #[test]
    fn a_recorded_green_check_is_injected_at_the_next_turn_start() {
        let mut runtime = static_runtime();
        runtime.record_verified_state_from_turn(&turn_summary_with(
            vec![bash_use("c1", "cargo test -p runtime")],
            vec![ConversationMessage::tool_result(
                "c1",
                "bash",
                r#"{"stdout":"ok"}"#,
                false,
            )],
        ));

        runtime.clear_turn_start_transient_reminders();
        runtime.inject_verified_state_reminder();
        let reminders = runtime.transient_reminders.join("\n");
        assert!(reminders.contains(VERIFIED_STATE_REMINDER_PREFIX), "{reminders}");
        assert!(
            reminders.contains("`cargo test -p runtime` — ran green, nothing edited since"),
            "{reminders}"
        );

        // Turn-scoped: the next turn's clear drops it, and it never accumulates.
        runtime.clear_turn_start_transient_reminders();
        assert!(!runtime
            .transient_reminders
            .join("\n")
            .contains(VERIFIED_STATE_REMINDER_PREFIX));
        runtime.inject_verified_state_reminder();
        runtime.inject_verified_state_reminder();
        assert_eq!(
            runtime
                .transient_reminders
                .iter()
                .filter(|reminder| reminder.starts_with(VERIFIED_STATE_REMINDER_PREFIX))
                .count(),
            1
        );
    }

    /// (b) An edit recorded after the check surfaces as the invalidated form,
    /// naming the changed file.
    #[test]
    fn an_edit_after_the_check_surfaces_as_invalidated_with_the_path() {
        let mut runtime = static_runtime();
        runtime.record_verified_state_from_turn(&turn_summary_with(
            vec![bash_use("c1", "cargo test -p runtime")],
            vec![ConversationMessage::tool_result(
                "c1",
                "bash",
                r#"{"stdout":"ok"}"#,
                false,
            )],
        ));
        runtime.record_verified_state_from_turn(&turn_summary_with(
            Vec::new(),
            vec![ConversationMessage::tool_result(
                "e1",
                "edit_file",
                r#"{"filePath":"/repo/crates/runtime/src/a.rs"}"#,
                false,
            )],
        ));

        runtime.inject_verified_state_reminder();
        let reminders = runtime.transient_reminders.join("\n");
        assert!(
            reminders.contains(
                "ran green BUT these files changed since: /repo/crates/runtime/src/a.rs"
            ),
            "{reminders}"
        );
    }

    /// (c) Byte neutrality: with no green check on record the turn start
    /// installs nothing, so the request is byte-identical to one built without
    /// this feature — including on a turn that only edited files.
    #[test]
    fn without_a_green_check_the_turn_start_is_byte_neutral() {
        let mut runtime = static_runtime();
        let before = runtime.transient_reminders.clone();
        runtime.inject_verified_state_reminder();
        assert_eq!(runtime.transient_reminders, before);

        runtime.record_verified_state_from_turn(&turn_summary_with(
            Vec::new(),
            vec![ConversationMessage::tool_result(
                "e1",
                "edit_file",
                r#"{"filePath":"/repo/a.rs"}"#,
                false,
            )],
        ));
        runtime.inject_verified_state_reminder();
        assert_eq!(
            runtime.transient_reminders, before,
            "an edit-only session must stay byte-neutral"
        );
    }

    /// A standing observation is retired the moment the turn mutates anything,
    /// so the absorber's tail re-anchor can never re-assert "nothing edited
    /// since" below the edits that falsified it. A FAILED edit changed nothing
    /// and must leave the observation standing.
    #[test]
    fn a_successful_mutation_retires_the_standing_observation() {
        let mut runtime = static_runtime();
        runtime.record_verified_state_from_turn(&turn_summary_with(
            vec![bash_use("c1", "cargo test -p runtime")],
            vec![ConversationMessage::tool_result(
                "c1",
                "bash",
                r#"{"stdout":"ok"}"#,
                false,
            )],
        ));
        runtime.inject_verified_state_reminder();
        assert!(runtime
            .transient_reminders
            .join("\n")
            .contains(VERIFIED_STATE_REMINDER_PREFIX));

        // A read is not a mutation; a failed edit is not one either.
        runtime.record_tool_finished(
            1,
            &ConversationMessage::tool_result("r1", "read_file", r#"{"path":"/repo/a.rs"}"#, false),
        );
        runtime.record_tool_finished(
            1,
            &ConversationMessage::tool_result(
                "e0",
                "edit_file",
                r#"{"filePath":"/repo/a.rs"}"#,
                true,
            ),
        );
        assert!(
            runtime
                .transient_reminders
                .join("\n")
                .contains(VERIFIED_STATE_REMINDER_PREFIX),
            "only a SUCCESSFUL mutation may retire the observation"
        );

        runtime.record_tool_finished(
            1,
            &ConversationMessage::tool_result(
                "e1",
                "edit_file",
                r#"{"filePath":"/repo/a.rs"}"#,
                false,
            ),
        );
        assert!(
            !runtime
                .transient_reminders
                .join("\n")
                .contains(VERIFIED_STATE_REMINDER_PREFIX),
            "a successful edit must stop the observation being re-asserted"
        );
    }

    /// (d) Sidecar round trip THROUGH the runtime seam: a stage records, a
    /// freshly built runtime rebinds the same sidecar, and the observation
    /// survives the process boundary. This is the multi-stage bench case.
    #[test]
    fn the_observation_survives_a_runtime_rebuild_via_the_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "zo-verified-state-runtime-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let sidecar = dir.join("session.verified-state.json");

        let mut stage_one = static_runtime();
        stage_one.reset_verified_state_session(Some(sidecar.clone()));
        stage_one.record_verified_state_from_turn(&turn_summary_with(
            vec![bash_use("c1", "cargo test -p runtime")],
            vec![ConversationMessage::tool_result(
                "c1",
                "bash",
                r#"{"stdout":"ok"}"#,
                false,
            )],
        ));

        // A brand-new runtime — the stage-2 process — sees it only because the
        // ledger was rebound to the sidecar.
        let mut stage_two = static_runtime();
        assert!(
            stage_two.verified_state.render_observation(None).is_none(),
            "an unbound runtime starts empty"
        );
        stage_two.reset_verified_state_session(Some(sidecar));
        stage_two.inject_verified_state_reminder();
        assert!(
            stage_two
                .transient_reminders
                .join("\n")
                .contains("cargo test -p runtime"),
            "the sidecar must carry the ledger across the stage boundary"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
