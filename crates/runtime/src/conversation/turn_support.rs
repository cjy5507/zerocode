//! Per-turn request assembly and session-trace recording for
//! [`ConversationRuntime`], split out of `mod.rs` so the turn loop there reads
//! as orchestration. Behaviour-preserving: these were `impl ConversationRuntime`
//! methods, now `pub(super)` so the loop in `mod.rs` still calls them.

use std::borrow::Cow;
use std::sync::Arc;

use serde_json::{Value, json};
use telemetry::SessionTracer;

use super::{
    ApiClient, ApiRequest, CompactionConfig, ContentBlock, ConversationMessage,
    ConversationRuntime, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MEMORY_RECALL_LIMIT, MemoryRetriever,
    MessageRole, RuntimeError, StreamingTurnError, ToolExecutor, TurnSummary,
    estimate_session_tokens, estimate_system_prompt_tokens, render_recalled_memory_section,
    trace_attrs,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnStopOrigin {
    User,
    HostFailure,
}

impl TurnStopOrigin {
    const fn candidate_kind(self) -> decision_core::dreamer::CandidateKind {
        match self {
            Self::User => decision_core::dreamer::CandidateKind::UserCancelled,
            Self::HostFailure => decision_core::dreamer::CandidateKind::TurnFailure,
        }
    }

    const fn summary(self) -> &'static str {
        match self {
            Self::User => "turn cancelled by user or host",
            Self::HostFailure => "turn failed because the host stopped consuming",
        }
    }

    const fn trace_outcome(self) -> crate::turn_trace::TurnOutcome {
        match self {
            Self::User => crate::turn_trace::TurnOutcome::Cancelled,
            Self::HostFailure => crate::turn_trace::TurnOutcome::Failed,
        }
    }

    const fn trace_event(self) -> &'static str {
        match self {
            Self::User => "turn_cancelled",
            Self::HostFailure => "turn_host_failure",
        }
    }
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    pub(super) fn record_turn_started(&self, user_input: &str) {
        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        session_tracer.record(
            "turn_started",
            trace_attrs(json!({"user_input": user_input})),
        );
    }

    /// Wire reminders for the request being built right now: the toggled
    /// transient reminders plus a query-recalled memory section. These ride the
    /// newest user-role wire message (see [`crate::append_wire_reminders`]);
    /// the system prompt stays frozen so its cache blocks — and every message
    /// breakpoint behind them — keep hitting.
    pub(super) fn request_wire_reminders(&self) -> Arc<[String]> {
        let recall = self.memory_retriever.as_ref().and_then(|retriever| {
            let query = self.recall_query_text()?;
            let hits = retriever.recall(query.as_ref(), DEFAULT_MEMORY_RECALL_LIMIT);
            render_recalled_memory_section(&hits)
        });
        let mut reminders = self.transient_reminders.clone();
        reminders.extend(recall);
        Arc::from(reminders)
    }

    /// Async sibling of [`Self::request_wire_reminders`]' recall half: runs the
    /// memory recall in `spawn_blocking` so a dense (ONNX) embedding forward
    /// pass — which holds a mutex and can take tens of ms — never runs on the
    /// streaming drive-loop `select!` task and starve render/reveal/input
    /// (FREEZE-1). The streaming turn path awaits this; the synchronous
    /// `build_request`/headless paths keep the sync recall (no render loop
    /// there to starve).
    /// Takes OWNED inputs — the caller clones them out of `&self` first — so
    /// the enclosing spawned turn future never holds a `&self` borrow across
    /// this await. That matters because the runtime is `Send` but not `Sync`
    /// (it holds `Send`-only trait objects), so a `&self` borrow spanning an
    /// await would fail to compile.
    pub(super) async fn recall_reminder_section(
        retriever: Option<Arc<dyn MemoryRetriever + Send + Sync>>,
        query: Option<String>,
        tracer: Option<SessionTracer>,
    ) -> Option<String> {
        let (Some(retriever), Some(query)) = (retriever, query) else {
            return None;
        };
        let section = match tokio::task::spawn_blocking(move || {
            let hits = retriever.recall(&query, DEFAULT_MEMORY_RECALL_LIMIT);
            render_recalled_memory_section(&hits)
        })
        .await
        {
            Ok(section) => section,
            // A panic inside recall (e.g. a poisoned embedding mutex) must not
            // abort the turn — degrade to no recalled context — but is surfaced so
            // a silently-degraded dense retriever is visible. eprintln reaches the
            // TUI's redirected stderr (zo.log) and headless stderr; the trace
            // event reaches OTLP operators who watch the telemetry stream rather
            // than the log (the owned `tracer` keeps this fn self-less, so the
            // spawned turn future stays `Send`).
            Err(error) => {
                eprintln!(
                    "zo: memory recall task failed ({error}); continuing this turn without recalled context"
                );
                if let Some(tracer) = &tracer {
                    tracer.record(
                        "memory_recall_failed",
                        trace_attrs(json!({ "error": error.to_string() })),
                    );
                }
                None
            }
        };
        section
    }

    pub(super) fn latest_user_text(&self) -> Option<Cow<'_, str>> {
        self.session.messages.iter().rev().find_map(|message| {
            if message.role != MessageRole::User {
                return None;
            }
            message_text(message)
        })
    }

    pub(super) fn recall_query_text(&self) -> Option<Cow<'_, str>> {
        let latest = self.latest_user_text()?;
        if !is_short_follow_up(latest.as_ref()) {
            return Some(latest);
        }

        let mut saw_latest_user = false;
        for message in self.session.messages.iter().rev() {
            if message.role != MessageRole::User {
                continue;
            }
            let Some(text) = message_text(message) else {
                continue;
            };
            if !saw_latest_user {
                saw_latest_user = true;
                continue;
            }
            if is_short_follow_up(text.as_ref()) {
                continue;
            }
            return Some(Cow::Owned(format!("{}\n{}", text.trim(), latest.trim())));
        }

        Some(latest)
    }

    pub(super) fn build_request(&mut self, tool_choice: Option<::api::ToolChoice>) -> ApiRequest {
        let wire_reminders = self.request_wire_reminders();
        self.enforce_request_overflow_guard(&wire_reminders);
        self.assemble_request(wire_reminders, tool_choice)
    }

    /// Emergency context-overflow guard: if the estimated payload exceeds the
    /// model window (minus output headroom), compact now so the request does
    /// not 400. **Synchronous and potentially expensive** — a fired guard runs
    /// `compact_with_api_fallback`, a blocking LLM summary round-trip on the
    /// caller's thread. On the streaming path that thread is the TUI `select!`
    /// loop, so the streaming caller runs async preflight compaction *before*
    /// building the request and then assembles via [`Self::assemble_request`]
    /// directly, skipping this guard. The synchronous `run_turn` paths (no async
    /// preflight) still call it through [`Self::build_request`]. Split out of
    /// `build_request` so the two responsibilities — mutate-to-fit vs. assemble
    /// a snapshot — read separately (SRP).
    pub(super) fn enforce_request_overflow_guard(&mut self, wire_reminders: &[String]) {
        let profile = crate::turn_profiling_enabled();
        let headroom = DEFAULT_MAX_OUTPUT_TOKENS;
        let budget = self.context_window.saturating_sub(headroom);
        let system_prompt_tokens = estimate_system_prompt_tokens(&self.system_prompt)
            + estimate_system_prompt_tokens(wire_reminders);
        let estimated = estimate_session_tokens(&self.session) as u64 + system_prompt_tokens;
        if estimated <= budget {
            return;
        }
        let guard_t = profile.then(std::time::Instant::now);
        // Emergency overflow compaction is not user-directed: no focus.
        let result = self.compact_with_api_fallback(
            CompactionConfig {
                max_estimated_tokens: 0,
                ..CompactionConfig::default()
            },
            None,
        );
        if result.removed_message_count > 0 {
            self.session = result.compacted_session;
        }

        // Re-check after compaction — if still over budget, aggressively
        // trim preserved messages down to the most recent pair to avoid
        // sending an oversized payload that the API will reject.
        let after_compaction = estimate_session_tokens(&self.session) as u64 + system_prompt_tokens;
        if after_compaction > budget && self.session.messages.len() > 2 {
            let result = self.compact_with_api_fallback(
                CompactionConfig {
                    max_estimated_tokens: 0,
                    preserve_recent_messages: 2,
                },
                None,
            );
            if result.removed_message_count > 0 {
                self.session = result.compacted_session;
            }
        }
        if let Some(t) = guard_t {
            log_build_segment("overflow_guard_compaction (BLOCKING LLM)", t);
        }
    }

    /// Assemble the request snapshot from current state. Pure and cheap — no
    /// compaction, no LLM call, no deep clone — so it is safe to call on the TUI
    /// `select!` thread every streaming iteration. `wire_reminders` is passed in
    /// (already query-recalled by the caller) so a streaming caller computes it
    /// once and reuses it for both the overflow estimate and the request.
    pub(super) fn assemble_request(
        &self,
        wire_reminders: Arc<[String]>,
        tool_choice: Option<::api::ToolChoice>,
    ) -> ApiRequest {
        // `session.messages` 는 `Arc<Vec<_>>` 이므로 요청 스냅샷은 전체 메시지
        // 를 deep clone 하지 않고 `Arc::clone`(포인터 복사) 으로 공유한다.
        // 메시지 변경은 `Session` 이 `Arc::make_mut`(COW) 로 처리하므로 이
        // 스냅샷은 그 시점 이력의 불변 뷰로 안전하다. (종전의 길이-기반
        // 캐시 + `Arc::new(messages.clone())` deep clone 을 대체.)
        //
        // `tool_consistent_messages` 는 turn 이 mid-flight 로 취소돼 결과 없이
        // 남은 `tool_use`(고아) 를 합성 `tool_result` 로 봉인한 뷰를 돌려준다.
        // 봉인할 게 없으면 위 `Arc::clone` 그대로(제로 카피). 이게 없으면
        // 다음 요청이 `400 tool_use ... without tool_result` 로 거절돼 세션이
        // 영구 손상된다.
        ApiRequest {
            system_prompt: Arc::clone(&self.system_prompt),
            wire_reminders,
            messages: self.session.tool_consistent_messages(),
            tool_choice,
            // Effort floor for this turn (deep-gate escalation); `None` on an
            // ordinary turn leaves the client's configured effort unchanged.
            effort_override: self.effort_override,
            // Per-turn wire-model override: the refusal fallback wins over a
            // confidence-cascade escalation (a refusal on the escalated model
            // must still swap to the safe fallback); `None` on an ordinary
            // turn leaves the client's bound model in use.
            //
            // Suppressed entirely whenever this request will NOT dispatch on
            // the main bound client: a deep PLAN/VERIFY leg runs on its swapped
            // client, and a quota-fallback turn runs on a different provider's
            // client — an override from
            // the main model's world riding either wire is a foreign model
            // id (400/404) or silently hijacks the verifier.
            model_override: if self.deep_plan_leg_active
                || self.deep_verify_leg_active
                || self.quota_fallback_active
            {
                None
            } else {
                self.refusal_fallback_model
                    .clone()
                    .or_else(|| self.escalation_model_override.clone())
            },
        }
    }

    pub(super) fn record_assistant_iteration(
        &self,
        iteration: usize,
        assistant_message: &ConversationMessage,
        pending_tool_use_count: usize,
    ) {
        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        session_tracer.record(
            "assistant_iteration_completed",
            trace_attrs(json!({
                "iteration": iteration,
                "assistant_blocks": assistant_message.blocks.len(),
                "pending_tool_use_count": pending_tool_use_count,
            })),
        );
    }

    pub(super) fn record_tool_started(&self, iteration: usize, tool_name: &str) {
        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        session_tracer.record(
            "tool_execution_started",
            trace_attrs(json!({"iteration": iteration, "tool_name": tool_name})),
        );
        session_tracer.record_security_audit(
            "tool_execution_started",
            trace_attrs(json!({"iteration": iteration, "tool_name": tool_name})),
        );
    }

    pub(super) fn record_tool_finished(
        &self,
        iteration: usize,
        result_message: &ConversationMessage,
    ) {
        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        let Some(ContentBlock::ToolResult {
            tool_name,
            output,
            is_error,
            ..
        }) = result_message.blocks.first()
        else {
            return;
        };
        session_tracer.record(
            "tool_execution_finished",
            trace_attrs(json!({
                "iteration": iteration,
                "tool_name": tool_name,
                "is_error": is_error,
            })),
        );
        let mut audit_attrs = trace_attrs(json!({
            "iteration": iteration,
            "tool_name": tool_name,
            "is_error": is_error,
        }));
        if *is_error {
            if let Some(preview) = tool_error_preview(output) {
                audit_attrs.insert("error_preview".to_string(), Value::String(preview));
            }
        }
        session_tracer.record_security_audit("tool_execution_finished", audit_attrs);
    }

    pub(super) fn record_turn_completed(&mut self, summary: &TurnSummary) {
        // Externalize the turn into the durable, compaction-proof trace under
        // `.zo/turns/` (Harness-1: state lives outside the context window).
        // Best-effort: a recording failure must never affect the turn. This runs
        // regardless of whether an OTLP tracer is attached — the durable trace is
        // the audit trail, independent of telemetry export. Rooted at the
        // session's stable workspace so it survives `EnterWorktree` chdirs.
        if let Some(cwd) = self.trace_cwd() {
            let _ = crate::turn_trace::record_completed(
                &cwd,
                &self.session.session_id,
                summary,
                self.session.session_goal.as_deref(),
            );
        }

        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        session_tracer.record(
            "turn_completed",
            trace_attrs(json!({
                "iterations": summary.iterations,
                "assistant_messages": summary.assistant_messages.len(),
                "tool_results": summary.tool_results.len(),
                "prompt_cache_events": summary.prompt_cache_events.len(),
                // Token counts feed the OTLP `zo_code.token.usage` metric
                // (CC monitoring parity) — keep the attr names stable.
                "input_tokens": summary.usage.input_tokens,
                "output_tokens": summary.usage.output_tokens,
                "cache_read_input_tokens": summary.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": summary.usage.cache_creation_input_tokens,
            })),
        );
    }

    pub(super) fn record_turn_failed(&mut self, iteration: usize, error: &RuntimeError) {
        if let Some(cwd) = self.trace_cwd() {
            let _ = crate::turn_trace::record_terminal(
                &cwd,
                &self.session.session_id,
                crate::turn_trace::TurnOutcome::Failed,
                iteration,
                self.session.session_goal.as_deref(),
            );
            // Segment the candidate by error signature (bounded class + status
            // + provider code), not by one fixed summary: the summary keys the
            // candidate id, and a single "turn failed" bucket mixed every root
            // cause into one candidate no advisor could act on. The evidence
            // detail keeps the leading error text (bounded by the recorder) so
            // the fusion root-cause advisor has concrete signals to cite.
            let signature = decision_core::dreamer::error_signature_label(
                error.failure_signature(),
                &error.to_string(),
            );
            let _ = crate::memory::record_self_improve_pulse_if_enabled(
                self.dream_automation_enabled,
                &cwd,
                decision_core::dreamer::CandidateKind::TurnFailure,
                &self.session.session_id,
                "turn",
                &format!("turn failure: {signature}"),
                &error.to_string(),
                false,
            );
        }

        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        session_tracer.record(
            "turn_failed",
            trace_attrs(json!({"iteration": iteration, "error": error.to_string()})),
        );
    }

    fn record_turn_stopped(
        &mut self,
        iteration: usize,
        reason: &str,
        origin: TurnStopOrigin,
    ) {
        if let Some(cwd) = self.trace_cwd() {
            let _ = crate::turn_trace::record_terminal(
                &cwd,
                &self.session.session_id,
                origin.trace_outcome(),
                iteration,
                self.session.session_goal.as_deref(),
            );
            let _ = crate::memory::record_self_improve_pulse_if_enabled(
                self.dream_automation_enabled,
                &cwd,
                origin.candidate_kind(),
                &self.session.session_id,
                "turn",
                origin.summary(),
                reason,
                false,
            );
        }

        let Some(session_tracer) = &self.session_tracer else {
            return;
        };
        session_tracer.record(
            origin.trace_event(),
            trace_attrs(json!({"iteration": iteration, "reason": reason})),
        );
    }

    pub(super) fn record_turn_cancelled(&mut self, iteration: usize, reason: &str) {
        self.record_turn_stopped(iteration, reason, TurnStopOrigin::User);
    }

    pub(super) fn record_turn_host_failure(&mut self, iteration: usize, reason: &str) {
        self.record_turn_stopped(iteration, reason, TurnStopOrigin::HostFailure);
    }

    /// Trace a host-side consumer failure and yield the matching streaming
    /// cancellation error. Explicit abort signals use `record_turn_cancelled`
    /// instead, so only intentional cancellation becomes non-actionable.
    pub(super) fn cancel_turn(&mut self, iteration: usize, reason: &str) -> StreamingTurnError {
        self.record_turn_host_failure(iteration, reason);
        StreamingTurnError::Cancelled
    }
}

fn tool_error_preview(output: &str) -> Option<String> {
    const MAX_CHARS: usize = 240;

    let preview = output
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(output)
        .trim();
    if preview.is_empty() {
        return None;
    }
    Some(preview.chars().take(MAX_CHARS).collect())
}

/// Emit one `ZO_PROFILE_TURN` segment line for a `build_request` sub-step
/// that took a non-trivial amount of wall-clock time. Kept here (not inline) so
/// `build_request` reads as orchestration and the >=10ms noise floor lives in
/// one place. Only ever called when profiling is already enabled, so the format
/// cost is off the hot path.
fn log_build_segment(label: &str, started_at: std::time::Instant) {
    let ms = started_at.elapsed().as_millis();
    if ms >= 10 {
        eprintln!("[TURN-SEG]   build_request/{label} = {ms}ms (synchronous; starves render_tick)");
    }
}

fn message_text(message: &ConversationMessage) -> Option<Cow<'_, str>> {
    let mut texts = message.blocks.iter().filter_map(|block| match block {
        ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.as_str()),
        _ => None,
    });
    let first = texts.next()?;
    let Some(second) = texts.next() else {
        return Some(Cow::Borrowed(first));
    };

    let mut joined = String::with_capacity(first.len() + second.len() + 1);
    joined.push_str(first);
    joined.push('\n');
    joined.push_str(second);
    for text in texts {
        joined.push('\n');
        joined.push_str(text);
    }
    Some(Cow::Owned(joined))
}

fn is_short_follow_up(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_lowercase();
    matches!(
        lower.as_str(),
        "y" | "yes" | "ok" | "okay" | "continue" | "계속" | "ㅇㅇ"
    ) || is_numeric_choice(&lower)
}

fn is_numeric_choice(text: &str) -> bool {
    let digits = text.strip_suffix("번").unwrap_or(text);
    matches!(digits, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
}

#[cfg(test)]
mod tests {
    use super::TurnStopOrigin;

    #[test]
    fn stop_origin_keeps_user_cancellation_separate_from_host_failure() {
        assert_eq!(
            TurnStopOrigin::User.candidate_kind(),
            decision_core::dreamer::CandidateKind::UserCancelled
        );
        assert!(!TurnStopOrigin::User.candidate_kind().is_actionable());
        assert_eq!(
            TurnStopOrigin::HostFailure.candidate_kind(),
            decision_core::dreamer::CandidateKind::TurnFailure
        );
        assert!(TurnStopOrigin::HostFailure.candidate_kind().is_actionable());
        assert_eq!(
            TurnStopOrigin::User.trace_outcome(),
            crate::turn_trace::TurnOutcome::Cancelled
        );
        assert_eq!(
            TurnStopOrigin::HostFailure.trace_outcome(),
            crate::turn_trace::TurnOutcome::Failed
        );
    }
}
