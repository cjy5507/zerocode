//! The streaming turn loop for [`ConversationRuntime`] — the `RenderBlock`-emitting
//! counterpart to the synchronous `run_turn` path in `mod.rs`. Split out verbatim
//! so `mod.rs` reads as the sync orchestrator; behaviour-preserving. The three
//! empty-stream helpers here are `pub(super)` because the sync loop in `mod.rs`
//! shares them.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::{
    ask_user_question_async, budget_exhausted_notice, build_assistant_message,
    build_async_permission_request, collect_pending_tool_uses, denial_banner,
    empty_stream_exhausted_message, format_tool_result_from_raw, is_edit_or_write_tool,
    is_refusal_stop_reason, is_truncation_stop_reason, is_verify_class_tool, merge_hook_feedback,
    normalize_empty_assistant_stream,
    parallel_safe_tool_indices, pre_hook_denial_outcome, quota_fallback_prearm_info,
    quota_fallback_swap_warn, quota_wait_hold_warn, refusal_surfaced_message,
    sleep_tool_execution_input,
    agent_notification_text, steering_message, tool_execution_input,
    take_truncation_continuation, tool_preview_from,
    tool_result_message, tool_summary_line, unblock_tool_execute, ApiClient, AssistantEvent,
    AssistantTurn, AsyncPermissionDecision, AsyncPermissionPrompter, BlockIdGen, BudgetExhausted,
    CapturePrompter, ContentBlock, ConversationMessage, ConversationRuntime, EmptyAssistantAction,
    HookEvent, HookRunResult, PermissionContext, PermissionOutcome, PermissionPromptDecision,
    PromptCacheEvent, QuotaEscape, RefusalDecision, RenderBlock, RuntimeError, StreamingTurnError,
    SystemLevel,
    TokenUsage, ToolBatchRepetitionHardStops, ToolCallId, ToolCallStatus, ToolExecutor, TurnSummary,
    EMPTY_STREAM_CONTINUATION_REMINDER, EMPTY_STREAM_CONTINUATION_REMINDER_PREFIX,
    EMPTY_STREAM_EXHAUSTED_FALLBACK_TEXT, EMPTY_STREAM_RETRY_REMINDER,
    EMPTY_STREAM_RETRY_REMINDER_PREFIX, EMPTY_STREAM_TRUNCATION_RETRY_REMINDER,
    MAX_EMPTY_STREAM_RETRIES, MAX_PARALLEL_SAFE_TOOL_DISPATCHES, REFUSAL_DRY_PREARM_WARN,
    REFUSAL_FALLBACK_WARN, REFUSAL_SURFACED_NOTICE, STEERING_ECHO_PREFIX,
    TRUNCATION_CONTINUATION_REMINDER,
};

// Rough token estimate for the system prompt sections.
//
// Uses the same heuristic as `estimate_message_tokens` (chars / 4) so
// that `build_request`'s overflow guard accounts for the system prompt
// ============================================================================
// Streaming variant — L7a
// ============================================================================
//
// Section header cites `.zo/code-rules.md` R-numbers that this block
// satisfies:
//
// * R1 (RenderBlock is the only currency that crosses the tui boundary).
// * R3 (no `block_on` / `block_in_place` inside an async context — we
//   reach the async prompter exclusively via `.await`).
// * R6 (stream pipeline errors propagate, never panic, never silently
//   drop).
// * R8 (bounded mpsc channels only — see
//   [`DEFAULT_STREAMING_CHANNEL_CAPACITY`]).

struct PreparedStreamingTool {
    tool_use_id: String,
    tool_name: String,
    effective_input: String,
    pre_hook_result: HookRunResult,
    permission_outcome: PermissionOutcome,
    tool_call_id: ToolCallId,
}

struct PrecomputedStreamingToolResult {
    output: String,
    is_error: bool,
    tool_start: std::time::Instant,
}

struct StreamingToolRenderContext<'a> {
    render_tx: &'a mpsc::Sender<RenderBlock>,
    id_gen: &'a BlockIdGen,
    rollback_message_count: usize,
}

struct StreamingToolFinalizeOptions {
    tool_start: std::time::Instant,
    render_result: bool,
    notify_slow: bool,
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    /// Streaming variant of [`ConversationRuntime::run_turn`].
    ///
    /// Emits [`RenderBlock`] values into `render_tx` as the agent loop
    /// progresses (text deltas, tool-call cards, tool results, system
    /// notices) and returns a [`TurnSummary`] on completion — matching
    /// the logical sequence the synchronous `run_turn` path produces
    /// for identical inputs.
    ///
    /// # Permission model
    ///
    /// When the permission policy needs human approval, the loop
    /// awaits `prompter.decide(..)`. Any
    /// [`crate::permission::PermissionPrompter`] implementation works;
    /// the canonical choice is L3's `ChannelPrompter`, which forwards
    /// requests over a bounded mpsc to the TUI event loop. No
    /// modification to L3 was required.
    ///
    /// # Cancellation
    ///
    /// If the consumer drops the `RenderBlock` receiver at any point,
    /// the next send fails with a closed-channel error and the loop
    /// returns [`StreamingTurnError::Cancelled`]. No new tool dispatch
    /// is started after cancellation is observed; an already-running
    /// tool will finish its in-flight execution before the loop
    /// notices the drop, but its result is simply discarded.
    ///
    /// # Backpressure
    ///
    /// `render_tx` is bounded (see [`DEFAULT_STREAMING_CHANNEL_CAPACITY`]).
    /// If the TUI cannot keep up, `send().await` suspends the loop
    /// until the consumer drains the channel, giving honest end-to-end
    /// backpressure per code-rules R8.
    ///
    /// # Relationship to sync `run_turn`
    ///
    /// This remains an additive, parallel implementation: the synchronous
    /// [`ConversationRuntime::run_turn`] and this streaming loop stay as
    /// separate orchestration paths so the non-TTY `-p` CLI path and the
    /// `RenderBlock` emission boundary do not get semantically merged. Small
    /// phase helpers are shared only where the two paths are byte-identical or
    /// trivially parameterized (empty-response cleanup/retry state, truncation
    /// continuation gating, pre-hook denial mapping, concurrency-safe batch
    /// selection, and final tool-result message construction). Streaming-only
    /// rendering, async permission prompts, request assembly, cancellation, and
    /// steering remain local to this loop. See
    /// `.zo/tasks/L7a-runtime-streaming.handoff.md` for the original
    /// rationale behind the "duplicate the loop body" decision.
    // cohesive async streaming turn; loop body intentionally parallel to the
    // sync path with only identical/parameterized phase helpers shared
    pub async fn run_turn_streaming(
        &mut self,
        user_input: impl Into<String>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        self.run_turn_streaming_with_images(user_input, Vec::new(), render_tx, prompter)
            .await
    }

    /// Like [`run_turn_streaming`] but with optional image attachments.
    ///
    /// Each image is a `(media_type, base64_data)` pair prepended to the
    /// user message before the text block.
    // cohesive async streaming turn (image-carrying variant); see the sibling
    // run_turn_streaming note on the intentional loop-body duplication
    #[allow(clippy::too_many_lines)]
    /// Clear the transient "empty stream retry" system reminder if one was set
    /// this turn. No-op when `empty_retries == 0`; shared by the many early-exit
    /// paths of both turn loops so the cleanup lives in one place.
    pub(super) fn clear_empty_retry_reminder(&mut self, empty_retries: usize) {
        if empty_retries > 0 {
            self.replace_transient_system_reminder_by_prefix(
                EMPTY_STREAM_RETRY_REMINDER_PREFIX,
                None,
            );
        }
    }

    pub(super) fn accept_assistant_content_turn(
        &mut self,
        empty_retries: &mut usize,
        empty_recovery_attempted: &mut bool,
        message: ConversationMessage,
        usage: Option<TokenUsage>,
        prompt_cache_events: Vec<PromptCacheEvent>,
        stop_reason: Option<String>,
    ) -> (
        ConversationMessage,
        Option<TokenUsage>,
        Vec<PromptCacheEvent>,
        Option<String>,
    ) {
        self.clear_empty_retry_reminder(*empty_retries);
        self.replace_transient_system_reminder_by_prefix(
            EMPTY_STREAM_CONTINUATION_REMINDER_PREFIX,
            None,
        );
        *empty_retries = 0;
        *empty_recovery_attempted = false;
        (message, usage, prompt_cache_events, stop_reason)
    }

    pub(super) fn handle_empty_assistant_turn(
        &mut self,
        usage: Option<TokenUsage>,
        stop_reason: Option<&str>,
        empty_retries: &mut usize,
        empty_recovery_attempted: &mut bool,
    ) -> EmptyAssistantAction {
        if let Some(usage) = usage {
            self.usage_tracker.record(usage);
        }
        if *empty_retries < MAX_EMPTY_STREAM_RETRIES {
            *empty_retries += 1;
            let reminder = if stop_reason.is_some_and(is_truncation_stop_reason) {
                EMPTY_STREAM_TRUNCATION_RETRY_REMINDER
            } else {
                EMPTY_STREAM_RETRY_REMINDER
            };
            self.replace_transient_system_reminder_by_prefix(
                EMPTY_STREAM_RETRY_REMINDER_PREFIX,
                Some(reminder),
            );
            return EmptyAssistantAction::Retry;
        }
        self.replace_transient_system_reminder_by_prefix(
            EMPTY_STREAM_CONTINUATION_REMINDER_PREFIX,
            Some(EMPTY_STREAM_CONTINUATION_REMINDER),
        );
        self.replace_transient_system_reminder_by_prefix(EMPTY_STREAM_RETRY_REMINDER_PREFIX, None);
        if *empty_recovery_attempted {
            EmptyAssistantAction::Exhausted
        } else {
            *empty_recovery_attempted = true;
            *empty_retries = 0;
            EmptyAssistantAction::ContinueOnce
        }
    }

    /// Turn-start prologue for [`Self::run_turn_streaming_with_images`]: record
    /// turn-started telemetry, capture the output-token baseline and the pre-turn
    /// message index (for rollback on an early failure), push the user input
    /// (text-only or image-carrying), and reset the per-turn transient state.
    ///
    /// Returns `(turn_start_output_tokens, message_count_before)`. The streaming
    /// loop differs from the sync [`Self::begin_turn_once`] in that it must also
    /// expose `message_count_before` so a first-iteration stream failure can
    /// truncate the orphaned user message back off the session.
    pub(super) fn begin_streaming_turn(
        &mut self,
        user_input: String,
        images: Vec<(String, String)>,
        internal_subturn: bool,
    ) -> Result<(u32, usize), StreamingTurnError> {
        if !internal_subturn {
            self.run_user_prompt_submit_for_streaming_user_entry(&user_input)?;
        }
        self.record_turn_started(&user_input);
        let turn_start_output_tokens = self.usage_tracker.cumulative_usage().output_tokens;
        let message_count_before = self.session.messages.len();
        if images.is_empty() {
            self.session
                .push_user_text(user_input)
                .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
        } else {
            self.session
                .push_user_with_images(user_input, images)
                .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
        }

        self.tool_fingerprint_counts.clear();
        self.tool_repetition_pending_hard_stop_fps.clear();
        self.tool_repetition_hard_stop_fps.clear();
        self.read_file_ranges_by_path.clear();
        self.read_file_redundant_advised_paths.clear();
        self.cross_turn_tool_repetition_pending_hard_stop_fps.clear();
        self.mode_denial_counts.clear();
        self.tool_loop_break_requested = false;
        self.verify_treadmill_run = 0;
        self.full_compactions_this_turn = 0;
        // Fold refusal history only at a PUBLIC boundary. Internal subturns are
        // additional legs of the same user turn and must not double-count it.
        if !internal_subturn {
            self.fold_finished_refusal_turn();
        }
        // Reset the per-leg refusal override before deciding whether the
        // session cooldown should re-arm it below.
        self.refusal_fallback_model = None;
        // Reset the per-turn quota fallback, pre-arming onto it when the session
        // is still inside a recorded quota-dry cooldown (applies to internal
        // subturns too — a quota-dry session applies to every leg). See
        // [`Self::begin_turn_quota_fallback`].
        self.begin_turn_quota_fallback();
        // Escalation freshness: keep a freshly-installed confidence-cascade
        // escalation for THIS turn, drop a stale one from a prior turn.
        // Internal deep-lane subturns run inside the escalated turn and are
        // exempt. See [`Self::begin_turn_escalation`].
        self.begin_turn_escalation(internal_subturn);
        // Refusal-dry sees the would-be wire model only after the ordinary
        // refusal reset and quota/escalation state have settled. It applies to
        // every leg, even though only public begins fold the streak above.
        self.begin_turn_refusal_fallback();
        // A fresh PUBLIC user turn (not an internal deep-lane subturn) is genuine
        // new intent: reset the cross-turn re-read tally so legitimately
        // re-reading the same file across separate user-driven turns never
        // accumulates into a false cross-turn stop. Internal subturns
        // (auto-continuation) deliberately KEEP the tally so their own re-read
        // loop — the one this guard exists to catch — still trips it.
        if !internal_subturn {
            self.cross_turn_tool_fingerprints.clear();
            self.cross_turn_tool_repetition_pending_hard_stop_fps.clear();
            self.cross_turn_tool_repetition_hard_stop_fps.clear();
        }
        // NOTE: `consecutive_microcompacts` is intentionally NOT reset here — it
        // is a cross-turn thrash signal that must survive turn boundaries — see
        // `begin_turn_once`.
        Ok((turn_start_output_tokens, message_count_before))
    }

    pub(super) fn cancel_streaming_turn(
        &mut self,
        iteration: usize,
        reason: &str,
        message_count_before: usize,
    ) -> StreamingTurnError {
        let error = self.cancel_turn(iteration, reason);
        Arc::make_mut(&mut self.session.messages).truncate(message_count_before);
        self.session.mark_transcript_dirty();
        error
    }

    fn settle_streaming_abort(
        &mut self,
        iteration: usize,
        reason: &str,
        message_count_before: usize,
    ) -> StreamingTurnError {
        if !self.hook_abort_signal.is_handled() {
            match self.hook_abort_signal.origin() {
                Some(crate::HookAbortOrigin::User) => {
                    self.record_turn_cancelled(iteration, reason);
                }
                Some(crate::HookAbortOrigin::Host) | None => {
                    self.record_turn_host_failure(iteration, reason);
                }
            }
            self.hook_abort_signal.mark_handled();
        }
        Arc::make_mut(&mut self.session.messages).truncate(message_count_before);
        self.session.mark_transcript_dirty();
        StreamingTurnError::Cancelled
    }

    fn cancel_streaming_turn_if_aborted(
        &mut self,
        iteration: usize,
        reason: &str,
        message_count_before: usize,
    ) -> Option<StreamingTurnError> {
        self.hook_abort_signal
            .is_aborted()
            .then(|| self.settle_streaming_abort(iteration, reason, message_count_before))
    }

    /// Cancel a streaming turn at an explicit user/host boundary while the
    /// caller owns the runtime future. This is the public counterpart to the
    /// render-channel failure path above: it preserves typed cancellation
    /// provenance and rolls back the partial turn to the host-captured baseline.
    pub fn cancel_streaming_turn_by_user(
        &mut self,
        reason: &str,
        message_count_before: usize,
    ) -> StreamingTurnError {
        self.hook_abort_signal.abort();
        self.settle_streaming_abort(0, reason, message_count_before)
    }

    /// Abort a streaming turn because its host failed independently of the
    /// user. Keeping this distinct prevents transport/render failures from being
    /// filtered as non-actionable user cancellations by Dreamer.
    pub fn cancel_streaming_turn_by_host(
        &mut self,
        reason: &str,
        message_count_before: usize,
    ) -> StreamingTurnError {
        self.hook_abort_signal.abort_host();
        self.settle_streaming_abort(0, reason, message_count_before)
    }

    #[allow(clippy::too_many_lines)] // cohesive streaming-turn core: ingest → stream → tool-loop → settle, one scope
    pub async fn run_turn_streaming_with_images(
        &mut self,
        user_input: impl Into<String>,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        let result = self
            .run_turn_streaming_with_images_inner(user_input, images, render_tx, prompter, false)
            .await;
        self.settle_team_inbox_turn_for_result(&result);
        result
    }

    pub(crate) async fn run_internal_subturn_streaming_with_images(
        &mut self,
        user_input: impl Into<String>,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        self.run_turn_streaming_with_images_inner(user_input, images, render_tx, prompter, true)
            .await
    }

    #[allow(clippy::too_many_lines)] // cohesive streaming-turn core: ingest → stream → tool-loop → settle, one scope
    async fn run_turn_streaming_with_images_inner(
        &mut self,
        user_input: impl Into<String>,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
        internal_subturn: bool,
    ) -> Result<TurnSummary, StreamingTurnError> {
        let id_gen = BlockIdGen::default();
        let user_input = user_input.into();
        // Baseline token count + pre-turn message index for rollback (see
        // [`Self::begin_streaming_turn`]). Internal deep-lane subturns carry
        // program-generated prompts, so they preserve the pre-A1 streaming
        // lifecycle behavior: no TurnStart or UserPromptSubmit policy.
        let (turn_start_output_tokens, message_count_before) =
            self.begin_streaming_turn(user_input, images, internal_subturn)?;
        // Input-side baseline for the third cost breaker; captured here (after
        // `begin_streaming_turn`, which makes no provider call) so both token
        // baselines describe the same instant.
        let turn_start_input_tokens = self.usage_tracker.cumulative_usage().input_tokens;

        let mut assistant_messages = Vec::new();
        let mut tool_results = Vec::new();
        let mut prompt_cache_events = Vec::new();
        let mut iterations = 0;
        let mut tool_calls = 0;
        let mut empty_retries = 0;
        let mut empty_recovery_attempted = false;
        let mut truncation_continuations = 0;
        let mut turn_end_gate_reprompts = 0;
        let mut auto_compaction = None;
        let mut provider_overflow_recovery_attempted = false;
        let mut microcompact = None;
        let mut budget_exhausted: Option<BudgetExhausted> = None;
        // Progress-gated deadline extensions granted so far this turn, and the
        // progress-result count at the last grant — each further extension
        // requires FRESH progress since the previous window, so a turn that
        // stops externalizing progress stops earning extensions.
        let mut deadline_extensions_used: u8 = 0;
        let mut extension_progress_marker: usize = 0;

        'outer: loop {
            if self.tool_loop_break_requested {
                break 'outer;
            }
            iterations += 1;
            if iterations > self.max_iterations {
                let error = RuntimeError::new(
                    "conversation loop exceeded the maximum number of iterations",
                );
                self.clear_empty_retry_reminder(empty_retries);
                self.record_turn_failed(iterations, &error);
                // Budget exhausted, not failed: this boundary is only reached
                // after a prior iteration closed the session well-formed
                // (assistant `tool_use` + user tool-results), so preserve the
                // work instead of rolling back. Append a synthetic closer, warn
                // on the render channel, and end the turn Ok(..) with the budget
                // marker so the caller (or user) can continue in a follow-up.
                budget_exhausted = Some(BudgetExhausted::Iterations);
                self.push_budget_exhausted_closer(
                    BudgetExhausted::Iterations,
                    iterations,
                    &mut assistant_messages,
                )
                .map_err(StreamingTurnError::runtime)?;
                // Streaming notice (sync path uses `eprintln`). Inlined — not a
                // `&self` helper — so the send future never holds `&self` across
                // the await, which the spawned turn's `Send` bound forbids.
                let _ = render_tx
                    .send(RenderBlock::System {
                        id: id_gen.next(),
                        level: SystemLevel::Warn,
                        text: budget_exhausted_notice(BudgetExhausted::Iterations, iterations),
                    })
                    .await;
                break 'outer;
            }

            // Wall-clock budget: a spawned sub-agent bounds a straggler, and the
            // interactive host sets `now + turn budget` each turn as the runaway
            // circuit breaker. Checked at the iteration boundary — cooperative,
            // but the per-chunk stream idle timeout bounds the in-flight read so
            // this check is always reached. Same as the iteration cap above:
            // preserve the well-formed work and end Ok(..) with the budget marker.
            if self
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                // Progress-gated extension: a turn that is demonstrably still
                // externalizing progress (fresh successful edit/write/plan
                // results since the last window) earns a bounded deadline push
                // instead of a mid-work stop — the legitimate long audit or
                // deploy pipeline the blunt 60-minute cut kept interrupting.
                // Fake-progress grinds are still bounded: the extension count
                // is capped, reads/probes don't count as progress, and the
                // cross-turn escalation ladder catches the repeat pattern.
                let extension_granted = match self.deadline_extension {
                    Some((max_extensions, step))
                        if deadline_extensions_used < max_extensions
                            && super::count_progress_tool_results(&tool_results)
                                > extension_progress_marker =>
                    {
                        deadline_extensions_used += 1;
                        extension_progress_marker =
                            super::count_progress_tool_results(&tool_results);
                        self.deadline = Some(std::time::Instant::now() + step);
                        let _ = render_tx
                            .send(RenderBlock::System {
                                id: id_gen.next(),
                                level: SystemLevel::Info,
                                text: format!(
                                    "[budget] deadline extended +{}m ({deadline_extensions_used}/{max_extensions}) — fresh progress detected",
                                    step.as_secs() / 60
                                ),
                            })
                            .await;
                        true
                    }
                    _ => false,
                };
                if !extension_granted {
                    let error = RuntimeError::new("agent exceeded its time budget");
                    self.clear_empty_retry_reminder(empty_retries);
                    self.record_turn_failed(iterations, &error);
                    budget_exhausted = Some(BudgetExhausted::Deadline);
                    self.push_budget_exhausted_closer(
                        BudgetExhausted::Deadline,
                        iterations,
                        &mut assistant_messages,
                    )
                    .map_err(StreamingTurnError::runtime)?;
                    let _ = render_tx
                        .send(RenderBlock::System {
                            id: id_gen.next(),
                            level: SystemLevel::Warn,
                            text: budget_exhausted_notice(BudgetExhausted::Deadline, iterations),
                        })
                        .await;
                    break 'outer;
                }
            }

            // Output-token budget (cost circuit breaker): the in-turn output
            // delta from turn start. Bounds an agentic loop that keeps generating
            // (re-planning an unachievable goal, re-invoking Workflow/agents)
            // without converging — the multi-day-runaway case the iteration cap
            // misses when a few iterations each fan out huge multi-agent work.
            if self.turn_output_token_budget.is_some_and(|budget| {
                self.usage_tracker
                    .cumulative_usage()
                    .output_tokens
                    .saturating_sub(turn_start_output_tokens)
                    > budget
            }) {
                let error = RuntimeError::new("turn exceeded its output-token budget");
                self.clear_empty_retry_reminder(empty_retries);
                self.record_turn_failed(iterations, &error);
                budget_exhausted = Some(BudgetExhausted::OutputTokens);
                self.push_budget_exhausted_closer(
                    BudgetExhausted::OutputTokens,
                    iterations,
                    &mut assistant_messages,
                )
                .map_err(StreamingTurnError::runtime)?;
                let _ = render_tx
                    .send(RenderBlock::System {
                        id: id_gen.next(),
                        level: SystemLevel::Warn,
                        text: budget_exhausted_notice(BudgetExhausted::OutputTokens, iterations),
                    })
                    .await;
                break 'outer;
            }

            // Input-token budget (cache-miss cost circuit breaker): the
            // in-turn full-price-input delta from turn start. Bounds a
            // cache-dead loop that re-sends the whole transcript uncached on
            // every call — millions of input tokens with little output,
            // invisible to the output breaker above and comfortably inside
            // the wall clock.
            if self.turn_input_token_budget.is_some_and(|budget| {
                self.usage_tracker
                    .cumulative_usage()
                    .input_tokens
                    .saturating_sub(turn_start_input_tokens)
                    > budget
            }) {
                let error = RuntimeError::new("turn exceeded its input-token budget");
                self.clear_empty_retry_reminder(empty_retries);
                self.record_turn_failed(iterations, &error);
                budget_exhausted = Some(BudgetExhausted::InputTokens);
                self.push_budget_exhausted_closer(
                    BudgetExhausted::InputTokens,
                    iterations,
                    &mut assistant_messages,
                )
                .map_err(StreamingTurnError::runtime)?;
                let _ = render_tx
                    .send(RenderBlock::System {
                        id: id_gen.next(),
                        level: SystemLevel::Warn,
                        text: budget_exhausted_notice(BudgetExhausted::InputTokens, iterations),
                    })
                    .await;
                break 'outer;
            }

            if render_tx.is_closed() {
                self.clear_empty_retry_reminder(empty_retries);
                return Err(self.cancel_streaming_turn(
                    iterations,
                    "render channel closed before request",
                    message_count_before,
                ));
            }

            // This turn pre-armed onto the quota fallback from the session
            // cooldown: announce it once, before the first request, so the user
            // knows why the model differs without re-spending the main model's
            // retry budget. Consumed here so it fires exactly once per turn.
            if self.quota_prearm_notice_pending {
                self.quota_prearm_notice_pending = false;
                if let Some((_, model)) = self.quota_fallback_client.as_ref() {
                    let text = quota_fallback_prearm_info(model);
                    let _ = render_tx
                        .send(RenderBlock::System {
                            id: id_gen.next(),
                            level: SystemLevel::Info,
                            text,
                        })
                        .await;
                }
            }
            if self.refusal_prearm_notice_pending {
                self.refusal_prearm_notice_pending = false;
                let _ = render_tx
                    .send(RenderBlock::System {
                        id: id_gen.next(),
                        level: SystemLevel::Warn,
                        text: REFUSAL_DRY_PREARM_WARN.to_string(),
                    })
                    .await;
            }

            // Proactive compaction: compact *before* building the request.
            // The first iteration must use a local request estimate rather than
            // stale provider usage from the previous turn; later iterations use
            // fresh provider usage plus the local estimate. This prevents a
            // resumed or fast-growing session from jumping past the 85% trigger
            // before the runtime has a chance to shrink it.
            if iterations == 1 {
                if let Some(event) = self
                    .maybe_microcompact_preflight_streaming(&render_tx, &id_gen)
                    .await
                {
                    microcompact.get_or_insert(event);
                }
                if let Some(event) = self
                    .maybe_auto_compact_preflight_streaming(&render_tx, &id_gen)
                    .await
                {
                    auto_compaction.get_or_insert(event);
                } else {
                    self.maybe_state_distill_preflight();
                    self.maybe_precompaction_warn_streaming(&render_tx, &id_gen)
                        .await;
                }
            } else {
                if let Some(event) = self.maybe_microcompact_streaming(&render_tx, &id_gen).await {
                    microcompact.get_or_insert(event);
                }
                if let Some(event) = self.maybe_auto_compact_streaming(&render_tx, &id_gen).await {
                    auto_compaction.get_or_insert(event);
                } else {
                    self.maybe_state_distill();
                    self.maybe_precompaction_warn_streaming(&render_tx, &id_gen)
                        .await;
                }
            }

            // Assemble the request without re-running the synchronous overflow
            // guard: the async preflight/iteration compaction just above already
            // shrank the session to fit (it owns the same emergency
            // context-window seam), so calling the guard here would at best
            // duplicate that work and at worst make a *second* blocking LLM
            // summary round-trip on this very `select!` thread — freezing the
            // spinner/stream for the whole summary (the reported first-turn
            // freeze). `assemble_request` is pure (Arc clones only), so it is
            // safe to run every iteration on the render thread.
            let __req_t = std::time::Instant::now();
            // Recall runs in `spawn_blocking` (FREEZE-1): the dense ONNX embedding
            // forward pass no longer executes on this `select!` task, so it cannot
            // starve render_tick. The recall inputs are cloned out of `&self`
            // here, BEFORE the await, so the spawned turn future never holds a
            // `&self` borrow across it (the runtime is `Send` but not `Sync`).
            // `assemble_request` itself is pure (Arc clones only).
            let recall_section = Self::recall_reminder_section(
                self.memory_retriever.clone(),
                self.recall_query_text().map(std::borrow::Cow::into_owned),
                self.session_tracer.clone(),
            )
            .await;
            let mut wire_reminders = self.transient_reminders.clone();
            wire_reminders.extend(recall_section);
            let request = self.assemble_request(Arc::from(wire_reminders), None);
            // The compaction/resume status reminder has now ridden this request;
            // drop it so it does not re-instruct `session_recall` on every later
            // turn (a re-orientation loop). A fresh compaction re-seeds it, so it
            // still surfaces once per compaction event.
            self.drop_compaction_status_reminder();
            if __req_t.elapsed().as_millis() >= 50 && crate::turn_profiling_enabled() {
                eprintln!(
                    "[TURN-SEG] request_system_prompt+assemble = {}ms (recall off-thread)",
                    __req_t.elapsed().as_millis()
                );
            }
            let text_block_id = id_gen.next();
            let events_for_build: Vec<AssistantEvent> =
                if let Some(async_client) = self.active_async_client() {
                    // Live async path (L7c seam) with retry on transient errors.
                    // `active_async_client` routes through the cross-provider
                    // quota fallback for the rest of the turn once one has armed,
                    // so a native sync-only session (no `async_api_client`) still
                    // reaches this branch after a quota swap.
                    // The hook abort flag cuts a transient-error backoff short so
                    // a foreground Ctrl+C is observed during the wait, not only at
                    // the next iteration boundary (WI-G).
                    let notice_tx = render_tx.clone();
                    let notice_ids = id_gen.clone();
                    // The model this turn streams on, cloned out before the
                    // closures so a foreground 429 can feed the SAME per-provider
                    // cool-down state (`api::quota`) the sub-agent admission path
                    // reads — the foreground main turn used to ride out the window
                    // here without ever recording the throttle (quota-view/router
                    // gap).
                    let rate_limit_model = self
                        .rate_limit_model_for_active_stream()
                        .map(str::to_string);
                    let verifier_rate_limit_failover = self.deep_verify_leg_active
                        && self
                            .deep_verify_candidates
                            .get(self.deep_verify_candidate_idx)
                            .is_some_and(|(_, current_model)| {
                                let current_provider = api::detect_provider_kind(current_model);
                                self.deep_verify_candidates
                                    .iter()
                                    .skip(self.deep_verify_candidate_idx + 1)
                                    .any(|(_, candidate)| {
                                        api::detect_provider_kind(candidate) != current_provider
                                    })
                            });
                    let stream_result = crate::retry::retry_async(
                        "stream_async",
                        Some(self.hook_abort_signal.flag()),
                        verifier_rate_limit_failover,
                        |attempt, error: &RuntimeError| {
                            if let Some(model) = rate_limit_model.as_deref() {
                                crate::retry::mark_foreground_rate_limit(
                                    model,
                                    &error.to_string(),
                                    attempt,
                                );
                            }
                        },
                        |attempt, delay, error: &RuntimeError| {
                            // Keep the live UI honest while L2 backs off: a
                            // capacity stall (overload/429/5xx) is otherwise a
                            // silent multi-second pause before the next attempt,
                            // which reads as a freeze. Mirrors the api-level (L0)
                            // retry notice, but also fires for mid-stream errors
                            // the establish-time path never sees. The label
                            // vocabulary lives in `core_types::retry_signal` so
                            // it stays in lockstep with the backoff classifier.
                            let error_text = error.to_string();
                            let label = core_types::retry_signal::retry_notice_label(&error_text);
                            let _ = notice_tx.try_send(RenderBlock::System {
                                id: notice_ids.next(),
                                level: SystemLevel::Warn,
                                text: format!(
                                    "{label}; retrying in {}s (attempt {attempt})",
                                    delay.as_secs().max(1)
                                ),
                            });
                        },
                        |_attempt| {
                            let req = request.clone();
                            let tx = render_tx.clone();
                            let client = async_client.clone();
                            async move { client.stream_async(req, tx, text_block_id).await }
                        },
                    )
                    .await;
                    match stream_result {
                        Ok(events) => events,
                        Err(error) => {
                            if let Some(cancelled) = self.cancel_streaming_turn_if_aborted(
                                iterations,
                                "turn stopped by abort signal",
                                message_count_before,
                            ) {
                                self.clear_empty_retry_reminder(empty_retries);
                                return Err(cancelled);
                            }
                            if !provider_overflow_recovery_attempted
                                && error.provider_error_class()
                                    == Some(crate::ProviderErrorClass::ContextOverflow)
                            {
                                provider_overflow_recovery_attempted = true;
                                if let Some(event) = self
                                    .recover_provider_context_overflow_streaming(
                                        &render_tx, &id_gen,
                                    )
                                    .await
                                {
                                    auto_compaction.get_or_insert(event);
                                    continue 'outer;
                                }
                            }
                            // Main model's quota window is exhausted (RateLimit
                            // survived the retry budget): HOLD on the main model
                            // when its window lifts within the wait band, else
                            // swap to the cross-provider fallback — re-requesting
                            // this turn either way rather than killing it.
                            match self.decide_quota_escape(&error) {
                                QuotaEscape::Wait(wait) => {
                                    let model = self.context_model.clone().unwrap_or_default();
                                    let _ = render_tx
                                        .send(RenderBlock::System {
                                            id: id_gen.next(),
                                            level: SystemLevel::Warn,
                                            text: quota_wait_hold_warn(&model, wait),
                                        })
                                        .await;
                                    tokio::time::sleep(wait).await;
                                    continue 'outer;
                                }
                                QuotaEscape::Fallback(model) => {
                                    let _ = render_tx
                                        .send(RenderBlock::System {
                                            id: id_gen.next(),
                                            level: SystemLevel::Warn,
                                            text: quota_fallback_swap_warn(&model),
                                        })
                                        .await;
                                    continue 'outer;
                                }
                                QuotaEscape::None => {}
                            }
                            self.clear_empty_retry_reminder(empty_retries);
                            self.record_turn_failed(iterations, &error);
                            if iterations == 1 {
                                Arc::make_mut(&mut self.session.messages)
                                    .truncate(message_count_before);
                                self.session.mark_transcript_dirty();
                            }
                            return Err(StreamingTurnError::from(error));
                        }
                    }
                } else {
                    // Default legacy path: synchronous collect-then-replay.
                    let events = match self.api_client.stream(request) {
                        Ok(events) => events,
                        Err(error) => {
                            if let Some(cancelled) = self.cancel_streaming_turn_if_aborted(
                                iterations,
                                "turn stopped by abort signal",
                                message_count_before,
                            ) {
                                self.clear_empty_retry_reminder(empty_retries);
                                return Err(cancelled);
                            }
                            if !provider_overflow_recovery_attempted
                                && error.provider_error_class()
                                    == Some(crate::ProviderErrorClass::ContextOverflow)
                            {
                                provider_overflow_recovery_attempted = true;
                                if let Some(event) = self
                                    .recover_provider_context_overflow_streaming(
                                        &render_tx, &id_gen,
                                    )
                                    .await
                                {
                                    auto_compaction.get_or_insert(event);
                                    continue 'outer;
                                }
                            }
                            // Same escape as the async branch: a native sync-only
                            // session either holds on the main model (wait band)
                            // or swaps to the cross-provider fallback (reached
                            // next iteration via `active_async_client`).
                            match self.decide_quota_escape(&error) {
                                QuotaEscape::Wait(wait) => {
                                    let model = self.context_model.clone().unwrap_or_default();
                                    let _ = render_tx
                                        .send(RenderBlock::System {
                                            id: id_gen.next(),
                                            level: SystemLevel::Warn,
                                            text: quota_wait_hold_warn(&model, wait),
                                        })
                                        .await;
                                    tokio::time::sleep(wait).await;
                                    continue 'outer;
                                }
                                QuotaEscape::Fallback(model) => {
                                    let _ = render_tx
                                        .send(RenderBlock::System {
                                            id: id_gen.next(),
                                            level: SystemLevel::Warn,
                                            text: quota_fallback_swap_warn(&model),
                                        })
                                        .await;
                                    continue 'outer;
                                }
                                QuotaEscape::None => {}
                            }
                            self.clear_empty_retry_reminder(empty_retries);
                            self.record_turn_failed(iterations, &error);
                            if iterations == 1 {
                                Arc::make_mut(&mut self.session.messages)
                                    .truncate(message_count_before);
                                self.session.mark_transcript_dirty();
                            }
                            return Err(StreamingTurnError::from(error));
                        }
                    };

                    // Replay events into RenderBlocks while preserving
                    // them for `build_assistant_message`, which remains
                    // the source of truth for the session bookkeeping.
                    let mut text_emitted = false;
                    for event in &events {
                        if let AssistantEvent::TextDelta(delta) = &event {
                            text_emitted = true;
                            if render_tx
                                .send(RenderBlock::TextDelta {
                                    id: text_block_id,
                                    text: delta.clone(),
                                    done: false,
                                })
                                .await
                                .is_err()
                            {
                                self.clear_empty_retry_reminder(empty_retries);
                                return Err(self.cancel_streaming_turn(
                                    iterations,
                                    "render channel closed during text streaming",
                                    message_count_before,
                                ));
                            }
                        }
                    }
                    if text_emitted
                        && render_tx
                            .send(RenderBlock::TextDelta {
                                id: text_block_id,
                                text: String::new(),
                                done: true,
                            })
                            .await
                            .is_err()
                    {
                        self.clear_empty_retry_reminder(empty_retries);
                        return Err(self.cancel_streaming_turn(
                            iterations,
                            "render channel closed finalizing text",
                            message_count_before,
                        ));
                    }
                    events
                };

            let __ba_t = std::time::Instant::now();
            let __ba_result =
                build_assistant_message(normalize_empty_assistant_stream(events_for_build));
            if __ba_t.elapsed().as_millis() >= 50 && crate::turn_profiling_enabled() {
                eprintln!(
                    "[TURN-SEG] build_assistant_message = {}ms (synchronous; starves render_tick)",
                    __ba_t.elapsed().as_millis()
                );
            }
            // Anthropic safety-classifier refusal (`stop_reason: "refusal"`):
            // drop the refused partial (never pushed to history — it stays on
            // screen as the already-streamed deltas, but the retry renders under
            // a fresh block id) and either retry once on Opus 4.8 (Fable/Mythos)
            // with a warn line, or surface a notice and end (already fell back,
            // or a non-Fable model). Anthropic-only; a non-Anthropic model yields
            // `Proceed` and falls through unchanged.
            if is_refusal_stop_reason(__ba_result.stop_reason().unwrap_or_default()) {
                let refused_usage = __ba_result.usage();
                match self.decide_refusal_fallback() {
                    RefusalDecision::Retry => {
                        if let Some(usage) = refused_usage {
                            self.usage_tracker.record(usage);
                        }
                        let _ = render_tx
                            .send(RenderBlock::System {
                                id: id_gen.next(),
                                level: SystemLevel::Warn,
                                text: REFUSAL_FALLBACK_WARN.to_string(),
                            })
                            .await;
                        continue 'outer;
                    }
                    RefusalDecision::Surface => {
                        if let Some(usage) = refused_usage {
                            self.usage_tracker.record(usage);
                        }
                        let _ = render_tx
                            .send(RenderBlock::System {
                                id: id_gen.next(),
                                level: SystemLevel::Warn,
                                text: REFUSAL_SURFACED_NOTICE.to_string(),
                            })
                            .await;
                        let assistant_message = refusal_surfaced_message();
                        self.record_assistant_iteration(iterations, &assistant_message, 0);
                        self.session
                            .push_message(assistant_message)
                            .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
                        if let Some(msg) = self.session.messages.last().cloned() {
                            assistant_messages.push(msg);
                        }
                        break 'outer;
                    }
                    RefusalDecision::Proceed => {}
                }
            }
            let (assistant_message, usage, turn_prompt_cache_events, stop_reason) =
                match __ba_result {
                    AssistantTurn::Content {
                        message,
                        usage,
                        prompt_cache_events,
                        stop_reason,
                    } => self.accept_assistant_content_turn(
                        &mut empty_retries,
                        &mut empty_recovery_attempted,
                        message,
                        usage,
                        prompt_cache_events,
                        stop_reason,
                    ),
                    // Clean stop with no content (thinking-only / transient
                    // empty): record telemetry and re-request a bounded number
                    // of times before failing, so a one-off empty completion
                    // doesn't discard the turn's work.
                    AssistantTurn::Empty { usage, stop_reason } => {
                        match self.handle_empty_assistant_turn(
                            usage,
                            stop_reason.as_deref(),
                            &mut empty_retries,
                            &mut empty_recovery_attempted,
                        ) {
                            EmptyAssistantAction::Retry | EmptyAssistantAction::ContinueOnce => {
                                continue 'outer;
                            }
                            EmptyAssistantAction::Exhausted => {}
                        }
                        let assistant_message = empty_stream_exhausted_message();
                        if render_tx
                            .send(RenderBlock::TextDelta {
                                id: text_block_id,
                                text: EMPTY_STREAM_EXHAUSTED_FALLBACK_TEXT.to_string(),
                                done: false,
                            })
                            .await
                            .is_err()
                        {
                            return Err(self.cancel_streaming_turn(
                                iterations,
                                "render channel closed delivering empty-response fallback",
                                message_count_before,
                            ));
                        }
                        if render_tx
                            .send(RenderBlock::TextDelta {
                                id: text_block_id,
                                text: String::new(),
                                done: true,
                            })
                            .await
                            .is_err()
                        {
                            return Err(self.cancel_streaming_turn(
                                iterations,
                                "render channel closed finalizing empty-response fallback",
                                message_count_before,
                            ));
                        }
                        self.record_assistant_iteration(iterations, &assistant_message, 0);
                        self.session
                            .push_message(assistant_message)
                            .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
                        if let Some(msg) = self.session.messages.last().cloned() {
                            assistant_messages.push(msg);
                        }
                        break 'outer;
                    }
                };
            if let Some(usage) = usage {
                self.usage_tracker.record(usage);
            }
            // Forward a real usage snapshot so the HUD shows accurate ctx/cost
            // mid-turn instead of a char-count estimate. Non-critical telemetry,
            // so a closed render channel is ignored (the turn is unwinding anyway).
            //
            // `ctx_tokens` is the *provider's* count of what occupied the
            // context window on the latest request (input + cache read/write),
            // not a chars/4 transcript guess. When the latest turn carries no
            // usage yet (e.g. a tool-only iteration before the first billed
            // response), fall back to the local estimate so the ledger still
            // advances instead of snapping to zero.
            let ctx_tokens = {
                let provider_ctx = self.usage_tracker.current_turn_usage().context_tokens();
                if provider_ctx > 0 {
                    u64::from(provider_ctx)
                } else {
                    u64::try_from(self.estimated_tokens()).unwrap_or(u64::MAX)
                }
            };
            let _ = render_tx
                .send(RenderBlock::Usage {
                    ctx_tokens,
                    cumulative: self.usage_tracker.cumulative_usage(),
                    current: self.usage_tracker.current_turn_usage(),
                })
                .await;
            // Surface cache-efficiency warnings live. The low-cache-hit streak
            // warning previously reached only the headless JSON event surface —
            // an interactive session re-billing its whole transcript uncached
            // every call (the leak class the input-token breaker also guards)
            // burned for hours with a working smoke detector and no bell.
            // One line per streak (the record layer edge-triggers it), Warn
            // level, non-critical: a closed render channel is ignored.
            for event in &turn_prompt_cache_events {
                if let Some(warning) = &event.warning {
                    let _ = render_tx
                        .send(RenderBlock::System {
                            id: id_gen.next(),
                            level: SystemLevel::Warn,
                            text: format!("[cache] {warning}"),
                        })
                        .await;
                }
            }
            prompt_cache_events.extend(turn_prompt_cache_events);
            let pending_tool_use_count = assistant_message
                .blocks
                .iter()
                .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
                .count();
            self.record_assistant_iteration(iterations, &assistant_message, pending_tool_use_count);

            if pending_tool_use_count == 0 {
                let final_visible_text = assistant_message
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                self.session
                    .push_message(assistant_message)
                    .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
                if let Some(msg) = self.session.messages.last().cloned() {
                    assistant_messages.push(msg);
                }
                // Text-only turn boundary (A). The tool-result drain below is
                // never reached on a turn the model answers with prose alone,
                // so steering typed during it would otherwise strand in the
                // queue until the *next* user turn. Instead, fold any pending
                // steering into a fresh user message and run one more
                // iteration: the last message is now this assistant reply, so a
                // new user turn is well-formed (no two consecutive user roles).
                //
                // The same fresh-user-turn mechanism also carries a truncation
                // continuation: if the provider cut this response off at the
                // output-token limit (so the model never reached the tool call
                // it was working toward), preserve the partial output and nudge
                // it to continue rather than ending the turn empty-handed.
                // Bounded so a model that keeps over-spending the window can't
                // loop forever.
                let steers = self.drain_steering();
                let truncated = take_truncation_continuation(
                    stop_reason.as_deref(),
                    &mut truncation_continuations,
                );
                if steers.is_empty() && !truncated {
                    // Turn-end gate: a reply that ends on a promise of undone
                    // work (or, on an autonomous surface, a question nobody
                    // can answer) gets a bounded re-prompt instead of ending
                    // the turn — the harness-enforced version of the prompt's
                    // "check your last paragraph" discipline.
                    if let Some(issue) = self.take_turn_end_gate_issue(
                        &final_visible_text,
                        &mut turn_end_gate_reprompts,
                    ) {
                        let _ = render_tx
                            .send(RenderBlock::System {
                                id: id_gen.next(),
                                level: SystemLevel::Info,
                                text: super::turn_end_gate::turn_end_gate_banner(issue)
                                    .to_string(),
                            })
                            .await;
                        self.session
                            .push_user_text(
                                super::turn_end_gate::turn_end_gate_reminder(issue).to_string(),
                            )
                            .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
                        continue;
                    }
                    break 'outer;
                }
                let mut continuation_text = String::new();
                if truncated {
                    continuation_text.push_str(TRUNCATION_CONTINUATION_REMINDER);
                }
                for steer in steers {
                    let _ = render_tx
                        .send(RenderBlock::System {
                            id: id_gen.next(),
                            level: SystemLevel::Info,
                            text: format!("{STEERING_ECHO_PREFIX}{steer}"),
                        })
                        .await;
                    if !continuation_text.is_empty() {
                        continuation_text.push_str("\n\n");
                    }
                    continuation_text.push_str(&steering_message(&steer));
                }
                self.session
                    .push_user_text(continuation_text)
                    .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
                continue;
            }
            if let Err(error) = self.check_tool_call_budget(tool_calls, pending_tool_use_count) {
                self.record_turn_failed(iterations, &error);
                // Tool-call budget: the over-budget assistant `tool_use` batch is
                // not yet in the session (it is pushed just below), so the
                // session still ends on the prior well-formed `user` message.
                // Drop the pending batch — no results will be produced for it —
                // append a closer, warn, and end Ok(..) with the marker so the
                // work already done survives instead of being rolled back.
                budget_exhausted = Some(BudgetExhausted::ToolCalls);
                self.push_budget_exhausted_closer(
                    BudgetExhausted::ToolCalls,
                    iterations,
                    &mut assistant_messages,
                )
                .map_err(StreamingTurnError::runtime)?;
                let _ = render_tx
                    .send(RenderBlock::System {
                        id: id_gen.next(),
                        level: SystemLevel::Warn,
                        text: budget_exhausted_notice(BudgetExhausted::ToolCalls, iterations),
                    })
                    .await;
                break 'outer;
            }
            tool_calls += pending_tool_use_count;

            let tool_uses = collect_pending_tool_uses(&assistant_message);

            self.session
                .push_message(assistant_message)
                .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
            if let Some(msg) = self.session.messages.last().cloned() {
                assistant_messages.push(msg);
            }

            // ── Pass 1: Permission checks & pre-hooks (sequential) ──
            // Hooks and permission prompts require &mut self and may
            // await user input, so they must run one at a time.
            let mut prepared = Vec::with_capacity(tool_uses.len());
            for tool_use in &tool_uses {
                let tool_use_id = &tool_use.id;
                let tool_name = &tool_use.name;
                let input = &tool_use.input;
                let __hook_t = std::time::Instant::now();
                // Async seam: the blocking hook subprocess runs on a
                // `spawn_blocking` worker (see `run_pre_tool_use_hook_async`), so
                // this streaming task keeps servicing render/progress/cancel
                // polling while the hook runs instead of starving the render tick.
                let pre_hook_result = self.run_pre_tool_use_hook_async(tool_name, input).await;
                if __hook_t.elapsed().as_millis() >= 50 && crate::turn_profiling_enabled() {
                    eprintln!(
                        "[TURN-SEG] pre_tool_hook({tool_name}) = {}ms (off-task; render_tick free)",
                        __hook_t.elapsed().as_millis()
                    );
                }
                let effective_input = pre_hook_result
                    .updated_input()
                    .map_or_else(|| input.clone(), ToOwned::to_owned);
                let permission_context = PermissionContext::new(
                    pre_hook_result.permission_override(),
                    pre_hook_result.permission_reason().map(ToOwned::to_owned),
                );

                let permission_outcome = if let Some(outcome) =
                    pre_hook_denial_outcome(&pre_hook_result, tool_name)
                {
                    outcome
                } else if let Some(reason) = self.architect_edit_gate_denial(tool_name) {
                    // Architect contract backstop: a reserved foreground model
                    // editing directly is redirected to delegation before the
                    // ordinary permission policy runs.
                    PermissionOutcome::Deny { reason }
                } else {
                    let mut capture = CapturePrompter::new(PermissionPromptDecision::Allow);
                    let tentative = self.permission_policy.authorize_with_context(
                        tool_name,
                        &effective_input,
                        &permission_context,
                        Some(&mut capture),
                    );

                    match (capture.take(), tentative) {
                        (None, outcome) => outcome,
                        (Some(sync_request), PermissionOutcome::Allow) => {
                            let async_request = build_async_permission_request(&sync_request);
                            let reason_for_deny = sync_request.reason.clone();
                            let tool_label = tool_name.to_owned();
                            // The user is about to see a permission prompt:
                            // fire PermissionRequest plus Notification (the
                            // event Claude Code uses for OS-level pings).
                            self.fire_lifecycle_hook(
                                HookEvent::PermissionRequest,
                                &serde_json::json!({
                                    "tool_name": tool_name,
                                    "input": effective_input.as_str(),
                                    "reason": sync_request.reason,
                                }),
                            );
                            self.fire_lifecycle_hook(
                                HookEvent::Notification,
                                &serde_json::json!({
                                    "message": format!(
                                        "Zo needs permission to run `{tool_name}`"
                                    ),
                                    "tool_name": tool_name,
                                }),
                            );
                            let decision = match prompter.decide(async_request).await {
                                Ok(decision) => decision,
                                Err(error) => {
                                    if let Some(cancelled) = self.cancel_streaming_turn_if_aborted(
                                        iterations,
                                        "permission prompt interrupted by abort signal",
                                        message_count_before,
                                    ) {
                                        return Err(cancelled);
                                    }
                                    self.record_turn_host_failure(
                                        iterations,
                                        "permission prompt abandoned",
                                    );
                                    Arc::make_mut(&mut self.session.messages)
                                        .truncate(message_count_before);
                                    self.session.mark_transcript_dirty();
                                    return Err(StreamingTurnError::Permission(error));
                                }
                            };
                            match decision {
                                // "Always allow" grants a durable rule: it takes
                                // effect for the rest of this session and is
                                // queued for the host to persist to settings.
                                AsyncPermissionDecision::Allow => {
                                    self.permission_policy
                                        .grant_always(tool_name, &effective_input);
                                    PermissionOutcome::Allow
                                }
                                AsyncPermissionDecision::AllowOnce => PermissionOutcome::Allow,
                                AsyncPermissionDecision::Deny => PermissionOutcome::Deny {
                                    reason: reason_for_deny.unwrap_or_else(|| {
                                        format!("user denied tool '{tool_label}'")
                                    }),
                                },
                            }
                        }
                        (Some(_), PermissionOutcome::Deny { reason }) => {
                            PermissionOutcome::Deny { reason }
                        }
                    }
                };
                if let PermissionOutcome::Deny { reason } = &permission_outcome {
                    self.fire_lifecycle_hook(
                        HookEvent::PermissionDenied,
                        &serde_json::json!({ "tool_name": tool_name, "reason": reason }),
                    );
                }

                let tool_block_id = id_gen.next();
                let tool_call_id = ToolCallId(tool_use_id.clone());
                if matches!(permission_outcome, PermissionOutcome::Allow)
                    && render_tx
                        .send(RenderBlock::ToolCall {
                            id: tool_block_id,
                            tool_call_id: tool_call_id.clone(),
                            name: tool_name.clone(),
                            summary: tool_summary_line(tool_name, &effective_input),
                            preview: tool_preview_from(tool_name, &effective_input),
                            status: ToolCallStatus::Running,
                        })
                        .await
                        .is_err()
                {
                    return Err(self.cancel_streaming_turn(
                        iterations,
                        "render channel closed before tool dispatch",
                        message_count_before,
                    ));
                }

                prepared.push(PreparedStreamingTool {
                    tool_use_id: tool_use_id.clone(),
                    tool_name: tool_name.clone(),
                    effective_input,
                    pre_hook_result,
                    permission_outcome,
                    tool_call_id,
                });
            }

            // ── Pass 2: Parallel execution of concurrency-safe tools ──
            // Fire off permitted read-only tools via spawn_blocking only when
            // the whole allowed batch is read-only. If an ordered/mutating tool
            // is present, reads must stay in Pass 3 so `Edit -> Read` observes
            // the edit instead of racing ahead of it.
            let mut precomputed: HashMap<usize, PrecomputedStreamingToolResult> = HashMap::new();
            let can_parallel_streaming_tools = prepared
                .iter()
                .all(|p| matches!(p.permission_outcome, PermissionOutcome::Allow))
                && self.hook_runner.lifecycle_command_count(HookEvent::PostToolUse) == 0
                && self
                    .hook_runner
                    .lifecycle_command_count(HookEvent::PostToolUseFailure)
                    == 0
                && !self.streaming_parallel_batch_has_repetition_risk(&prepared);
            if can_parallel_streaming_tools {
                if let Some(dispatch) = self.concurrent_dispatch.as_ref().map(Arc::clone) {
                    // Dispatch eligible read-only tools in capped waves. Within each
                    // wave, consume JoinHandles by completion order (not model order):
                    // a slow first read must not hide later fast results from the TUI.
                    // Only batches with all tools allowed and no post-tool hooks take
                    // this path; otherwise Pass 3 preserves denial/hook ordering by
                    // executing tools serially in model order.
                    if let Some(eligible) = parallel_safe_tool_indices(
                    prepared
                        .iter()
                        .enumerate()
                        .map(|(idx, p)| (idx, p.tool_name.as_str(), &p.permission_outcome)),
                ) {
                    for batch in eligible.chunks(MAX_PARALLEL_SAFE_TOOL_DISPATCHES) {
                        let mut handles = FuturesUnordered::new();
                        for &idx in batch {
                            let p = &prepared[idx];
                            self.record_tool_started(iterations, &p.tool_name);
                            let tool_start = std::time::Instant::now();
                            let dispatch = Arc::clone(&dispatch);
                            let name = p.tool_name.clone();
                            let input = p.effective_input.clone();
                            let handle = tokio::task::spawn_blocking(move || {
                                match dispatch(&name, &input) {
                                    Ok(output) => (output, false),
                                    Err(e) => (e.to_string(), true),
                                }
                            });
                            handles.push(async move { (idx, tool_start, handle.await) });
                        }
                        while let Some((idx, tool_start, joined)) = handles.next().await {
                            let (output, is_error) = match joined {
                                Ok(result) => result,
                                Err(join_err) => (join_err.to_string(), true),
                            };
                            crate::notifications::notify_if_slow(
                                &prepared[idx].tool_name,
                                tool_start,
                                std::time::Duration::from_secs(10),
                            );
                            self.render_precomputed_streaming_tool_result(
                                iterations,
                                &prepared[idx],
                                &output,
                                is_error,
                                StreamingToolRenderContext {
                                    render_tx: &render_tx,
                                    id_gen: &id_gen,
                                    rollback_message_count: message_count_before,
                                },
                            )
                            .await?;
                            precomputed.insert(
                                idx,
                                PrecomputedStreamingToolResult {
                                    output,
                                    is_error,
                                    tool_start,
                                },
                            );
                        }
                    }
                }
            }
            }

            // ── Pass 3: Process results in order ──
            // Precomputed parallel tools have already rendered as they completed;
            // this pass only appends them to the transcript in model order.
            // Non-precomputed tools execute one-by-one. In the live TUI path a
            // `concurrent_dispatch` seam is installed; send every ordinary tool
            // through `spawn_blocking` there so even single Read/Edit/Skill calls
            // yield the turn future instead of freeze-then-bursting the render
            // loop. Sequential `.await`s preserve mutating tool order.
            let mut batch_hard_stops = ToolBatchRepetitionHardStops::default();
            for (idx, p) in prepared.iter().enumerate() {
                let result_message = if let Some(precomputed_result) = precomputed.remove(&idx) {
                    self.finalize_streaming_tool_result(
                        iterations,
                        p,
                        (precomputed_result.output, precomputed_result.is_error),
                        StreamingToolRenderContext {
                            render_tx: &render_tx,
                            id_gen: &id_gen,
                            rollback_message_count: message_count_before,
                        },
                        StreamingToolFinalizeOptions {
                            tool_start: precomputed_result.tool_start,
                            render_result: false,
                            notify_slow: false,
                        },
                        &mut batch_hard_stops,
                    )
                    .await?
                } else {
                    match &p.permission_outcome {
                        PermissionOutcome::Allow => {
                            let synthetic_output = batch_hard_stops.preflight_notice(
                                &p.tool_name,
                                &p.effective_input,
                                || {
                                    self.next_tool_repetition_hard_stop_notice(
                                        &p.tool_name,
                                        &p.effective_input,
                                    )
                                },
                            );
                            if let Some((output, terminates)) = synthetic_output {
                                if terminates {
                                    self.tool_loop_break_requested = true;
                                }
                                self.render_synthetic_streaming_tool_result(
                                    iterations,
                                    p,
                                    &output,
                                    &render_tx,
                                    &id_gen,
                                    message_count_before,
                                )
                                .await?;
                                ConversationMessage::tool_result(
                                    &p.tool_use_id,
                                    &p.tool_name,
                                    merge_hook_feedback(
                                        p.pre_hook_result.messages(),
                                        output,
                                        true,
                                    ),
                                    true,
                                )
                            } else {
                            self.record_tool_started(iterations, &p.tool_name);
                            let tool_start = std::time::Instant::now();

                            let (output, is_error) = if p.tool_name == "AskUserQuestion" {
                                // AskUserQuestion must be handled async:
                                // `unblock_tool_execute` uses `block_in_place`
                                // which blocks the current task, deadlocking
                                // the TUI select! loop that needs to drain
                                // the prompt and relay the user's answer.
                                match ask_user_question_async(&p.effective_input, &render_tx, &id_gen)
                                    .await
                                {
                                    Ok(answer) => (answer, false),
                                    Err(e) => (e, true),
                                }
                            } else {
                                let execution_input = if let Some((delay, input)) =
                                    sleep_tool_execution_input(&p.tool_name, &p.effective_input)
                                {
                                    if !delay.is_zero() {
                                        tokio::time::sleep(delay).await;
                                    }
                                    Cow::Owned(input)
                                } else if let Some(input) = tool_execution_input(
                                    &p.tool_name,
                                    &p.tool_use_id,
                                    &p.effective_input,
                                ) {
                                    Cow::Owned(input)
                                } else {
                                    Cow::Borrowed(p.effective_input.as_str())
                                };
                                if let Some(dispatch) = &self.concurrent_dispatch {
                                    let dispatch = std::sync::Arc::clone(dispatch);
                                    let name = p.tool_name.clone();
                                    let input = execution_input.into_owned();
                                    let join =
                                        tokio::task::spawn_blocking(move || dispatch(&name, &input))
                                            .await;
                                    match join {
                                        Ok(Ok(output)) => (output, false),
                                        Ok(Err(e)) => (e.to_string(), true),
                                        Err(join_err) => (join_err.to_string(), true),
                                    }
                                } else {
                                    unblock_tool_execute(
                                        &mut self.tool_executor,
                                        &p.tool_name,
                                        execution_input.as_ref(),
                                    )
                                }
                            };

                            self.finalize_streaming_tool_result(
                                iterations,
                                p,
                                (output, is_error),
                                StreamingToolRenderContext {
                                    render_tx: &render_tx,
                                    id_gen: &id_gen,
                                    rollback_message_count: message_count_before,
                                },
                                StreamingToolFinalizeOptions {
                                    tool_start,
                                    render_result: true,
                                    notify_slow: true,
                                },
                                &mut batch_hard_stops,
                            )
                            .await?
                            }
                        }
                        PermissionOutcome::Deny { reason } => {
                        // Settle the tool card first: the streaming parser
                        // already flipped it to Running, and the TUI only
                        // reconciles a card's status when a ToolResult render
                        // block arrives — without one a denied tool spins
                        // forever under the denial banner.
                        let result_block_id = id_gen.next();
                        if render_tx
                            .send(RenderBlock::ToolResult {
                                id: result_block_id,
                                tool_call_id: p.tool_call_id.clone(),
                                is_error: true,
                                body: format_tool_result_from_raw(&p.tool_name, reason, true),
                            })
                            .await
                            .is_err()
                        {
                            return Err(self.cancel_streaming_turn(
                                iterations,
                                "render channel closed delivering denial result",
                                message_count_before,
                            ));
                        }
                        let deny_block_id = id_gen.next();
                        if render_tx
                            .send(RenderBlock::System {
                                id: deny_block_id,
                                level: SystemLevel::Warn,
                                text: denial_banner(&p.tool_name, reason),
                            })
                            .await
                            .is_err()
                        {
                            return Err(self.cancel_streaming_turn(
                                iterations,
                                "render channel closed delivering denial",
                                message_count_before,
                            ));
                        }
                        let body = self.fold_repeated_mode_denial(
                            &p.tool_name,
                            super::denial_result_body(reason),
                        );
                        ConversationMessage::tool_result(
                            &p.tool_use_id,
                            &p.tool_name,
                            merge_hook_feedback(p.pre_hook_result.messages(), body, true),
                            true,
                        )
                    }
                }
                };
                self.session
                    .push_message(result_message.clone())
                    .map_err(|error| StreamingTurnError::runtime(error.to_string()))?;
                self.record_tool_finished(iterations, &result_message);
                tool_results.push(result_message);
            }

            self.arm_tool_repetition_hard_stops();

            // Verification-treadmill circuit breaker (mirrors the sync loop): a
            // batch that plans/validates/spawns (verify-class) but changes no file
            // is a self-verification round; too many in a row without progress stop
            // the turn gracefully. Preserve the well-formed work and end with the
            // marker, like the other budgets.
            let had_verify = tool_uses
                .iter()
                .any(|tool_use| is_verify_class_tool(&tool_use.name));
            let had_mutation = tool_uses
                .iter()
                .any(|tool_use| is_edit_or_write_tool(&tool_use.name));
            if self.note_verify_treadmill(had_verify, had_mutation) {
                let error = RuntimeError::new("turn hit the verification treadmill");
                self.clear_empty_retry_reminder(empty_retries);
                self.record_turn_failed(iterations, &error);
                budget_exhausted = Some(BudgetExhausted::VerificationTreadmill);
                self.push_budget_exhausted_closer(
                    BudgetExhausted::VerificationTreadmill,
                    iterations,
                    &mut assistant_messages,
                )
                .map_err(StreamingTurnError::runtime)?;
                let _ = render_tx
                    .send(RenderBlock::System {
                        id: id_gen.next(),
                        level: SystemLevel::Warn,
                        text: budget_exhausted_notice(
                            BudgetExhausted::VerificationTreadmill,
                            iterations,
                        ),
                    })
                    .await;
                break 'outer;
            }

            // Mid-turn steering boundary. Fold any user-typed steering into the
            // *last* tool-result message as extra Text blocks rather than a new
            // message: tool-result messages serialize as wire role "user", so a
            // separate user message here would be two consecutive "user" turns
            // (which the API rejects). A user turn may carry tool_result blocks
            // followed by text, so appending keeps one valid turn. `make_mut`
            // mirrors the truncate path already used in this function.
            let steers = self.drain_steering();
            if !steers.is_empty() {
                let messages = Arc::make_mut(&mut self.session.messages);
                if let Some(last) = messages.last_mut() {
                    for steer in steers {
                        let _ = render_tx
                            .send(RenderBlock::System {
                                id: id_gen.next(),
                                level: SystemLevel::Info,
                                text: format!("{STEERING_ECHO_PREFIX}{steer}"),
                            })
                            .await;
                        last.blocks.push(ContentBlock::Text {
                            text: steering_message(&steer),
                        });
                    }
                    self.session.mark_transcript_dirty();
                }
            }
            // Mid-turn agent-notification boundary (CC's task-notification
            // contract). Background agents that finished during this tool
            // batch are folded into the same last tool-result message as
            // extra Text blocks, so the main model keeps working through
            // completions instead of ending its turn to receive them. The
            // transcript gets the same collapsible agent-result card the
            // follow-up-turn path renders — mid-turn delivery must not make
            // the result invisible to the user.
            let notifications = self.drain_agent_notifications();
            if !notifications.is_empty() {
                let messages = Arc::make_mut(&mut self.session.messages);
                if let Some(last) = messages.last_mut() {
                    for notification in notifications {
                        let _ = render_tx
                            .send(RenderBlock::AgentResult {
                                id: id_gen.next(),
                                label: notification.label.clone(),
                                status: notification.status,
                                body: notification.text.clone(),
                            })
                            .await;
                        last.blocks.push(ContentBlock::Text {
                            text: agent_notification_text(&notification),
                        });
                    }
                    self.session.mark_transcript_dirty();
                }
            }
            // Re-anchor the live plan after this tool batch (mirrors the sync
            // turn loop) so the next streamed request keeps the plan in view.
            self.reinject_todo_progress_reminder();
        }

        if let Some(event) = self.maybe_microcompact_streaming(&render_tx, &id_gen).await {
            microcompact.get_or_insert(event);
        }
        if let Some(event) = self.maybe_auto_compact_streaming(&render_tx, &id_gen).await {
            auto_compaction.get_or_insert(event);
        } else {
            self.maybe_state_distill();
            self.maybe_precompaction_warn_streaming(&render_tx, &id_gen)
                .await;
        }

        let summary = self.build_turn_summary(
            assistant_messages,
            tool_results,
            prompt_cache_events,
            iterations,
            auto_compaction,
            microcompact,
            turn_start_output_tokens,
            budget_exhausted,
        );
        self.record_turn_completed(&summary);

        Ok(summary)
    }
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    fn streaming_parallel_batch_has_repetition_risk(&self, prepared: &[PreparedStreamingTool]) -> bool {
        self.parallel_batch_has_repetition_risk(prepared.iter().map(|p| {
            (
                p.tool_name.as_str(),
                p.effective_input.as_str(),
                &p.permission_outcome,
            )
        }))
    }

    async fn render_precomputed_streaming_tool_result(
        &mut self,
        iterations: usize,
        p: &PreparedStreamingTool,
        output: &str,
        is_error: bool,
        render: StreamingToolRenderContext<'_>,
    ) -> Result<(), StreamingTurnError> {
        let body = if is_error {
            let output = merge_hook_feedback(
                p.pre_hook_result.messages(),
                output.to_string(),
                false,
            );
            format_tool_result_from_raw(&p.tool_name, &output, true)
        } else {
            format_tool_result_from_raw(&p.tool_name, output, false)
        };
        let result_block_id = render.id_gen.next();
        if render.render_tx
            .send(RenderBlock::ToolResult {
                id: result_block_id,
                tool_call_id: p.tool_call_id.clone(),
                is_error,
                body,
            })
            .await
            .is_err()
        {
            return Err(self.cancel_streaming_turn(
                iterations,
                "render channel closed delivering tool result",
                render.rollback_message_count,
            ));
        }
        Ok(())
    }

    async fn render_synthetic_streaming_tool_result(
        &mut self,
        iterations: usize,
        p: &PreparedStreamingTool,
        output: &str,
        render_tx: &mpsc::Sender<RenderBlock>,
        id_gen: &BlockIdGen,
        rollback_message_count: usize,
    ) -> Result<(), StreamingTurnError> {
        let result_block_id = id_gen.next();
        if render_tx
            .send(RenderBlock::ToolResult {
                id: result_block_id,
                tool_call_id: p.tool_call_id.clone(),
                is_error: true,
                body: format_tool_result_from_raw(&p.tool_name, output, true),
            })
            .await
            .is_err()
        {
            return Err(self.cancel_streaming_turn(
                iterations,
                "render channel closed delivering synthetic tool result",
                rollback_message_count,
            ));
        }
        Ok(())
    }

    async fn finalize_streaming_tool_result(
        &mut self,
        iterations: usize,
        p: &PreparedStreamingTool,
        result: (String, bool),
        render: StreamingToolRenderContext<'_>,
        options: StreamingToolFinalizeOptions,
        batch_hard_stops: &mut ToolBatchRepetitionHardStops,
    ) -> Result<ConversationMessage, StreamingTurnError> {
        let (mut output, mut is_error) = result;
        if options.notify_slow {
            crate::notifications::notify_if_slow(
                &p.tool_name,
                options.tool_start,
                std::time::Duration::from_secs(10),
            );
        }
        // Successful structured renders must use the pristine tool output,
        // not hook-merged text, or edit/read cards degrade into generic
        // strings. Preserve that pristine string only when pre-hook feedback
        // will overwrite `output` before we know the final render status.
        // If there is no pre-hook feedback, `output` remains the pristine
        // string until after the success render body is built, so no clone is
        // needed even when post-hook feedback will later be appended for the
        // model-facing transcript.
        let pure_output_for_render = (!is_error && !p.pre_hook_result.messages().is_empty())
            .then(|| output.clone());
        output = merge_hook_feedback(p.pre_hook_result.messages(), output, false);

        // Async seam (same off-task offload as the pre-hook): the blocking
        // post-hook subprocess runs on a `spawn_blocking` worker so this
        // streaming task stays responsive while the hook runs.
        let post_hook_result = if is_error {
            self.run_post_tool_use_failure_hook_async(&p.tool_name, &p.effective_input, &output)
                .await
        } else {
            self.run_post_tool_use_hook_async(&p.tool_name, &p.effective_input, &output, false)
                .await
        };
        let post_hook_is_error = post_hook_result.is_denied()
            || post_hook_result.is_failed()
            || post_hook_result.is_cancelled();
        if post_hook_is_error {
            is_error = true;
        }

        let body = if is_error {
            output = merge_hook_feedback(post_hook_result.messages(), output, post_hook_is_error);
            format_tool_result_from_raw(&p.tool_name, &output, true)
        } else {
            let render_output = pure_output_for_render.as_deref().unwrap_or(&output);
            let body = format_tool_result_from_raw(&p.tool_name, render_output, false);
            output = merge_hook_feedback(post_hook_result.messages(), output, false);
            body
        };
        if options.render_result {
            let result_block_id = render.id_gen.next();
            if render
                .render_tx
                .send(RenderBlock::ToolResult {
                    id: result_block_id,
                    tool_call_id: p.tool_call_id.clone(),
                    is_error,
                    body,
                })
                .await
                .is_err()
            {
                return Err(self.cancel_streaming_turn(
                    iterations,
                    "render channel closed delivering tool result",
                    render.rollback_message_count,
                ));
            }
        }
        // Enforcer-layer denials surface as tool errors here; fold same-class
        // repeats on the model-facing result like the policy-layer deny arm.
        if is_error {
            output = self.fold_repeated_mode_denial(&p.tool_name, output);
        }
        // Append (after rendering, so the user-facing body stays clean) only to
        // the model-facing result, nudging the agent out of a tight
        // identical-call loop.
        self.append_tool_repetition_notice(
            &mut output,
            &p.tool_name,
            &p.effective_input,
            is_error,
            batch_hard_stops,
        );

        // Drain images the tool staged. The real live dispatcher shares the
        // image sink through cloned contexts, so this remains correct whether
        // the serial tool executed directly or through the spawn_blocking
        // dispatch seam.
        let images = self.tool_executor.take_pending_images();
        Ok(tool_result_message(
            &p.tool_use_id,
            &p.tool_name,
            output,
            is_error,
            images,
        ))
    }
}
