use std::borrow::Cow;
use std::time::Duration;

use api::ProviderErrorClass;

use runtime::message_stream::{AgentResultStatus, SystemLevel};
use zo_cli::tui::App;
use zo_cli::tui::app::{AgentResultMeta, QueuedMessage};
use tools::{
    background_completion_matches_session, clear_background_agent, is_background_agent,
    wait_for_agent_completions, AGENT_STARVED_STATUS, AgentCompletion,
    provider_error_class_from_completion,
};

/// Maximum chars of a background agent's result re-injected verbatim into the
/// conversation. A long agent transcript would otherwise blow the main model's
/// context, so an oversized result keeps its head and tail with a clear elision
/// notice in the middle.
const MAX_REINJECTED_RESULT_CHARS: usize = 16_000;

/// Build the model-facing result message (header + size-capped body) and the
/// render-card meta for a terminal **background** agent completion, consuming
/// the background marker. `None` when the completion is not deliverable: not a
/// background agent, a `stopped` agent (cancelled by the user — resurrecting
/// its result would be surprising), or a completed agent with an empty result
/// (the `⎿ Done` row already notified the user).
///
/// Shared by BOTH delivery paths — the mid-turn task-notification fold and the
/// follow-up-turn re-injection — so the model reads byte-identical content
/// regardless of when the agent finished. The broadcast channel strips the full
/// result to keep the renderer light, so the answer is read back from the
/// completion store by id with a zero timeout (it is recorded before the
/// channel signal fires, so it is already present). The background marker is
/// cleared up front so the id set never grows without bound; the channel is
/// single-consumer, so each completion is delivered exactly once.
fn build_background_agent_result_message(
    completion: &AgentCompletion,
    active_session_id: &str,
) -> Option<(AgentResultMeta, String)> {
    if !is_background_agent(&completion.agent_id) {
        return None;
    }
    if suppress_mismatched_background_task_completion(completion, active_session_id) {
        return None;
    }
    clear_background_agent(&completion.agent_id);
    if completion.status != "completed" && completion.status != "failed" {
        return None;
    }
    let label = agent_display_label(completion).into_owned();
    // The broadcast event strips `result`; read the full answer back from the
    // store by id. A missing id (never recorded / TTL-evicted) yields a
    // `still_running` placeholder with `result: None`, which the gates below
    // treat as "nothing to re-inject".
    let full = wait_for_agent_completions(std::slice::from_ref(&completion.agent_id), Duration::ZERO)
        .into_iter()
        .find(|stored| stored.agent_id == completion.agent_id);
    let body = if completion.status == "completed" {
        // Only deliver a non-empty result. An empty/absent result (evicted, or
        // an agent that produced no text) needs no delivery — the `⎿ Done`
        // row already notified the user.
        match full.and_then(|stored| stored.result) {
            Some(result) if !result.trim().is_empty() => result,
            _ => return None,
        }
    } else {
        // Failed/other terminal. Prefer an explicit error message, but fall back
        // to the result payload: a failed background *bash* task carries its
        // output (stdout/stderr + the `[exit N]` line) in `result` with
        // `error: None` (see `notify_background_task_completion`), so reading
        // only `error` would hand the model a useless "failed without a message"
        // and hide the actual failure output. The channel event strips
        // `result`, so the store read-back (`full`) is the copy that still
        // carries it. The error text also rides the channel event, so it
        // survives even a store eviction.
        let (stored_error, stored_result) = full
            .map_or((None, None), |stored| (stored.error, stored.result));
        stored_error
            .or_else(|| completion.error.clone())
            .or_else(|| stored_result.filter(|result| !result.trim().is_empty()))
            .unwrap_or_else(|| "agent failed without an error message".to_string())
    };
    // The follow-up pointer names the agent ID (always resolvable by the
    // SendMessage lookup; a display label may not be) so the model reaches for
    // a context-preserving resume instead of re-spawning and re-explaining.
    let header = if completion.status == "completed" {
        format!(
            "[background agent `{label}` finished — its result follows. To follow up or go \
             deeper, call SendMessage(to: \"{id}\") — it resumes this agent with its context \
             intact instead of re-spawning]",
            id = completion.agent_id
        )
    } else {
        format!(
            "[background agent `{label}` failed. You can resume it with its context via \
             SendMessage(to: \"{id}\")]",
            id = completion.agent_id
        )
    };
    let message = format!(
        "{header}\n\n{}",
        truncate_for_reinjection(&body, MAX_REINJECTED_RESULT_CHARS)
    );
    let status = if completion.status == "completed" {
        AgentResultStatus::Completed
    } else {
        AgentResultStatus::Failed
    };
    Some((AgentResultMeta { label, status }, message))
}

/// Drop a background-task push that belongs to another session. The full task
/// output remains in the process-scoped registry for `TaskOutput`; consuming
/// the marker prevents the compact channel event from surfacing in this
/// session as either a follow-up turn or a generic completion notice.
pub(crate) fn suppress_mismatched_background_task_completion(
    completion: &AgentCompletion,
    active_session_id: &str,
) -> bool {
    if !is_background_agent(&completion.agent_id)
        || background_completion_matches_session(&completion.agent_id, active_session_id)
    {
        return false;
    }
    clear_background_agent(&completion.agent_id);
    true
}

/// When `completion` is the terminal completion of an agent the model launched
/// in **background** mode (`AgentInput::background`), queue its result as a
/// fresh user turn so the main model picks it up on the next REPL iteration —
/// the idle-host half of background delivery (a turn in flight uses
/// [`deliver_background_agent_completion_mid_turn`] instead). Returns `true`
/// when a re-injection was queued.
///
/// The message text still submits as a normal user-role turn (the model must
/// read the result to continue), but it is tagged so the transcript renders a
/// collapsible agent-result card authored by the agent instead of an amber
/// `You` message — otherwise a long result floods the transcript as raw
/// markdown. See `RenderBlock::AgentResult`.
pub(crate) fn reinject_background_agent_completion(
    app: &mut App,
    completion: &AgentCompletion,
    active_session_id: &str,
) -> bool {
    let Some((meta, message)) =
        build_background_agent_result_message(completion, active_session_id)
    else {
        return false;
    };
    app.queue_agent_result_message(message, meta).is_ok()
}

/// Stage a background agent's terminal completion for **mid-turn delivery**
/// (CC's task-notification contract): the live turn drains the inbox at its
/// next tool-result boundary and folds the result in, so a main model that
/// kept working after spawning learns of the finished agent without ending
/// its turn. Returns `true` when the completion was staged (or fell back to
/// the follow-up-turn queue on a poisoned inbox — the result is never lost).
///
/// The turn controller drains whatever the turn never reached a boundary to
/// fold back out of the inbox after the turn and re-queues it as follow-up
/// turns, keeping delivery exactly-once.
pub(crate) fn deliver_background_agent_completion_mid_turn(
    app: &mut App,
    inbox: &runtime::AgentNotificationInbox,
    completion: &AgentCompletion,
    active_session_id: &str,
) -> bool {
    let Some((meta, message)) =
        build_background_agent_result_message(completion, active_session_id)
    else {
        return false;
    };
    match inbox.lock() {
        Ok(mut inbox) => {
            inbox.push(runtime::AgentNotification {
                label: meta.label,
                status: meta.status,
                text: message,
            });
            true
        }
        // A poisoned inbox must not eat the result (the background marker is
        // already consumed): degrade to the follow-up-turn queue.
        Err(_) => app.queue_agent_result_message(message, meta).is_ok(),
    }
}

/// Re-queue every notification the finished turn never reached a tool-result
/// boundary to fold (or that arrived after its last boundary) as follow-up
/// agent-result turns — the second and final drain point of the mid-turn
/// inbox. The caller runs this only after the turn task has handed the
/// runtime back, so it can never race the mid-turn fold: delivery stays
/// exactly-once. Returns the number of notifications re-queued.
pub(crate) fn requeue_undelivered_agent_notifications(
    app: &mut App,
    inbox: &runtime::AgentNotificationInbox,
) -> usize {
    let Ok(mut leftovers) = inbox.lock() else {
        return 0;
    };
    let mut requeued = 0;
    for notification in leftovers.drain(..) {
        if app
            .queue_agent_result_message(
                notification.text,
                AgentResultMeta {
                    label: notification.label,
                    status: notification.status,
                },
            )
            .is_ok()
        {
            requeued += 1;
        }
    }
    requeued
}

/// Keep the head and tail of an oversized re-injected result, eliding the middle
/// with an explicit notice (shared SSOT helper).
fn truncate_for_reinjection(body: &str, max: usize) -> String {
    core_types::text::elide_middle(body, max)
}

/// Fold a popped agent-result turn together with every other agent-result
/// still sitting in the queue into ONE combined submit. N background tasks
/// that finish during one long turn would otherwise pop as N consecutive
/// follow-up turns — a parade of near-identical alarm cards, each burning a
/// full model turn on "that result is stale". Each body already leads with
/// its own `[background agent … finished/failed …]` header, so the joined
/// sections stay self-describing for the model; the combined card label
/// carries the batch size. `head` must be an agent-result message (the
/// caller's pop gate); `rest` comes from
/// [`App::drain_queued_agent_results`], so every entry carries meta.
pub(crate) fn coalesce_agent_result_messages(
    head: QueuedMessage,
    rest: Vec<QueuedMessage>,
) -> QueuedMessage {
    if rest.is_empty() {
        return head;
    }
    let mut texts = Vec::with_capacity(rest.len() + 1);
    let mut labels = Vec::with_capacity(rest.len() + 1);
    let mut status = AgentResultStatus::Completed;
    for message in std::iter::once(head).chain(rest) {
        if let Some(meta) = message.agent_result {
            if matches!(meta.status, AgentResultStatus::Failed) {
                status = AgentResultStatus::Failed;
            }
            labels.push(meta.label);
        }
        texts.push(message.text);
    }
    let label = match labels.split_first() {
        Some((first, tail)) if tail.iter().all(|label| label == first) => {
            format!("{first} ×{}", labels.len())
        }
        _ => format!("{} background agents", labels.len()),
    };
    QueuedMessage {
        text: texts.join("\n\n---\n\n"),
        images: Vec::new(),
        goal_owned: false,
        loop_id: None,
        agent_result: Some(AgentResultMeta { label, status }),
        steered: false,
    }
}

pub(crate) fn format_agent_completion(completion: &AgentCompletion) -> (SystemLevel, String) {
    // W9-3: a starvation notice is a *live* warning, not a terminal failure —
    // the agent keeps retrying after posting it. Branch before the error-text
    // heuristics below so its "rate-limit" wording can't be misread as the
    // canned gave-up-after-retries failure.
    if completion.status == AGENT_STARVED_STATUS {
        let label = agent_display_label(completion);
        let detail = completion
            .error
            .as_deref()
            .unwrap_or("starved by rate-limit");
        return (SystemLevel::Warn, format!("Agent '{label}': {detail}"));
    }
    if completion.status == "completed" {
        let label = agent_display_label(completion);
        return (SystemLevel::Info, format!("Agent '{label}' finished"));
    }
    if completion.status == "stopped" {
        let label = agent_display_label(completion);
        let detail = completion.error.as_deref().unwrap_or("cancelled");
        return (
            SystemLevel::Warn,
            format!("Agent '{label}' stopped: {detail}"),
        );
    }

    let label = agent_display_label(completion);
    // An unexplained failure that still returned a result (e.g. a legacy
    // manifest from before failure reasons were recorded) should say so —
    // "unknown error" next to a visible result reads as a contradiction.
    let detail = completion.error.as_deref().unwrap_or(
        if completion.result.as_deref().is_some_and(|r| !r.trim().is_empty()) {
            "no error detail recorded — partial result attached"
        } else {
            "unknown error"
        },
    );
    let message = if agent_completion_is_auth_failure(completion) {
        format!("agent '{label}' auth failed · /login or ZO_AGENT_MODEL")
    } else if agent_completion_is_rate_limit_failure(completion) {
        // Sub-agents already default to concurrency 1, so the old "lower to 1"
        // advice was self-contradictory. They run on the *same* account quota as
        // the foreground turn (identical OAuth credentials as Claude Code), so a
        // 429 here is the provider throttling rapid back-to-back requests — not a
        // separate budget. It clears on its own; retry shortly or fan out less.
        format!(
            "agent '{label}' rate limited — gave up after retries · provider throttled rapid requests (sub-agents share your account limit); retry shortly or run fewer agents at once"
        )
    } else {
        format!("Agent '{label}' failed: {detail}")
    };
    (SystemLevel::Error, message)
}

/// Internal plumbing agents whose lifecycle the auto fan-out controller already
/// narrates (launch / fallback / synthesis notes). The `decompose` and `triage`
/// agents run the pre-analysis split and the semantic route classification, and
/// both results are consumed synchronously from the completion store, so their
/// raw channel completions are pure noise — including the benign
/// `stopped: auto fan-out collection window closed` reap that fires when their
/// wait window elapses while the model is still streaming. Drop them at the
/// display boundary so that reap never surfaces as a user-facing warning.
pub(crate) fn agent_completion_is_internal(completion: &AgentCompletion) -> bool {
    matches!(completion.name.as_str(), "decompose" | "triage")
}

/// W9-3 starvation notice marker — rendered as a one-shot warning line and
/// kept away from the agent tree's `⎿ Done` flip and the failure dedup slots
/// (the agent is still running).
pub(crate) fn agent_completion_is_starvation_notice(completion: &AgentCompletion) -> bool {
    completion.status == AGENT_STARVED_STATUS
}

pub(crate) fn agent_completion_is_auth_failure(completion: &AgentCompletion) -> bool {
    match provider_error_class_from_completion(completion) {
        Some(ProviderErrorClass::AuthExpired) => true,
        Some(_) => false,
        None => completion.error.as_deref().is_some_and(is_auth_failure),
    }
}

pub(crate) fn agent_completion_is_rate_limit_failure(completion: &AgentCompletion) -> bool {
    match provider_error_class_from_completion(completion) {
        Some(ProviderErrorClass::RateLimit { .. }) => true,
        Some(_) => false,
        None => completion
            .error
            .as_deref()
            .is_some_and(is_rate_limit_failure),
    }
}

fn agent_display_label(completion: &AgentCompletion) -> Cow<'_, str> {
    let name = completion.name.trim();
    if name == "decompose" {
        return Cow::Borrowed("decomposition");
    }
    if !name.is_empty() {
        return Cow::Borrowed(name);
    }
    let agent_id = completion.agent_id.trim();
    if !agent_id.is_empty() {
        return Cow::Borrowed(agent_id);
    }
    Cow::Borrowed("agent")
}

fn is_auth_failure(detail: &str) -> bool {
    let normalized = detail.to_ascii_lowercase();
    normalized.contains("401")
        || normalized.contains("unauthorized")
        || normalized.contains("authentication")
        || normalized.contains("api key")
        || normalized.contains("credentials")
}

fn is_rate_limit_failure(detail: &str) -> bool {
    let normalized = detail.to_ascii_lowercase();
    normalized.contains("429")
        || normalized.contains("too many requests")
        || normalized.contains("rate_limit")
        || normalized.contains("rate limit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tools::AGENT_STARVED_STATUS;

    fn completion(status: &str, error: Option<&str>) -> AgentCompletion {
        AgentCompletion {
            agent_id: "agent-1#starved".to_string(),
            name: "explorer".to_string(),
            status: status.to_string(),
            result: None,
            structured: None,
            error: error.map(str::to_string),
            output_tokens: 0,
        }
    }

    fn classified_completion(
        status: &str,
        error: Option<&str>,
        provider_error_class: ProviderErrorClass,
    ) -> AgentCompletion {
        let mut completion = completion(status, error);
        completion.structured = Some(tools::provider_error_class_metadata(provider_error_class));
        completion
    }

    /// W9-3: starved 통지는 경고 레벨로, rate-limit 문구가 들어 있어도 종결
    /// 실패("gave up after retries")로 오인 렌더되지 않는다.
    #[test]
    fn empty_agent_name_falls_back_to_agent_id_in_notice() {
        let completion = AgentCompletion {
            agent_id: "agent-pre-analysis-2".to_string(),
            name: "   ".to_string(),
            status: "completed".to_string(),
            result: None,
            structured: None,
            error: None,
            output_tokens: 0,
        };

        let (level, text) = format_agent_completion(&completion);

        assert_eq!(level, SystemLevel::Info);
        assert_eq!(text, "Agent 'agent-pre-analysis-2' finished");
        assert!(!text.contains("Agent ''"));
    }

    #[test]
    fn starved_notice_renders_as_warning_not_terminal_failure() {
        let notice = completion(
            AGENT_STARVED_STATUS,
            Some("rate-limit starved for 5m (retry 6 on claude-opus-4-8)"),
        );
        assert!(agent_completion_is_starvation_notice(&notice));
        let (level, text) = format_agent_completion(&notice);
        assert_eq!(level, SystemLevel::Warn);
        assert!(text.contains("explorer"));
        assert!(text.contains("starved for 5m"));
        assert!(!text.contains("gave up after retries"));
    }

    /// The live "'poc-functional' failed: unknown error" contradiction: a
    /// failed completion CARRYING a result must not fabricate "unknown error"
    /// next to it — and a recorded failure reason (e.g. the budget-exhausted
    /// kind the spawn path now persists) is always preferred verbatim.
    #[test]
    fn failed_notice_with_result_never_claims_unknown_error() {
        let mut with_result = completion("failed", None);
        with_result.result = Some("[budget exhausted: output tokens]\npartial".to_string());
        let (level, text) = format_agent_completion(&with_result);
        assert_eq!(level, SystemLevel::Error);
        assert!(text.contains("no error detail recorded — partial result attached"));
        assert!(!text.contains("unknown error"));

        let with_reason = completion("failed", Some("budget exhausted: output tokens — partial result preserved"));
        let (_, text) = format_agent_completion(&with_reason);
        assert!(text.contains("budget exhausted: output tokens"));

        // No result, no reason: the honest fallback stays.
        let bare = completion("failed", None);
        let (_, text) = format_agent_completion(&bare);
        assert!(text.contains("unknown error"));
    }

    #[test]
    fn agent_notice_prefers_provider_error_class_over_text() {
        let rate_limit = classified_completion(
            "failed",
            Some("401 stale auth diagnostic in provider body"),
            ProviderErrorClass::RateLimit { retry_after: None },
        );
        assert!(agent_completion_is_rate_limit_failure(&rate_limit));
        assert!(!agent_completion_is_auth_failure(&rate_limit));

        let auth = classified_completion(
            "failed",
            Some("rate limit diagnostics from quota dashboard"),
            ProviderErrorClass::AuthExpired,
        );
        assert!(agent_completion_is_auth_failure(&auth));
        assert!(!agent_completion_is_rate_limit_failure(&auth));
    }

    #[test]
    fn agent_notice_formatter_prefers_provider_error_class_over_text() {
        let rate_limit = classified_completion(
            "failed",
            Some("401 stale auth diagnostic in provider body"),
            ProviderErrorClass::RateLimit { retry_after: None },
        );
        let (_, text) = format_agent_completion(&rate_limit);
        assert!(text.contains("rate limited"));
        assert!(!text.contains("auth failed"));

        let auth = classified_completion(
            "failed",
            Some("api returned 429 Too Many Requests"),
            ProviderErrorClass::AuthExpired,
        );
        let (_, text) = format_agent_completion(&auth);
        assert!(text.contains("auth failed"));
        assert!(!text.contains("rate limited"));
    }

    #[test]
    fn agent_notice_keeps_legacy_string_fallback() {
        let auth = completion("failed", Some("401 Unauthorized: invalid api key"));
        assert!(agent_completion_is_auth_failure(&auth));

        let rate_limit = completion("failed", Some("api returned 429 Too Many Requests"));
        assert!(agent_completion_is_rate_limit_failure(&rate_limit));
    }

    fn agent_result_message(text: &str, label: &str, status: AgentResultStatus) -> QueuedMessage {
        QueuedMessage {
            text: text.to_string(),
            images: Vec::new(),
            goal_owned: false,
            loop_id: None,
            agent_result: Some(AgentResultMeta {
                label: label.to_string(),
                status,
            }),
            steered: false,
        }
    }

    /// 배치 없는 단일 완료는 그대로 통과한다 — fold가 항상 새 메시지를
    /// 만들면 라벨/텍스트가 불필요하게 재조립된다.
    #[test]
    fn coalesce_with_empty_rest_is_identity() {
        let head = agent_result_message("[bg a] done", "background bash", AgentResultStatus::Completed);
        let folded = coalesce_agent_result_messages(head, Vec::new());
        assert_eq!(folded.text, "[bg a] done");
        let meta = folded.agent_result.expect("meta preserved");
        assert_eq!(meta.label, "background bash");
        assert_eq!(meta.status, AgentResultStatus::Completed);
    }

    /// 같은 턴 동안 쌓인 N개 완료는 한 턴으로 합쳐진다 — 07-13 라이브에서
    /// 백그라운드 bash 7건이 알람 7턴 퍼레이드로 팝된 버그의 직접 회귀.
    #[test]
    fn coalesce_folds_batch_into_one_turn_with_counted_label() {
        let head = agent_result_message("[bg a] done", "background bash", AgentResultStatus::Completed);
        let rest = vec![
            agent_result_message("[bg b] done", "background bash", AgentResultStatus::Completed),
            agent_result_message("[bg c] done", "background bash", AgentResultStatus::Completed),
        ];
        let folded = coalesce_agent_result_messages(head, rest);
        assert_eq!(
            folded.text,
            "[bg a] done\n\n---\n\n[bg b] done\n\n---\n\n[bg c] done"
        );
        let meta = folded.agent_result.expect("meta");
        assert_eq!(meta.label, "background bash ×3");
        assert_eq!(meta.status, AgentResultStatus::Completed);
        assert!(!folded.goal_owned);
        assert!(folded.loop_id.is_none());
    }

    /// 하나라도 실패면 배치 카드는 Failed 틴트 — 성공 라벨 아래 실패가
    /// 묻히지 않는다. 라벨이 섞이면 개수 요약으로 떨어진다.
    #[test]
    fn coalesce_mixed_labels_and_any_failure_surface_in_meta() {
        let head = agent_result_message("[bg a] done", "background bash", AgentResultStatus::Completed);
        let rest = vec![agent_result_message(
            "[scout] boom",
            "runtime-scout",
            AgentResultStatus::Failed,
        )];
        let folded = coalesce_agent_result_messages(head, rest);
        let meta = folded.agent_result.expect("meta");
        assert_eq!(meta.label, "2 background agents");
        assert_eq!(meta.status, AgentResultStatus::Failed);
    }
}
