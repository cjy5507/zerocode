//! Retry with jittered exponential backoff for transient API errors.
//!
//! Claude Code CLI retries transient HTTP errors (429, 500, 502, 503, 529)
//! with exponential backoff. This module provides the same capability for
//! the Rust runtime, giving it parity and — thanks to lower per-retry
//! overhead — an edge in recovery speed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use core_types::retry_signal::classify_error_text;

/// Maximum number of retry attempts (excluding the initial attempt).
const MAX_RETRIES: u32 = 4;

/// Polling slice for [`cancellable_sleep`]. Short enough that a cooperative
/// cancel during a multi-second backoff is observed promptly, long enough to
/// stay effectively free when no cancel arrives.
const CANCEL_POLL_SLICE: Duration = Duration::from_millis(100);

/// Base delay for the first retry (doubles each subsequent attempt).
const BASE_DELAY: Duration = Duration::from_millis(500);

/// Base delay for rate-limit (429) errors — longer to respect limits.
const RATE_LIMIT_BASE_DELAY: Duration = Duration::from_secs(5);

/// Maximum per-retry delay cap.
const MAX_DELAY: Duration = Duration::from_secs(30);

/// Wall-clock budget for riding out a rate-limit / overload (429 / 529 /
/// "overloaded") window before giving up. A subscription throttle or a provider
/// overload routinely lasts *minutes*, so the old tiny attempt budget (≈65 s)
/// gave up mid-window and returned `Fail` — which kills the turn and forces the
/// user to manually re-issue "continue" into a still-throttled limit. With a
/// wall-clock budget we keep retrying (on the 30 s-capped rate-limit backoff)
/// until capacity frees up and the turn resumes on its own. Long enough to
/// outlast a per-minute throttle, short enough that a hard daily cap doesn't
/// hang forever — and the caller's cancel flag (Ctrl+C / esc) aborts the wait at
/// any point regardless.
const RATE_LIMIT_MAX_ELAPSED: Duration = Duration::from_secs(300);

/// Env override (milliseconds) for [`RATE_LIMIT_MAX_ELAPSED`]. `0` opts out of
/// the wall-clock budget and falls back to the bounded [`MAX_RETRIES`] attempt
/// count (the pre-wall-clock behaviour); a bad value uses the default.
const RATE_LIMIT_MAX_ELAPSED_ENV: &str = "ZO_RATE_LIMIT_MAX_WAIT_MS";

/// Resolve the rate-limit wall-clock budget. `None` means "no wall-clock budget"
/// (env opt-out), in which case rate-limit errors fall back to the bounded
/// attempt schedule.
fn rate_limit_max_elapsed() -> Option<Duration> {
    match std::env::var(RATE_LIMIT_MAX_ELAPSED_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(ms) => Some(Duration::from_millis(ms)),
            Err(_) => Some(RATE_LIMIT_MAX_ELAPSED),
        },
        Err(_) => Some(RATE_LIMIT_MAX_ELAPSED),
    }
}

/// Transient error classifier result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryVerdict {
    /// The error is transient — retry after the recommended delay.
    Retry { delay: Duration },
    /// The error is permanent or retries are exhausted — propagate.
    Fail,
}

/// Determine whether an error message represents a transient failure
/// that should be retried.
///
/// The classifier inspects the stringified error for known patterns:
/// - HTTP 429 (rate limit / overloaded)
/// - HTTP 500, 502, 503, 529 (server errors)
/// - Connection/timeout errors
///
/// `attempt` is 0-indexed (0 = first attempt just failed); `elapsed` is the
/// wall-clock time already spent retrying this operation (used only for the
/// rate-limit budget — pass `Duration::ZERO` for a pure attempt-based check).
#[must_use]
pub fn classify_for_retry(error_message: &str, attempt: u32, elapsed: Duration) -> RetryVerdict {
    // Provider clients already spent their own retry budget before returning
    // this wrapper. Re-entering the whole provider ladder here turns one retry
    // into another six HTTP attempts and defeats verifier failover.
    let normalized = error_message.to_ascii_lowercase();
    if normalized.contains("api failed after ") && normalized.contains(" attempts:") {
        return RetryVerdict::Fail;
    }

    // The retry *vocabulary* (which words mean "rate limit" vs "transient" vs
    // "fatal") lives in one place — `core_types::retry_signal` — so the api
    // layer, the conversation UI, and the stream parser all classify from the
    // same words. This site owns only the *consequence*.
    let signal = classify_error_text(error_message);
    if !signal.is_retryable() {
        return RetryVerdict::Fail;
    }

    // A capacity stall (429 / 529 / overloaded) is bounded by a *wall-clock*
    // budget, not the tiny attempt count: these windows last minutes, so giving
    // up after ~65 s returns `Fail` → the turn dies and the user must manually
    // re-issue "continue" into a still-throttled limit. Riding it out on the
    // 30 s-capped rate-limit backoff lets the turn resume by itself the moment
    // capacity frees up. The caller's cancel flag still aborts the wait.
    if signal.is_rate_limit() {
        match rate_limit_max_elapsed() {
            Some(budget) if elapsed < budget => {
                return RetryVerdict::Retry {
                    delay: rate_limit_backoff(attempt),
                };
            }
            Some(_) => return RetryVerdict::Fail,
            // Env opt-out: fall through to the bounded attempt schedule below.
            None => {}
        }
    }

    // Generic transient blips (and rate-limit when the wall-clock budget is
    // disabled) use the bounded attempt schedule.
    if attempt >= MAX_RETRIES {
        return RetryVerdict::Fail;
    }
    let delay = if signal.is_rate_limit() {
        rate_limit_backoff(attempt)
    } else {
        backoff_delay(attempt)
    };
    RetryVerdict::Retry { delay }
}

/// Feed a foreground main-turn capacity stall into the process-global
/// per-provider cool-down state (`api::quota`) so the router's headroom penalty
/// and the sub-agent admission gate observe the SAME throttle the main turn
/// just hit — closing the gap where only sub-agent 429s were recorded (the
/// foreground `retry_async` path here previously rode out the window without
/// ever marking the shared state).
///
/// Fires only for genuine rate-limit / overload signals — classified from the
/// same `core_types::retry_signal` vocabulary this module's backoff classifier
/// uses, so a generic 5xx / timeout blip (which does not consume provider
/// quota) is ignored. `attempt` is 0-indexed (the just-failed attempt), matching
/// [`api::quota::mark_rate_limit_cooldown_from`]'s exponential ladder. No
/// structured `Retry-After` is available at this `Display`-only error seam, so
/// the ladder drives the window (`None`); the migrated helper still honors a
/// present server hint verbatim wherever one IS available (the sub-agent path).
pub fn mark_foreground_rate_limit(model: &str, error_message: &str, attempt: u32) {
    if !classify_error_text(error_message).is_rate_limit() {
        return;
    }
    api::quota::mark_rate_limit_cooldown_from(api::detect_provider_kind(model), None, attempt);
}

/// Longer backoff for rate-limit (429) errors.
fn rate_limit_backoff(attempt: u32) -> Duration {
    backoff_delay_from(attempt, RATE_LIMIT_BASE_DELAY)
}

/// Compute the backoff delay for the given attempt with jitter.
///
/// Uses exponential backoff with ±25% jitter to prevent thundering herd.
fn backoff_delay(attempt: u32) -> Duration {
    backoff_delay_from(attempt, BASE_DELAY)
}

/// Shared exponential-backoff core for both the standard and rate-limit
/// schedules. They differ only in `base` — the exponential growth, the
/// [`MAX_DELAY`] cap, and the ±25% deterministic jitter are identical, so the
/// single implementation lives here and the two named entry points select the
/// base. Keeping one body means the cap/jitter logic can never drift between
/// the two schedules.
///
/// The jitter is a cheap hash of the attempt number (no RNG) — it just spreads
/// retry storms so concurrent clients don't re-fire in lockstep.
fn backoff_delay_from(attempt: u32, base: Duration) -> Duration {
    let base_ms = u64::try_from(base.as_millis()).unwrap_or(u64::MAX);
    // `attempt` can grow unbounded while riding out a rate-limit window on the
    // wall-clock budget, so cap the shift exponent — the result is `MAX_DELAY`-
    // capped just below anyway, and `1u64 << 64+` would overflow/panic.
    let exponential_ms = base_ms.saturating_mul(1u64 << attempt.min(32));
    let max_delay_ms = u64::try_from(MAX_DELAY.as_millis()).unwrap_or(u64::MAX);
    let capped_ms = exponential_ms.min(max_delay_ms);

    let jitter_range = capped_ms / 4;
    let jitter_offset = u64::from(attempt).wrapping_mul(2_654_435_761) % (jitter_range * 2 + 1);
    let jittered_ms = capped_ms
        .saturating_sub(jitter_range)
        .saturating_add(jitter_offset);

    Duration::from_millis(jittered_ms)
}

/// Sleep for `delay`, but wake early if `cancel` flips to `true`. Returns
/// `true` when it was cancelled mid-sleep, `false` when the full delay elapsed.
///
/// Polls a shared `AtomicBool` in short slices instead of taking a new
/// dependency on a cancellation primitive — the codebase's cooperative-cancel
/// convention. With `cancel == None` it is a plain `sleep`.
async fn cancellable_sleep(delay: Duration, cancel: Option<&AtomicBool>) -> bool {
    let Some(cancel) = cancel else {
        tokio::time::sleep(delay).await;
        return false;
    };
    let mut remaining = delay;
    while !remaining.is_zero() {
        if cancel.load(Ordering::Relaxed) {
            return true;
        }
        let slice = remaining.min(CANCEL_POLL_SLICE);
        tokio::time::sleep(slice).await;
        remaining = remaining.saturating_sub(slice);
    }
    cancel.load(Ordering::Relaxed)
}

/// Async retry helper that retries a future-producing closure on transient
/// errors.
///
/// `operation_name` is used for logging. `make_future` is called with the
/// current attempt number (0-indexed). `on_error(attempt, error)` fires for
/// every failed call before classification, including terminal/fail-fast
/// errors, so state such as provider cool-downs is not coupled to whether a
/// retry is scheduled. `on_retry(next_attempt, delay, error)` fires once per
/// scheduled retry *before* the backoff sleep — `next_attempt` is the upcoming
/// 1-based retry number — so a live UI can surface "retrying in Ns" instead of
/// going silent for the whole wait. `cancel`, when set, aborts the backoff sleep
/// and stops retrying — the last error propagates — so a foreground Ctrl+C is
/// observed during the wait instead of only at retry boundaries.
/// `fail_fast_on_rate_limit` leaves generic transient retries intact but returns
/// the first capacity error so a higher-level cross-provider failover can run
/// immediately.
pub async fn retry_async<F, Fut, T, E, O, R>(
    _operation_name: &str,
    cancel: Option<&AtomicBool>,
    fail_fast_on_rate_limit: bool,
    mut on_error: O,
    mut on_retry: R,
    mut make_future: F,
) -> Result<T, E>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
    O: FnMut(u32, &E),
    R: FnMut(u32, Duration, &E),
{
    let started = Instant::now();
    let mut attempt = 0u32;
    loop {
        match make_future(attempt).await {
            Ok(value) => return Ok(value),
            Err(error) => {
                on_error(attempt, &error);
                let error_text = error.to_string();
                let verdict = if fail_fast_on_rate_limit
                    && classify_error_text(&error_text).is_rate_limit()
                {
                    RetryVerdict::Fail
                } else {
                    classify_for_retry(&error_text, attempt, started.elapsed())
                };
                match verdict {
                    RetryVerdict::Retry { delay } => {
                        on_retry(attempt + 1, delay, &error);
                        if cancellable_sleep(delay, cancel).await {
                            return Err(error);
                        }
                        attempt += 1;
                    }
                    RetryVerdict::Fail => return Err(error),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_429_as_transient() {
        let verdict = classify_for_retry("HTTP 429 Too Many Requests", 0, Duration::ZERO);
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn classifies_500_as_transient() {
        let verdict = classify_for_retry("HTTP 500 Internal Server Error", 0, Duration::ZERO);
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn classifies_520_unknown_status_as_transient() {
        let verdict = classify_for_retry(
            "api returned 520 <unknown status code>: error code: 520",
            0,
            Duration::ZERO,
        );
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn classifies_overloaded_as_transient() {
        let verdict = classify_for_retry("overloaded_error: Overloaded", 0, Duration::ZERO);
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn classifies_timeout_as_transient() {
        let verdict = classify_for_retry("request timed out", 0, Duration::ZERO);
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn classifies_rate_limit_as_transient() {
        let verdict =
            classify_for_retry("rate_limit_error: rate limit exceeded", 0, Duration::ZERO);
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn classifies_backend_terminal_stream_failure_as_transient() {
        let verdict = classify_for_retry(
            "turn: runtime: provider stream: transport error: api stream error: backend reported a terminal stream failure",
            0,
            Duration::ZERO,
        );
        assert!(matches!(verdict, RetryVerdict::Retry { .. }));
    }

    #[test]
    fn exhausted_provider_retry_ladder_is_terminal() {
        let verdict = classify_for_retry(
            "api failed after 6 attempts: api returned 429 Too Many Requests",
            0,
            Duration::ZERO,
        );
        assert_eq!(
            verdict,
            RetryVerdict::Fail,
            "the runtime must not re-enter a provider retry ladder that already exhausted"
        );
    }

    #[test]
    fn classifies_auth_error_as_permanent() {
        let verdict =
            classify_for_retry("authentication_error: invalid API key", 0, Duration::ZERO);
        assert_eq!(verdict, RetryVerdict::Fail);
    }

    #[test]
    fn classifies_validation_error_as_permanent() {
        let verdict =
            classify_for_retry("invalid_request_error: messages too long", 0, Duration::ZERO);
        assert_eq!(verdict, RetryVerdict::Fail);
    }

    #[test]
    fn exhausts_transient_retries_after_max() {
        // A generic transient (non-capacity) blip is bounded by the attempt count.
        let verdict =
            classify_for_retry("HTTP 500 Internal Server Error", MAX_RETRIES, Duration::ZERO);
        assert_eq!(verdict, RetryVerdict::Fail);
    }

    #[test]
    fn rate_limit_retries_past_attempt_budget_within_wall_clock() {
        // A 429 / overload must NOT give up at MAX_RETRIES: it rides the window
        // out on the wall-clock budget so the turn resumes instead of dying and
        // forcing a manual "continue" into a still-throttled limit.
        let verdict = classify_for_retry(
            "HTTP 429 Too Many Requests",
            MAX_RETRIES + 10,
            Duration::from_secs(10),
        );
        assert!(
            matches!(verdict, RetryVerdict::Retry { .. }),
            "rate-limit must keep retrying past the attempt budget while inside the wall-clock window"
        );
    }

    #[test]
    fn rate_limit_gives_up_after_wall_clock_budget() {
        let verdict = classify_for_retry(
            "HTTP 429 Too Many Requests",
            2,
            RATE_LIMIT_MAX_ELAPSED + Duration::from_secs(1),
        );
        assert_eq!(
            verdict,
            RetryVerdict::Fail,
            "rate-limit must give up once the wall-clock budget is spent"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn backoff_increases_exponentially() {
        let d0 = backoff_delay(0);
        let d1 = backoff_delay(1);
        let d2 = backoff_delay(2);
        // Each delay should be roughly double the previous (within jitter)
        assert!(d1.as_millis() > d0.as_millis());
        assert!(d2.as_millis() > d1.as_millis());
        // But never exceeds the cap
        assert!(d2 <= MAX_DELAY + Duration::from_millis(MAX_DELAY.as_millis() as u64 / 4));
    }

    #[test]
    fn backoff_is_capped() {
        let d10 = backoff_delay(10);
        // Even at attempt 10, delay should be near MAX_DELAY (not overflow)
        assert!(d10.as_millis() <= (MAX_DELAY.as_millis() * 5 / 4));
    }

    #[tokio::test]
    async fn cancellable_sleep_wakes_early_on_cancel() {
        let cancel = AtomicBool::new(true);
        let start = std::time::Instant::now();
        let was_cancelled = cancellable_sleep(Duration::from_secs(30), Some(&cancel)).await;
        assert!(was_cancelled);
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "a pre-set cancel must not wait out the full delay"
        );
    }

    #[tokio::test]
    async fn cancellable_sleep_completes_without_cancel() {
        let start = std::time::Instant::now();
        let was_cancelled = cancellable_sleep(Duration::from_millis(50), None).await;
        assert!(!was_cancelled);
        assert!(start.elapsed() >= Duration::from_millis(40));
    }

    #[tokio::test]
    async fn retry_sleep_aborts_on_cancel() {
        // A transient error schedules a backoff; with the cancel flag set, the
        // sleep is abandoned and the last error propagates instead of retrying.
        let cancel = AtomicBool::new(true);
        let attempts = std::cell::Cell::new(0u32);
        let start = std::time::Instant::now();
        let result: Result<(), String> = retry_async(
            "test",
            Some(&cancel),
            false,
            |_, _| {},
            |_, _, _| {},
            |attempt| {
                attempts.set(attempt + 1);
                async move { Err::<(), String>("HTTP 429 Too Many Requests".to_string()) }
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.get(), 1, "cancel must stop further retry attempts");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "the rate-limit backoff must be cut short by cancel"
        );
    }

    #[tokio::test]
    async fn fail_fast_rate_limit_marks_cooldown_without_runtime_retry() {
        let model = "claude-sonnet-4-5";
        let kind = api::detect_provider_kind(model);
        let attempts = std::cell::Cell::new(0u32);
        let observed_errors = std::cell::Cell::new(0u32);
        let notices = std::cell::Cell::new(0u32);
        let result: Result<(), String> = retry_async(
            "test",
            None,
            true,
            |attempt, error: &String| {
                observed_errors.set(observed_errors.get() + 1);
                mark_foreground_rate_limit(model, error, attempt);
            },
            |_, _, _| notices.set(notices.get() + 1),
            |attempt| {
                attempts.set(attempt + 1);
                async move { Err::<(), String>("HTTP 429 Too Many Requests".to_string()) }
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.get(), 1, "fail-fast must return the first 429");
        assert_eq!(
            observed_errors.get(),
            1,
            "the terminal 429 must reach the cooldown observer"
        );
        assert!(
            api::quota::rate_limit_cooldown_remaining_ms(kind) > 0,
            "the fail-fast 429 must open the provider cooldown"
        );
        assert_eq!(notices.get(), 0, "fail-fast must not schedule a backoff");
    }

    #[test]
    fn overloaded_uses_the_longer_rate_limit_backoff() {
        // An `overloaded_error` (HTTP 529) is a capacity signal: it must back off
        // on the rate-limit schedule (seconds), not the short transient one
        // (sub-second), so the provider pool has time to recover.
        let RetryVerdict::Retry { delay: overload } = classify_for_retry(
            "transport error: api stream error (overloaded_error): Overloaded",
            0,
            Duration::ZERO,
        ) else {
            panic!("overloaded must be retried");
        };
        let RetryVerdict::Retry { delay: server } =
            classify_for_retry("HTTP 500 Internal Server Error", 0, Duration::ZERO)
        else {
            panic!("500 must be retried");
        };
        assert!(
            overload >= RATE_LIMIT_BASE_DELAY.saturating_sub(RATE_LIMIT_BASE_DELAY / 4),
            "overload backoff ({overload:?}) must use the rate-limit schedule"
        );
        assert!(
            overload > server,
            "overload ({overload:?}) must wait longer than a generic 5xx ({server:?})"
        );
    }

    #[tokio::test]
    async fn on_retry_fires_before_each_backoff() {
        // The notice hook lets a live UI report the stall; assert it is invoked
        // once per scheduled retry with the 1-based attempt number and a delay.
        let cancel = AtomicBool::new(true); // abort after the first notice
        let notices = std::cell::RefCell::new(Vec::new());
        let _: Result<(), String> = retry_async(
            "test",
            Some(&cancel),
            false,
            |_, _| {},
            |attempt, delay, error: &String| {
                notices.borrow_mut().push((attempt, delay, error.clone()));
            },
            |_attempt| async move { Err::<(), String>("overloaded_error: Overloaded".to_string()) },
        )
        .await;
        let notices = notices.into_inner();
        assert_eq!(
            notices.len(),
            1,
            "one notice before the (cancelled) backoff"
        );
        assert_eq!(notices[0].0, 1, "first retry is reported as attempt 1");
        assert!(
            notices[0].1 > Duration::ZERO,
            "a positive backoff is reported"
        );
    }

    /// The foreground main-turn 429 must feed the SAME per-provider cool-down
    /// state (`api::quota`) the sub-agent path reads — and only for genuine
    /// rate-limit/overload signals. We read the provider `detect_provider_kind`
    /// resolves for a fixed model and assert the transition on that exact slot
    /// (robust to whichever kind the env maps it to); no other runtime test
    /// marks the cool-down state, so the slot is clean.
    #[test]
    fn foreground_rate_limit_marks_provider_cooldown() {
        let model = "claude-sonnet-4-5";
        let kind = api::detect_provider_kind(model);
        // A generic 5xx is not a capacity signal → it must NOT open a cool-down.
        let before = api::quota::rate_limit_cooldown_remaining_ms(kind);
        mark_foreground_rate_limit(model, "HTTP 500 Internal Server Error", 0);
        assert_eq!(
            api::quota::rate_limit_cooldown_remaining_ms(kind),
            before,
            "a 5xx blip must not mark a rate-limit cool-down"
        );
        // A 429 is a capacity stall → it must open the provider's cool-down.
        mark_foreground_rate_limit(model, "HTTP 429 Too Many Requests", 0);
        assert!(
            api::quota::rate_limit_cooldown_remaining_ms(kind) > 0,
            "a foreground 429 must open the provider cool-down the router/spawn gate reads"
        );
    }
}
