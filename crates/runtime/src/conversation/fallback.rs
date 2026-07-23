//! Refusal and cross-provider quota fallback for [`ConversationRuntime`], split
//! out of `mod.rs` so the turn loops there read as orchestration.
//! Behaviour-preserving: these were `ConversationRuntime` methods and
//! module-level helpers, now `pub(super)` where the loops in `mod.rs` still
//! reach them.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::message_stream::types::{BlockIdGen, RenderBlock};
use crate::session::{ContentBlock, ConversationMessage};

use super::{
    ApiClient, ApiRequest, AssistantEvent, AsyncApiClient, ConversationRuntime, RuntimeError,
    ToolExecutor, DEFAULT_STREAMING_CHANNEL_CAPACITY,
};

/// Anthropic's recommended client-side fallback for a Fable/Mythos
/// safety-classifier refusal. A Claude Fable 5 request that trips the
/// classifier (aggressive cyber, life-sciences, or reasoning-extraction) returns
/// HTTP 200 with `stop_reason: "refusal"`; benign security/bio work can trip it
/// too, so Anthropic's guidance is to retry the same turn on Opus 4.8.
const REFUSAL_FALLBACK_MODEL: &str = "claude-opus-4-8";
/// System-level warn shown when a Fable/Mythos refusal is auto-retried on the
/// fallback model.
pub(super) const REFUSAL_FALLBACK_WARN: &str = core_types::REFUSAL_FALLBACK_WARN;
/// System-level warn shown once when the session first pre-arms the refusal
/// fallback during its cooldown.
pub(super) const REFUSAL_DRY_PREARM_WARN: &str =
    core_types::retry_signal::REFUSAL_DRY_PREARM_WARN;
/// System-level notice shown when a refusal cannot be auto-retried: the turn is
/// already on the fallback model (Opus 4.8 refused too), or the active model is
/// not a Fable/Mythos model. Surfaced honestly instead of looping forever.
pub(super) const REFUSAL_SURFACED_NOTICE: &str =
    "The model's safety classifier declined this request; no automatic retry was possible.";

/// Consecutive public turns that must hit the refusal fallback before the
/// session stops probing Fable/Mythos on every new turn. Two catches a sticky
/// context-triggered classifier without parking the session after one isolated
/// refusal. See [`ConversationRuntime::refusal_consecutive_turns`].
const REFUSAL_DRY_TURN_THRESHOLD: u8 = 2;

/// Session-scoped refusal cooldown after [`REFUSAL_DRY_TURN_THRESHOLD`]
/// consecutive public turns. Thirty minutes avoids paying one doomed Fable
/// request per turn in a long classifier-sticky session while still probing
/// Fable again automatically. Process memory only, like the quota cooldown.
const REFUSAL_DRY_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// Session-scoped cooldown applied after a quota fallback fires when the
/// provider gave no `retry_after` hint. Fifteen minutes: long enough to ride
/// out a typical subscription/quota throttle window without re-spending the
/// main model's rate-limit retry budget every turn, short enough that a
/// transient exhaustion never parks the whole session off its configured model
/// for the rest of the day. While the session is inside this window every turn
/// pre-arms straight onto the fallback client; once it elapses the session
/// returns to the main model on its own. See [`ConversationRuntime::quota_dry_until`].
const QUOTA_FALLBACK_DEFAULT_COOLDOWN: std::time::Duration =
    std::time::Duration::from_secs(15 * 60);

/// Loud warn line for the mid-turn quota swap: the main model's quota window is
/// exhausted, so this turn continues on the cross-provider fallback and the
/// session returns to the main model once the recorded cooldown clears.
pub(super) fn quota_fallback_swap_warn(model: &str) -> String {
    format!(
        "{prefix}{model}; the main model is rate-limited (quota exhausted), so this turn \
         continues on {model}. The main model resumes automatically once the cooldown clears.",
        prefix = core_types::QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX,
    )
}

/// Short info line for a turn that pre-arms onto the fallback because the
/// session is still inside the quota-dry cooldown from an earlier turn.
pub(super) fn quota_fallback_prearm_info(model: &str) -> String {
    format!(
        "{prefix}{model}; the main model is still cooling down from a rate limit, so this turn \
         continues on {model}.",
        prefix = core_types::QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX,
    )
}

/// Loud warn line for holding the turn on the main model instead of falling
/// back: the quota window lifts within the configured wait band, so the turn
/// waits it out rather than switching providers.
pub(super) fn quota_wait_hold_warn(model: &str, wait: std::time::Duration) -> String {
    let secs = wait.as_secs();
    let human = if secs >= 60 {
        format!("~{}m", secs.div_ceil(60))
    } else {
        format!("{}s", secs.max(1))
    };
    // Starts with the shared prefix so the TUI can flip the spinner into the
    // quota-hold state — see `core_types::QUOTA_HOLD_NOTICE_PREFIX`.
    format!(
        "{prefix} ({model}); its quota window resets in {human}, so this turn \
         holds on {model} rather than switching providers. Press esc to interrupt.",
        prefix = core_types::QUOTA_HOLD_NOTICE_PREFIX,
    )
}

/// How the turn loop should escape a hard `RateLimit` on the main model.
pub(super) enum QuotaEscape {
    /// Hold on the main model: its exhausted window lifts within the wait band.
    /// The caller sleeps this long (the TUI keeps its live elapsed + interrupt
    /// affordance), then re-requests the SAME turn on the SAME model — no
    /// provider swap and no session cooldown recorded.
    Wait(std::time::Duration),
    /// Swap this turn onto the cross-provider fallback (`model`) and re-request.
    Fallback(String),
    /// Neither applies — the caller fails the turn as it did before the feature.
    None,
}

/// Whether a provider stop reason is a safety-classifier refusal. Case- and
/// whitespace-insensitive so a provider reporting `"Refusal"` or a padded value
/// still matches. Only Anthropic emits this; the fallback path additionally
/// gates on the active model family (see [`is_anthropic_claude_model`]) so a
/// non-Anthropic provider is never affected even if one ever reported it.
pub(super) fn is_refusal_stop_reason(reason: &str) -> bool {
    reason.trim().eq_ignore_ascii_case("refusal")
}

/// Whether `model` is an Anthropic Claude model — the only family that emits a
/// `refusal` stop reason. Gates the refusal-fallback path so Gemini/OpenAI/
/// `DeepSeek` are never touched. Matches on the family substrings rather than an
/// exact id so aliases (`fable`, `mythos`) and dated variants still classify.
fn is_anthropic_claude_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("claude") || lower.contains("fable") || lower.contains("mythos")
}

/// Whether `model` is a Fable/Mythos model — the models whose safety classifier
/// can decline a benign request, so a one-time Opus 4.8 fallback is warranted.
/// An Opus/Sonnet/Haiku refusal is surfaced honestly instead of retried.
fn is_fable_or_mythos_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("fable") || lower.contains("mythos")
}

/// How the turn loop reacts to a `stop_reason: "refusal"`.
pub(super) enum RefusalDecision {
    /// Not a refusal this runtime handles (non-Anthropic or unknown model) —
    /// consume the turn through the ordinary content/empty path.
    Proceed,
    /// Fell back to Opus 4.8 for this turn (the override is now set); the caller
    /// drops the refused partial and re-requests the same turn once.
    Retry,
    /// Cannot retry (already fell back this turn, or the active model is not a
    /// Fable/Mythos model). Surface a notice and end the turn.
    Surface,
}

/// The synthetic assistant message recorded when a refusal is surfaced (rather
/// than retried), so the turn is well-formed — a user turn is never left with no
/// assistant response — and the notice is visible on the headless path.
pub(super) fn refusal_surfaced_message() -> ConversationMessage {
    ConversationMessage::assistant(vec![ContentBlock::Text {
        text: REFUSAL_SURFACED_NOTICE.to_string(),
    }])
}
/// Drive an [`AsyncApiClient`] to completion from a synchronous context and
/// return its assembled [`AssistantEvent`] sequence, discarding the render
/// deltas. Used only by the sync turn loop's quota-fallback dispatch
/// ([`ConversationRuntime::sync_stream_events`]): the cross-provider fallback
/// client is async, but `run_turn_once` runs without an ambient runtime on the
/// headless `-p` text path. Mirrors the runtime-flavor-aware sync→async bridge
/// in [`crate::bash`] so it is correct whether called with no runtime (fresh
/// current-thread runtime), inside a multi-thread runtime (`block_in_place`), or
/// inside a current-thread runtime such as a `#[tokio::test]` (a dedicated
/// thread, since `block_in_place` panics on single-threaded executors).
fn block_on_async_client(
    client: Arc<dyn AsyncApiClient>,
    request: ApiRequest,
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    // A bounded render channel drained concurrently with the stream: the client
    // "should short-circuit as soon as the next send().await fails", so the
    // receiver must stay alive until the stream finishes. Discard every block.
    async fn drive(
        client: Arc<dyn AsyncApiClient>,
        request: ApiRequest,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let (tx, mut rx) = mpsc::channel::<RenderBlock>(DEFAULT_STREAMING_CHANNEL_CAPACITY);
        let block_id = BlockIdGen::default().next();
        let stream = client.stream_async(request, tx, block_id);
        tokio::pin!(stream);
        loop {
            tokio::select! {
                result = &mut stream => {
                    while rx.try_recv().is_ok() {}
                    return result;
                }
                _ = rx.recv() => {}
            }
        }
    }

    use tokio::runtime::{Builder, Handle, RuntimeFlavor};
    if let Ok(handle) = Handle::try_current() {
        return if handle.runtime_flavor() == RuntimeFlavor::CurrentThread {
            std::thread::spawn(move || {
                let runtime = Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                runtime.block_on(drive(client, request))
            })
            .join()
            .map_err(|_| RuntimeError::new("quota fallback bridge thread panicked"))?
        } else {
            tokio::task::block_in_place(|| handle.block_on(drive(client, request)))
        };
    }
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| RuntimeError::new(error.to_string()))?;
    runtime.block_on(drive(client, request))
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    /// Install (or clear with `None`) the cross-provider client the turn loop
    /// swaps to when the main model's subscription/quota window is exhausted,
    /// together with its model id. The host re-derives and re-installs this on
    /// every turn entry — the top-ranked alternative on a *different* provider —
    /// so a model switch, a `/smart quota-fallback off`, or a change in
    /// connected providers never leaves a stale fallback installed. `None`
    /// disables the feature for the turn (a quota-exhausted turn then fails as
    /// it did before this feature existed). Mirrors [`Self::set_deep_verify_client`].
    pub fn set_quota_fallback_client(
        &mut self,
        client: Option<(Arc<dyn AsyncApiClient>, String)>,
    ) {
        self.quota_fallback_client = client;
    }

    /// The model id of the currently installed quota fallback client, or `None`
    /// when no fallback is installed. A read accessor so a host contract test
    /// can assert that a turn-entry install set (or a `None` route cleared) the
    /// fallback without reaching into private state.
    pub fn quota_fallback_model(&self) -> Option<&str> {
        self.quota_fallback_client.as_ref().map(|(_, model)| model.as_str())
    }

    /// Install (or clear) the per-turn confidence-cascade model escalation:
    /// the wire model this turn's requests run on instead of the session
    /// model. Host-managed set-or-clear at every turn entry (mirrors
    /// [`Self::set_quota_fallback_client`]) so an escalation is scoped to
    /// exactly the one armed turn. Same-provider only — this changes the wire
    /// model id on the bound client, never the client itself; the host
    /// resolver enforces the provider match.
    pub fn set_escalation_model_override(&mut self, model: Option<String>) {
        self.escalation_model_override = model.filter(|model| !model.trim().is_empty());
        // Freshness latch: the NEXT turn begin consumes this install; any
        // later begin without a fresh install clears the override (see the
        // field doc on `escalation_armed_fresh`).
        self.escalation_armed_fresh = self.escalation_model_override.is_some();
    }

    /// Escalation freshness at a turn begin (see `escalation_armed_fresh`):
    /// the first PUBLIC turn begin after the host installed an escalation
    /// keeps it — that IS the escalated turn; any later public begin without
    /// a fresh install drops the stale override, so a slash-command or
    /// queued-text turn on the same runtime never silently runs on a
    /// previous turn's Deep model. Internal deep-lane subturns are exempt:
    /// they run INSIDE the escalated turn (unlike the per-leg refusal reset,
    /// which deliberately gives each leg its own fallback budget).
    pub(super) fn begin_turn_escalation(&mut self, internal_subturn: bool) {
        if internal_subturn {
            return;
        }
        if self.escalation_armed_fresh {
            self.escalation_armed_fresh = false;
        } else {
            self.escalation_model_override = None;
        }
    }

    /// Set how close to a quota reset the turn loop holds on the main model
    /// instead of falling back (see [`Self::quota_wait_band`]). `ZERO` disables
    /// the band. Re-applied by the host each turn from `smart.quotaWaitBandMinutes`.
    pub fn set_quota_wait_band(&mut self, band: std::time::Duration) {
        self.quota_wait_band = band;
    }

    /// The quota-wait band currently armed on this runtime. A read accessor so a
    /// host contract test can assert that a turn-entry install refreshed the
    /// band from settings.
    pub fn quota_wait_band(&self) -> std::time::Duration {
        self.quota_wait_band
    }

    /// The async client the current streaming request runs on: the quota
    /// fallback when this turn has swapped to it (mid-turn exhaustion or a
    /// cooldown pre-arm), else the natively-installed streaming client. Keeps
    /// the swap in one place so the request-dispatch site stays a single
    /// expression. A `quota_fallback_active` turn with no fallback client
    /// installed (impossible today — arming requires the client) degrades to
    /// the native client rather than panicking.
    pub(super) fn active_async_client(&self) -> Option<&Arc<dyn AsyncApiClient>> {
        // A deep-gate leg's swapped client (cross-model planner/verifier or the
        // Architect implementer) wins over the quota fallback.
        if !self.deep_plan_leg_active
            && !self.deep_verify_leg_active
            && !self.exec_impl_leg_active
            && self.quota_fallback_active
        {
            if let Some((client, _)) = &self.quota_fallback_client {
                return Some(client);
            }
        }
        self.async_api_client.as_ref()
    }
    /// The model id the current request runs on, accounting for Architect
    /// implementer/PLAN swaps before the ordinary quota/refusal/session model
    /// ladder. `None` when the host never reported a model (bare test harness).
    ///
    /// The quota fallback takes precedence so the refusal path
    /// ([`Self::decide_refusal_fallback`]) judges the *active* model: a fallback
    /// on a non-Anthropic provider makes `is_anthropic_claude_model` false, so a
    /// spurious `refusal` from that provider yields `Proceed` instead of
    /// arming the Opus override (which would then force `claude-opus-4-8` onto a
    /// non-Anthropic client). This is the consistency the two fallbacks need to
    /// coexist.
    pub(super) fn effective_request_model(&self) -> Option<&str> {
        // During an Architect EXEC leg the request genuinely runs on the
        // implementer client, so refusal judgment must follow the model on the
        // wire: a GPT implementer's spurious `refusal` yields `Proceed`
        // instead of arming the Anthropic Opus override onto a GPT client —
        // the same consistency rule the quota fallback established below.
        if self.exec_impl_leg_active {
            if let Some(contract) = &self.exec_contract {
                return Some(contract.impl_model.as_str());
            }
        }
        if self.deep_plan_leg_active {
            if let Some((_, model)) = &self.deep_plan_client {
                return Some(model.as_str());
            }
        }
        // During a deep-gate VERIFY leg the request runs on the swapped verifier,
        // not the quota fallback, so judge refusals against the ordinary model
        // (unchanged pre-quota-fallback behavior).
        if !self.deep_plan_leg_active
            && !self.deep_verify_leg_active
            && self.quota_fallback_active
        {
            if let Some((_, model)) = &self.quota_fallback_client {
                return Some(model.as_str());
            }
        }
        self.refusal_fallback_model
            .as_deref()
            .or(self.escalation_model_override.as_deref())
            .or(self.context_model.as_deref())
    }

    /// Model whose provider owns capacity errors from the active stream.
    /// Deep PLAN/VERIFY swap only the client, so the leg model must override
    /// the ordinary turn model for quota accounting.
    pub(super) fn rate_limit_model_for_active_stream(&self) -> Option<&str> {
        if self.deep_plan_leg_active {
            return self
                .deep_plan_client
                .as_ref()
                .map(|(_, model)| model.as_str());
        }
        if self.deep_verify_leg_active {
            // The verifier client is a temporary swap that intentionally does
            // not alter `effective_request_model()` (refusal routing still
            // belongs to the ordinary turn model). Capacity accounting must
            // nevertheless follow the client actually on the wire.
            return self
                .deep_verify_candidates
                .get(self.deep_verify_candidate_idx)
                .map(|(_, model)| model.as_str());
        }
        if self.exec_impl_leg_active {
            // The Architect implementer swap: capacity errors belong to the
            // implementer's provider (`effective_request_model` already
            // reports it during this leg, but keep the explicit arm so the
            // wire-truth rule reads the same for both leg swaps).
            if let Some(contract) = &self.exec_contract {
                return Some(contract.impl_model.as_str());
            }
        }
        self.effective_request_model()
    }

    /// Decide how to react to a `stop_reason: "refusal"`. On [`RefusalDecision::Retry`]
    /// this arms the per-turn model override to [`REFUSAL_FALLBACK_MODEL`] as a
    /// side effect; the caller is responsible for dropping the refused partial
    /// (by not pushing it) and re-requesting. Anthropic-only: a non-Anthropic or
    /// unknown model yields [`RefusalDecision::Proceed`] so those providers keep
    /// their ordinary handling. The fallback is capped at one per turn — a second
    /// refusal (Opus 4.8 declined too) yields [`RefusalDecision::Surface`].
    pub(super) fn decide_refusal_fallback(&mut self) -> RefusalDecision {
        // Own the string so the `&self` borrow is dropped before the mutation.
        let Some(model) = self.effective_request_model().map(str::to_string) else {
            return RefusalDecision::Proceed;
        };
        if !is_anthropic_claude_model(&model) {
            return RefusalDecision::Proceed;
        }
        if self.refusal_fallback_model.is_some() {
            // Already swapped to the fallback this turn and it refused too.
            return RefusalDecision::Surface;
        }
        if is_fable_or_mythos_model(&model) {
            self.refusal_fallback_model = Some(REFUSAL_FALLBACK_MODEL.to_string());
            if !self.refusal_turn_hit {
                self.refusal_turn_hit = true;
                if self
                    .refusal_consecutive_turns
                    .saturating_add(1)
                    >= REFUSAL_DRY_TURN_THRESHOLD
                    && self
                        .refusal_dry_until
                        .is_none_or(|until| std::time::Instant::now() >= until)
                {
                    self.refusal_dry_until =
                        Some(std::time::Instant::now() + REFUSAL_DRY_COOLDOWN);
                    // The first begin that actually pre-arms owns the notice;
                    // do not let this mid-turn retry emit it prematurely.
                    self.refusal_prearm_notice_pending = false;
                    self.refusal_prearm_notice_latched = false;
                }
            }
            RefusalDecision::Retry
        } else {
            // A refusal on Opus/Sonnet/Haiku: surface honestly, do not loop.
            RefusalDecision::Surface
        }
    }

    /// Fold the just-finished PUBLIC turn into the consecutive-refusal streak.
    /// Callers deliberately skip this for internal subturns/continuations so
    /// multiple deep-lane legs cannot count as multiple user turns. The hit
    /// flag is cleared only at this public boundary; each successful
    /// [`RefusalDecision::Retry`] within the new turn can set it once.
    pub(super) fn fold_finished_refusal_turn(&mut self) {
        if self.refusal_turn_hit {
            self.refusal_consecutive_turns = self.refusal_consecutive_turns.saturating_add(1);
        } else {
            self.refusal_consecutive_turns = 0;
        }
        self.refusal_turn_hit = false;
    }

    /// Turn-start refusal-cooldown management, called after the ordinary
    /// per-leg override reset and after quota/escalation turn state has settled:
    ///
    /// - an elapsed cooldown clears itself and lets Fable/Mythos serve natively;
    /// - an active cooldown pre-arms Opus 4.8 only when the model that would
    ///   otherwise serve this request is Fable/Mythos on the native client;
    /// - the first such pre-arm latches one warning, while later dry turns stay
    ///   silent.
    ///
    /// This still runs for internal subturns because each leg resets its
    /// one-shot override. Only the streak fold above is public-turn-only.
    pub(super) fn begin_turn_refusal_fallback(&mut self) {
        if self
            .refusal_dry_until
            .is_some_and(|until| std::time::Instant::now() >= until)
        {
            self.refusal_dry_until = None;
            self.refusal_prearm_notice_pending = false;
            self.refusal_prearm_notice_latched = false;
        }
        let should_prearm = self.refusal_dry_until.is_some()
            // A same-provider refusal override must never ride the
            // cross-provider quota-fallback client.
            && !self.quota_fallback_active
            && self
                .effective_request_model()
                .is_some_and(is_fable_or_mythos_model);
        if !should_prearm {
            return;
        }
        self.refusal_fallback_model = Some(REFUSAL_FALLBACK_MODEL.to_string());
        if !self.refusal_prearm_notice_latched {
            self.refusal_prearm_notice_pending = true;
            self.refusal_prearm_notice_latched = true;
        }
    }

    /// Turn-start quota-fallback state management, called from both turn entry
    /// points (`begin_turn_once`, `begin_streaming_turn`). Mirrors the per-turn
    /// refusal-fallback reset but with the session-cooldown twist:
    ///
    /// - a stale/elapsed [`Self::quota_dry_until`] is dropped so the session
    ///   returns to the main model on its own (natural recovery);
    /// - while still inside the cooldown window AND a fallback client is
    ///   installed, the turn PRE-arms straight onto the fallback so it does not
    ///   re-spend the main model's rate-limit retry budget just to rediscover
    ///   the wall is still up (a short notice is latched for the turn loop);
    /// - otherwise the turn starts on the main model (a live mid-turn
    ///   exhaustion can still arm the fallback within the turn).
    ///
    /// Applies uniformly to internal deep-lane subturns too: a quota-dry session
    /// applies to every leg.
    pub(super) fn begin_turn_quota_fallback(&mut self) {
        // The wait-band one-shot is per turn: a fresh turn may wait again.
        self.quota_waited_this_turn = false;
        if self
            .quota_dry_until
            .is_some_and(|until| std::time::Instant::now() >= until)
        {
            self.quota_dry_until = None;
        }
        let cooldown_active = self.quota_dry_until.is_some();
        if cooldown_active && self.quota_fallback_client.is_some() {
            self.quota_fallback_active = true;
            self.quota_prearm_notice_pending = true;
        } else {
            // No cooldown, or the cooldown outlived its fallback client (e.g.
            // `/smart quota-fallback off` mid-session cleared it): start native.
            self.quota_fallback_active = false;
            self.quota_prearm_notice_pending = false;
        }
    }

    /// Decide how a turn-killing error should escape a hard `RateLimit` on the
    /// main model. Only a [`api::ProviderErrorClass::RateLimit`] that survived
    /// the retry budget (the wall is still up) is a candidate; anything else, or
    /// a turn already on the fallback, yields [`QuotaEscape::None`] so the caller
    /// fails exactly as before.
    ///
    /// Preference order:
    /// 1. **Wait** — when the exhausted window lifts within [`Self::quota_wait_band`]
    ///    (from the 429's `retry_after` hint and/or the measured window resets,
    ///    per [`api::quota::reset_wait_within_band`]) and this turn has not waited
    ///    yet. The caller sleeps and re-requests on the SAME model; no swap, no
    ///    cooldown. A single one-shot ([`Self::quota_waited_this_turn`]) prevents
    ///    a lying "reset now" header from looping the wait.
    /// 2. **Fallback** — otherwise, when the provider's measured quota permits
    ///    a model swap and a cross-provider client is installed: arm the swap
    ///    (records [`Self::quota_dry_until`] and clears any refusal→Opus
    ///    override, which would otherwise force `claude-opus-4-8` onto the
    ///    fallback client whose own target model is correct).
    /// 3. **None** — no band match, fallback is below the Anthropic 95% gate,
    ///    or no fallback client is installed.
    pub(super) fn decide_quota_escape(&mut self, error: &RuntimeError) -> QuotaEscape {
        self.decide_quota_escape_with_gate(error, ::api::quota::quota_fallback_permitted)
    }

    pub(super) fn decide_quota_escape_with_gate(
        &mut self,
        error: &RuntimeError,
        fallback_permitted: impl FnOnce(::api::ProviderKind) -> bool,
    ) -> QuotaEscape {
        // A deep-gate PLAN/VERIFY leg runs on a swapped deep client, so a hard
        // `RateLimit` here belongs to that provider, not the main model's quota
        // window. Never arm the main-turn quota fallback from it:
        // that would poison `quota_fallback_active`/`quota_dry_until` and force
        // every later main turn onto the fallback client. The verifier's own
        // 429 failover is handled by the deep gate's ranked-candidate loop
        // ([`verify_subturn`]), which never touches this state.
        if self.deep_plan_leg_active || self.deep_verify_leg_active {
            return QuotaEscape::None;
        }
        if self.quota_fallback_active {
            return QuotaEscape::None;
        }
        let Some(::api::ProviderErrorClass::RateLimit { retry_after }) =
            error.provider_error_class()
        else {
            return QuotaEscape::None;
        };
        let main_provider = self
            .context_model
            .as_deref()
            .map(::api::detect_provider_kind);
        // Prefer holding on the main model when its wall lifts within the band.
        if !self.quota_waited_this_turn && !self.quota_wait_band.is_zero() {
            if let Some(provider) = main_provider {
                if let Some(wait) = ::api::quota::reset_wait_within_band(
                    provider,
                    retry_after,
                    self.quota_wait_band,
                ) {
                    self.quota_waited_this_turn = true;
                    return QuotaEscape::Wait(wait);
                }
            }
        }
        // A blocked swap gets at most the one bounded wait above. If that wait
        // was unavailable or has already been consumed, end the turn normally
        // instead of creating an unbounded same-model retry loop.
        if main_provider.is_some_and(|provider| !fallback_permitted(provider)) {
            return QuotaEscape::None;
        }
        let Some((_, model)) = self.quota_fallback_client.as_ref() else {
            return QuotaEscape::None;
        };
        let model = model.clone();
        self.quota_fallback_active = true;
        self.refusal_fallback_model = None;
        // Sister clear to the refusal one above, same invariant: a per-turn
        // wire-model override from the main model's world must never ride
        // the cross-provider fallback client (assemble_request also guards
        // on `quota_fallback_active` — this keeps the state itself honest).
        self.escalation_model_override = None;
        let cooldown = retry_after.unwrap_or(QUOTA_FALLBACK_DEFAULT_COOLDOWN);
        self.quota_dry_until = Some(std::time::Instant::now() + cooldown);
        QuotaEscape::Fallback(model)
    }

    /// Drive one request through whichever client the SYNC turn loop
    /// ([`Self::run_turn_once`]) should use: the native [`ApiClient::stream`]
    /// normally, or — when this turn has swapped to the cross-provider quota
    /// fallback — the async fallback client driven to completion on a scoped
    /// tokio runtime (mirrors `bash.rs`'s sync→async bridge). The fallback is an
    /// [`AsyncApiClient`]; the sync loop has no ambient runtime (the headless
    /// `-p` text path is a sync context), so the bridge picks the same three
    /// cases `bash.rs` does. The render deltas the async client emits are
    /// drained and discarded — the sync loop rebuilds its own view from the
    /// returned events, exactly as it does for the native client.
    pub(super) fn sync_stream_events(
        &mut self,
        request: ApiRequest,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        if self.quota_fallback_active {
            if let Some((client, _)) = self.quota_fallback_client.clone() {
                return block_on_async_client(client, request);
            }
        }
        self.api_client.stream(request)
    }
}

#[cfg(test)]
mod tests {
    use super::{quota_fallback_prearm_info, quota_fallback_swap_warn};

    #[test]
    fn quota_fallback_notices_round_trip_the_active_model() {
        let model = "openai:gpt-5.6-sol";
        for notice in [
            quota_fallback_swap_warn(model),
            quota_fallback_prearm_info(model),
        ] {
            assert_eq!(core_types::parse_quota_fallback_model(&notice), Some(model));
            assert!(!notice.starts_with(core_types::QUOTA_HOLD_NOTICE_PREFIX));
        }
    }
}
