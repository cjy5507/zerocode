//! Live turn activity indicator.
//!
//! Shows a one-line, human-readable status while a turn is in flight, e.g.:
//!
//! ```text
//! ✦ Thinking… (1m 51s · ↓ 2.3k tokens · esc to interrupt)
//! ```
//!
//! Owned by `App`; rendered into the `rule_top` row of the layout when
//! a turn is active. The glyph cadence is time-based so it stays smooth
//! even when redraws are event-driven instead of fixed-tick.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthChar;

use super::glyphs;
use super::heat::IGNITION_MS;
use super::hud::{HudState, HudViewModel};
use super::term::reduce_motion_enabled;
use super::theme::Theme;

/// Neutral live-reasoning verbs used only while prose is streaming. The
/// selected entry is latched in [`TurnActivity`] from the host's stable turn
/// generation so the word never changes while one response streams.
pub const ZO_VERBS: [&str; 6] = [
    "Thinking",
    "Planning",
    "Exploring",
    "Solving",
    "Reviewing",
    "Working",
];

/// Seconds without any observable progress event before the spinner flips to
/// its "no output" stall badge. Long enough not to fire during normal model
/// thinking / a slow-but-live tool, short enough to flag a genuine hang fast.
pub const STALL_THRESHOLD_SECS: u64 = 20;

/// Longer stall grace while an active *tool* is the current unit of work — a
/// build/test/command, a slow fetch, a sub-agent. These legitimately run 90-120s
/// with no streamed token, so the 20s text threshold cried "no output / stuck"
/// on a healthy `cargo test`. Only text/reasoning gaps use [`STALL_THRESHOLD_SECS`].
pub const STALL_THRESHOLD_TOOL_SECS: u64 = 120;

/// Stall grace *before any content has streamed this turn*. Between the request
/// landing (`message_start`) and the first reasoning/text delta, the model can
/// legitimately compute server-side for a while — Opus 4.8 front-loads a long
/// reasoning pass, Gemini Code Assist has slow first-frame latency. Crying "no
/// output" during that normal warm-up reads as a hang when the turn is healthy.
/// Once a token streams the threshold tightens back to [`STALL_THRESHOLD_SECS`]
/// so a genuine mid-stream freeze still surfaces fast.
pub const STALL_THRESHOLD_PREFIRST_SECS: u64 = 60;

/// Sliding window over which live output throughput is measured. Only the last
/// few seconds of *active* streaming count, so tool-execution / network / idle
/// gaps age out of the figure instead of diluting a lifetime average toward
/// zero (the old `tokens_out / elapsed_secs` "1 tok/s" bug).
const RATE_WINDOW: Duration = Duration::from_secs(3);

/// Minimum span (ms) between the oldest and newest in-window sample before a
/// rate is reported, so a single burst can't produce a wild first number.
const RATE_MIN_SPAN_MS: u128 = 300;

/// Last-resort memory bound on retained throughput samples. Trimming is
/// primarily by age ([`RATE_WINDOW`]); this cap only guards against an absurdly
/// chatty turn and is set well above what a fast stream produces in one window
/// (~one sample per text delta), so it never collapses the measured span itself.
const RATE_SAMPLE_CAP: usize = 256;

/// Snapshot of an in-flight turn used by the spinner widget.
#[derive(Debug, Clone)]
pub struct TurnActivity {
    /// When the current turn started.
    pub started_at: Instant,
    /// When the most recent observable progress event landed (text/reasoning
    /// delta, tool call, tool result, or token update). Drives the
    /// "stuck vs. still working" signal: if nothing arrives for a while the
    /// spinner flips to a muted "no output" badge so a hung turn (blocked tool,
    /// rate-limit, network stall) is visibly distinct from active streaming.
    pub last_event_at: Instant,
    /// Cumulative input tokens streamed so far this turn.
    pub tokens_in: u32,
    /// Monotonic display count of MAIN-MODEL output tokens generated this turn.
    /// Never decreases mid-turn: a smaller authoritative figure (a short
    /// follow-up iteration's `current.output_tokens`) can no longer clobber a
    /// larger accumulated value (the "counter goes down" bug). Fan-out sub-agent
    /// totals are tracked separately in `agent_tokens_out`, so a large sub-agent
    /// sum can never read as — or inflate — the main output count.
    pub tokens_out: u32,
    /// Fan-out sub-agent output aggregate (summed across live sub-agents),
    /// surfaced as a distinct "↑ N agent tokens" figure during the Delegating
    /// prelude. Kept apart from `tokens_out` so the next main-model count never
    /// looks like it dropped from the sub-agent total.
    agent_tokens_out: u32,
    /// Turn-start baseline of the *session*-cumulative output tokens, latched on
    /// the first usage snapshot of the turn. The live count is then
    /// `cumulative_output - baseline` — this turn's running total, monotonic by
    /// construction (the tracker only ever grows cumulative) and exact at turn
    /// end. `None` until the first authoritative usage lands, during which the
    /// chars/4 warm-up estimate drives the count instead.
    cumulative_out_baseline: Option<u32>,
    /// Running chars/4 estimate of streamed output, accumulated on every text
    /// delta. This is the throughput signal: it advances *continuously while
    /// text streams* (unlike the authoritative usage, which lands only at each
    /// iteration boundary), so the rate stays live between usage snapshots. It
    /// never drives the displayed count once authoritative usage takes over.
    rate_tokens: u32,
    /// Recent `(when, rate_tokens)` samples feeding the sliding-window
    /// throughput. Pruned by age at read time so tool-wait gaps fall out of the
    /// window rather than dragging the rate toward zero.
    rate_samples: VecDeque<(Instant, u32)>,
    /// Human-readable provider-neutral summary of the active work.
    current_action: Option<String>,
    /// Prose-streaming verb selected once from the host's stable turn
    /// generation. Kept separate from `current_action` so tool labels remain
    /// action-first and can temporarily take over without changing the turn's
    /// prose verb.
    prose_streaming_verb: &'static str,
    /// Whether `current_action` describes an active *tool* (a running
    /// command/build/test/fetch/sub-agent) rather than text/reasoning. Tools run
    /// long with no streamed token, so they get the longer stall grace
    /// ([`STALL_THRESHOLD_TOOL_SECS`]); set via [`Self::set_tool_action`] and
    /// cleared by any plain [`Self::set_current_action`].
    action_is_tool: bool,
    /// Latched by the stream's quiet-reasoning heartbeat: keep-alive chunks are
    /// verifiably arriving but the model has emitted no visible event — deep
    /// reasoning on a large context. While set, the stall badge reads as a calm
    /// "reasoning · stream alive Nm" instead of the alarming "no output Nm"
    /// (which users read as a hang and Esc out of, losing the whole reasoning
    /// pass). Cleared by every real progress event, so a subsequent genuine
    /// freeze still surfaces as "no output".
    stream_alive_quiet: bool,
    /// Latched by the runtime's quota-hold warning: a hard 429 parked this turn
    /// on the same model until its quota window resets (up to the wait band,
    /// ~15 minutes). While set, the stall badge reads
    /// "rate-limited · quota hold Nm" instead of "no output Nm" — the live
    /// report this fixes: a silent quota park read as a hang. Cleared by every
    /// real progress event (the resumed request's first delta).
    quota_hold: bool,
}

/// Compact context shown beside the live action while a turn streams.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActivityContext {
    /// Resolved model id already normalized for display (`gpt-5.5-fast`, not
    /// generic `gpt` or a provider-prefixed raw string).
    pub model: Option<String>,
    /// Dynamic workflow phase badge, e.g. `phase 2/4 read-code running \u{2192} test`.
    pub workflow: Option<String>,
    /// Auto-compaction pressure (0–100), shown as `· 73% ctx`. This uses the
    /// same ceiling as the bottom HUD and sidebar; when that ceiling is unknown,
    /// it falls back to context-window fill.
    pub ctx_percent: Option<u8>,
}

impl ActivityContext {
    #[must_use]
    pub fn from_hud(state: &HudState) -> Self {
        let view = HudViewModel::from_state(state);
        let ctx_percent = crate::tui::hud::context_pressure_percent(state)
            .map(|pct| u8::try_from(pct).unwrap_or(100));
        Self {
            model: non_empty_label(&view.model),
            workflow: view.workflow.as_deref().and_then(non_empty_label),
            ctx_percent,
        }
    }
}

impl TurnActivity {
    /// Begin a new turn at `now` with zeroed token counters.
    #[must_use]
    pub fn new(now: Instant) -> Self {
        Self::new_for_turn(now, 0)
    }

    /// Begin a new turn using the host's stable, monotonic turn generation to
    /// select the prose-streaming verb.
    #[must_use]
    pub fn new_for_turn(now: Instant, turn_generation: u64) -> Self {
        Self {
            started_at: now,
            last_event_at: now,
            tokens_in: 0,
            tokens_out: 0,
            agent_tokens_out: 0,
            cumulative_out_baseline: None,
            rate_tokens: 0,
            rate_samples: VecDeque::new(),
            current_action: None,
            prose_streaming_verb: zo_verb_for_turn(turn_generation),
            action_is_tool: false,
            stream_alive_quiet: false,
            quota_hold: false,
        }
    }

    /// Elapsed seconds since the turn started.
    #[must_use]
    pub fn elapsed_secs(&self) -> u64 {
        Instant::now()
            .saturating_duration_since(self.started_at)
            .as_secs()
    }

    /// Mark that an observable progress event just landed, resetting the
    /// stall clock. Called from the host on every streamed delta, tool
    /// transition, and real token update.
    pub fn mark_event(&mut self) {
        self.last_event_at = Instant::now();
        self.stream_alive_quiet = false;
        self.quota_hold = false;
    }

    /// Latch the quiet-reasoning state (see [`Self::stream_alive_quiet`]):
    /// the stream heartbeat proved the connection alive while the model
    /// reasons silently. Deliberately does NOT reset the stall clock — the
    /// badge keeps counting honestly, it just says what is actually
    /// happening instead of crying "no output" on a healthy turn.
    pub fn note_stream_alive_quiet(&mut self) {
        self.stream_alive_quiet = true;
    }

    /// Whether the quiet-reasoning heartbeat is currently latched.
    #[must_use]
    pub fn stream_alive_quiet(&self) -> bool {
        self.stream_alive_quiet
    }

    /// Latch the quota-hold state (see [`Self::quota_hold`]): the runtime
    /// announced it is parking this turn until the model's quota window
    /// resets. Like [`Self::note_stream_alive_quiet`], the stall clock keeps
    /// counting honestly — only the badge wording changes.
    pub fn note_quota_hold(&mut self) {
        self.quota_hold = true;
    }

    /// Whether the quota-hold park is currently latched.
    #[must_use]
    pub fn quota_hold(&self) -> bool {
        self.quota_hold
    }

    /// Seconds since the last observable progress event.
    #[must_use]
    pub fn idle_secs(&self) -> u64 {
        Instant::now()
            .saturating_duration_since(self.last_event_at)
            .as_secs()
    }

    /// Live output generation rate in tokens/sec over the last [`RATE_WINDOW`]
    /// of *active* streaming, or `None` until there is enough recent signal.
    ///
    /// Unlike a lifetime `tokens_out / elapsed` average — which counts every
    /// tool-wait and idle second in the denominator and so decays toward zero —
    /// this reflects how fast text is *currently* arriving: idle gaps age out of
    /// the window. The arithmetic is split into [`window_rate_from_samples`] so
    /// it stays unit-testable with an injected clock.
    #[must_use]
    pub fn output_tokens_per_sec(&self) -> Option<u32> {
        self.window_rate(Instant::now())
    }

    /// Throughput over the window ending at `now`. Samples are filtered by age
    /// at read time (not just pruned on write) so a long tool-wait gap — during
    /// which no sample is recorded and nothing prunes the ring — does not leave
    /// a stale wide span in the calculation. Single pass, no per-frame allocation.
    #[must_use]
    fn window_rate(&self, now: Instant) -> Option<u32> {
        let mut oldest: Option<(Instant, u32)> = None;
        let mut newest: (Instant, u32) = (now, 0);
        let mut count = 0usize;
        for &(at, total) in &self.rate_samples {
            if now.saturating_duration_since(at) > RATE_WINDOW {
                continue;
            }
            if oldest.is_none() {
                oldest = Some((at, total));
            }
            newest = (at, total);
            count += 1;
        }
        rate_from_endpoints(oldest?, newest, count)
    }

    /// Streaming warm-up: advance the chars/4 output estimate by `added` on every
    /// text delta. This always feeds the throughput window (so the rate stays
    /// live while text streams, between sparse usage snapshots), and *also* drives
    /// the visible count until the first authoritative usage latches a baseline.
    pub fn bump_output_estimate(&mut self, added: u32, now: Instant) {
        if added == 0 {
            return;
        }
        self.rate_tokens = self.rate_tokens.saturating_add(added);
        self.push_rate_sample(now);
        // Before authoritative usage exists, the estimate is the best count we
        // have. Once a baseline is latched the real count owns the display, so a
        // chars/4 over-estimate (low-entropy ASCII — long whitespace/punctuation/
        // base64 runs where chars-per-token ≫ 4) can't inflate it past real data.
        if self.cumulative_out_baseline.is_none() {
            self.tokens_out = self.tokens_out.max(self.rate_tokens);
        }
        self.last_event_at = now;
        self.stream_alive_quiet = false;
        self.quota_hold = false;
    }

    /// Authoritative path. On the first non-empty usage snapshot of the turn,
    /// latch the turn-start cumulative baseline (`cumulative - current`, i.e. the
    /// session total just before this turn). The displayed count then tracks this
    /// turn's running cumulative output, which only grows — never the smaller
    /// per-iteration `current` that made the counter drop. Throughput stays on
    /// the streaming estimate, so usage snapshots don't perturb the rate.
    pub fn record_output_usage(&mut self, cumulative_out: u32, current_out: u32, now: Instant) {
        if cumulative_out == 0 {
            return;
        }
        let baseline = *self
            .cumulative_out_baseline
            .get_or_insert_with(|| cumulative_out.saturating_sub(current_out));
        let turn_out = cumulative_out.saturating_sub(baseline);
        self.tokens_out = self.tokens_out.max(turn_out);
        self.last_event_at = now;
        self.stream_alive_quiet = false;
        self.quota_hold = false;
    }

    /// Fan-out path: a *snapshot sum* of spawned sub-agent output. Tracked on its
    /// own monotonic field (NOT `tokens_out`) so it shows as a distinct "↑ N agent
    /// tokens" figure and never makes the main-model count look like it dropped.
    /// Excluded from the throughput window — sub-agent aggregate is not main-model
    /// generation speed.
    pub fn record_agent_output(&mut self, agent_total: u32, now: Instant) {
        self.agent_tokens_out = self.agent_tokens_out.max(agent_total);
        self.last_event_at = now;
        self.stream_alive_quiet = false;
        self.quota_hold = false;
    }

    /// Append a throughput sample at `now`. Trim primarily by age — drop front
    /// samples older than [`RATE_WINDOW`] so the retained span tracks the window
    /// (a raw count cap would collapse the span on a fast stream and intermittently
    /// hide the rate). [`RATE_SAMPLE_CAP`] is only a last-resort memory bound.
    fn push_rate_sample(&mut self, now: Instant) {
        self.rate_samples.push_back((now, self.rate_tokens));
        while self.rate_samples.len() > 1 {
            match self.rate_samples.front() {
                Some(&(at, _)) if now.saturating_duration_since(at) > RATE_WINDOW => {
                    self.rate_samples.pop_front();
                }
                _ => break,
            }
        }
        while self.rate_samples.len() > RATE_SAMPLE_CAP {
            self.rate_samples.pop_front();
        }
    }

    /// `true` when no progress event has landed for `threshold_secs`, i.e. the
    /// turn looks stalled rather than actively streaming.
    #[must_use]
    pub fn is_stalled(&self, threshold_secs: u64) -> bool {
        self.idle_secs() >= threshold_secs
    }

    /// Stall threshold appropriate to the current unit of work: a longer grace
    /// while a tool runs (builds/tests/commands are silent for 90-120s by
    /// nature), the tight text threshold otherwise. SSOT for the per-action
    /// stall cutoff so the spinner never cries "stuck" on a healthy long tool.
    #[must_use]
    pub const fn stall_threshold_secs(&self) -> u64 {
        if self.action_is_tool {
            STALL_THRESHOLD_TOOL_SECS
        } else {
            STALL_THRESHOLD_SECS
        }
    }

    /// `true` once authoritative main-model output has streamed this turn (the
    /// first real `usage` snapshot landed). Stays `false` through the warm-up
    /// window — `message_start` carries an empty cumulative, and sub-agent
    /// (`agent_tokens_out`) totals never touch `tokens_out` — so it cleanly
    /// distinguishes "model is still warming up" from "model has produced
    /// output". Gates the pre-first-content stall grace.
    #[must_use]
    pub const fn has_streamed_content(&self) -> bool {
        self.tokens_out > 0
    }

    /// `true` when the current unit of work is an active tool (longer grace).
    #[must_use]
    pub const fn action_is_tool(&self) -> bool {
        self.action_is_tool
    }

    /// Update the live status line with the current unit of work. Clears the
    /// tool flag — a plain action is text/reasoning unless [`Self::set_tool_action`]
    /// re-marks it — so a stale tool grace never lingers after a tool finishes.
    pub fn set_current_action(&mut self, action: impl Into<String>) {
        self.action_is_tool = false;
        let action = sanitize_action(&action.into());
        if !action.is_empty() {
            self.current_action = Some(action);
        }
    }

    /// Like [`Self::set_current_action`] but marks the work as an active tool, so
    /// `stall_threshold_secs` grants it the longer grace.
    pub fn set_tool_action(&mut self, action: impl Into<String>) {
        self.set_current_action(action);
        self.action_is_tool = true;
    }

    /// Switch the activity line to this turn's latched prose-streaming verb.
    pub fn set_prose_streaming_action(&mut self) {
        self.set_current_action(self.prose_streaming_verb);
    }

    /// Live-reasoning verb latched for prose streaming in this turn.
    #[must_use]
    pub const fn prose_streaming_verb(&self) -> &'static str {
        self.prose_streaming_verb
    }

    /// Human-readable action, falling back to a calm generic label.
    #[must_use]
    pub fn current_action(&self) -> &str {
        self.current_action.as_deref().unwrap_or("Working")
    }
}

fn zo_verb_for_turn(turn_generation: u64) -> &'static str {
    let idx = usize::try_from(turn_generation % ZO_VERBS.len() as u64).unwrap_or(0);
    ZO_VERBS[idx]
}

/// The "no output {elapsed}" stall badge for the live spinner, or `None` while
/// the turn still reads as working. Pure so the threshold policy is unit-tested
/// without a live turn. The stall cutoff depends on the unit of work:
/// - an active tool gets the long [`STALL_THRESHOLD_TOOL_SECS`] grace,
/// - once content has streamed, the tight [`STALL_THRESHOLD_SECS`] catches a
///   genuine mid-stream freeze fast,
/// - before the first content delta, the longer [`STALL_THRESHOLD_PREFIRST_SECS`]
///   grace covers the legitimate server-side warm-up (Opus 4.8 reasoning,
///   Gemini first-frame latency) so a healthy turn doesn't read as hung.
#[must_use]
pub fn stall_badge(has_streamed: bool, is_tool: bool, idle_secs: u64) -> Option<String> {
    stall_badge_with_liveness(has_streamed, is_tool, idle_secs, StallLiveness::None)
}

/// Why a stalled turn is still alive, selecting the calm badge wording instead
/// of the alarming "no output". A latched liveness signal keeps the badge
/// honest — the stall clock still counts — while saying what is actually
/// happening, so a healthy pass no longer reads as a hang (users Esc'd out of
/// minutes of "no output" and lost the work).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StallLiveness {
    /// No liveness signal — a genuine stall, shown as "no output".
    None,
    /// Keep-alive chunks are verifiably arriving while the model reasons with
    /// no visible delta (deep reasoning on a large context).
    Reasoning,
    /// The turn is parked on the same model until its quota window resets. A
    /// parked request emits nothing at all — not even keep-alives — so this
    /// outranks [`Self::Reasoning`]; both can never be true at once.
    QuotaHold,
}

/// [`stall_badge`] with the stream-liveness signal (see [`StallLiveness`]): the
/// badge reads as a calm "reasoning · stream alive" / "rate-limited · quota
/// hold" instead of "no output" when the turn is verifiably still alive.
#[must_use]
pub fn stall_badge_with_liveness(
    has_streamed: bool,
    is_tool: bool,
    idle_secs: u64,
    liveness: StallLiveness,
) -> Option<String> {
    let threshold = if is_tool {
        STALL_THRESHOLD_TOOL_SECS
    } else if has_streamed {
        STALL_THRESHOLD_SECS
    } else {
        STALL_THRESHOLD_PREFIRST_SECS
    };
    (idle_secs >= threshold).then(|| {
        let elapsed = format_elapsed(idle_secs);
        match liveness {
            StallLiveness::QuotaHold => format!("rate-limited · quota hold {elapsed}"),
            StallLiveness::Reasoning => format!("reasoning · stream alive {elapsed}"),
            StallLiveness::None => format!("no output {elapsed}"),
        }
    })
}

fn effective_idle_secs(model_idle_secs: u64, child_output_age: Option<Duration>) -> u64 {
    child_output_age.map_or(model_idle_secs, |age| model_idle_secs.min(age.as_secs()))
}

fn sanitize_action(action: &str) -> String {
    let mut collapsed = String::with_capacity(action.len());
    let mut last_was_space = false;
    for ch in action.chars() {
        let ch = if ch == '\n' || ch == '\r' || ch == '\t' {
            ' '
        } else {
            ch
        };
        if ch.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
            }
            last_was_space = true;
        } else {
            collapsed.push(ch);
            last_was_space = false;
        }
    }
    collapse_repeated_action_detail(collapsed.trim())
}

fn collapse_repeated_action_detail(action: &str) -> String {
    let Some((head, tail)) = action.split_once(':') else {
        return action.to_string();
    };
    if normalize_action_phrase(head) == normalize_action_phrase(tail) {
        head.trim().to_string()
    } else {
        action.to_string()
    }
}

fn normalize_action_phrase(text: &str) -> String {
    text.trim()
        .trim_end_matches('\u{2026}')
        .trim_end_matches('.')
        .trim()
        .to_ascii_lowercase()
}

fn non_empty_label(label: &str) -> Option<String> {
    let trimmed = label.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Format an elapsed seconds count: `Xs`, `Xm Ys`, or `Xh Ym` once it rolls
/// past an hour (so a long-running turn/agent never shows an unbounded minute
/// count). Shared by the spinner, sidebar, and workflow viewer.
#[must_use]
pub fn format_elapsed(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

/// Format a token count with a `k` suffix (`2.3k`) once it reaches 1k.
#[must_use]
pub fn format_tokens(n: u32) -> String {
    if n >= 1000 {
        let thousands = f64::from(n) / 1000.0;
        format!("{thousands:.1}k")
    } else {
        n.to_string()
    }
}

/// Keep the status phrase within a single line on narrow terminals.
#[must_use]
pub fn truncate_status(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    if display_width(text) <= limit {
        text.to_string()
    } else {
        let budget = limit.saturating_sub(1);
        let mut out = String::new();
        let mut width = 0usize;
        for ch in text.chars() {
            let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if width + char_width > budget {
                break;
            }
            width += char_width;
            out.push(ch);
        }
        out.push('\u{2026}');
        out
    }
}

use crate::tui::text_metrics::display_width;

#[derive(Debug, PartialEq, Eq)]
struct ActivityParts<'a> {
    phase: &'a str,
    detail: Option<&'a str>,
}

fn activity_parts(action: &str) -> ActivityParts<'_> {
    let action = action.trim();
    if let Some(parts) = split_activity_once(action, ';') {
        return parts;
    }
    if let Some(parts) = action.split_once(": ").and_then(|(phase, detail)| {
        let phase = phase.trim();
        let detail = detail.trim();
        if phase.is_empty() || detail.is_empty() {
            None
        } else {
            Some(ActivityParts {
                phase,
                detail: Some(detail),
            })
        }
    }) {
        return parts;
    }
    ActivityParts {
        phase: action,
        detail: None,
    }
}

fn split_activity_once(action: &str, separator: char) -> Option<ActivityParts<'_>> {
    let (phase, detail) = action.split_once(separator)?;
    let phase = phase.trim();
    let detail = detail.trim();
    if phase.is_empty() || detail.is_empty() {
        return None;
    }
    Some(ActivityParts {
        phase,
        detail: Some(detail),
    })
}

fn truncate_activity_parts(
    phase: &str,
    detail: Option<&str>,
    limit: usize,
) -> (String, Option<String>) {
    let Some(detail) = detail else {
        return (truncate_status(phase, limit), None);
    };
    let phase_width = display_width(phase);
    let separator_width = 3;
    if phase_width + separator_width >= limit {
        return (truncate_status(phase, limit), None);
    }
    let detail_limit = limit.saturating_sub(phase_width + separator_width);
    if detail_limit == 0 {
        return (truncate_status(phase, limit), None);
    }
    if display_width(detail) > detail_limit {
        if let Some(progress) = progress_detail_prefix(detail) {
            if display_width(&progress) <= detail_limit {
                return (phase.to_string(), Some(progress));
            }
        }
    }
    (
        phase.to_string(),
        Some(truncate_status(detail, detail_limit)),
    )
}

const MAX_ACTIVITY_CHARS: usize = 88;

fn activity_limit_for_width(
    width: u16,
    elapsed: &str,
    token_label: &str,
    context_label: Option<&str>,
) -> usize {
    // Prefix: 2-cell leading rule + space (the shared 3-cell `── ` leader —
    // same gutter as the HUD line and the input's `│❯ `, so all three bottom
    // boundaries start content at col 3), then spinner glyph + space. Suffix:
    // the optional context badge plus the parenthesized metrics/control hint.
    // The activity phrase gets the remaining budget so phase/action, elapsed,
    // tokens, model, and interrupt stay visible on medium and narrow panes.
    let prefix_width = 2 + 1 + 1 + 1;
    let context_width = context_label.map_or(0, |label| 3 + display_width(label));
    let suffix_width = activity_suffix(elapsed, token_label, usize::from(width)).width();
    usize::from(width)
        .saturating_sub(prefix_width + context_width + suffix_width)
        .min(MAX_ACTIVITY_CHARS)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActivitySuffix {
    elapsed: Option<String>,
    token_label: Option<String>,
    interrupt_label: &'static str,
}

impl ActivitySuffix {
    fn full(elapsed: &str, token_label: &str) -> Self {
        Self {
            elapsed: Some(elapsed.to_string()),
            token_label: Some(token_label.to_string()),
            interrupt_label: "esc to interrupt",
        }
    }

    fn compact(elapsed: &str, token_label: &str) -> Self {
        Self {
            elapsed: Some(elapsed.to_string()),
            token_label: Some(compact_token_label(token_label)),
            interrupt_label: "esc",
        }
    }

    fn minimal(token_label: &str) -> Self {
        Self {
            elapsed: None,
            token_label: Some(compact_token_label(token_label)),
            interrupt_label: "esc",
        }
    }

    fn escape_only() -> Self {
        Self {
            elapsed: None,
            token_label: None,
            interrupt_label: "esc",
        }
    }

    fn text(&self) -> String {
        let mut parts = Vec::new();
        if let Some(elapsed) = &self.elapsed {
            parts.push(elapsed.as_str());
        }
        if let Some(token_label) = &self.token_label {
            parts.push(token_label.as_str());
        }
        parts.push(self.interrupt_label);
        format!(" ({}) ", parts.join(" · "))
    }

    fn width(&self) -> usize {
        display_width(&self.text())
    }
}

fn activity_suffix(elapsed: &str, token_label: &str, width: usize) -> ActivitySuffix {
    let full = ActivitySuffix::full(elapsed, token_label);
    if full.width() <= width {
        return full;
    }

    let compact = ActivitySuffix::compact(elapsed, token_label);
    if compact.width() <= width {
        return compact;
    }

    let minimal = ActivitySuffix::minimal(token_label);
    if minimal.width() <= width {
        return minimal;
    }

    ActivitySuffix::escape_only()
}

fn compact_token_label(label: &str) -> String {
    let label = label.trim();
    if label == "tokens pending" || label == "agent tokens pending" {
        return "pending".to_string();
    }
    if label == "output pending" || label == "agent output pending" {
        return "pending".to_string();
    }
    if let Some((input, pending)) = label.split_once(" · ") {
        if pending.ends_with("output pending") {
            let input = input
                .trim()
                .strip_prefix("↑ ")
                .unwrap_or(input.trim())
                .trim_end_matches(" input")
                .trim();
            return format!("↑{input} · pending");
        }
    }
    // Combined `↑ {in} ↓ ~{out} tokens [· {rate} tok/s]` → `↑{in} ↓{out} [· {rate}/s]`.
    // Reuse the `↓`-branch below for the output half so both stay consistent.
    if let Some(rest) = label.strip_prefix("↑ ") {
        if let Some((up, down)) = rest.split_once(" ↓ ~") {
            let down_compact = compact_token_label(&format!("↓ ~{down}"));
            return format!("↑{} {down_compact}", up.trim());
        }
    }
    if let Some(rest) = label.strip_prefix("↓ ~") {
        // `↓ ~Nk tokens` or `↓ ~Nk tokens · R tok/s`. Compact the count but keep
        // the live rate — the throughput is the signal the user asked to keep
        // visible, so it survives the narrow-width pass as `↓Nk · R/s`.
        if let Some((count_part, rate_part)) = rest.split_once(" · ") {
            let count = count_part.strip_suffix(" tokens").unwrap_or(count_part);
            let rate = rate_part
                .trim()
                .strip_suffix(" tok/s")
                .map(str::trim)
                .map_or_else(|| rate_part.trim().to_string(), |r| format!("{r}/s"));
            return format!("↓{count} · {rate}");
        }
        if let Some(tokens) = rest.strip_suffix(" tokens") {
            return format!("↓{tokens}");
        }
    }
    label.to_string()
}

/// Minimum activity-row width for the inline workflow phase badge. Below this
/// the badge would crowd out the action phrase, so the phase moves to the
/// HUD's dedicated second row instead (`app/render.rs` keys the two-row HUD
/// grant on this same constant — the phase must always be visible somewhere).
pub const WORKFLOW_BADGE_MIN_COLS: u16 = 108;

fn activity_context_label(context: Option<&ActivityContext>, width: u16) -> Option<String> {
    let context = context?;
    let mut parts = Vec::new();

    // The workflow phase is most useful on wide panes: it explains where the
    // current action sits in a multi-step run. Keep it bounded so the action
    // itself still has room.
    if width >= WORKFLOW_BADGE_MIN_COLS {
        if let Some(workflow) = context.workflow.as_deref() {
            // 124 (not 132): a 160-col terminal whose spinner pane is narrowed
            // to ~128 by the HUD sidebar is still wide enough for the full phase
            // badge — the old 132 cutoff truncated it to "phase N/N re…".
            let limit = if width >= 124 { 42 } else { 30 };
            parts.push(truncate_status(workflow, limit));
        }
    }

    // The model id matters even on medium panes because `gpt`/`opus` aliases hide
    // what is really streaming. It is shorter than the workflow badge, so show it
    // earlier.
    if width >= 72 {
        if let Some(model) = context.model.as_deref() {
            parts.push(truncate_status(model, 24));
        }
    }

    // Context-window fill % is the primary "am I near the limit?" signal. Keep
    // it last (it's short) and only on panes wide enough to spare the cells.
    if width >= 90 {
        if let Some(pct) = context.ctx_percent {
            parts.push(format!("{pct}% ctx"));
        }
    }

    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// Time-based marker glyph for inline transcript markers (the live tool
/// group's per-row running markers). Shares one process-relative clock so
/// every active marker pulses `✦`/`✧` in unison with the activity line's
/// heartbeat — many parallel rows read as ONE calm brand pulse instead of a
/// frenetic per-row rotation (the retired 64-frame spinner read as borrowed
/// chrome). Reduce-motion settles on the solid spark.
#[must_use]
pub fn marker_glyph() -> &'static str {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let elapsed_ms = u64::try_from(
        Instant::now()
            .saturating_duration_since(*EPOCH.get_or_init(Instant::now))
            .as_millis(),
    )
    .unwrap_or(0);
    spark_glyph_for_elapsed(elapsed_ms, reduce_motion_enabled(), true)
}

/// Return a pulsing dots suffix that cycles through `""`, `"."`,
/// `".."`, `"..."` at roughly ~2Hz, giving a "thinking" heartbeat
/// effect on the verb line.
#[must_use]
pub fn pulsing_dots(activity: &TurnActivity) -> &'static str {
    pulsing_dots_for_elapsed(activity_animation_elapsed_ms(activity), reduce_motion_enabled())
}

fn activity_animation_elapsed_ms(activity: &TurnActivity) -> u64 {
    u64::try_from(
        Instant::now()
            .saturating_duration_since(activity.started_at)
            .as_millis(),
    )
    .unwrap_or(0)
}

/// Reuse the active-turn scheduler's 33 ms animation cadence for the ignition
/// crest. This is an index calculation only; it does not introduce a clock or
/// request any additional frames.
const IGNITION_TICK_MS: u64 = 33;

fn ignition_wave_line(
    elapsed_ms: u64,
    width: u16,
    reduced_motion: bool,
    theme: &Theme,
) -> Option<Line<'static>> {
    if elapsed_ms >= IGNITION_MS || width == 0 || theme.no_color {
        return None;
    }

    let width = usize::from(width);
    let phase = if reduced_motion {
        0
    } else {
        usize::try_from(elapsed_ms / IGNITION_TICK_MS).unwrap_or(0) % 16
    };
    let spans = (0..width)
        .map(|cell_idx| {
            let color_idx = (cell_idx * 16 / width + phase) % 16;
            Span::styled(
                glyphs::ANVIL_LINE,
                Style::new().fg(theme.heat().ignition[color_idx]),
            )
        })
        .collect::<Vec<_>>();
    Some(Line::from(spans))
}

const fn heartbeat_phase(elapsed_ms: u64) -> u64 {
    (elapsed_ms / 500) % 4
}

/// Pure inner for [`pulsing_dots`], split out so the reduce-motion gate is
/// testable without a clock. Under reduce-motion the dots settle to the phase-0
/// value (`""`), which is already the width the caller reserves.
#[must_use]
fn pulsing_dots_for_elapsed(elapsed_ms: u64, reduced: bool) -> &'static str {
    if reduced {
        return "";
    }
    // Cycle every ~500ms through 4 phases.
    match heartbeat_phase(elapsed_ms) {
        0 => "",
        1 => ".",
        2 => "..",
        _ => "...",
    }
}

/// Two-frame Zo spark pulse on the same ~2Hz heartbeat as the trailing
/// dots. Reduced motion settles on the solid spark; no-color uses the stable
/// one-cell ASCII sibling for both phases.
#[must_use]
pub(crate) fn spark_glyph_for_elapsed(elapsed_ms: u64, reduced: bool, color: bool) -> &'static str {
    if reduced || heartbeat_phase(elapsed_ms).is_multiple_of(2) {
        glyphs::pick(color, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC)
    } else {
        glyphs::pick(
            color,
            glyphs::ZO_SPARK_HOLLOW,
            glyphs::ZO_SPARK_HOLLOW_NC,
        )
    }
}

/// Draw the spinner line into `area`. Caller is responsible for only
/// invoking this when a turn is actually active.
pub fn draw(frame: &mut Frame<'_>, area: Rect, activity: &TurnActivity, theme: &Theme) {
    draw_with_context(frame, area, activity, None, theme);
}

/// Draw the spinner line with optional model/workflow context.
#[allow(clippy::too_many_lines)] // cohesive single-row spinner composition
pub fn draw_with_context(
    frame: &mut Frame<'_>,
    area: Rect,
    activity: &TurnActivity,
    context: Option<&ActivityContext>,
    theme: &Theme,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    // Use a single-row sub-rect so the caller can reserve a blank row
    // above the input box for visual separation.
    let line_area = Rect::new(area.x, area.y, area.width, 1);
    // Sample the existing turn-relative heartbeat once so the spark pulse and
    // trailing dots share one clock and one ~2Hz cadence.
    let animation_elapsed_ms = activity_animation_elapsed_ms(activity);
    let reduced_motion = reduce_motion_enabled();
    if let Some(ignition) =
        ignition_wave_line(animation_elapsed_ms, area.width, reduced_motion, theme)
    {
        frame.render_widget(Paragraph::new(ignition), line_area);
        return;
    }
    let glyph = spark_glyph_for_elapsed(animation_elapsed_ms, reduced_motion, !theme.no_color);
    // Stall detection: no observable progress for a while means the turn is
    // likely blocked (a slow/hung tool, a rate-limit wait, or a network stall)
    // rather than actively streaming. Surface that distinctly so the user can
    // tell "still working" from "stuck" — the core "is it alive?" signal.
    // Foreground Bash output is captured below the model event stream. Treat a
    // more recent child write as progress for badge purposes, while leaving the
    // model activity clock untouched; no bytes means no synthetic activity.
    let child_output_age = activity
        .action_is_tool()
        .then(|| tools::live_output::current(0))
        .flatten()
        .and_then(|snapshot| snapshot.last_output_age);
    let idle = effective_idle_secs(activity.idle_secs(), child_output_age);
    // The pre-first-content grace keeps a normal server-side warm-up (long Opus
    // reasoning, slow first frame) from reading as a hang; once a token streams
    // the cutoff tightens so a genuine mid-stream freeze still surfaces fast.
    // QuotaHold outranks Reasoning: a parked request emits nothing at all.
    let liveness = if activity.quota_hold() {
        StallLiveness::QuotaHold
    } else if activity.stream_alive_quiet() {
        StallLiveness::Reasoning
    } else {
        StallLiveness::None
    };
    let stall_badge_text = stall_badge_with_liveness(
        activity.has_streamed_content(),
        activity.action_is_tool(),
        idle,
        liveness,
    );
    // The glyph turns warn only when the badge shows AND the stream is not
    // verifiably alive — a latched quiet-reasoning stretch is healthy, and a
    // quota hold is a deliberate park, so both keep the calm spark glyph and
    // a dim badge instead of the warn tint.
    let stalled = stall_badge_text.is_some() && liveness == StallLiveness::None;
    let action = activity_parts(activity.current_action());
    let elapsed = format_elapsed(activity.elapsed_secs());
    let token_label = turn_token_label(
        activity.current_action(),
        activity.tokens_in,
        activity.tokens_out,
        activity.agent_tokens_out,
        activity.output_tokens_per_sec(),
    );
    let mut context_label = activity_context_label(context, area.width);
    let mut activity_limit =
        activity_limit_for_width(area.width, &elapsed, &token_label, context_label.as_deref());
    if context_label.is_some()
        && activity_progress_prefix_width(&action).is_some_and(|needed| activity_limit < needed)
    {
        // During fan-out, the completion fraction/percent is the user's
        // primary "is this moving?" signal. Prefer preserving that over a
        // model/context badge when the row is tight.
        context_label = None;
        activity_limit = activity_limit_for_width(area.width, &elapsed, &token_label, None);
    }
    if activity_limit == 0 && context_label.is_some() {
        // On awkward widths, prefer keeping a readable action phrase and the
        // interrupt hint over showing context that would consume the whole row.
        context_label = None;
        activity_limit = activity_limit_for_width(area.width, &elapsed, &token_label, None);
    }
    let (action_phase, action_detail) =
        truncate_activity_parts(action.phase, action.detail, activity_limit);

    let glyph_style = if stalled {
        Style::new()
            .fg(theme.palette.warn)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new()
            .fg(theme.heat().spark)
            .add_modifier(Modifier::BOLD)
    };
    let dim_style = Style::new().fg(theme.palette.dim);
    let value_style = Style::new().fg(theme.palette.fg);
    let warn_style = Style::new().fg(theme.palette.warn);

    let rule_char = if theme.no_color {
        glyphs::HORIZONTAL_RULE_NC
    } else {
        glyphs::HORIZONTAL_RULE
    };

    // Build prefix wave first so the rest of the line can compute its
    // remaining width against the actual prefix length. 2 rule cells + the
    // space below = the same 3-cell `── ` leader as the HUD line, so the
    // spinner glyph lands on col 3 — the shared content column of the bottom
    // stack (HUD text, input text after `│❯ `).
    let prefix_len: usize = 2;
    let prefix_spans = waved_rule_spans(rule_char, prefix_len, 0, theme, activity.started_at);

    let mut left_spans: Vec<Span<'_>> = Vec::with_capacity(24);
    left_spans.extend(prefix_spans);
    left_spans.push(Span::raw(" "));
    left_spans.push(Span::styled(glyph.to_string(), glyph_style));
    left_spans.push(Span::raw(" "));
    let dots = if action_phase.is_empty() {
        ""
    } else {
        pulsing_dots_for_elapsed(animation_elapsed_ms, reduced_motion)
    };
    left_spans.extend(waved_verb_spans(
        &action_phase,
        dots,
        theme,
        activity.started_at,
    ));
    if let Some(detail) = action_detail {
        left_spans.push(Span::styled(" · ", dim_style));
        left_spans.push(Span::styled(detail, dim_style));
    }
    // Stuck-vs-working badge: when the turn has streamed nothing for a while,
    // call it out explicitly so the user knows the spinner is not just idly
    // animating over a hung request. Placed right after the action so it reads
    // as part of the live status, before the model/context badge.
    if let Some(text) = &stall_badge_text {
        let badge_style = if stalled { warn_style } else { value_style };
        left_spans.push(Span::styled(" · ", dim_style));
        left_spans.push(Span::styled(text.clone(), badge_style));
    }
    if let Some(label) = context_label {
        left_spans.push(Span::styled(" · ", dim_style));
        left_spans.push(Span::styled(label, value_style));
    }
    let suffix = activity_suffix(&elapsed, &token_label, usize::from(area.width));
    let mut right_spans = Vec::new();
    right_spans.push(Span::styled(" (", dim_style));
    if let Some(elapsed) = suffix.elapsed {
        right_spans.push(Span::styled(elapsed, value_style));
        right_spans.push(Span::styled(" · ", dim_style));
    }
    if let Some(token_label) = suffix.token_label {
        right_spans.push(Span::styled(token_label, value_style));
        right_spans.push(Span::styled(" · ", dim_style));
    }
    right_spans.push(Span::styled(suffix.interrupt_label, dim_style));
    right_spans.push(Span::styled(") ", dim_style));

    let left_width = left_spans
        .iter()
        .map(|s| display_width(s.content.as_ref()))
        .sum::<usize>();
    let right_width = right_spans
        .iter()
        .map(|s| display_width(s.content.as_ref()))
        .sum::<usize>();
    let total_width = left_width + right_width;

    let mut spans = left_spans;
    if total_width < usize::from(area.width) {
        // Trailing/middle rule stays uniformly dim — only the *leading* rule
        // gets the wave so the line doesn't look like it's truncated
        // with bright/dim glyphs scattered at the right edge.
        let trailing_style = Style::new().fg(theme.palette.faint);
        spans.push(Span::styled(
            rule_char.repeat(usize::from(area.width).saturating_sub(total_width)),
            trailing_style,
        ));
    }
    spans.extend(right_spans);

    let para = Paragraph::new(Line::from(spans));
    frame.render_widget(para, line_area);
}

fn activity_progress_prefix_width(action: &ActivityParts<'_>) -> Option<usize> {
    let detail = progress_detail_prefix(action.detail?)?;
    Some(
        display_width(action.phase)
            .saturating_add(display_width(" · "))
            .saturating_add(display_width(&detail)),
    )
}

fn progress_detail_prefix(detail: &str) -> Option<String> {
    let mut segments = Vec::new();
    let mut saw_percent = false;
    for segment in detail.split(" · ") {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        segments.push(segment);
        if segment.contains('%') {
            saw_percent = true;
        }
        if saw_percent && segment.contains("left") {
            return Some(segments.join(" · "));
        }
    }
    if saw_percent {
        Some(segments.join(" · "))
    } else {
        None
    }
}

fn turn_token_label(
    action: &str,
    tokens_in: u32,
    tokens_out: u32,
    agent_tokens: u32,
    rate: Option<u32>,
) -> String {
    if tokens_out > 0 {
        // CC-style: show BOTH directions — `↑` input and `↓` output — so the
        // figure isn't a lone down-arrow. Input is prepended only when known
        // (`↑ {in} `); the live tok/s rides the output count. The rate is omitted
        // below ~1s / no output (warm-up) so it never flashes a wild first number.
        let counts = if tokens_in > 0 {
            format!(
                "↑ {} ↓ ~{} tokens",
                format_tokens(tokens_in),
                format_tokens(tokens_out)
            )
        } else {
            format!("↓ ~{} tokens", format_tokens(tokens_out))
        };
        return match rate {
            Some(rate) if rate > 0 => format!("{counts} · {rate} tok/s"),
            _ => counts,
        };
    }

    // Fan-out prelude: surface the sub-agent aggregate as a DISTINCT figure, not
    // as the main `↓ tokens` count — so when the main stream starts (with its own,
    // typically smaller, output) it never reads as the number dropping.
    if agent_tokens > 0 {
        return format!("↑ ~{} agent tokens", format_tokens(agent_tokens));
    }

    if tokens_in > 0 {
        let pending = if is_agent_prelude_action(action) {
            "agent output pending"
        } else {
            "output pending"
        };
        return format!("↑ {} input · {pending}", format_tokens(tokens_in));
    }

    if is_agent_prelude_action(action) {
        "agent output pending".to_string()
    } else {
        "tokens pending".to_string()
    }
}

/// Tokens/sec between the oldest and newest in-window samples, or `None` when
/// there is not enough recent signal. Needs at least two samples spanning
/// [`RATE_MIN_SPAN_MS`], and *rounds* (never floors) so a sub-1 rate that the old
/// integer division suppressed to zero now shows. Pure integer arithmetic over
/// the two endpoints, so it is testable without a wall clock and free of
/// float-cast truncation.
#[must_use]
fn rate_from_endpoints(oldest: (Instant, u32), newest: (Instant, u32), count: usize) -> Option<u32> {
    if count < 2 {
        return None;
    }
    let span_ms = newest.0.saturating_duration_since(oldest.0).as_millis();
    if span_ms < RATE_MIN_SPAN_MS {
        return None;
    }
    let delta = u128::from(newest.1.saturating_sub(oldest.1));
    if delta == 0 {
        return None;
    }
    // tokens/sec, rounded to nearest: (delta * 1000 + span/2) / span.
    let rate = (delta * 1000 + span_ms / 2) / span_ms;
    u32::try_from(rate).ok().filter(|r| *r >= 1)
}

fn is_agent_prelude_action(action: &str) -> bool {
    let action = action.to_ascii_lowercase();
    action.contains("pre-analysis")
        || action.contains("auto fan-out")
        || action.contains("delegating")
}

/// Returns one `Span` per column of the rule, brightening characters
/// near the traveling crest so the line looks like a wave passing
/// through. Crest position advances ~12 columns/sec.
fn waved_rule_spans(
    rule_char: &str,
    width: usize,
    col_offset: usize,
    theme: &Theme,
    started_at: Instant,
) -> Vec<Span<'static>> {
    if reduce_motion_enabled() {
        // Settled: a plain, uniform rule with no traveling crest.
        let faint = Style::new().fg(theme.palette.faint);
        return (0..width)
            .map(|_| Span::styled(rule_char.to_string(), faint))
            .collect();
    }
    let elapsed_ms = u64::try_from(
        Instant::now()
            .saturating_duration_since(started_at)
            .as_millis(),
    )
    .unwrap_or(0);
    // Total cycle distance — the period the crest takes to cross a
    // generous span before wrapping. We add buffer so the crest is
    // off-screen between cycles instead of teleporting.
    let cycle = 96_u64;
    // ~12 cols/sec: one column every ~83ms.
    let phase = (elapsed_ms / 83) % cycle;
    let crest = i64::try_from(phase).unwrap_or(0);

    let accent = Style::new()
        .fg(theme.palette.accent)
        .add_modifier(Modifier::BOLD);
    let bright = Style::new().fg(theme.palette.fg);
    let dim = Style::new().fg(theme.palette.dim);
    let faint = Style::new().fg(theme.palette.faint);

    let mut spans = Vec::with_capacity(width);
    for i in 0..width {
        let abs_col = i64::try_from(col_offset + i).unwrap_or(i64::MAX);
        // Distance from the crest, modulo cycle length, signed.
        let mut delta = (abs_col - crest).rem_euclid(i64::try_from(cycle).unwrap_or(96));
        // Map [0..cycle) to a symmetric distance around the crest at 0.
        let half = i64::try_from(cycle / 2).unwrap_or(48);
        if delta > half {
            delta = i64::try_from(cycle).unwrap_or(96) - delta;
        }
        let style = match delta {
            0 => accent,
            1 | 2 => bright,
            3..=5 => dim,
            _ => faint,
        };
        spans.push(Span::styled(rule_char.to_string(), style));
    }
    spans
}

/// Build per-character spans for the verb so a brightness crest travels
/// across it in a wave pattern — the heart of the "alive" feel users
/// asked for. Each character is styled based on its distance from the
/// crest position, which advances ~10 cols/sec.
fn waved_verb_spans(
    verb: &str,
    dots: &str,
    theme: &Theme,
    started_at: Instant,
) -> Vec<Span<'static>> {
    let spark_bold = Style::new()
        .fg(theme.heat().spark)
        .add_modifier(Modifier::BOLD);
    let bright_bold = Style::new()
        .fg(theme.palette.bright)
        .add_modifier(Modifier::BOLD);
    let fg_bold = Style::new()
        .fg(theme.palette.fg)
        .add_modifier(Modifier::BOLD);
    let combined: String = format!("{verb}{dots}");
    waved_text_spans(&combined, started_at, spark_bold, bright_bold, fg_bold)
}

/// Build per-character spans with a brightness crest traveling across `text`,
/// advancing ~one column / 80ms off `started_at`. The crest character gets
/// `peak`, its immediate neighbors `near`, everything else `base`; multiplying
/// the cycle by 2 leaves a pause between sweeps so it reads as a wave, not a
/// strobe. Shared by the spinner verb and the live plan-progress header so the
/// crest math has a single owner (`pub(crate)` for the `app::render` reuse).
pub(crate) fn waved_text_spans(
    text: &str,
    started_at: Instant,
    peak: Style,
    near: Style,
    base: Style,
) -> Vec<Span<'static>> {
    if reduce_motion_enabled() {
        // Settled: no traveling crest — every glyph holds the base brightness.
        return text
            .chars()
            .map(|ch| Span::styled(ch.to_string(), base))
            .collect();
    }
    let elapsed_ms = u64::try_from(
        Instant::now()
            .saturating_duration_since(started_at)
            .as_millis(),
    )
    .unwrap_or(0);
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len().max(1);
    let cycle = (len as u64).saturating_mul(2).max(8);
    let phase_ms_per_col: u64 = 80;
    let crest = (elapsed_ms / phase_ms_per_col) % cycle;
    let signed_cycle = i64::try_from(cycle).unwrap_or(16);
    let half = signed_cycle / 2;

    let mut spans = Vec::with_capacity(chars.len());
    for (i, ch) in chars.iter().enumerate() {
        let signed_crest = i64::try_from(crest).unwrap_or(0);
        let signed_i = i64::try_from(i).unwrap_or(0);
        let mut delta = (signed_i - signed_crest).rem_euclid(signed_cycle);
        if delta > half {
            delta = signed_cycle - delta;
        }
        let style = match delta {
            0 => peak,
            1 | 2 => near,
            _ => base,
        };
        spans.push(Span::styled(ch.to_string(), style));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared tool-row marker is the same ✦/✧ spark heartbeat the
    /// activity line uses — the retired 64-frame 3D rotation must not return.
    #[test]
    fn marker_glyph_is_a_spark_pulse_frame() {
        let marker = marker_glyph();
        assert!(
            marker == glyphs::ZO_SPARK || marker == glyphs::ZO_SPARK_HOLLOW,
            "tool-row marker must be a spark pulse frame: {marker:?}"
        );
    }

    #[test]
    fn pulsing_dots_reduce_motion_settles_to_empty() {
        for elapsed_ms in (0..4096).step_by(53) {
            let phase = match (elapsed_ms / 500) % 4 {
                0 => "",
                1 => ".",
                2 => "..",
                _ => "...",
            };
            assert_eq!(pulsing_dots_for_elapsed(elapsed_ms, false), phase);
            assert_eq!(pulsing_dots_for_elapsed(elapsed_ms, true), "");
        }
    }

    #[test]
    fn activity_spark_pulses_on_the_dot_heartbeat_and_reduces_motion() {
        assert_eq!(spark_glyph_for_elapsed(0, false, true), glyphs::ZO_SPARK);
        assert_eq!(spark_glyph_for_elapsed(499, false, true), glyphs::ZO_SPARK);
        assert_eq!(
            spark_glyph_for_elapsed(500, false, true),
            glyphs::ZO_SPARK_HOLLOW
        );
        assert_eq!(
            spark_glyph_for_elapsed(1_000, false, true),
            glyphs::ZO_SPARK
        );
        assert_eq!(
            spark_glyph_for_elapsed(500, true, true),
            glyphs::ZO_SPARK
        );
        assert_eq!(
            spark_glyph_for_elapsed(500, false, false),
            glyphs::ZO_SPARK_HOLLOW_NC
        );
    }

    #[test]
    fn ignition_wave_hands_off_at_exact_500ms_boundary() {
        let theme = Theme::zo();
        let wave = ignition_wave_line(499, 32, false, &theme).expect("499ms stays ignition");

        assert_eq!(wave.spans.len(), 32);
        assert!(
            wave.spans
                .iter()
                .all(|span| span.content == glyphs::ANVIL_LINE)
        );
        assert_eq!(
            wave.spans[0].style.fg,
            Some(theme.heat().ignition[15]),
            "499ms uses the existing 33ms tick phase"
        );
        assert!(
            ignition_wave_line(500, 32, false, &theme).is_none(),
            "500ms deterministically hands the row back to the normal activity line"
        );
    }

    #[test]
    fn ignition_wave_advances_on_animation_ticks_and_skips_no_color() {
        let theme = Theme::zo();
        let before = ignition_wave_line(32, 16, false, &theme).expect("ignition before tick");
        let after = ignition_wave_line(33, 16, false, &theme).expect("ignition on next tick");
        assert_ne!(before.spans[0].style.fg, after.spans[0].style.fg);

        assert!(ignition_wave_line(0, 16, false, &Theme::no_color()).is_none());
    }

    #[test]
    fn zo_verb_is_stable_within_a_turn_and_rotates_across_turns() {
        let now = Instant::now();
        let selected: Vec<&str> = (0..ZO_VERBS.len() as u64)
            .map(|turn_generation| {
                let mut activity = TurnActivity::new_for_turn(now, turn_generation);
                let selected = activity.prose_streaming_verb();
                activity.set_prose_streaming_action();
                assert_eq!(activity.current_action(), selected);
                activity.set_current_action("Reading tool output; choosing next step");
                activity.set_prose_streaming_action();
                assert_eq!(activity.current_action(), selected);
                selected
            })
            .collect();

        assert_eq!(selected, ZO_VERBS);
    }

    #[test]
    fn zo_verb_does_not_replace_action_first_tool_labels() {
        let mut activity = TurnActivity::new_for_turn(Instant::now(), 4);
        activity.set_tool_action("Running command: cargo test");

        assert_eq!(activity.current_action(), "Running command: cargo test");
        assert!(activity.action_is_tool());
    }

    #[test]
    fn elapsed_under_minute() {
        assert_eq!(format_elapsed(0), "0s");
        assert_eq!(format_elapsed(45), "45s");
    }

    #[test]
    fn elapsed_minutes_and_seconds() {
        assert_eq!(format_elapsed(60), "1m 0s");
        assert_eq!(format_elapsed(111), "1m 51s");
    }

    #[test]
    fn tokens_compact() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(2_345), "2.3k");
    }

    #[test]
    fn turn_activity_tracks_current_action() {
        let mut activity = TurnActivity::new(Instant::now());
        assert_eq!(activity.current_action(), "Working");
        activity.set_current_action("Running command:\n cargo test");
        assert_eq!(activity.current_action(), "Running command: cargo test");
    }

    #[test]
    fn turn_activity_collapses_repeated_action_detail() {
        let mut activity = TurnActivity::new(Instant::now());
        activity.set_current_action("Smart: smart...");
        assert_eq!(activity.current_action(), "Smart");
    }

    #[test]
    fn agent_prelude_hides_zero_main_stream_tokens() {
        assert_eq!(
            turn_token_label("Smart: 6 pre-analysis agents running", 0, 0, 0, None),
            "agent output pending"
        );
        assert_eq!(
            turn_token_label("Drafting response", 0, 0, 0, None),
            "tokens pending"
        );
        assert_eq!(
            turn_token_label("Drafting response", 1_200, 0, 0, None),
            "↑ 1.2k input · output pending"
        );
        assert_eq!(
            turn_token_label("Smart: 6 pre-analysis agents running", 1_200, 0, 0, None),
            "↑ 1.2k input · agent output pending"
        );
        // Both directions present → `↑ input ↓ output` (the user's "위도 나와야" fix).
        assert_eq!(
            turn_token_label("Smart: 6 pre-analysis agents running", 1_200, 42, 0, None),
            "↑ 1.2k ↓ ~42 tokens"
        );
    }

    #[test]
    fn fan_out_aggregate_shows_as_distinct_agent_tokens_not_main_count() {
        // Sub-agent aggregate surfaces as its own "↑ N agent tokens" figure while
        // the main output is still zero — so it never reads as the main count
        // dropping when the main stream later starts smaller.
        assert_eq!(
            turn_token_label("Smart: 6 pre-analysis agents running", 0, 0, 736, None),
            "↑ ~736 agent tokens"
        );
        // Once the main model produces output, the main `↓ tokens` wins.
        assert_eq!(
            turn_token_label("Drafting response", 0, 120, 736, Some(80)),
            "↓ ~120 tokens · 80 tok/s"
        );
    }

    #[test]
    fn rate_from_endpoints_needs_recent_signal() {
        let t0 = Instant::now();
        // Fewer than two in-window samples → no rate.
        assert_eq!(rate_from_endpoints((t0, 0), (t0, 0), 1), None);
        // Span under the minimum → no rate (avoids a wild first-sample number).
        assert_eq!(
            rate_from_endpoints((t0, 0), (t0 + Duration::from_millis(100), 50), 2),
            None
        );
        // No growth across the window → no rate.
        assert_eq!(
            rate_from_endpoints((t0, 100), (t0 + Duration::from_secs(2), 100), 2),
            None
        );
        // Real signal → rounded tokens/sec over the window span.
        assert_eq!(
            rate_from_endpoints((t0, 0), (t0 + Duration::from_secs(3), 240), 5),
            Some(80)
        );
        // Sub-1-tok/s-lifetime case that the OLD integer division floored to
        // None now survives because it rounds the recent-window delta.
        assert_eq!(
            rate_from_endpoints((t0, 0), (t0 + Duration::from_secs(1), 3), 2),
            Some(3)
        );
    }

    #[test]
    fn window_rate_reflects_recent_streaming_not_lifetime() {
        let t0 = Instant::now();
        let mut a = TurnActivity::new(t0);
        // Streaming estimate accumulates 240 tokens over the first 3s.
        a.bump_output_estimate(80, t0 + Duration::from_secs(1));
        a.bump_output_estimate(160, t0 + Duration::from_secs(3));
        // Over the in-window span: (240 - 80) / (3s - 1s) = 80 tok/s.
        assert_eq!(a.window_rate(t0 + Duration::from_secs(3)), Some(80));
        // Tool runs 30s with no token updates: the lifetime average would be
        // ~240/33 ≈ 7, but the window has aged out → no stale recent rate.
        assert_eq!(a.window_rate(t0 + Duration::from_secs(33)), None);
    }

    #[test]
    fn output_count_never_decreases_on_smaller_iteration_usage() {
        let t0 = Instant::now();
        let mut a = TurnActivity::new(t0);
        // Warm-up chars/4 estimate climbs to ~700 before any usage lands.
        a.bump_output_estimate(700, t0 + Duration::from_secs(1));
        assert_eq!(a.tokens_out, 700);
        // Iteration-1 authoritative usage is SMALLER than the estimate. The
        // count must hold, not drop to 180 (the reported "goes down" bug).
        a.record_output_usage(180, 180, t0 + Duration::from_secs(2));
        assert_eq!(a.tokens_out, 700);
        // Iteration-2: cumulative grew (270) but its per-iteration `current`
        // (90) is smaller again. The monotonic cumulative-delta still never
        // lowers the displayed count.
        a.record_output_usage(270, 90, t0 + Duration::from_secs(3));
        assert!(
            a.tokens_out >= 700,
            "count must stay monotonic: {}",
            a.tokens_out
        );
    }

    #[test]
    fn fan_out_agent_total_is_monotonic_and_separate_from_main_count() {
        let t0 = Instant::now();
        let mut a = TurnActivity::new(t0);
        a.record_agent_output(5_000, t0 + Duration::from_secs(1));
        // A later snapshot summing fewer agent tokens must not lower the figure.
        a.record_agent_output(3_000, t0 + Duration::from_secs(2));
        assert_eq!(a.agent_tokens_out, 5_000, "agent aggregate stays monotonic");
        // Crucially it must NOT bleed into the main output count (the bug where a
        // fan-out anchor made the next main figure look like it dropped).
        assert_eq!(a.tokens_out, 0, "fan-out must not touch main tokens_out");
        // Fan-out bursts never seed the throughput window.
        assert_eq!(a.window_rate(t0 + Duration::from_secs(2)), None);
    }

    #[test]
    fn turn_token_label_shows_live_rate_next_to_output() {
        // CC-style: the running output count carries a live tok/s so the figure
        // visibly moves while streaming.
        assert_eq!(
            turn_token_label("Drafting response", 0, 2_345, 0, Some(142)),
            "↓ ~2.3k tokens · 142 tok/s"
        );
        // Rate omitted (warm-up) → falls back to the count-only label.
        assert_eq!(
            turn_token_label("Drafting response", 0, 2_345, 0, None),
            "↓ ~2.3k tokens"
        );
    }

    #[test]
    fn compact_token_label_keeps_live_rate() {
        // The narrow-width pass shortens the count but keeps the throughput,
        // since the rate is the signal the user asked to keep visible.
        assert_eq!(
            compact_token_label("↓ ~2.3k tokens · 142 tok/s"),
            "↓2.3k · 142/s"
        );
        // Without a rate it still compacts the count as before.
        assert_eq!(compact_token_label("↓ ~2.3k tokens"), "↓2.3k");
        // Combined input+output keeps both arrows on the narrow pass.
        assert_eq!(
            compact_token_label("↑ 1.2k ↓ ~2.3k tokens · 142 tok/s"),
            "↑1.2k ↓2.3k · 142/s"
        );
        assert_eq!(compact_token_label("↑ 1.2k ↓ ~2.3k tokens"), "↑1.2k ↓2.3k");
    }

    #[test]
    fn activity_suffix_compacts_without_losing_pending_or_escape() {
        let suffix = activity_suffix("1m 51s", "↑ 1.2k input · output pending", 36);

        assert!(
            suffix.width() <= 36,
            "compact suffix must fit its budget: {suffix:?}"
        );
        let text = suffix.text();
        assert!(text.contains("↑1.2k"), "input tokens survive: {text:?}");
        assert!(text.contains("pending"), "pending state survives: {text:?}");
        assert!(text.contains("esc"), "interrupt hint survives: {text:?}");
        assert!(
            !text.contains("output pending"),
            "compact suffix should shorten low-value wording: {text:?}"
        );
    }

    #[test]
    fn activity_suffix_drops_elapsed_before_pending_or_escape() {
        let suffix = activity_suffix("1m 51s", "agent output pending", 18);
        let text = suffix.text();

        assert!(suffix.width() <= 18, "suffix should fit: {text:?}");
        assert!(
            !text.contains("1m 51s"),
            "elapsed should be dropped before pending/esc: {text:?}"
        );
        assert!(text.contains("pending"), "pending survives: {text:?}");
        assert!(text.contains("esc"), "escape survives: {text:?}");
    }

    #[test]
    fn status_truncates_with_ellipsis() {
        assert_eq!(truncate_status("abcdef", 10), "abcdef");
        assert_eq!(truncate_status("abcdef", 3), "ab\u{2026}");
        assert_eq!(truncate_status("abcdef", 0), "");
    }

    #[test]
    fn status_truncates_by_terminal_cell_width() {
        let text = truncate_status("한국어상태표시", 7);
        assert!(text.ends_with('\u{2026}'), "text: {text:?}");
        assert!(
            display_width(&text) <= 7,
            "wide status must fit terminal cells: {text:?}"
        );
    }

    #[test]
    fn activity_parts_split_semicolon_detail() {
        assert_eq!(
            activity_parts("Reading tool output; choosing next step"),
            ActivityParts {
                phase: "Reading tool output",
                detail: Some("choosing next step")
            }
        );
    }

    #[test]
    fn activity_parts_split_colon_detail() {
        assert_eq!(
            activity_parts("Running command: cargo test"),
            ActivityParts {
                phase: "Running command",
                detail: Some("cargo test")
            }
        );
    }

    #[test]
    fn activity_parts_keep_plain_action() {
        assert_eq!(
            activity_parts("Drafting response"),
            ActivityParts {
                phase: "Drafting response",
                detail: None
            }
        );
    }

    #[test]
    fn activity_detail_truncates_after_phase() {
        let (phase, detail) =
            truncate_activity_parts("Reading tool output", Some("choosing next step"), 30);
        assert_eq!(phase, "Reading tool output");
        assert_eq!(detail.as_deref(), Some("choosin…"));
    }

    #[test]
    fn activity_detail_preserves_progress_and_left_before_tail() {
        let (phase, detail) = truncate_activity_parts(
            "Delegating",
            Some("2/4 complete · 50% · 50% left · 2 pre-analysis agents active (2 running)"),
            52,
        );
        assert_eq!(phase, "Delegating");
        assert_eq!(detail.as_deref(), Some("2/4 complete · 50% · 50% left"));
    }

    #[test]
    fn activity_detail_truncates_korean_by_terminal_cell_width() {
        let (phase, detail) =
            truncate_activity_parts("분석 중", Some("한국어 파일 내용을 확인하는 중"), 18);
        assert_eq!(phase, "분석 중");
        let detail = detail.expect("detail should fit after phase");
        assert!(detail.ends_with('\u{2026}'), "detail: {detail:?}");
        assert!(
            display_width(&phase) + 3 + display_width(&detail) <= 18,
            "phase + separator + detail must fit: phase={phase:?} detail={detail:?}"
        );
    }

    #[test]
    fn activity_limit_preserves_spinner_metrics_on_medium_widths() {
        let wide = activity_limit_for_width(140, "1m 51s", "↓ ~2.3k tokens", None);
        let medium = activity_limit_for_width(60, "1m 51s", "↓ ~2.3k tokens", None);
        assert_eq!(wide, MAX_ACTIVITY_CHARS);
        assert!(
            medium < wide,
            "medium panes should shorten the activity before clipping metrics"
        );
        assert!(medium > 0, "medium panes still keep an activity phrase");
    }

    #[test]
    fn activity_context_label_shows_model_on_medium_width() {
        let context = ActivityContext {
            model: Some("gpt-5.5-fast".to_string()),
            workflow: Some("phase 2/4 read-code running \u{2192} synthesize".to_string()),
            ..Default::default()
        };

        let label = activity_context_label(Some(&context), 90).expect("context label");

        assert!(label.starts_with("gpt-5.5-fast"));
    }

    #[test]
    fn activity_context_label_shows_workflow_and_model_on_wide_width() {
        let context = ActivityContext {
            model: Some("gpt-5.5-fast".to_string()),
            workflow: Some("phase 2/4 read-code running \u{2192} synthesize".to_string()),
            ..Default::default()
        };

        let label = activity_context_label(Some(&context), 132).expect("context label");

        assert!(label.contains("phase 2/4 read-code running \u{2192} synthesize"));
        assert!(label.contains("gpt-5.5-fast"));
    }

    #[test]
    fn activity_context_label_includes_ctx_percent_on_wide_pane() {
        let context = ActivityContext {
            model: Some("opus-4.8".to_string()),
            ctx_percent: Some(73),
            ..Default::default()
        };
        let label = activity_context_label(Some(&context), 120).expect("context label");
        assert!(
            label.contains("73% ctx"),
            "ctx fill must surface: {label:?}"
        );
    }

    #[test]
    fn turn_activity_reports_stall_after_idle_threshold() {
        use std::time::Duration;
        let now = Instant::now();
        let mut activity = TurnActivity::new(now);
        // Backdate the last event well past the stall threshold.
        activity.last_event_at = now
            .checked_sub(Duration::from_secs(STALL_THRESHOLD_SECS + 5))
            .expect("backdate");
        assert!(activity.is_stalled(STALL_THRESHOLD_SECS));
        // A fresh event clears the stall.
        activity.mark_event();
        assert!(!activity.is_stalled(STALL_THRESHOLD_SECS));
    }

    #[test]
    fn tool_action_gets_longer_stall_grace_than_text() {
        // Roadmap ⑥: a uniform 20s stall threshold cried "stuck" on a healthy
        // 90s `cargo test`. A tool action now gets the longer grace; text/
        // reasoning keeps the tight one; a later plain action clears the flag so
        // no stale grace lingers after the tool finishes.
        let mut a = TurnActivity::new(Instant::now());
        a.set_current_action("Drafting response");
        assert_eq!(a.stall_threshold_secs(), STALL_THRESHOLD_SECS);
        a.set_tool_action("Running command: cargo test");
        assert_eq!(a.stall_threshold_secs(), STALL_THRESHOLD_TOOL_SECS);
        a.set_current_action("Drafting response");
        assert_eq!(a.stall_threshold_secs(), STALL_THRESHOLD_SECS);
    }

    #[test]
    fn stall_badge_suppressed_before_first_content_until_longer_grace() {
        // Pre-first-content (warm-up): the tight 20s text cutoff must NOT fire —
        // a long server-side reasoning pass is healthy, not hung. The badge only
        // appears after the longer pre-first-content grace.
        assert_eq!(stall_badge(false, false, STALL_THRESHOLD_SECS + 5), None);
        assert_eq!(stall_badge(false, false, STALL_THRESHOLD_PREFIRST_SECS - 1), None);
        let badge = stall_badge(false, false, STALL_THRESHOLD_PREFIRST_SECS + 1)
            .expect("badge after the pre-first-content grace");
        assert!(badge.starts_with("no output "), "badge: {badge}");
    }

    #[test]
    fn stall_badge_fires_at_tight_threshold_once_content_streamed() {
        // Once output has streamed, a genuine mid-stream freeze must surface fast
        // at the tight cutoff (not wait the longer warm-up grace).
        assert!(stall_badge(true, false, STALL_THRESHOLD_SECS - 1).is_none());
        assert!(stall_badge(true, false, STALL_THRESHOLD_SECS).is_some());
    }

    #[test]
    fn stall_badge_tool_keeps_long_grace() {
        // A running tool keeps the 120s grace regardless of streamed content.
        assert!(stall_badge(false, true, STALL_THRESHOLD_SECS + 30).is_none());
        assert!(stall_badge(true, true, STALL_THRESHOLD_TOOL_SECS - 1).is_none());
        assert!(stall_badge(false, true, STALL_THRESHOLD_TOOL_SECS + 1).is_some());
    }

    #[test]
    fn recent_child_output_wins_over_stale_model_event_age() {
        assert_eq!(
            effective_idle_secs(21 * 60, Some(Duration::from_secs(3))),
            3
        );
        assert_eq!(effective_idle_secs(21 * 60, None), 21 * 60);
    }

    /// 라이브 리포트: sol/xhigh가 몇 분씩 침묵 리즈닝하는 동안 배지가
    /// "no output"으로 떠서 유저가 행으로 오인하고 Esc로 턴을 죽였다.
    /// 하트비트가 래치되면 같은 임계에서 차분한 "reasoning · stream alive"로
    /// 바뀌고, 임계 미만이면 래치돼도 배지는 없다.
    #[test]
    fn quiet_reasoning_latch_flips_badge_wording() {
        let calm = stall_badge_with_liveness(true, false, STALL_THRESHOLD_SECS + 10, StallLiveness::Reasoning)
            .expect("threshold crossed");
        assert!(
            calm.starts_with("reasoning · stream alive"),
            "calm badge: {calm}"
        );
        assert!(!calm.contains("no output"));
        // Same inputs without the latch keep the alarming wording.
        let alarming = stall_badge_with_liveness(true, false, STALL_THRESHOLD_SECS + 10, StallLiveness::None)
            .expect("threshold crossed");
        assert!(alarming.starts_with("no output"), "badge: {alarming}");
        // Below threshold the latch alone shows nothing.
        assert!(stall_badge_with_liveness(true, false, 1, StallLiveness::Reasoning).is_none());
    }

    /// 래치는 실제 진행 이벤트(스트림 델타·토큰 업데이트·`mark_event`)가 풀어
    /// 이후의 진짜 프리즈는 다시 "no output"으로 드러난다.
    #[test]
    fn quiet_reasoning_latch_clears_on_real_progress() {
        let now = Instant::now();
        let mut activity = TurnActivity::new(now);
        assert!(!activity.stream_alive_quiet(), "fresh turn is not latched");
        activity.note_stream_alive_quiet();
        assert!(activity.stream_alive_quiet());
        activity.mark_event();
        assert!(!activity.stream_alive_quiet(), "mark_event clears the latch");

        activity.note_stream_alive_quiet();
        activity.bump_output_estimate(4, now);
        assert!(!activity.stream_alive_quiet(), "streamed text clears the latch");

        activity.note_stream_alive_quiet();
        activity.record_output_usage(120, 120, now);
        assert!(!activity.stream_alive_quiet(), "usage snapshot clears the latch");

        activity.note_stream_alive_quiet();
        activity.record_agent_output(500, now);
        assert!(!activity.stream_alive_quiet(), "agent progress clears the latch");
    }

    /// 라이브 리포트: 하드 429 → 쿼터 대기 밴드(최대 ~15분) 파킹이 "no output
    /// 5m"로 떠서 유저가 세션이 죽은 줄 알았다. 래치되면 배지가 "rate-limited ·
    /// quota hold"로 바뀌고(quiet 래치보다 우선 — 파킹 중엔 keep-alive도 없다),
    /// 재개된 요청의 첫 진행 이벤트가 래치를 푼다.
    #[test]
    fn quota_hold_latch_names_the_park_and_clears_on_progress() {
        let badge =
            stall_badge_with_liveness(true, false, STALL_THRESHOLD_SECS + 10, StallLiveness::QuotaHold)
                .expect("threshold crossed");
        assert!(
            badge.starts_with("rate-limited · quota hold"),
            "badge: {badge}"
        );
        // Quota hold outranks the quiet-reasoning latch.
        let both = stall_badge_with_liveness(true, false, STALL_THRESHOLD_SECS + 10, StallLiveness::QuotaHold)
            .expect("threshold crossed");
        assert!(both.starts_with("rate-limited · quota hold"), "badge: {both}");

        let now = Instant::now();
        let mut activity = TurnActivity::new(now);
        assert!(!activity.quota_hold(), "fresh turn is not latched");
        activity.note_quota_hold();
        assert!(activity.quota_hold());
        activity.bump_output_estimate(4, now);
        assert!(!activity.quota_hold(), "resumed stream clears the latch");
    }

    #[test]
    fn has_streamed_content_tracks_main_model_output_only() {
        let now = Instant::now();
        let mut a = TurnActivity::new(now);
        assert!(!a.has_streamed_content(), "fresh turn has no output");
        // A message_start-style snapshot (cumulative 0) does NOT mark content.
        a.record_output_usage(0, 0, now);
        assert!(!a.has_streamed_content(), "empty cumulative is not content");
        // Sub-agent totals are not main-model content.
        a.record_agent_output(5_000, now);
        assert!(!a.has_streamed_content(), "agent tokens are not main output");
        // The first authoritative main-model usage flips it true.
        a.record_output_usage(120, 120, now);
        assert!(a.has_streamed_content());
    }

    #[test]
    fn long_tool_under_tool_threshold_is_not_stalled_but_same_text_gap_is() {
        use std::time::Duration;
        let now = Instant::now();
        // ~50s of silence: past the 20s text cutoff, well under the 120s tool one.
        let backdated = now
            .checked_sub(Duration::from_secs(STALL_THRESHOLD_SECS + 30))
            .expect("backdate");

        let mut tool = TurnActivity::new(now);
        tool.last_event_at = backdated;
        tool.set_tool_action("Running command: cargo build");
        assert!(
            !tool.is_stalled(tool.stall_threshold_secs()),
            "a 50s tool is healthy, not stalled"
        );

        let mut text = TurnActivity::new(now);
        text.last_event_at = backdated;
        text.set_current_action("Drafting response");
        assert!(
            text.is_stalled(text.stall_threshold_secs()),
            "a 50s text gap reads as stalled"
        );
    }

    #[test]
    fn activity_context_reduces_action_budget_but_keeps_it_positive() {
        let plain = activity_limit_for_width(120, "1m 51s", "↓ ~2.3k tokens", None);
        let with_context = activity_limit_for_width(
            120,
            "1m 51s",
            "↓ ~2.3k tokens",
            Some("phase 2/4 read-code running · gpt-5.5-fast"),
        );

        assert!(
            with_context < plain,
            "context badge should be budgeted instead of overflowing the row"
        );
        assert!(
            with_context > 0,
            "wide panes still keep a readable activity phrase"
        );
    }
}
