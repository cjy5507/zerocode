//! Pure render scheduling policy for stream-heavy TUI paths.
//!
//! The event loops own IO, input handling, and the actual `draw()` call. This
//! module owns only the stream pacing decision: first visible output is
//! painted immediately, rapid stream bursts are coalesced into frame-sized
//! updates, and animation ticks recover deferred frames once they are eligible.

use std::time::{Duration, Instant};

/// Minimum gap between stream-driven full-screen redraws (~60 fps).
///
/// This does not delay text: streamed content lands in the transcript as it
/// arrives. The gate only caps how often the terminal is asked to repaint.
pub const STREAM_FRAME_INTERVAL: Duration = Duration::from_millis(16);

/// Shared animation/render tick interval (~30 fps).
///
/// Ticks recover deferred stream frames and keep width-stable animations moving
/// without racing the stream gate on a mismatched cadence.
pub const ANIMATION_TICK_INTERVAL: Duration = Duration::from_millis(33);

/// Ceiling for the adaptive stream interval: even the slowest terminal keeps
/// at least ~4 fps, so a throttled stream still visibly moves.
pub const MAX_STREAM_FRAME_INTERVAL: Duration = Duration::from_millis(250);

/// Multiple of the measured draw cost reserved for non-draw work.
///
/// 3× bounds draw time to ~25% of the loop under sustained streaming; the
/// remaining time keeps input/event handling responsive on terminals whose
/// paint is slower than the configured frame grid.
const DRAW_COST_HEADROOM: u32 = 3;

/// The outcome of a render scheduling decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawDecision {
    /// Draw on this event-loop iteration.
    DrawNow,
    /// Do not draw now; a later tick is responsible for the settled frame.
    DeferToTick,
}

impl DrawDecision {
    /// Returns true when the caller should perform a draw immediately.
    #[must_use]
    pub const fn draws_now(self) -> bool {
        matches!(self, Self::DrawNow)
    }
}

/// Coalesces stream/wheel redraw requests while preserving fast first paint.
#[derive(Debug)]
pub struct StreamFrameGate {
    min_interval: Duration,
    last_stream_draw: Instant,
    deferred_stream_frame: bool,
    /// Smoothed wall-clock cost of recent `draw()` calls, fed by
    /// [`Self::note_draw_cost`]. Zero until the first measurement, which keeps
    /// the configured `min_interval` in charge on fast terminals.
    draw_cost_ewma: Duration,
}

impl StreamFrameGate {
    /// Create a gate whose first stream event can draw immediately.
    #[must_use]
    pub fn new_ready(now: Instant, min_interval: Duration) -> Self {
        let ceiling = MAX_STREAM_FRAME_INTERVAL.max(min_interval);
        let last_stream_draw = now.checked_sub(ceiling).unwrap_or(now);
        Self {
            min_interval,
            last_stream_draw,
            deferred_stream_frame: false,
            draw_cost_ewma: Duration::ZERO,
        }
    }

    /// Feed the measured wall-clock cost of a completed draw back into the
    /// gate.
    ///
    /// On a fast terminal the cost is a millisecond or two and the gate keeps
    /// its configured cadence. When the terminal itself is the bottleneck
    /// (Apple Terminal.app painting a full frame in 50–200 ms — the `write`
    /// blocks once the tty buffer fills), the effective interval stretches to
    /// [`DRAW_COST_HEADROOM`]× the smoothed cost: a fast stream then degrades
    /// frame rate instead of input latency, because the loop spends a bounded
    /// fraction of its time inside `draw()`.
    pub fn note_draw_cost(&mut self, cost: Duration) {
        // EWMA (70% old, 30% new): smooths one-off spikes (resize, first
        // paint) while converging within a few frames on a regime change.
        let blended = (self.draw_cost_ewma.as_micros() * 7 + cost.as_micros() * 3) / 10;
        self.draw_cost_ewma = Duration::from_micros(u64::try_from(blended).unwrap_or(u64::MAX));
    }

    /// Effective minimum gap between stream draws: the configured interval,
    /// stretched by measured draw cost, capped so streams never look frozen.
    fn current_interval(&self) -> Duration {
        let ceiling = MAX_STREAM_FRAME_INTERVAL.max(self.min_interval);
        self.draw_cost_ewma
            .saturating_mul(DRAW_COST_HEADROOM)
            .clamp(self.min_interval, ceiling)
    }

    /// Decide whether a stream-like event should repaint now.
    ///
    /// Use this for provider block arrivals and high-frequency wheel scrolls:
    /// both mutate visible state quickly and should be capped to terminal frame
    /// cadence while still landing their final frame on the next tick.
    pub fn on_stream_update(&mut self, now: Instant) -> DrawDecision {
        if now.duration_since(self.last_stream_draw) >= self.current_interval() {
            self.mark_stream_drawn(now);
            DrawDecision::DrawNow
        } else {
            self.deferred_stream_frame = true;
            DrawDecision::DeferToTick
        }
    }

    /// Decide whether a render tick should repaint.
    ///
    /// `has_tick_work` is supplied by the event loop and covers non-stream work
    /// such as active animations, startup intro frames, HUD refreshes, or dirty
    /// input state. A deferred stream frame is also recovered here once doing so
    /// still respects the stream frame interval.
    pub fn on_tick(&mut self, now: Instant, has_tick_work: bool) -> DrawDecision {
        if self.deferred_stream_frame {
            if now.duration_since(self.last_stream_draw) >= self.current_interval() {
                self.mark_stream_drawn(now);
                DrawDecision::DrawNow
            } else {
                DrawDecision::DeferToTick
            }
        } else if has_tick_work {
            DrawDecision::DrawNow
        } else {
            DrawDecision::DeferToTick
        }
    }

    /// Decide whether a tick frame that represents active stream/turn work
    /// should repaint now. Unlike [`Self::on_tick`], this shares the stream
    /// frame budget in both directions: a provider-arrival draw prevents an
    /// immediate tick overpaint, and a tick draw prevents an immediate provider
    /// overpaint. That is the provider-agnostic fast→pause trigger: GPT chunks
    /// and Claude token deltas both enter through the same redraw drivers.
    pub fn on_stream_tick(&mut self, now: Instant, has_tick_work: bool) -> DrawDecision {
        if !has_tick_work {
            return DrawDecision::DeferToTick;
        }
        if now.duration_since(self.last_stream_draw) >= self.current_interval() {
            self.mark_stream_drawn(now);
            DrawDecision::DrawNow
        } else {
            self.deferred_stream_frame = true;
            DrawDecision::DeferToTick
        }
    }

    /// Record that the caller performed a full-screen draw for stream/turn work
    /// outside [`Self::on_stream_update`]. This lets animation/tick-driven
    /// frames share the same budget as block-arrival frames instead of
    /// immediately overpainting them and flooding slower terminal emulators.
    pub fn note_stream_draw(&mut self, now: Instant) {
        self.mark_stream_drawn(now);
    }

    fn mark_stream_drawn(&mut self, now: Instant) {
        self.last_stream_draw = now;
        self.deferred_stream_frame = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_stream_update_draws_immediately() {
        let now = Instant::now();
        let mut gate = StreamFrameGate::new_ready(now, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_stream_update(now), DrawDecision::DrawNow);
    }

    #[test]
    fn burst_stream_updates_are_coalesced_then_recovered_by_tick() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(1)),
            DrawDecision::DeferToTick
        );
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(2)),
            DrawDecision::DeferToTick
        );
        assert_eq!(
            gate.on_tick(start + ANIMATION_TICK_INTERVAL, false),
            DrawDecision::DrawNow
        );
    }

    #[test]
    fn deferred_stream_tick_respects_frame_interval() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(1)),
            DrawDecision::DeferToTick
        );
        assert_eq!(
            gate.on_tick(start + Duration::from_millis(2), false),
            DrawDecision::DeferToTick,
            "tick recovery must not create back-to-back full redraws"
        );
        assert_eq!(
            gate.on_tick(start + STREAM_FRAME_INTERVAL, false),
            DrawDecision::DrawNow
        );
    }

    #[test]
    fn tick_with_animation_work_draws_even_without_stream() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_tick(start, true), DrawDecision::DrawNow);
    }

    #[test]
    fn tick_without_work_stays_idle() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_tick(start, false), DrawDecision::DeferToTick);
    }

    #[test]
    fn animation_tick_does_not_delay_first_stream_update() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_tick(start, true), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(1)),
            DrawDecision::DrawNow,
            "non-stream animation draws must not consume the stream frame budget"
        );
    }

    #[test]
    fn noted_tick_draw_throttles_immediate_stream_overpaint() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        gate.note_stream_draw(start);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(1)),
            DrawDecision::DeferToTick,
            "a tick-driven turn draw should share the stream frame budget so block arrivals do not immediately overpaint it"
        );
        assert_eq!(
            gate.on_stream_update(start + STREAM_FRAME_INTERVAL),
            DrawDecision::DrawNow
        );
    }

    #[test]
    fn stream_tick_after_provider_draw_respects_shared_budget() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_tick(start + Duration::from_millis(1), true),
            DrawDecision::DeferToTick,
            "turn/render ticks must not immediately overpaint a provider-driven stream draw"
        );
        assert_eq!(
            gate.on_stream_tick(start + STREAM_FRAME_INTERVAL, true),
            DrawDecision::DrawNow,
            "the deferred tick frame should recover as soon as the shared frame budget allows"
        );
    }

    #[test]
    fn stream_tick_without_work_stays_idle() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_stream_tick(start, false), DrawDecision::DeferToTick);
    }

    #[test]
    fn fast_terminal_keeps_configured_cadence() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        // Millisecond-class draws (fast emulator): the adaptive interval must
        // stay pinned at the configured frame grid.
        for _ in 0..10 {
            gate.note_draw_cost(Duration::from_millis(1));
        }
        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + STREAM_FRAME_INTERVAL),
            DrawDecision::DrawNow,
            "cheap draws must not stretch the interval past the frame grid"
        );
    }

    #[test]
    fn slow_terminal_stretches_interval_to_bound_draw_share() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        // Sustained 100ms draws (Apple Terminal.app class): the EWMA converges
        // near 100ms, so the effective interval approaches 300ms — the loop
        // keeps ~2/3 of its time free for input instead of saturating on draws.
        for _ in 0..20 {
            gate.note_draw_cost(Duration::from_millis(100));
        }
        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(100)),
            DrawDecision::DeferToTick,
            "a stream burst must not repaint while the terminal is still the bottleneck"
        );
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(249)),
            DrawDecision::DeferToTick
        );
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(251)),
            DrawDecision::DrawNow
        );
    }

    #[test]
    fn adaptive_interval_recovers_when_terminal_speeds_up() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        for _ in 0..20 {
            gate.note_draw_cost(Duration::from_millis(100));
        }
        // Regime change back to a fast terminal (window shrunk, alt screen):
        // cheap draws pull the EWMA — and the interval — back down.
        for _ in 0..20 {
            gate.note_draw_cost(Duration::from_millis(1));
        }
        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(33)),
            DrawDecision::DrawNow,
            "the interval must recover once draws are cheap again"
        );
    }

    #[test]
    fn adaptive_interval_is_capped_so_streams_never_look_frozen() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        for _ in 0..30 {
            gate.note_draw_cost(Duration::from_secs(1));
        }
        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + MAX_STREAM_FRAME_INTERVAL),
            DrawDecision::DrawNow,
            "even pathological draw costs must keep at least the capped cadence"
        );
    }

    #[test]
    fn draw_recovered_by_tick_resets_stream_gate() {
        let start = Instant::now();
        let mut gate = StreamFrameGate::new_ready(start, STREAM_FRAME_INTERVAL);

        assert_eq!(gate.on_stream_update(start), DrawDecision::DrawNow);
        assert_eq!(
            gate.on_stream_update(start + Duration::from_millis(1)),
            DrawDecision::DeferToTick
        );
        assert_eq!(
            gate.on_tick(start + ANIMATION_TICK_INTERVAL, false),
            DrawDecision::DrawNow
        );
        assert_eq!(
            gate.on_stream_update(start + ANIMATION_TICK_INTERVAL + Duration::from_millis(1)),
            DrawDecision::DeferToTick,
            "a tick-drawn deferred frame should prevent an immediate extra repaint"
        );
    }
}
