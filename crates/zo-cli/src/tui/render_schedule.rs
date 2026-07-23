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
}

impl StreamFrameGate {
    /// Create a gate whose first stream event can draw immediately.
    #[must_use]
    pub fn new_ready(now: Instant, min_interval: Duration) -> Self {
        let last_stream_draw = now.checked_sub(min_interval).unwrap_or(now);
        Self {
            min_interval,
            last_stream_draw,
            deferred_stream_frame: false,
        }
    }

    /// Decide whether a stream-like event should repaint now.
    ///
    /// Use this for provider block arrivals and high-frequency wheel scrolls:
    /// both mutate visible state quickly and should be capped to terminal frame
    /// cadence while still landing their final frame on the next tick.
    pub fn on_stream_update(&mut self, now: Instant) -> DrawDecision {
        if now.duration_since(self.last_stream_draw) >= self.min_interval {
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
            if now.duration_since(self.last_stream_draw) >= self.min_interval {
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
        if now.duration_since(self.last_stream_draw) >= self.min_interval {
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
