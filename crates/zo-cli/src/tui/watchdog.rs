//! Liveness heartbeat + phase marker for the interactive TUI event loops.
//!
//! Both live loops — the idle [`crate::tui::app::App::run`] select loop and the
//! mid-turn render loop in the `zo` bin (`session::turn_controller::drive_turn`)
//! — call [`beat`] once per iteration, and the bin marks [`set_phase`] as the
//! main async task moves between coarse stages. A background watchdog (spawned
//! by the bin at session start) samples [`beat_count`] every second.
//!
//! The point is to settle, with a *fact* rather than a guess, the intermittent
//! "the TUI freezes when I go to type again" report: if the counter stops
//! advancing the async event loop itself is wedged (a zo-side hang, e.g. a
//! blocking call or a lock held across a retrying request that slipped onto the
//! main task); if the counter keeps advancing while the user sees a frozen
//! screen, zo is healthy and the freeze is downstream in the terminal
//! emulator. The watchdog writes its verdict — including the [`phase_label`] of
//! the stalled stage — to the redirected-stderr log (`~/.zo/logs/zo.log`).
//!
//! One relaxed add/store per frame is free; the watchdog only ever writes on a
//! multi-second stall, so this is dormant in normal operation.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// Monotonic count of live TUI event-loop iterations. See the module docs.
static MAIN_LOOP_BEAT: AtomicU64 = AtomicU64::new(0);

/// Coarse stage the main async task is currently in, so a stall can be named
/// without symbols on a stripped release binary.
static MAIN_PHASE: AtomicU8 = AtomicU8::new(0);

/// What the main async task is doing. Set at each stage boundary so the freeze
/// watchdog can report *where* a stall happened.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Phase {
    /// Idle prompt — `App::run` select loop.
    Idle = 0,
    /// Per-turn setup before the turn is spawned (client build, MCP refresh,
    /// route hint) — `run_live_turn_with_images`.
    PreTurnSetup = 1,
    /// Pre-turn OAuth/client refresh (pumped, bounded).
    OauthRefresh = 2,
    /// Smart prelude / semantic-triage work — `maybe_apply_auto_fanout_live`.
    FanoutPrelude = 3,
    /// The streaming turn's render/select loop — `drive_turn`.
    TurnRender = 4,
    /// Post-turn work on the main task: persist, checkpoint, goal advance.
    PostTurn = 5,
}

/// Mark the main task's current stage (see [`Phase`]).
#[inline]
pub fn set_phase(phase: Phase) {
    MAIN_PHASE.store(phase as u8, Ordering::Relaxed);
}

/// Human-readable label for the current phase, for the watchdog's stall report.
#[must_use]
pub fn phase_label() -> &'static str {
    match MAIN_PHASE.load(Ordering::Relaxed) {
        0 => "idle (App::run)",
        1 => "pre-turn setup (run_live_turn_with_images: client/MCP/route)",
        2 => "pre-turn OAuth/client refresh",
        3 => "Smart prelude (maybe_apply_auto_fanout_live)",
        4 => "drive_turn render loop",
        5 => "post-turn (persist/checkpoint/goal-advance)",
        _ => "unknown",
    }
}

/// Record one live event-loop iteration. Call at the top of each TUI loop body.
#[inline]
pub fn beat() {
    MAIN_LOOP_BEAT.fetch_add(1, Ordering::Relaxed);
}

/// Current beat count, sampled by the freeze watchdog thread.
#[must_use]
pub fn beat_count() -> u64 {
    MAIN_LOOP_BEAT.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fanout_prelude_phase_label_uses_smart_wording() {
        set_phase(Phase::FanoutPrelude);
        assert_eq!(phase_label(), "Smart prelude (maybe_apply_auto_fanout_live)");
        set_phase(Phase::Idle);
    }
}
