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

/// What the captured stack says the stalled main thread is actually doing.
///
/// The watchdog used to assert "this is a ZO-SIDE hang" from the beat counter
/// alone, which is only half the story: a render-loop write to a terminal that
/// has stopped draining also stops the beat, and that stall is downstream, not
/// a wedged async loop. Reading the sample the watchdog just captured turns the
/// verdict into an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreezeVerdict {
    /// Blocked in `write` — the terminal is not consuming output.
    TerminalWriteBlocked,
    /// Parked on a lock/condvar: a genuine zo-side hang.
    LockWait,
    /// Parked reading input or in an event wait, while the loop reports stalled.
    InputWait,
    /// No dominant leaf syscall — say so instead of guessing.
    Unknown,
}

impl FreezeVerdict {
    /// The sentence the watchdog logs for this verdict.
    #[must_use]
    pub const fn sentence(self) -> &'static str {
        match self {
            Self::TerminalWriteBlocked => {
                "the main thread is blocked in write() — the terminal is not draining zo's output (busy/suspended terminal, flow control, or a multiplexer that stopped reading). This stall is downstream of zo and clears when the terminal does."
            }
            Self::LockWait => {
                "the main thread is parked on a lock/condvar — this is a ZO-SIDE hang."
            }
            Self::InputWait => {
                "the main thread is parked in an input/event wait while the loop counter is frozen — suspect a loop that stopped beating rather than a blocking call."
            }
            Self::Unknown => {
                "the captured stack has no dominant blocking leaf; read the sample before concluding whose stall this is."
            }
        }
    }
}

/// Fraction of the main thread's samples a single leaf must own before it is
/// called the cause. Below this the thread moved during the capture, so no
/// single frame explains the stall.
const DOMINANT_LEAF_SHARE: f64 = 0.8;

/// Classify a macOS `sample(1)` report by what the main thread's dominant leaf
/// frame is. Pure so the real captures in `~/.zo/logs/zo-freeze-*.sample` can
/// be replayed as fixtures.
#[must_use]
pub fn classify_freeze_sample(report: &str) -> FreezeVerdict {
    let mut total: Option<u64> = None;
    let mut leaves: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    let mut in_main = false;
    for line in report.lines() {
        let Some((count, symbol)) = parse_sample_frame(line) else {
            continue;
        };
        if line.contains("com.apple.main-thread") {
            in_main = true;
            total = Some(count);
            continue;
        }
        if symbol.starts_with("Thread_") {
            // A sibling thread's block starts here; the main thread's is done.
            if in_main {
                break;
            }
            continue;
        }
        if in_main {
            *leaves.entry(symbol).or_default() += count;
        }
    }
    let Some(total) = total.filter(|total| *total > 0) else {
        return FreezeVerdict::Unknown;
    };
    #[allow(clippy::cast_precision_loss)] // sample counts are small
    let dominant = leaves
        .into_iter()
        .filter(|(symbol, _)| leaf_verdict(symbol).is_some())
        .filter(|(_, count)| *count as f64 >= total as f64 * DOMINANT_LEAF_SHARE)
        .max_by_key(|(_, count)| *count);
    dominant.map_or(FreezeVerdict::Unknown, |(symbol, _)| {
        leaf_verdict(symbol).unwrap_or(FreezeVerdict::Unknown)
    })
}

/// The verdict a known blocking syscall leaf implies, or `None` when the symbol
/// is not a leaf we can reason about.
fn leaf_verdict(symbol: &str) -> Option<FreezeVerdict> {
    match symbol {
        "write" | "writev" | "__write_nocancel" => Some(FreezeVerdict::TerminalWriteBlocked),
        "__psynch_cvwait" | "__psynch_mutexwait" | "__ulock_wait" | "__ulock_wait2"
        | "semaphore_wait_trap" => Some(FreezeVerdict::LockWait),
        "read" | "__read_nocancel" | "kevent" | "kevent_id" | "poll" | "select"
        | "mach_msg2_trap" => Some(FreezeVerdict::InputWait),
        _ => None,
    }
}

/// Pull `(sample_count, symbol)` out of one `sample(1)` call-graph line, which
/// looks like `+   444 write  (in libsystem_kernel.dylib) + 8  [0x198c74820]`
/// under a variable prefix of tree-drawing characters.
fn parse_sample_frame(line: &str) -> Option<(u64, &str)> {
    let body = line.trim_start_matches([' ', '\t', '+', '!', '|', ':', '*']);
    let mut parts = body.splitn(2, char::is_whitespace);
    let count = parts.next()?.parse::<u64>().ok()?;
    let rest = parts.next()?.trim_start();
    let symbol = rest
        .split_once("  ")
        .map_or(rest, |(symbol, _)| symbol)
        .trim();
    (!symbol.is_empty()).then_some((count, symbol))
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

    /// Shape of a real `~/.zo/logs/zo-freeze-*.sample` main-thread block, from
    /// the 2026-08-07 captures: every sample sits in a render-loop `write`.
    const BLOCKED_ON_WRITE: &str = "\
Call graph:
    444 Thread_372581952   DispatchQueue_1: com.apple.main-thread  (serial)
    + 444 start  (in dyld) + 7184  [0x1988e9d54]
    +   444 ???  (in zo)  load address 0x102394000 + 0x2539a8  [0x1025e79a8]
    +     444 ???  (in zo)  load address 0x102394000 + 0xf7f684  [0x103313684]
    +       444 write  (in libsystem_kernel.dylib) + 8  [0x198c74820]
    12 Thread_372581960
    + 12 __psynch_cvwait  (in libsystem_kernel.dylib) + 8  [0x198c744f8]
";

    #[test]
    fn a_render_loop_write_is_not_reported_as_a_zo_side_hang() {
        assert_eq!(
            classify_freeze_sample(BLOCKED_ON_WRITE),
            FreezeVerdict::TerminalWriteBlocked
        );
        assert!(
            FreezeVerdict::TerminalWriteBlocked
                .sentence()
                .contains("terminal is not draining"),
            "the verdict names the terminal, not zo"
        );
    }

    #[test]
    fn a_condvar_park_is_still_called_a_zo_side_hang() {
        let report = BLOCKED_ON_WRITE.replace("444 write ", "444 __psynch_cvwait ");
        assert_eq!(
            classify_freeze_sample(&report),
            FreezeVerdict::LockWait,
            "a lock wait keeps the old zo-side verdict"
        );
        assert!(FreezeVerdict::LockWait.sentence().contains("ZO-SIDE"));
    }

    /// A thread that moved during the capture has no dominant leaf, and the
    /// watchdog must say that rather than pick the biggest scrap.
    #[test]
    fn a_thread_that_kept_moving_yields_no_verdict() {
        let report = "\
Call graph:
    100 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
    + 40 write  (in libsystem_kernel.dylib) + 8  [0x1]
    + 30 __psynch_cvwait  (in libsystem_kernel.dylib) + 8  [0x2]
    + 30 kevent  (in libsystem_kernel.dylib) + 8  [0x3]
";
        assert_eq!(classify_freeze_sample(report), FreezeVerdict::Unknown);
    }

    /// Sibling threads must not contribute: only the main thread's block counts.
    #[test]
    fn sibling_thread_frames_are_ignored() {
        let report = "\
Call graph:
    10 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
    + 10 __psynch_cvwait  (in libsystem_kernel.dylib) + 8  [0x2]
    900 Thread_2
    + 900 write  (in libsystem_kernel.dylib) + 8  [0x1]
";
        assert_eq!(classify_freeze_sample(report), FreezeVerdict::LockWait);
    }
}
