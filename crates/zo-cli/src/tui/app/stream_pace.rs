//! Lightweight streaming coalescer — smooths provider-sized text bursts without
//! turning small deltas into a slow typewriter.
//! (Claude/DeepSeek native streams) reads as a smooth type-in instead of whole
//! chunks slamming onto the screen one "뭉탱이" at a time.
//!
//! This is intentionally **not** the old smooth-reveal controller (deleted in
//! `9e908c2f`). That one felt slow for two concrete reasons, and this pacer
//! avoids both by construction:
//!
//! 1. **No rate ceiling.** The old controller capped sustained reveal at
//!    `MAX_REVEAL_RATE = 1200 c/s`, so a fast stream was *throttled* below its
//!    generation speed. Here the instantaneous rate is `backlog / WINDOW`, so a
//!    larger backlog drains proportionally faster (an ease-out curve): big
//!    content comes out quicker, never slower.
//! 2. **No fixed tail reserve.** The old controller always retained ~0.18 s of
//!    buffer and kept typing a tail after the model had already finished (up to
//!    ~0.95 s of trailing latency). Here there is no reserve: a provider-closed
//!    text block lands immediately when small, while a large final provider
//!    burst is smoothed over the shorter finish window and then sealed. An
//!    aborted/open tail that is only marked done at turn end uses the same short
//!    finish window and flushes the last sub-frame remainder.
//!
//! The net effect is faster than a slow typewriter while still avoiding a huge
//! open-stream burst slamming onto the screen. Pacing only applies to live,
//! still-open streamed prose during an active turn; replay/resume (no active
//! turn) bypasses it and lands whole, and any non-prose block flushes the paced
//! tail first so true arrival order is always preserved without an ordering
//! queue.

use std::time::{Duration, Instant};

use crate::tui::render_schedule::ANIMATION_TICK_INTERVAL;

use super::App;

/// One render frame at the shared 30 fps tick grid — used only to size the
/// "near-done, just finish it" threshold below.
const FRAME: Duration = ANIMATION_TICK_INTERVAL;

/// Target time to drain the *current* backlog while the stream is still open.
/// The per-frame reveal is `backlog * (dt / WINDOW)`, i.e. ~37 % of the backlog
/// per 33 ms frame — a burst is spread across ~2-3 frames, enough to read as a
/// type-in without ever falling behind generation. Smaller than a human
/// reaction time, so it never adds perceptible latency.
const WINDOW: Duration = Duration::from_millis(60);

/// Upper bound on the adaptive drain window. When the provider delivers in
/// widely-spaced clumps (Claude's ~480ms inter-token gaps), a small backlog is
/// spread across the estimated gap until the next delta instead of draining in
/// one `WINDOW` and leaving the rest of the gap blank — that is what turned a
/// genuine per-token stream into the "뭉텅이 → pause → 뭉텅이" cadence. Capped so
/// a genuinely long pause (the model thinking) never stalls the visible reveal
/// for more than this; beyond it the backlog drains at the normal `WINDOW`/floor
/// rate. Sits just under a human's ~0.5s "is it stuck?" threshold.
const MAX_ADAPTIVE_WINDOW: Duration = Duration::from_millis(450);

/// After a stream is marked `done`, drain any large remaining tail against this
/// shorter window so final provider bursts and aborted/terminal edge cases
/// settle promptly instead of trailing. Small final arrivals still land whole
/// immediately via [`DoneArrivalPolicy::RevealImmediately`].
const FINISH_WINDOW: Duration = Duration::from_millis(30);

/// Floor on the drain rate (chars/sec). Guarantees the tail always terminates
/// (the `backlog / WINDOW` term alone decays geometrically and would only
/// asymptote toward empty) and keeps a very slow trickle still visibly typing.
///
/// Set to ~200 c/s (≈ 6-7 chars per 33 ms frame), not the old 360. At 360 a
/// single frame drained ~12 chars, so any small mid-cadence continuation backlog
/// (a ~10-char Claude delta arriving every ~50 ms) was dumped *whole* on its
/// first frame and the reveal then sat idle until the next delta — a periodic
/// catch-up-then-idle micro-hitch (`[10,10,0,10,10,0,…]`) the user still felt as
/// not-smooth. A 200 c/s floor meters that same backlog across the gap as a
/// steady glide (`[10,6,4,6,8,…]`, idle frames eliminated) while staying well
/// above a readable trickle so the tail never looks frozen. Large/fast backlogs
/// are unaffected: their rate is the much higher `backlog / WINDOW` term, so this
/// floor only ever binds the small-backlog steady-state where smoothness matters.
const FLOOR_RATE: f32 = 200.0;

/// Gentler drain floor used only while smoothing a small continuation backlog
/// across an estimated inter-arrival gap (the adaptive path). Low enough that a
/// ~20-char Claude clump spreads across most of a ~480ms gap instead of the
/// ~100ms the normal `FLOOR_RATE` would force, but still a steady visible
/// trickle (~60 c/s ≈ 2 chars per frame) so it never looks frozen and always
/// terminates.
const ADAPTIVE_FLOOR_RATE: f32 = 60.0;

/// Hard cap on a single drip. Only relevant for a pathological one-shot dump
/// (a giant pasted/replayed block routed through the pacer): it still reads as
/// a fast type-in rather than a single instant paint. Far above any real
/// per-frame backlog, so a normal fast stream is never throttled by it.
const MAX_CHUNK: usize = 4096;

/// Land-whole threshold: a delta at or below this many chars is revealed whole
/// on arrival (zero added latency); anything larger is gently typed in across a
/// frame or two at [`FLOOR_RATE`]. Set to a *phrase* (~24 chars), not a full
/// sentence, so mid-sized provider chunks stop slamming in as one "뭉탱이" — they
/// flow in instead. Kept high enough that genuinely tiny word-sized deltas still
/// land instantly, so this reads as smooth web-chat streaming, never a slow
/// typewriter. Lowered 64 → 24 for gentler per-delta smoothing; the drain-finish
/// promotion uses the separate [`TAIL_PROMOTE_CHARS`] so large backlogs still
/// settle just as fast.
const IMMEDIATE_CHARS: usize = 24;

/// Drain-finish promotion threshold: when a drain frame would leave only this
/// many chars behind, take them all on that frame instead of trailing a small
/// remainder over yet another frame. Deliberately a full phrase-and-a-bit and
/// kept independent of the smaller [`IMMEDIATE_CHARS`] land-whole threshold, so
/// dialing per-delta smoothing down never slows how promptly a large provider
/// burst settles.
const TAIL_PROMOTE_CHARS: usize = 64;

/// Largest same-block continuation backlog that is treated as a *single delta*
/// to be smoothly typed in rather than a catch-up burst to drain fast. A live
/// Claude stream delivers token clumps and sentence-sized deltas (observed
/// 16-70 chars per provider `content_block_delta`, occasionally larger when a
/// network read batches several), and at the 16-33 ms drip cadence the per-frame
/// backlog rarely exceeds this. Anything above it is the model running ahead of
/// the reveal (a real catch-up burst), which still drains fast against `WINDOW`
/// with the tail-promote finish. Set well above a sentence so a normal delta is
/// always metered out across frames instead of being tail-promoted whole — the
/// "뭉텅이 → pause → 뭉텅이" cadence — while staying below the multi-hundred-char
/// bursts that should settle quickly.
const SMOOTH_CONTINUATION_MAX: usize = 128;

/// Characters revealed on the very first drip of a larger freshly opened block.
/// Phrase-sized, not glyph-sized: enough lands immediately for a web-chat feel
/// while preventing a large burst from slamming in at once. Matches
/// [`IMMEDIATE_CHARS`] so a block opening just above the land-whole threshold
/// still shows a full phrase on its first frame.
const STARTER_CHARS: f32 = 24.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DoneArrivalPolicy {
    /// Nothing is buffered; send only the terminal `done=true` seal.
    SealEmpty,
    /// A small final delta is already smooth enough to land on arrival.
    RevealImmediately,
    /// A large final burst should use [`FINISH_WINDOW`] instead of one-frame flush.
    PaceFinishWindow,
}

impl DoneArrivalPolicy {
    const fn for_pending_chars(pending_chars: usize) -> Self {
        match pending_chars {
            0 => Self::SealEmpty,
            1..=IMMEDIATE_CHARS => Self::RevealImmediately,
            _ => Self::PaceFinishWindow,
        }
    }
}

/// Per-block pacing buffer: characters that have arrived from the provider but
/// have not yet been revealed into the transcript. Exactly one is live at a
/// time (the open streaming prose block); a non-prose block or a block-id change
/// flushes it.
#[derive(Debug)]
pub(super) struct StreamPacer {
    /// Transcript block these characters belong to. Reveals are pushed as
    /// `TextDelta` with this id so the transcript merges them onto the block.
    block_id: runtime::message_stream::BlockId,
    /// Received-but-not-yet-revealed characters.
    pending: String,
    /// `pending.chars().count()`, maintained incrementally so each drip is O(1)
    /// in the backlog size instead of rescanning the whole buffer per frame.
    pending_chars: usize,
    /// The provider closed this block; once `pending` drains, forward `done`.
    done: bool,
    /// Wall-clock instant of the last drip, so the next reveals an amount
    /// proportional to the time actually elapsed — a real cadence independent of
    /// how often the drip is driven.
    last_drip: Instant,
    /// Fractional character carried between drips (the earned count is rarely a
    /// whole number); without it, rounding would bias the long-run rate.
    carry: f32,
    /// `true` only for the initial arrival of a newly opened block. This keeps
    /// genuinely tiny openings zero-latency while preventing later same-block
    /// continuation bursts from using the small-buffer land-whole shortcut.
    allow_small_immediate: bool,
    /// A same-block continuation arrived since the last reveal. Small Claude
    /// token clumps should be metered by frame cadence, not promoted whole by
    /// the tail-finish shortcut on the next tick.
    continuation_burst: bool,
    /// Wall-clock instant of the last *arrival* (a `buffer_paced_at` call), as
    /// opposed to the last drip. Used to estimate the provider's inter-arrival
    /// gap so a small backlog can be spread across the time until the *next*
    /// delta is expected, instead of draining in ~60ms and then showing nothing
    /// for the rest of a ~480ms gap (the Claude "뭉텅이" cadence).
    last_arrival: Instant,
    /// Smoothed estimate (EWMA) of the gap between continuation arrivals. Seeded
    /// at `WINDOW` so the first continuation behaves exactly as before until a
    /// real cadence is observed. Bounded by `MAX_ADAPTIVE_WINDOW` so a long pause
    /// can never stall the reveal for seconds.
    arrival_interval: Duration,
}

impl App {
    /// `true` while paced characters are still waiting to be revealed — the
    /// idle/turn loops keep ticking until this drains so the tail types out.
    #[must_use]
    pub fn stream_pending(&self) -> bool {
        self.stream_pacer.is_some()
    }

    /// Buffer a streamed text delta for paced reveal, then drip the first slice
    /// on this same frame (so the block opens immediately). A delta for a
    /// different block id flushes the current tail first to preserve order.
    pub(super) fn buffer_paced(
        &mut self,
        id: runtime::message_stream::BlockId,
        text: String,
        done: bool,
    ) {
        self.buffer_paced_at(Instant::now(), id, text, done);
    }

    /// Time-injected core of [`Self::buffer_paced`]. `now` is the clock the drip
    /// math measures against — real in production, controlled in tests so a
    /// burst+gap cadence is deterministic without a real sleep.
    pub(super) fn buffer_paced_at(
        &mut self,
        now: Instant,
        id: runtime::message_stream::BlockId,
        text: String,
        done: bool,
    ) {
        let opened_new_block = match self.stream_pacer.as_mut() {
            Some(pacer) if pacer.block_id == id => {
                let appended_chars = text.chars().count();
                pacer.pending_chars += appended_chars;
                pacer.pending.push_str(&text);
                pacer.done |= done;
                if appended_chars > 0 && !done {
                    pacer.continuation_burst = true;
                    // Track the provider's inter-arrival cadence (EWMA) so the
                    // drip can spread a small backlog across the gap until the
                    // next delta is expected, rather than draining it in one
                    // 60ms WINDOW and going blank for the rest of a ~480ms gap.
                    //
                    // Only deltas that arrive a real frame or more apart inform
                    // the cadence: several tokens delivered in ONE network read
                    // land on the same instant (gap ≈ 0) and must still settle
                    // promptly — folding their 0ms gap into the EWMA would wrongly
                    // slow that whole-read burst down. So a sub-frame gap leaves
                    // the interval (and thus the fast default cadence) untouched.
                    let gap = now.saturating_duration_since(pacer.last_arrival);
                    if gap >= FRAME {
                        // 0.5 weight: responsive to a cadence shift within a
                        // couple of deltas without letting one outlier dominate.
                        pacer.arrival_interval = (pacer.arrival_interval + gap) / 2;
                    }
                }
                pacer.last_arrival = now;
                false
            }
            Some(_) => {
                // A new block opened before the previous tail finished: land the
                // old tail whole (order) and start pacing the new one.
                self.flush_stream();
                self.open_pacer(now, id, text, done);
                true
            }
            None => {
                self.open_pacer(now, id, text, done);
                true
            }
        };
        if done {
            let policy = self
                .stream_pacer
                .as_ref()
                .map_or(DoneArrivalPolicy::SealEmpty, |pacer| {
                    DoneArrivalPolicy::for_pending_chars(pacer.pending_chars)
                });
            match policy {
                DoneArrivalPolicy::SealEmpty => self.seal_paced_block(),
                DoneArrivalPolicy::RevealImmediately | DoneArrivalPolicy::PaceFinishWindow => {
                    self.drip_stream_elapsed(now, None);
                }
            }
        } else if opened_new_block {
            // Open the freshly started block on its arrival frame for a
            // low-latency first paint (the `STARTER_CHARS` carry shows a phrase
            // at once).
            //
            // A *continuation* delta is intentionally NOT dripped here (see the
            // sibling branch). Dripping on every arrival reset `last_drip` to
            // ~now on each token, so the wall-clock drip never saw a real elapsed
            // span and the small-delta land-whole / tail-promote shortcuts
            // revealed each token the instant it arrived — making the on-screen
            // cadence mirror the provider's bursty network delivery. That is why
            // a genuine per-token stream (Claude) read as clump-then-pause
            // stutter while a coarse-chunk provider looked smooth.
            self.drip_stream_elapsed(now, None);
        } else {
            // Continuation delta: do NOT drip on arrival. Let it accumulate and
            // be metered by the frame-driven drip (`advance_tick` and the gated
            // block-arrival `drip_stream`, both on the 30-60 fps grid, kept alive
            // by `stream_pending`), which decouples reveal cadence from arrival
            // cadence so a burst spreads across the following frames instead of
            // slamming in whole. A non-prose block flushes, and turn end
            // finishes, the accumulated tail, so nothing is lost or reordered.
        }
    }

    fn open_pacer(
        &mut self,
        now: Instant,
        id: runtime::message_stream::BlockId,
        text: String,
        done: bool,
    ) {
        self.stream_pacer = Some(StreamPacer {
            block_id: id,
            pending_chars: text.chars().count(),
            pending: text,
            done,
            last_drip: now,
            carry: STARTER_CHARS,
            allow_small_immediate: true,
            continuation_burst: false,
            last_arrival: now,
            arrival_interval: WINDOW,
        });
    }

    /// Reveal the characters the elapsed wall-clock time has earned since the
    /// last drip. Driven by [`App::advance_tick`] (the 30 fps grid) and by each
    /// block arrival, so the cadence tracks real time on whichever fires first.
    ///
    /// `pub` (like [`App::advance_tick`]) so the mid-turn loop in the `zo`
    /// bin (`session::turn_controller::drive_turn`) can drive the drip on each
    /// throttled block-arrival repaint, matching the idle loop; a lib-internal
    /// `pub(super)` would not reach the bin crate.
    pub fn drip_stream(&mut self) {
        self.drip_stream_elapsed(Instant::now(), None);
    }

    /// Time-injected drip for tests: drive the reveal with an explicit `now` and
    /// an optional forced elapsed span, so a burst cadence is fully
    /// deterministic without a real sleep.
    #[cfg(test)]
    pub(super) fn drip_stream_at(&mut self, now: Instant, forced: Option<Duration>) {
        self.drip_stream_elapsed(now, forced);
    }

    /// Core drip. `forced` overrides the measured `now - last_drip` span (tests);
    /// `None` measures the real elapsed time (production). Reveals on a UTF-8
    /// char boundary so a multibyte glyph (CJK / emoji) is never split.
    fn drip_stream_elapsed(&mut self, now: Instant, forced: Option<Duration>) {
        let Some(snapshot) = self.stream_pacer.as_ref() else {
            return;
        };
        let pending_chars = snapshot.pending_chars;
        let done = snapshot.done;

        // Nothing buffered: seal if the provider is done, otherwise keep the
        // (empty) pacer so the next delta merges onto the same block and pacing
        // stays smooth across an inter-burst gap.
        if pending_chars == 0 {
            if done {
                self.seal_paced_block();
            } else if let Some(pacer) = self.stream_pacer.as_mut() {
                // Keep the drip clock fresh while idling between deltas.
                // Leaving `last_drip` at the last actual reveal made the first
                // drip after a clumpy provider's ~470ms gap measure dt ≈ the
                // whole gap, earn ~a delta's worth of characters at once, and
                // dump the freshly arrived backlog in one frame — the exact
                // clump→pause stutter the adaptive spread exists to hide. With
                // the clock pinned to the tick grid, that first drip sees
                // dt ≈ one frame and meters the new delta smoothly.
                pacer.last_drip = now;
            }
            return;
        }

        let Some(pacer) = self.stream_pacer.as_mut() else {
            return;
        };

        let dt = forced.unwrap_or_else(|| now.saturating_duration_since(pacer.last_drip));
        pacer.last_drip = now;

        // Small openings and small terminal deltas are already perceptually
        // smooth, so keep them zero-latency. Do NOT apply this to same-block
        // continuation backlog: Claude often delivers several tiny tokens in
        // one network read, and revealing that <=24 char backlog whole on the
        // next frame preserves the exact clump→pause stutter this pacer exists
        // to hide.
        if pending_chars <= IMMEDIATE_CHARS && (pacer.allow_small_immediate || done) {
            pacer.carry = 0.0;
            pacer.allow_small_immediate = false;
            pacer.continuation_burst = false;
            self.reveal_paced(pending_chars);
            return;
        }

        let dt_secs = dt.as_secs_f32();
        // A same-block continuation backlog up to a delta's worth is smoothed
        // (typed in across frames) rather than tail-promoted whole. Gated by the
        // wide `SMOOTH_CONTINUATION_MAX`, not the smaller `TAIL_PROMOTE_CHARS`:
        // Claude routinely delivers sentence-sized deltas (~70 chars), and at
        // the old 64-char gate those fell straight through to the tail-promote
        // and slammed in whole — the exact "뭉텅이" the user still felt. Only a
        // genuine catch-up backlog (model running well ahead of the reveal)
        // exceeds this and drains fast against `WINDOW`.
        let smooth_small_continuation = pacer.continuation_burst
            && !pacer.done
            && pending_chars <= SMOOTH_CONTINUATION_MAX;
        // The *adaptive* path only engages when the provider is actually
        // delivering in widely-spaced clumps (estimated gap above one WINDOW):
        // then a small clump is spread across the gap until the next delta. A
        // burst delivered in one network read (gap ≈ 0, interval stays at WINDOW)
        // keeps the original fast settle so it never drags.
        let adaptive = smooth_small_continuation && pacer.arrival_interval > WINDOW;

        // Drain window: normally a tight 60ms so bursts type in fast; a clumpy
        // provider's small backlog is spread across the estimated inter-arrival
        // gap (bounded by MAX_ADAPTIVE_WINDOW). `done` always uses the short
        // finish window so the end never drags.
        let window = if pacer.done {
            FINISH_WINDOW
        } else if adaptive {
            pacer.arrival_interval.clamp(WINDOW, MAX_ADAPTIVE_WINDOW)
        } else {
            WINDOW
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "backlog char counts stay well under 2^24, so the f32 cast is exact"
        )]
        let backlog = pacer.pending_chars as f32;
        // Floor on the drain rate. Normally `FLOOR_RATE` keeps a fast stream
        // visibly typing, but for an adaptive (clumpy-provider) reveal that floor
        // would drain the small backlog in one frame and defeat the wider window,
        // so a smoothed continuation uses a gentler floor that still guarantees a
        // readable trickle (~1 char per couple of frames) and termination.
        let floor = if adaptive {
            ADAPTIVE_FLOOR_RATE
        } else {
            FLOOR_RATE
        };
        let rate = (backlog / window.as_secs_f32()).max(floor);

        let earned = pacer.carry + rate * dt_secs;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "earned is non-negative and floored before the usize cast"
        )]
        let mut take = earned.floor().max(0.0) as usize;
        let capped = take >= MAX_CHUNK;
        take = take.min(pacer.pending_chars).min(MAX_CHUNK);

        // Carry the sub-character remainder only when the reveal was rate-bound
        // (not when we drained the buffer or hit the per-frame cap).
        pacer.carry = if take == pacer.pending_chars || capped {
            0.0
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "take is a small per-frame count; the f32 round-trip is exact"
            )]
            let taken = take as f32;
            earned - taken
        };

        // If this real elapsed frame would leave only a phrase-sized remainder
        // behind, take it now so a large backlog settles promptly instead of
        // trailing one more frame. Uses the larger `TAIL_PROMOTE_CHARS` (not the
        // smaller per-delta land-whole threshold) so finish speed is unaffected
        // by how aggressively small deltas are smoothed. Not applied to the
        // arrival-frame starter (`dt == 0`), or a barely-large first burst would
        // dump all at once. Suppressed while smoothing a small continuation
        // backlog (same-instant burst's first frame, or the whole adaptive
        // spread) so the remainder is metered out rather than dumped whole —
        // promoting it is exactly the 뭉텅이 this pacer hides.
        if !smooth_small_continuation
            && dt > Duration::ZERO
            && pacer.pending_chars.saturating_sub(take) <= TAIL_PROMOTE_CHARS
        {
            take = pacer.pending_chars;
        }

        // Once the provider is done and only a sub-frame remainder is left, take
        // it all this frame so the answer never dribbles char-by-char at the end.
        if pacer.done && dt > Duration::ZERO {
            let frame_chars = (FLOOR_RATE * FRAME.as_secs_f32()).ceil();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "frame_chars is a small positive constant (~7)"
            )]
            let frame_chars = frame_chars as usize;
            if pacer.pending_chars.saturating_sub(take) <= frame_chars {
                take = pacer.pending_chars;
            }
        }

        if take == 0 {
            return;
        }

        pacer.allow_small_immediate = false;
        // Keep `continuation_burst` set for as long as a continuation backlog is
        // still draining, on BOTH the dense (same-read / sub-frame gap) and the
        // adaptive (widely-spaced clump) paths. Clearing it after the first
        // partial reveal was the residual "뭉텅이": a sentence-sized Claude delta
        // had its first frame metered, but the reset then re-enabled the
        // tail-promote shortcut on the very next frame, which dumped the whole
        // remainder at once — so the delta still landed in ~one frame and the
        // cadence mirrored the network burst. Persisting the flag while
        // `pending` drains keeps every continuation delta metered across frames
        // (a steady type-in) until it is fully revealed; it naturally clears once
        // the backlog hits zero, and the next arrival re-arms it. A genuine
        // catch-up backlog (above `SMOOTH_CONTINUATION_MAX`) still drains fast via
        // the `backlog / WINDOW` ease-out and the tail-promote finish, which only
        // re-engages once it falls back under the smoothing threshold.
        let still_draining = pacer.pending_chars.saturating_sub(take) > 0;
        pacer.continuation_burst = still_draining;
        self.reveal_paced(take);
    }

    /// Split `take` characters off the front of the pending buffer (on a char
    /// boundary) and push them into the transcript, forwarding `done` only when
    /// the buffer is now empty.
    fn reveal_paced(&mut self, take: usize) {
        let Some(pacer) = self.stream_pacer.as_mut() else {
            return;
        };
        // Byte offset of the `take`-th char boundary — O(take), not O(pending).
        let split = pacer
            .pending
            .char_indices()
            .nth(take)
            .map_or(pacer.pending.len(), |(byte_idx, _)| byte_idx);
        let revealed: String = pacer.pending.drain(..split).collect();
        pacer.pending_chars = pacer.pending_chars.saturating_sub(take);

        let drained = pacer.pending_chars == 0;
        let done = pacer.done && drained;
        let id = pacer.block_id;
        if drained && done {
            self.stream_pacer = None;
        }

        self.push_paced_text(id, revealed, done);
    }

    /// Land the entire pending buffer at once (with its `done` flag) and drop the
    /// pacer. Used when a non-prose block arrives (preserve order), when the
    /// transcript is reset/cleared, or when a different block supersedes this one.
    pub(super) fn flush_stream(&mut self) {
        let Some(mut pacer) = self.stream_pacer.take() else {
            return;
        };
        if pacer.pending.is_empty() {
            return;
        }
        let text = std::mem::take(&mut pacer.pending);
        self.push_paced_text(pacer.block_id, text, pacer.done);
    }

    /// Drop any paced tail without revealing it — for transcript resets
    /// (`/clear`, `/resume`, `/new`) where the block id no longer exists, so a
    /// later drip cannot resurrect stale text onto a fresh surface.
    pub(super) fn discard_stream(&mut self) {
        self.stream_pacer = None;
    }

    /// Mark the open paced block `done` so its tail finishes and seals on
    /// subsequent idle drips, without forcing a one-frame jump. Called from
    /// `end_turn`: the provider's final delta has usually already set `done`,
    /// but a turn that ended without a terminal delta (e.g. an aborted stream)
    /// still settles cleanly instead of leaving a caret blinking forever.
    pub(super) fn finish_stream(&mut self) {
        if let Some(pacer) = self.stream_pacer.as_mut() {
            pacer.done = true;
            if pacer.pending_chars == 0 {
                // Buffer already drained but the block was last pushed with
                // `done = false` (its caret still blinking) — emit the terminal
                // `done` so it seals instead of leaving the pacer holding an
                // unsealed block.
                self.seal_paced_block();
            }
        }
    }

    /// Drop the (now-empty) pacer and push a terminal `done` so the open block
    /// flips off its streaming caret. Only meaningful once at least one slice
    /// has been revealed; an unrevealed empty block seals as a suppressed
    /// height-0 phantom, which the transcript already hides.
    fn seal_paced_block(&mut self) {
        let Some(pacer) = self.stream_pacer.take() else {
            return;
        };
        self.push_paced_text(pacer.block_id, String::new(), true);
    }

    /// Push a revealed slice into the transcript and keep the tail pinned when
    /// auto-follow is on. Mirrors `push_transcript_block_now`'s tail without its
    /// steering-echo handling (which only applies to `System` blocks).
    fn push_paced_text(
        &mut self,
        id: runtime::message_stream::BlockId,
        text: String,
        done: bool,
    ) {
        self.transcript
            .push(runtime::message_stream::RenderBlock::TextDelta { id, text, done });
        if self.transcript_view.follow_output {
            self.transcript.scroll_to_bottom();
        }
    }
}
