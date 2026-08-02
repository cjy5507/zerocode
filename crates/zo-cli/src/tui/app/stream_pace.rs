//! See the repository-root `STREAM_PACING.md` for the streaming-pacing design rationale.
use std::time::{Duration, Instant};

use crate::tui::render_schedule::ANIMATION_TICK_INTERVAL;

use super::App;

/// One render frame at the shared 30 fps tick grid — used only to size the
/// "near-done, just finish it" threshold below.
const FRAME: Duration = ANIMATION_TICK_INTERVAL;

const WINDOW: Duration = Duration::from_millis(60);

const MAX_ADAPTIVE_WINDOW: Duration = Duration::from_millis(450);

/// After a stream is marked `done`, drain any large remaining tail against this
/// shorter window so final provider bursts and aborted/terminal edge cases
/// settle promptly instead of trailing. Small final arrivals still land whole
/// immediately via [`DoneArrivalPolicy::RevealImmediately`].
const FINISH_WINDOW: Duration = Duration::from_millis(30);

/// Floor on the drain rate (chars/sec). Guarantees the tail always terminates
/// (the `backlog / WINDOW` term alone decays geometrically and would only
/// asymptote toward empty) and keeps a very slow trickle still visibly typing.
const FLOOR_RATE: f32 = 200.0;

const ADAPTIVE_FLOOR_RATE: f32 = 60.0;

/// Hard cap on a single drip. Only relevant for a pathological one-shot dump
/// (a giant pasted/replayed block routed through the pacer): it still reads as
/// a fast type-in rather than a single instant paint. Far above any real
/// per-frame backlog, so a normal fast stream is never throttled by it.
const MAX_CHUNK: usize = 4096;

/// Reclaim a consumed prefix only after it is large enough to matter and at
/// least half the allocation. This keeps compaction amortized instead of moving
/// the remaining UTF-8 tail on every 30 fps reveal frame.
const COMPACT_PREFIX_BYTES: usize = 16 * 1024;

const IMMEDIATE_CHARS: usize = 24;

/// Drain-finish promotion threshold: when a drain frame would leave only this
/// many chars behind, take them all on that frame instead of trailing a small
/// remainder over yet another frame. Deliberately a full phrase-and-a-bit and
/// kept independent of the smaller [`IMMEDIATE_CHARS`] land-whole threshold, so
/// dialing per-delta smoothing down never slows how promptly a large provider
/// burst settles.
const TAIL_PROMOTE_CHARS: usize = 64;

const SMOOTH_CONTINUATION_MAX: usize = 128;

/// Characters revealed on the very first drip of a larger freshly opened block.
/// Phrase-sized, not glyph-sized: enough lands immediately for a web-chat feel
/// while preventing a large burst from slamming in at once. Matches
/// [`IMMEDIATE_CHARS`] so a block opening just above the land-whole threshold
/// still shows a full phrase on its first frame.
const STARTER_CHARS: f32 = 24.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DoneArrivalPolicy {
    SealEmpty,
    RevealImmediately,
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

#[derive(Debug)]
pub(super) struct StreamPacer {
    block_id: runtime::message_stream::BlockId,
    pending: String,
    /// First unconsumed byte in `pending`. Reveals advance this cursor instead
    /// of draining the String's front and memmoving the full tail every frame.
    pending_start: usize,
    /// Remaining character count, maintained incrementally so each drip is O(1)
    /// in the backlog size instead of rescanning the whole buffer per frame.
    pending_chars: usize,
    done: bool,
    /// Wall-clock instant of the last drip, so the next reveals an amount
    /// proportional to the time actually elapsed — a real cadence independent of
    /// how often the drip is driven.
    last_drip: Instant,
    /// Fractional character carried between drips (the earned count is rarely a
    /// whole number); without it, rounding would bias the long-run rate.
    carry: f32,
    allow_small_immediate: bool,
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
    #[must_use]
    pub fn stream_pending(&self) -> bool {
        self.stream_pacer.is_some()
    }

    pub(super) fn buffer_paced(
        &mut self,
        id: runtime::message_stream::BlockId,
        text: String,
        done: bool,
    ) {
        self.buffer_paced_at(Instant::now(), id, text, done);
    }

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
            self.drip_stream_elapsed(now, None);
        } else {
            // Continuations wait for tick-driven dripping so reveal cadence stays decoupled from arrival cadence.
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
            pending_start: 0,
            done,
            last_drip: now,
            carry: STARTER_CHARS,
            allow_small_immediate: true,
            continuation_burst: false,
            last_arrival: now,
            arrival_interval: WINDOW,
        });
    }

    pub fn drip_stream(&mut self) {
        self.drip_stream_elapsed(Instant::now(), None);
    }

    #[cfg(test)]
    pub(super) fn drip_stream_at(&mut self, now: Instant, forced: Option<Duration>) {
        self.drip_stream_elapsed(now, forced);
    }

    fn drip_stream_elapsed(&mut self, now: Instant, forced: Option<Duration>) {
        let Some(snapshot) = self.stream_pacer.as_ref() else {
            return;
        };
        let pending_chars = snapshot.pending_chars;
        let done = snapshot.done;

        if pending_chars == 0 {
            if done {
                self.seal_paced_block();
            } else if let Some(pacer) = self.stream_pacer.as_mut() {
                pacer.last_drip = now;
            }
            return;
        }

        let Some(pacer) = self.stream_pacer.as_mut() else {
            return;
        };

        let dt = forced.unwrap_or_else(|| now.saturating_duration_since(pacer.last_drip));
        pacer.last_drip = now;

        if pending_chars <= IMMEDIATE_CHARS && (pacer.allow_small_immediate || done) {
            pacer.carry = 0.0;
            pacer.allow_small_immediate = false;
            pacer.continuation_burst = false;
            self.reveal_paced(pending_chars);
            return;
        }

        let dt_secs = dt.as_secs_f32();
        let smooth_small_continuation = pacer.continuation_burst
            && !pacer.done
            && pending_chars <= SMOOTH_CONTINUATION_MAX;
        let adaptive = smooth_small_continuation && pacer.arrival_interval > WINDOW;

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
        let still_draining = pacer.pending_chars.saturating_sub(take) > 0;
        pacer.continuation_burst = still_draining;
        self.reveal_paced(take);
    }

    /// Split `take` characters off the unconsumed pending suffix (on a UTF-8
    /// boundary) and push them into the transcript. A byte cursor advances in
    /// O(revealed chars); consumed storage is compacted only amortized.
    fn reveal_paced(&mut self, take: usize) {
        let Some(pacer) = self.stream_pacer.as_mut() else {
            return;
        };
        let remaining = &pacer.pending[pacer.pending_start..];
        let relative_end = remaining
            .char_indices()
            .nth(take)
            .map_or(remaining.len(), |(byte_idx, _)| byte_idx);
        let end = pacer.pending_start.saturating_add(relative_end);
        let revealed = pacer.pending[pacer.pending_start..end].to_string();
        pacer.pending_start = end;
        pacer.pending_chars = pacer.pending_chars.saturating_sub(take);

        let drained = pacer.pending_chars == 0;
        let done = pacer.done && drained;
        let id = pacer.block_id;
        if drained {
            pacer.pending.clear();
            pacer.pending_start = 0;
        } else if pacer.pending_start >= COMPACT_PREFIX_BYTES
            && pacer.pending_start >= pacer.pending.len() / 2
        {
            pacer.pending.drain(..pacer.pending_start);
            pacer.pending_start = 0;
        }
        if drained && done {
            self.stream_pacer = None;
        }

        self.push_paced_text(id, revealed, done);
    }

    /// Land the entire unconsumed suffix at once (with its `done` flag) and drop
    /// the pacer. Used when a non-prose block arrives (preserve order), when the
    /// transcript is reset/cleared, or when a different block supersedes this one.
    pub(super) fn flush_stream(&mut self) {
        let Some(mut pacer) = self.stream_pacer.take() else {
            return;
        };
        if pacer.pending_chars == 0 {
            return;
        }
        let text = if pacer.pending_start == 0 {
            std::mem::take(&mut pacer.pending)
        } else {
            pacer.pending.split_off(pacer.pending_start)
        };
        self.push_paced_text(pacer.block_id, text, pacer.done);
    }

    /// Drop any paced tail without revealing it — for transcript resets
    /// (`/clear`, `/resume`, `/new`) where the block id no longer exists, so a
    /// later drip cannot resurrect stale text onto a fresh surface.
    pub(super) fn discard_stream(&mut self) {
        self.stream_pacer = None;
    }

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
