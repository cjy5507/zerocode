# Stream pacing

This design record holds the streaming-reveal rationale moved from `crates/zo-cli/src/tui/app/stream_pace.rs`. The module keeps only comments that express an implementation constraint, bound, invariant, or ordering requirement not evident from the code itself.

## Decision

Use an adaptive, backlog-proportional coalescer for live streamed prose. Keep replay/resume unpaced, preserve block-arrival order by flushing before a non-prose or replacement block, and make terminal tails settle quickly.

## Context and rejected alternatives

Lightweight streaming coalescer — smooths provider-sized text bursts without
turning small deltas into a slow typewriter.
(Claude/DeepSeek native streams) reads as a smooth type-in instead of whole
chunks slamming onto the screen one "뭉탱이" at a time.

This is intentionally **not** the old smooth-reveal controller (deleted in
`9e908c2f`). That one felt slow for two concrete reasons, and this pacer
avoids both by construction:

1. **No rate ceiling.** The old controller capped sustained reveal at
   `MAX_REVEAL_RATE = 1200 c/s`, so a fast stream was *throttled* below its
   generation speed. Here the instantaneous rate is `backlog / WINDOW`, so a
   larger backlog drains proportionally faster (an ease-out curve): big
   content comes out quicker, never slower.
2. **No fixed tail reserve.** The old controller always retained ~0.18 s of
   buffer and kept typing a tail after the model had already finished (up to
   ~0.95 s of trailing latency). Here there is no reserve: a provider-closed
   text block lands immediately when small, while a large final provider
   burst is smoothed over the shorter finish window and then sealed. An
   aborted/open tail that is only marked done at turn end uses the same short
   finish window and flushes the last sub-frame remainder.

The net effect is faster than a slow typewriter while still avoiding a huge
open-stream burst slamming onto the screen. Pacing only applies to live,
still-open streamed prose during an active turn; replay/resume (no active
turn) bypasses it and lands whole, and any non-prose block flushes the paced
tail first so true arrival order is always preserved without an ordering
queue.

## Rate, window, and threshold tuning

Target time to drain the *current* backlog while the stream is still open.
The per-frame reveal is `backlog * (dt / WINDOW)`, i.e. ~37 % of the backlog
per 33 ms frame — a burst is spread across ~2-3 frames, enough to read as a
type-in without ever falling behind generation. Smaller than a human
reaction time, so it never adds perceptible latency.

Upper bound on the adaptive drain window. When the provider delivers in
widely-spaced clumps (Claude's ~480ms inter-token gaps), a small backlog is
spread across the estimated gap until the next delta instead of draining in
one `WINDOW` and leaving the rest of the gap blank — that is what turned a
genuine per-token stream into the "뭉텅이 → pause → 뭉텅이" cadence. Capped so
a genuinely long pause (the model thinking) never stalls the visible reveal
for more than this; beyond it the backlog drains at the normal `WINDOW`/floor
rate. Sits just under a human's ~0.5s "is it stuck?" threshold.


Set to ~200 c/s (≈ 6-7 chars per 33 ms frame), not the old 360. At 360 a
single frame drained ~12 chars, so any small mid-cadence continuation backlog
(a ~10-char Claude delta arriving every ~50 ms) was dumped *whole* on its
first frame and the reveal then sat idle until the next delta — a periodic
catch-up-then-idle micro-hitch (`[10,10,0,10,10,0,…]`) the user still felt as
not-smooth. A 200 c/s floor meters that same backlog across the gap as a
steady glide (`[10,6,4,6,8,…]`, idle frames eliminated) while staying well
above a readable trickle so the tail never looks frozen. Large/fast backlogs
are unaffected: their rate is the much higher `backlog / WINDOW` term, so this
floor only ever binds the small-backlog steady-state where smoothness matters.

Gentler drain floor used only while smoothing a small continuation backlog
across an estimated inter-arrival gap (the adaptive path). Low enough that a
~20-char Claude clump spreads across most of a ~480ms gap instead of the
~100ms the normal `FLOOR_RATE` would force, but still a steady visible
trickle (~60 c/s ≈ 2 chars per frame) so it never looks frozen and always
terminates.

Land-whole threshold: a delta at or below this many chars is revealed whole
on arrival (zero added latency); anything larger is gently typed in across a
frame or two at [`FLOOR_RATE`]. Set to a *phrase* (~24 chars), not a full
sentence, so mid-sized provider chunks stop slamming in as one "뭉탱이" — they
flow in instead. Kept high enough that genuinely tiny word-sized deltas still
land instantly, so this reads as smooth web-chat streaming, never a slow
typewriter. Lowered 64 → 24 for gentler per-delta smoothing; the drain-finish
promotion uses the separate [`TAIL_PROMOTE_CHARS`] so large backlogs still
settle just as fast.

Largest same-block continuation backlog that is treated as a *single delta*
to be smoothly typed in rather than a catch-up burst to drain fast. A live
Claude stream delivers token clumps and sentence-sized deltas (observed
16-70 chars per provider `content_block_delta`, occasionally larger when a
network read batches several), and at the 16-33 ms drip cadence the per-frame
backlog rarely exceeds this. Anything above it is the model running ahead of
the reveal (a real catch-up burst), which still drains fast against `WINDOW`
with the tail-promote finish. Set well above a sentence so a normal delta is
always metered out across frames instead of being tail-promoted whole — the
"뭉텅이 → pause → 뭉텅이" cadence — while staying below the multi-hundred-char
bursts that should settle quickly.

## Pacer state model

Nothing is buffered; send only the terminal `done=true` seal.

A small final delta is already smooth enough to land on arrival.

A large final burst should use [`FINISH_WINDOW`] instead of one-frame flush.

Per-block pacing buffer: characters that have arrived from the provider but
have not yet been revealed into the transcript. Exactly one is live at a
time (the open streaming prose block); a non-prose block or a block-id change
flushes it.

Transcript block these characters belong to. Reveals are pushed as
`TextDelta` with this id so the transcript merges them onto the block.

Received-but-not-yet-revealed characters.

The provider closed this block; once `pending` drains, forward `done`.

`true` only for the initial arrival of a newly opened block. This keeps
genuinely tiny openings zero-latency while preventing later same-block
continuation bursts from using the small-buffer land-whole shortcut.

A same-block continuation arrived since the last reveal. Small Claude
token clumps should be metered by frame cadence, not promoted whole by
the tail-finish shortcut on the next tick.

## Arrival handling and cadence tracking

`true` while paced characters are still waiting to be revealed — the
idle/turn loops keep ticking until this drains so the tail types out.

Buffer a streamed text delta for paced reveal, then drip the first slice
on this same frame (so the block opens immediately). A delta for a
different block id flushes the current tail first to preserve order.

Time-injected core of [`Self::buffer_paced`]. `now` is the clock the drip
math measures against — real in production, controlled in tests so a
burst+gap cadence is deterministic without a real sleep.

Open the freshly started block on its arrival frame for a
low-latency first paint (the `STARTER_CHARS` carry shows a phrase
at once).

A *continuation* delta is intentionally NOT dripped here (see the
sibling branch). Dripping on every arrival reset `last_drip` to
~now on each token, so the wall-clock drip never saw a real elapsed
span and the small-delta land-whole / tail-promote shortcuts
revealed each token the instant it arrived — making the on-screen
cadence mirror the provider's bursty network delivery. That is why
a genuine per-token stream (Claude) read as clump-then-pause
stutter while a coarse-chunk provider looked smooth.

Continuation delta: do NOT drip on arrival. Let it accumulate and
be metered by the frame-driven drip (`advance_tick` and the gated
block-arrival `drip_stream`, both on the 30-60 fps grid, kept alive
by `stream_pending`), which decouples reveal cadence from arrival
cadence so a burst spreads across the following frames instead of
slamming in whole. A non-prose block flushes, and turn end
finishes, the accumulated tail, so nothing is lost or reordered.

## Reveal algorithm and tail behavior

Reveal the characters the elapsed wall-clock time has earned since the
last drip. Driven by [`App::advance_tick`] (the 30 fps grid) and by each
block arrival, so the cadence tracks real time on whichever fires first.

`pub` (like [`App::advance_tick`]) so the mid-turn loop in the `zo`
bin (`session::turn_controller::drive_turn`) can drive the drip on each
throttled block-arrival repaint, matching the idle loop; a lib-internal
`pub(super)` would not reach the bin crate.

Time-injected drip for tests: drive the reveal with an explicit `now` and
an optional forced elapsed span, so a burst cadence is fully
deterministic without a real sleep.

Core drip. `forced` overrides the measured `now - last_drip` span (tests);
`None` measures the real elapsed time (production). Reveals on a UTF-8
char boundary so a multibyte glyph (CJK / emoji) is never split.

Nothing buffered: seal if the provider is done, otherwise keep the
(empty) pacer so the next delta merges onto the same block and pacing
stays smooth across an inter-burst gap.

Keep the drip clock fresh while idling between deltas.
Leaving `last_drip` at the last actual reveal made the first
drip after a clumpy provider's ~470ms gap measure dt ≈ the
whole gap, earn ~a delta's worth of characters at once, and
dump the freshly arrived backlog in one frame — the exact
clump→pause stutter the adaptive spread exists to hide. With
the clock pinned to the tick grid, that first drip sees
dt ≈ one frame and meters the new delta smoothly.

Small openings and small terminal deltas are already perceptually
smooth, so keep them zero-latency. Do NOT apply this to same-block
continuation backlog: Claude often delivers several tiny tokens in
one network read, and revealing that <=24 char backlog whole on the
next frame preserves the exact clump→pause stutter this pacer exists
to hide.

A same-block continuation backlog up to a delta's worth is smoothed
(typed in across frames) rather than tail-promoted whole. Gated by the
wide `SMOOTH_CONTINUATION_MAX`, not the smaller `TAIL_PROMOTE_CHARS`:
Claude routinely delivers sentence-sized deltas (~70 chars), and at
the old 64-char gate those fell straight through to the tail-promote
and slammed in whole — the exact "뭉텅이" the user still felt. Only a
genuine catch-up backlog (model running well ahead of the reveal)
exceeds this and drains fast against `WINDOW`.

The *adaptive* path only engages when the provider is actually
delivering in widely-spaced clumps (estimated gap above one WINDOW):
then a small clump is spread across the gap until the next delta. A
burst delivered in one network read (gap ≈ 0, interval stays at WINDOW)
keeps the original fast settle so it never drags.

Drain window: normally a tight 60ms so bursts type in fast; a clumpy
provider's small backlog is spread across the estimated inter-arrival
gap (bounded by MAX_ADAPTIVE_WINDOW). `done` always uses the short
finish window so the end never drags.

Floor on the drain rate. Normally `FLOOR_RATE` keeps a fast stream
visibly typing, but for an adaptive (clumpy-provider) reveal that floor
would drain the small backlog in one frame and defeat the wider window,
so a smoothed continuation uses a gentler floor that still guarantees a
readable trickle (~1 char per couple of frames) and termination.

Carry the sub-character remainder only when the reveal was rate-bound
(not when we drained the buffer or hit the per-frame cap).

If this real elapsed frame would leave only a phrase-sized remainder
behind, take it now so a large backlog settles promptly instead of
trailing one more frame. Uses the larger `TAIL_PROMOTE_CHARS` (not the
smaller per-delta land-whole threshold) so finish speed is unaffected
by how aggressively small deltas are smoothed. Not applied to the
arrival-frame starter (`dt == 0`), or a barely-large first burst would
dump all at once. Suppressed while smoothing a small continuation
backlog (same-instant burst's first frame, or the whole adaptive
spread) so the remainder is metered out rather than dumped whole —
promoting it is exactly the 뭉텅이 this pacer hides.

Keep `continuation_burst` set for as long as a continuation backlog is
still draining, on BOTH the dense (same-read / sub-frame gap) and the
adaptive (widely-spaced clump) paths. Clearing it after the first
partial reveal was the residual "뭉텅이": a sentence-sized Claude delta
had its first frame metered, but the reset then re-enabled the
tail-promote shortcut on the very next frame, which dumped the whole
remainder at once — so the delta still landed in ~one frame and the
cadence mirrored the network burst. Persisting the flag while
`pending` drains keeps every continuation delta metered across frames
(a steady type-in) until it is fully revealed; it naturally clears once
the backlog hits zero, and the next arrival re-arms it. A genuine
catch-up backlog (above `SMOOTH_CONTINUATION_MAX`) still drains fast via
the `backlog / WINDOW` ease-out and the tail-promote finish, which only
re-engages once it falls back under the smoothing threshold.

## Lifecycle, ordering, and complexity

Byte offset of the `take`-th char boundary. This scan is O(take), but
the `drain` below is not: it shifts the remaining bytes forward, so a
reveal costs O(pending) overall. Acceptable because a live backlog is
tens to hundreds of bytes, and the one path that could grow it to
megabytes (replay/resume) bypasses the pacer entirely — see the module
header. A front cursor instead of `drain` is the fix if that changes.

Mark the open paced block `done` so its tail finishes and seals on
subsequent idle drips, without forcing a one-frame jump. Called from
`end_turn`: the provider's final delta has usually already set `done`,
but a turn that ended without a terminal delta (e.g. an aborted stream)
still settles cleanly instead of leaving a caret blinking forever.

Push a revealed slice into the transcript and keep the tail pinned when
auto-follow is on. Mirrors `push_transcript_block_now`'s tail without its
steering-echo handling (which only applies to `System` blocks).
