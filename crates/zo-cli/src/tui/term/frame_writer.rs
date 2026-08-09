//! The seam that keeps a slow terminal from freezing zo.
//!
//! Frames used to go straight from the async main task to `stdout`. A terminal
//! that stops draining — a busy or suspended emulator, flow control, a
//! multiplexer that stopped reading — makes that `write` block, and with it the
//! whole event loop: input, the spinner, even Ctrl+C. The captured stacks in
//! `~/.zo/logs/zo-freeze-*.sample` say this is what every freeze since
//! 2026-08-02 actually was, with the main thread parked in `write` for 17-36s.
//!
//! [`FrameWriter`] moves the blocking write onto its own thread. The render
//! loop hands over a finished frame and returns immediately; the thread absorbs
//! the stall. What matters for correctness is what happens when the terminal
//! stays blocked long enough for frames to pile up:
//!
//! - Frames are ratatui **cell diffs**, so a dropped frame poisons every later
//!   one. Whenever this drops anything it therefore sets a process-wide
//!   "repaint everything" flag that [`take_needs_full_redraw`] hands to the next
//!   draw, which clears the terminal and paints a complete frame. The backlog is
//!   discarded wholesale rather than trimmed, because that full repaint
//!   supersedes all of it.
//! - Only a cell diff may be dropped. Teardown escapes, scrollback insertions,
//!   and mode switches ride this same writer and no repaint reproduces them, so
//!   discarding is opt-in per write: see [`with_discardable_frames`].
//! - A frame may be bracketed in synchronized-output escapes (CSI ?2026), and
//!   the `Begin` can already be on its way out when the rest is dropped. That
//!   would strand the terminal in synchronized mode — the exact freeze that got
//!   an earlier synchronized-output attempt reverted — so a compensating `End`
//!   is queued in place of the discarded backlog.
//!
//! Anything that needs bytes to have actually reached the terminal — a viewport
//! rebuild over the same stdout, teardown, suspending for `$EDITOR` — must call
//! [`FrameWriter::drain_blocking`] instead of relying on `flush`, which now only
//! means "this frame is complete".

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Frames allowed to sit unwritten before the backlog is discarded. A healthy
/// terminal never has more than one in flight; this only fills when the
/// terminal has stopped consuming, and a few frames of slack absorb a brief
/// hiccup without a full repaint.
pub const DEFAULT_QUEUE_FRAMES: usize = 4;

/// Handoffs a bracketed frame costs: `Begin`, the cell diff, `End` — each one
/// its own `execute!`, so each is its own flush. The queue bound is stated in
/// frames, so it has to be scaled by this or "four frames of slack" is really
/// one and a third, and an ordinary hiccup turns into a full-screen repaint.
const HANDOFFS_PER_BRACKETED_FRAME: usize = 3;

/// `CSI ?2026l` — leave synchronized output. Queued in place of a discarded
/// backlog so a `Begin` that already went out cannot strand the terminal.
const END_SYNCHRONIZED_UPDATE: &[u8] = b"\x1b[?2026l";

/// Set whenever a frame is discarded, cleared by the next draw that honors it.
/// Process-global because a process has one terminal: threading it through
/// every generic `draw_frame` caller would buy nothing.
static NEEDS_FULL_REDRAW: AtomicBool = AtomicBool::new(false);

/// Whether the next draw must repaint everything because a frame was dropped.
/// Consumes the flag.
pub fn take_needs_full_redraw() -> bool {
    NEEDS_FULL_REDRAW.swap(false, Ordering::AcqRel)
}

/// Whether a repaint is owed, without consuming the flag. The discard happens
/// while the app may be otherwise idle, so the event loop has to count this as
/// work — nothing else would schedule the frame that repairs the screen.
pub fn full_redraw_pending() -> bool {
    NEEDS_FULL_REDRAW.load(Ordering::Acquire)
}

thread_local! {
    /// Whether writes from this thread are repaintable frames.
    static FRAMES_DISCARDABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark what `paint` writes as a frame the writer may drop when the terminal
/// stops draining.
///
/// The default is the opposite, and has to be: teardown escapes, a scrollback
/// insertion, and a mode switch all travel this same writer, and no repaint can
/// reproduce any of them. Dropping a `LeaveAlternateScreen` hands the user back
/// a shell stuck on the alternate screen; dropping an `insert_before` loses that
/// much transcript for good. Only a ratatui cell diff is safe to discard,
/// because [`take_needs_full_redraw`] makes the next frame a complete repaint.
pub fn with_discardable_frames<R>(paint: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FRAMES_DISCARDABLE.with(|discardable| discardable.set(self.0));
        }
    }

    let _restore = Restore(FRAMES_DISCARDABLE.with(|discardable| discardable.replace(true)));
    paint()
}

fn frames_are_discardable() -> bool {
    FRAMES_DISCARDABLE.with(std::cell::Cell::get)
}

/// Throw away everything queued and stop accepting more.
///
/// For the panic path only: the hook has already written the restore escapes
/// straight to stdout, so any frame still queued would land *after* them — alt
/// screen diffs dumped into the recovered shell, and a queued `Begin` stranding
/// the terminal in synchronized mode behind the hook's defensive `End`.
pub fn abandon_queued_frames() {
    let Some(shared) = ACTIVE_QUEUE
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(Arc::clone))
    else {
        return;
    };
    shared.abandoned.store(true, Ordering::Release);
    if let Ok(mut queue) = shared.queue.lock() {
        queue.blobs.clear();
    }
    shared.wake.notify_all();
}

/// Force a full repaint on the next draw, e.g. after a mode switch rebuilds the
/// viewport under the same writer.
pub fn request_full_redraw() {
    NEEDS_FULL_REDRAW.store(true, Ordering::Release);
}

/// The live writer's queue, so callers holding only a ratatui `Terminal` can
/// still wait for the terminal to actually have the bytes. Reaching the writer
/// through the backend would mean depending on ratatui's unstable
/// `backend-writer` accessors; the terminal is process-global anyway.
static ACTIVE_QUEUE: Mutex<Option<Arc<Shared>>> = Mutex::new(None);

/// Wait for the live writer's queue to reach the terminal, or `timeout` to
/// elapse; returns whether it drained. Callers must flush the backend first so
/// the in-progress frame is queued. With no live writer (headless, tests) there
/// is nothing outstanding, so this succeeds.
pub fn drain_active(timeout: Duration) -> bool {
    let Some(shared) = ACTIVE_QUEUE
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(Arc::clone))
    else {
        return true;
    };
    wait_until_drained(&shared, timeout)
}

/// Block until nothing is outstanding on `shared`, or `timeout` elapses.
fn wait_until_drained(shared: &Arc<Shared>, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let Ok(mut queue) = shared.queue.lock() else {
        return false;
    };
    while queue.outstanding() {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return false;
        };
        let (next, wait) = shared
            .wake
            .wait_timeout(queue, remaining)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue = next;
        if wait.timed_out() && queue.outstanding() {
            return false;
        }
    }
    true
}

#[derive(Default)]
struct Queue {
    blobs: VecDeque<Blob>,
    /// A blob the thread has taken but not finished writing. An empty `blobs`
    /// alone does not mean the terminal has the bytes — it usually means the
    /// thread is blocked inside that very write — and a `drain_blocking` that
    /// confused the two would let a second writer interleave with a frame still
    /// going out.
    in_flight: bool,
    /// Set when the writer half is dropped, so the thread can finish and exit.
    closed: bool,
    /// Frames discarded because the terminal was not keeping up. Surfaced for
    /// diagnosis: a nonzero count is a terminal that stalled, not a zo defect.
    dropped_frames: u64,
}

/// One handoff's bytes, and whether a full repaint would bring them back.
struct Blob {
    bytes: Vec<u8>,
    discardable: bool,
}

impl Queue {
    /// Whether anything is still on its way to the terminal.
    fn outstanding(&self) -> bool {
        !self.blobs.is_empty() || self.in_flight
    }

    /// Drop the repaintable backlog, keeping what no repaint reproduces, and
    /// report how many frames went. Retaining rather than clearing is the whole
    /// guarantee: a teardown escape queued behind four frames must not be swept
    /// away by the *next* overflow.
    fn discard_repaintable(&mut self) -> u64 {
        let before = self.blobs.len();
        self.blobs.retain(|blob| !blob.discardable);
        (before - self.blobs.len()) as u64
    }
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled on every enqueue, on close, and after each blob is written, so
    /// both the writer thread and `drain_blocking` can wait without polling.
    wake: Condvar,
    /// Set by [`abandon_queued_frames`]: the terminal has been restored behind
    /// this writer's back, so nothing more may be written to it.
    abandoned: AtomicBool,
}

/// A terminal writer that never blocks the caller.
///
/// Implements [`Write`] so it drops into `CrosstermBackend` where a
/// `BufWriter<Stdout>` used to sit: writes accumulate in memory and `flush`
/// hands the finished frame to the writer thread. Buffering here rather than in
/// a `BufWriter` also guarantees one frame is one blob no matter how large,
/// instead of splitting at a fixed capacity.
pub struct FrameWriter {
    pending: Vec<u8>,
    shared: Arc<Shared>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// Queued handoffs allowed before the repaintable backlog goes, already
    /// scaled from frames by [`HANDOFFS_PER_BRACKETED_FRAME`].
    max_queued: usize,
    /// Whether to queue a compensating synchronized-output `End` when a backlog
    /// is discarded. Only meaningful when the app brackets frames in CSI ?2026;
    /// emitting it otherwise would send an escape the terminal never opened.
    compensate_synchronized: bool,
}

impl FrameWriter {
    /// Start a writer thread over `sink` (the real terminal in production, a
    /// test double otherwise).
    pub fn spawn<W>(sink: W, max_queued: usize, compensate_synchronized: bool) -> Self
    where
        W: Write + Send + 'static,
    {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
            abandoned: AtomicBool::new(false),
        });
        let worker = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("zo-frame-writer".to_string())
            .spawn(move || drain_forever(&worker, sink))
            .ok();
        if let Ok(mut active) = ACTIVE_QUEUE.lock() {
            // Newest wins: a viewport rebuild briefly holds two writers, and the
            // one callers mean is the one frames are going to now.
            *active = Some(Arc::clone(&shared));
        }
        Self {
            pending: Vec::with_capacity(64 * 1024),
            shared,
            handle,
            max_queued: max_queued.max(1)
                * if compensate_synchronized {
                    HANDOFFS_PER_BRACKETED_FRAME
                } else {
                    1
                },
            compensate_synchronized,
        }
    }

    /// Production writer: stdout, the default backlog, synchronized-output
    /// compensation decided by the terminal profile that gates the brackets.
    #[must_use]
    pub fn stdout() -> Self {
        Self::spawn(
            io::stdout(),
            DEFAULT_QUEUE_FRAMES,
            super::TermProfile::current().synchronized_output,
        )
    }

    /// Frames discarded so far because the terminal stopped consuming.
    #[must_use]
    pub fn dropped_frames(&self) -> u64 {
        self.shared
            .queue
            .lock()
            .map_or(0, |queue| queue.dropped_frames)
    }

    /// Block until every queued frame has reached the sink, or `timeout`
    /// elapses. Returns whether the queue drained. Callers that are about to
    /// write to the same terminal by another route — a viewport rebuild,
    /// teardown, suspending for a child program — must call this, because
    /// `flush` only marks a frame complete.
    pub fn drain_blocking(&mut self, timeout: Duration) -> bool {
        let _ = self.flush();
        wait_until_drained(&self.shared, timeout)
    }

    /// Hand the accumulated frame to the writer thread, discarding the backlog
    /// if the terminal has fallen too far behind.
    fn enqueue_frame(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let blob = std::mem::take(&mut self.pending);
        self.pending = Vec::with_capacity(64 * 1024);
        if self.shared.abandoned.load(Ordering::Acquire) {
            return;
        }
        let discardable = frames_are_discardable();
        let Ok(mut queue) = self.shared.queue.lock() else {
            return;
        };
        if queue.blobs.len() >= self.max_queued {
            // The backlog's frames are diffs against a screen state the coming
            // full repaint will overwrite, so keeping any of them only delays
            // the recovery. This blob joins them only if the render path
            // vouched for it — otherwise it is a write that must still land.
            let stale = queue.discard_repaintable();
            queue.dropped_frames = queue
                .dropped_frames
                .saturating_add(stale + u64::from(discardable));
            if self.compensate_synchronized {
                queue.blobs.push_back(Blob {
                    bytes: END_SYNCHRONIZED_UPDATE.to_vec(),
                    discardable: false,
                });
            }
            NEEDS_FULL_REDRAW.store(true, Ordering::Release);
            if !discardable {
                queue.blobs.push_back(Blob { bytes: blob, discardable });
            }
        } else {
            queue.blobs.push_back(Blob { bytes: blob, discardable });
        }
        drop(queue);
        self.shared.wake.notify_all();
    }
}

impl Write for FrameWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        Ok(buf.len())
    }

    /// End of frame: hand it over. This never blocks and never reports the
    /// terminal's errors — the write happens later, on another thread.
    fn flush(&mut self) -> io::Result<()> {
        self.enqueue_frame();
        Ok(())
    }
}

impl Drop for FrameWriter {
    fn drop(&mut self) {
        self.enqueue_frame();
        // Give the terminal a bounded chance to take the last frames — teardown
        // escapes ride this same queue — then stop waiting, so a wedged
        // terminal cannot keep zo from exiting.
        let drained = self.drain_blocking(Duration::from_millis(500));
        if let Ok(mut queue) = self.shared.queue.lock() {
            queue.closed = true;
        }
        self.shared.wake.notify_all();
        // Join only when the queue actually emptied. A thread parked inside a
        // write to a wedged terminal never returns, and joining it here would
        // move the hang from the render loop to exit — the same freeze one step
        // later. Leaving it detached costs one blocked thread that the process
        // teardown reclaims.
        if let Some(handle) = self.handle.take() {
            if drained {
                let _ = handle.join();
            }
        }
        if let Ok(mut active) = ACTIVE_QUEUE.lock() {
            if active
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &self.shared))
            {
                *active = None;
            }
        }
    }
}

/// Writer-thread body: write blobs as they arrive, exit once the queue is both
/// closed and empty. A write error is dropped on purpose — there is no caller
/// left to report it to, and a terminal that refuses one frame will refuse the
/// next one too.
fn drain_forever<W: Write>(shared: &Arc<Shared>, mut sink: W) {
    loop {
        let blob = {
            let Ok(mut queue) = shared.queue.lock() else {
                return;
            };
            loop {
                if let Some(blob) = queue.blobs.pop_front() {
                    // Claim it before releasing the lock: until the write
                    // returns, this blob is still owed to the terminal.
                    queue.in_flight = true;
                    break blob;
                }
                if queue.closed {
                    return;
                }
                queue = shared
                    .wake
                    .wait(queue)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        let _ = sink.write_all(&blob.bytes);
        let _ = sink.flush();
        if let Ok(mut queue) = shared.queue.lock() {
            queue.in_flight = false;
        }
        shared.wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// `NEEDS_FULL_REDRAW` is process-global (a process has one terminal), so
    /// tests that assert on it must not run concurrently — in parallel they
    /// read each other's drops and the failure looks like a writer bug.
    static REDRAW_FLAG_LOCK: Mutex<()> = Mutex::new(());

    fn serialized() -> std::sync::MutexGuard<'static, ()> {
        let guard = REDRAW_FLAG_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        take_needs_full_redraw();
        guard
    }

    /// A sink that blocks in `write_all` until released, standing in for a
    /// terminal that has stopped draining.
    struct BlockingSink {
        gate: Gate,
        writes: mpsc::Sender<Vec<u8>>,
    }

    impl Write for BlockingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let (lock, cv) = &*self.gate;
            let mut open = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*open {
                open = cv
                    .wait(open)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            let _ = self.writes.send(buf.to_vec());
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// The gate a [`BlockingSink`] waits on: flip it to let writes through.
    type Gate = Arc<(Mutex<bool>, Condvar)>;

    fn blocking_sink() -> (BlockingSink, Gate, mpsc::Receiver<Vec<u8>>) {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (tx, rx) = mpsc::channel();
        (
            BlockingSink {
                gate: Arc::clone(&gate),
                writes: tx,
            },
            gate,
            rx,
        )
    }

    fn open(gate: &Gate) {
        let (lock, cv) = &**gate;
        *lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        cv.notify_all();
    }

    /// A rendered frame: a cell diff a later full repaint can reproduce, so the
    /// writer is allowed to drop it.
    fn frame(writer: &mut FrameWriter, body: &[u8]) {
        with_discardable_frames(|| {
            writer.write_all(body).expect("buffered write");
            writer.flush().expect("frame handoff");
        });
    }

    /// A write nothing can reproduce — a teardown escape, a scrollback
    /// insertion — handed over exactly the way the app's non-render paths do.
    fn control_write(writer: &mut FrameWriter, body: &[u8]) {
        writer.write_all(body).expect("buffered write");
        writer.flush().expect("control handoff");
    }

    /// The whole point: a terminal that has stopped reading must not slow the
    /// caller down. Every frame handoff stays effectively instant even while
    /// the sink is wedged.
    #[test]
    fn a_wedged_terminal_never_blocks_the_render_loop() {
        let _serial = serialized();
        let (sink, gate, _rx) = blocking_sink();
        let mut writer = FrameWriter::spawn(sink, 4, true);

        let start = Instant::now();
        for index in 0..64u8 {
            frame(&mut writer, &[index]);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "handoffs must not wait on the terminal: {elapsed:?}"
        );
        assert!(
            writer.dropped_frames() > 0,
            "a 64-frame backlog against a 4-frame queue has to discard"
        );
        open(&gate);
        drop(writer);
    }

    /// Dropping a frame poisons every later cell diff, so the drop must be
    /// paired with a full repaint — and with an escape that cannot leave the
    /// terminal stranded in synchronized mode.
    #[test]
    fn discarding_a_backlog_asks_for_a_full_repaint_and_closes_synchronized_output() {
        let _serial = serialized();
        let (sink, gate, rx) = blocking_sink();
        let mut writer = FrameWriter::spawn(sink, 2, true);
        for index in 0..16u8 {
            frame(&mut writer, &[index]);
        }
        assert!(
            take_needs_full_redraw(),
            "a discarded backlog must force a complete repaint"
        );
        assert!(
            !take_needs_full_redraw(),
            "the flag is consumed by the draw that honors it"
        );
        open(&gate);
        drop(writer);
        let written: Vec<Vec<u8>> = rx.try_iter().collect();
        assert!(
            written.iter().any(|blob| blob == END_SYNCHRONIZED_UPDATE),
            "the discarded backlog is replaced by a synchronized-output End: {written:?}"
        );
    }

    /// The discard policy is for cell diffs. `LeaveAlternateScreen` and the
    /// inline scrollback insertion travel this same writer, and no repaint
    /// brings them back: dropping one hands the user a shell stuck on the
    /// alternate screen, or silently loses that much transcript. They must
    /// survive a backlog that is being discarded around them.
    #[test]
    fn a_write_no_repaint_can_reproduce_outlives_a_discarded_backlog() {
        let _serial = serialized();
        let (sink, gate, rx) = blocking_sink();
        let mut writer = FrameWriter::spawn(sink, 2, true);
        for index in 0..16u8 {
            frame(&mut writer, &[index]);
        }
        assert!(writer.dropped_frames() > 0, "the frames overflowed the queue");
        control_write(&mut writer, b"leave-alternate-screen");
        // And it stays safe once more frames pile up behind it.
        for index in 16..32u8 {
            frame(&mut writer, &[index]);
        }

        open(&gate);
        assert!(writer.drain_blocking(Duration::from_secs(2)), "the sink is open");
        drop(writer);

        let written: Vec<Vec<u8>> = rx.try_iter().collect();
        assert!(
            written.iter().any(|blob| blob == b"leave-alternate-screen"),
            "the teardown escape must reach the terminal: {written:?}"
        );
    }

    /// The panic hook restores the terminal by writing straight to stdout, then
    /// unwinding drops the writer. Anything still queued would land on top of
    /// that restore — alternate-screen diffs in the recovered shell, or a
    /// `Begin` re-stranding it in synchronized mode behind the defensive `End`.
    #[test]
    fn an_abandoned_queue_never_paints_over_a_terminal_someone_else_restored() {
        let _serial = serialized();
        let (sink, gate, rx) = blocking_sink();
        let mut writer = FrameWriter::spawn(sink, 4, true);
        frame(&mut writer, b"half-painted frame");

        abandon_queued_frames();
        control_write(&mut writer, b"too late");
        open(&gate);
        drop(writer);

        let written: Vec<Vec<u8>> = rx.try_iter().collect();
        assert!(
            written.is_empty(),
            "nothing may reach the terminal after the queue is abandoned: {written:?}"
        );
    }

    /// A terminal that keeps up must see every byte, in order, with no repaint
    /// requested — the fast path has to stay byte-identical to the old direct
    /// writer.
    #[test]
    fn a_healthy_terminal_receives_every_frame_in_order() {
        let _serial = serialized();
        let (sink, gate, rx) = blocking_sink();
        open(&gate); // never blocks
        let mut writer = FrameWriter::spawn(sink, 4, true);
        for index in 0..8u8 {
            frame(&mut writer, &[index]);
            assert!(
                writer.drain_blocking(Duration::from_secs(2)),
                "an unblocked sink drains promptly"
            );
        }
        assert_eq!(writer.dropped_frames(), 0);
        assert!(!take_needs_full_redraw(), "nothing was dropped");
        drop(writer);
        let written: Vec<Vec<u8>> = rx.try_iter().collect();
        assert_eq!(
            written,
            (0..8u8).map(|index| vec![index]).collect::<Vec<_>>(),
            "every frame arrives once, in order"
        );
    }

    /// `drain_blocking` is what callers use before writing to the terminal by
    /// another route, so it must report honestly that it could not drain.
    #[test]
    fn drain_reports_failure_while_the_terminal_is_wedged() {
        let _serial = serialized();
        let (sink, gate, _rx) = blocking_sink();
        let mut writer = FrameWriter::spawn(sink, 4, false);
        frame(&mut writer, b"frame");
        assert!(
            !writer.drain_blocking(Duration::from_millis(50)),
            "a wedged terminal cannot drain"
        );
        open(&gate);
        assert!(
            writer.drain_blocking(Duration::from_secs(2)),
            "and drains once it resumes"
        );
        drop(writer);
    }

    /// Without synchronized-output brackets there is nothing to compensate, so
    /// no escape may be invented.
    #[test]
    fn no_synchronized_escape_is_sent_when_frames_are_not_bracketed() {
        let _serial = serialized();
        let (sink, gate, rx) = blocking_sink();
        let mut writer = FrameWriter::spawn(sink, 2, false);
        for index in 0..16u8 {
            frame(&mut writer, &[index]);
        }
        open(&gate);
        drop(writer);
        let written: Vec<Vec<u8>> = rx.try_iter().collect();
        assert!(
            !written.iter().any(|blob| blob == END_SYNCHRONIZED_UPDATE),
            "an End was sent for a bracket that never opened: {written:?}"
        );
    }
}
