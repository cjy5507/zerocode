//! Small, self-contained helpers for the `drive_turn` render loop
//! (`turn_controller`). These are the safe-floor pieces of the loop: pure
//! background-snapshot slot mechanics with no coupling to the `select!`
//! structure, the frame gate, or the terminal draw path — so they can be unit
//! tested in isolation and reused across the loop body and its teardown.

use tokio::task::JoinHandle;

/// Poll a background-snapshot slot: if its task has finished, clear the slot
/// and return the joined value; otherwise leave it running and return `None`.
///
/// The `is_finished` gate keeps the `await` non-blocking — it only resolves an
/// already-completed task, exactly like the inline `is_finished` guard it
/// replaces. A join error (the snapshot task panicked) clears the slot and
/// yields `None`, so the next cadence tick simply respawns: a HUD snapshot is
/// best-effort and must never be turn-fatal.
pub(super) async fn take_finished_snapshot<T>(slot: &mut Option<JoinHandle<T>>) -> Option<T> {
    if !slot.as_ref().is_some_and(JoinHandle::is_finished) {
        return None;
    }
    slot.take()?.await.ok()
}

/// Abort a background-snapshot slot on loop exit, if one is in flight. Consumes
/// the slot; a `None` slot is a no-op.
pub(super) fn abort_snapshot<T>(slot: Option<JoinHandle<T>>) {
    if let Some(handle) = slot {
        handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_thread_rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build current-thread test runtime")
    }

    #[test]
    fn running_slot_returns_none_and_stays_live() {
        current_thread_rt().block_on(async {
            let (tx, rx) = tokio::sync::oneshot::channel::<u32>();
            let mut slot = Some(tokio::spawn(async move { rx.await.unwrap_or(0) }));
            // The task is parked on `rx`, so it is not finished.
            assert!(take_finished_snapshot(&mut slot).await.is_none());
            assert!(slot.is_some(), "an unfinished snapshot must keep running");
            drop(tx);
        });
    }

    #[test]
    fn finished_slot_yields_value_and_clears() {
        current_thread_rt().block_on(async {
            let mut slot = Some(tokio::spawn(async { 42u32 }));
            while !slot.as_ref().is_some_and(JoinHandle::is_finished) {
                tokio::task::yield_now().await;
            }
            assert_eq!(take_finished_snapshot(&mut slot).await, Some(42));
            assert!(slot.is_none(), "a drained slot must clear so a respawn can arm");
        });
    }

    #[test]
    fn panicked_snapshot_is_swallowed_and_slot_clears() {
        current_thread_rt().block_on(async {
            let mut slot: Option<JoinHandle<u32>> =
                Some(tokio::spawn(async { panic!("intentional test panic") }));
            while !slot.as_ref().is_some_and(JoinHandle::is_finished) {
                tokio::task::yield_now().await;
            }
            // A join error must not propagate; the slot clears for a respawn.
            assert!(take_finished_snapshot(&mut slot).await.is_none());
            assert!(slot.is_none());
        });
    }

    #[test]
    fn abort_snapshot_handles_empty_slot() {
        abort_snapshot::<u32>(None);
    }
}
