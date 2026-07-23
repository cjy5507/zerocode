//! Dispatch context + typed error (`W4` plumbing).
//!
//! [`DispatchCtx`] bundles the three borrows every handler needs
//! (`cli`, `app`, `ids`) so handlers take one parameter instead of
//! three and the call sites read uniformly. Because the fields are
//! distinct, handlers can still borrow `ctx.cli` and `ctx.app`
//! disjointly in the same expression (e.g. reseeding the transcript
//! from the live session).
//!
//! [`DispatchError`] replaces the ~17 hand-written
//! `.map_err(|e| TuiLoopError::Tui(e.to_string()))` conversions with a
//! single `#[from]` bridge, preserving the error source chain up to the
//! one boundary where the loop only needs a message.

use runtime::message_stream::BlockIdGen;
use zo_cli::tui::App;

use super::super::tui_loop::TuiLoopError;
use super::super::LiveCli;

/// The borrows every slash-command handler operates on.
///
/// Handlers receive `&mut DispatchCtx` and reach fields directly
/// (`ctx.cli`, `ctx.app`, `ctx.ids`) so the borrow checker can split
/// the mutable/immutable field borrows it needs.
pub(super) struct DispatchCtx<'a> {
    pub(super) cli: &'a mut LiveCli,
    pub(super) app: &'a mut App,
    pub(super) ids: &'a BlockIdGen,
}

/// Errors raised while executing a slash command.
///
/// The `#[from]` arms let handlers use `?` on the underlying report
/// services (which return `Box<dyn Error>`) and on raw I/O without the
/// lossy per-call-site stringification the dispatcher used before.
#[derive(Debug, thiserror::Error)]
pub(super) enum DispatchError {
    /// A command/report service failed; its boxed source is preserved.
    #[error(transparent)]
    Service(#[from] Box<dyn std::error::Error>),

    /// Direct filesystem / process I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<DispatchError> for TuiLoopError {
    fn from(error: DispatchError) -> Self {
        // Collapse to a message only at the loop boundary; the source
        // chain stayed intact through every intermediate `?`.
        Self::Tui(error.to_string())
    }
}
