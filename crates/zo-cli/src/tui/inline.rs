//! Inline terminal mode and native-scrollback transcript emission.

use std::collections::VecDeque;

use ratatui::backend::{Backend, TestBackend};
use ratatui::layout::Position;
use ratatui::Terminal;

use super::image_protocol::ImageProtocol;
use super::theme::Theme;
use super::transcript::Transcript;

/// Fixed height of the live inline viewport.
pub const INLINE_VIEWPORT_HEIGHT: u16 = 12;

/// Terminal presentation strategy for an interactive session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TerminalMode {
    /// Existing alternate-screen, full-terminal presentation.
    #[default]
    Fullscreen,
    /// Primary-screen viewport with finalized output in native scrollback.
    Inline,
}

impl TerminalMode {
    /// Resolve the strategy from the effective inline feature flag.
    #[must_use]
    pub const fn from_inline(inline: bool) -> Self {
        if inline { Self::Inline } else { Self::Fullscreen }
    }

    /// Whether this strategy uses the primary-screen inline viewport.
    #[must_use]
    pub const fn is_inline(self) -> bool {
        matches!(self, Self::Inline)
    }
}

/// Ownership queue for settled transcript chunks awaiting native scrollback.
///
/// Finalization moves the entire live transcript into this queue and replaces
/// it with an empty one. A chunk therefore cannot remain visible in the live
/// viewport and cannot be enqueued twice. Failed sinks put the chunk back at
/// the front so terminal I/O errors never silently drop content.
#[derive(Debug, Default)]
pub(crate) struct FinalizedTranscriptQueue {
    pending: VecDeque<Transcript>,
}

impl FinalizedTranscriptQueue {
    /// Move all currently-live blocks into one settled chunk.
    pub(crate) fn finalize(&mut self, live: &mut Transcript) {
        if live.is_empty() {
            return;
        }
        self.pending.push_back(std::mem::take(live));
    }

    /// Drain chunks through `emit`, restoring the current chunk on failure.
    pub(crate) fn drain_with<E>(
        &mut self,
        mut emit: impl FnMut(&mut Transcript) -> Result<(), E>,
    ) -> Result<(), E> {
        while let Some(mut transcript) = self.pending.pop_front() {
            if let Err(error) = emit(&mut transcript) {
                self.pending.push_front(transcript);
                return Err(error);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pending.len()
    }
}

/// Render queued transcript chunks with the existing styled transcript path
/// and insert the resulting cells before an inline viewport.
pub(crate) fn insert_finalized<B: Backend>(
    queue: &mut FinalizedTranscriptQueue,
    terminal: &mut Terminal<B>,
    theme: &Theme,
    tick: u64,
    image_protocol: ImageProtocol,
) -> Result<(), B::Error> {
    // `insert_before` uses the terminal's cached viewport geometry. Refresh it
    // before measuring/rendering so a width change between frames cannot leave
    // the source and destination buffers with different row wrapping.
    terminal.autoresize()?;
    let width = terminal.size()?.width;
    queue.drain_with(|transcript| {
        let content_height = transcript.scrollback_height(width, theme, image_protocol);
        if content_height == 0 || width == 0 {
            return Ok(());
        }

        // The transcript includes two rows of bottom breathing room in its
        // viewport math. Giving the off-screen renderer those rows prevents it
        // from deciding a scrollbar is needed; only the real content rows are
        // copied into native scrollback below.
        let render_height = content_height.saturating_add(2);
        let backend = TestBackend::new(width, render_height);
        let mut rendered = Terminal::new(backend)
            .expect("TestBackend terminal construction is infallible");
        rendered
            .draw(|frame| {
                transcript.draw(frame, frame.area(), theme, tick, image_protocol);
            })
            .expect("TestBackend drawing is infallible");

        let source = rendered.backend().buffer();
        terminal.insert_before(content_height, |destination| {
            for y in 0..content_height {
                for x in 0..width {
                    let position = Position::new(x, y);
                    if let (Some(source), Some(destination)) =
                        (source.cell(position), destination.cell_mut(position))
                    {
                        *destination = source.clone();
                    }
                }
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use runtime::message_stream::{BlockId, RenderBlock};

    use super::FinalizedTranscriptQueue;
    use crate::tui::Transcript;

    fn user_block(id: u64, text: &str) -> RenderBlock {
        RenderBlock::UserMessage {
            id: BlockId(id),
            text: text.to_string(),
        }
    }

    #[test]
    fn finalized_chunks_emit_once_and_empty_finalize_is_a_noop() {
        let mut live = Transcript::new();
        live.push(user_block(1, "first"));
        let mut queue = FinalizedTranscriptQueue::default();

        queue.finalize(&mut live);
        queue.finalize(&mut live);
        assert!(live.is_empty());
        assert_eq!(queue.len(), 1);

        let mut emitted = Vec::new();
        queue
            .drain_with::<()>(|chunk| {
                emitted.push(chunk.blocks().len());
                Ok(())
            })
            .expect("recording sink");
        assert_eq!(emitted, vec![1]);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn failed_sink_requeues_interrupted_turn_without_dropping_it() {
        let mut live = Transcript::new();
        live.push(user_block(2, "keep this on interrupt"));
        let mut queue = FinalizedTranscriptQueue::default();
        queue.finalize(&mut live);

        assert_eq!(queue.drain_with(|_| Err("terminal unavailable")), Err("terminal unavailable"));
        assert_eq!(queue.len(), 1);

        let mut emitted = 0;
        queue
            .drain_with::<()>(|chunk| {
                emitted += chunk.blocks().len();
                Ok(())
            })
            .expect("retry sink");
        assert_eq!(emitted, 1);
        assert_eq!(queue.len(), 0);
    }
}
