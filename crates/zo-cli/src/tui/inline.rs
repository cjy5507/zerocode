//! Inline terminal mode and native-scrollback transcript emission.

use std::collections::VecDeque;

use ratatui::backend::{Backend, TestBackend};
use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::style::Style;
use ratatui::Terminal;
use unicode_width::UnicodeWidthStr;

use super::image_protocol::ImageProtocol;
use super::theme::Theme;
use super::transcript::Transcript;

/// Fallback height for a terminal that will not report its size.
pub const INLINE_VIEWPORT_MIN_HEIGHT: u16 = 12;

/// One row is deliberately left above the viewport. At shutdown, Ratatui needs
/// room to insert the retained transcript before an inline viewport as a chunk;
/// an exactly full-height viewport falls back to borrowing the top row and
/// flushing inserted lines one at a time.
const SCROLLBACK_HEADROOM: u16 = 1;

/// Rows the live inline viewport claims on the primary screen.
///
/// Effectively the whole terminal. Anything less leaves part of the window
/// unusable — the live region is where the conversation is composed and read.
/// The bounded transcript owns live history so captured wheel, click, and drag
/// events all address one data source; shutdown copies it to native scrollback.
///
/// This remains the primary screen, not the alternate one, so the conversation
/// survives the session instead of disappearing with an alternate screen.
#[must_use]
pub fn inline_viewport_height(terminal_rows: u16) -> u16 {
    if terminal_rows == 0 {
        return INLINE_VIEWPORT_MIN_HEIGHT;
    }
    terminal_rows.saturating_sub(SCROLLBACK_HEADROOM).max(1)
}

/// Terminal presentation strategy for an interactive session.
///
/// Inline is the default: it keeps the primary screen and captures pointer
/// events so wheel history, click-to-expand, and drag selection work against the
/// same bounded transcript. That transcript moves to native scrollback only when
/// the session ends. Fullscreen remains for surfaces that need a persistent
/// right-hand panel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TerminalMode {
    /// Primary-screen viewport with finalized output in native scrollback.
    #[default]
    Inline,
    /// Alternate-screen, full-terminal presentation with mouse capture.
    Fullscreen,
}

impl TerminalMode {
    /// Resolve the strategy from every input that can pin it.
    ///
    /// Precedence, first match wins:
    /// 1. `--fullscreen` — the explicit escape hatch,
    /// 2. `--inline` — explicit, and still accepted now that it is the default,
    /// 3. `settings.tui.inlineMode` — only when actually set,
    /// 4. the default (inline).
    #[must_use]
    pub const fn resolve(inline_flag: bool, fullscreen_flag: bool, configured: Option<bool>) -> Self {
        if fullscreen_flag {
            return Self::Fullscreen;
        }
        if inline_flag {
            return Self::Inline;
        }
        match configured {
            Some(false) => Self::Fullscreen,
            Some(true) | None => Self::Inline,
        }
    }

    /// Whether this strategy uses the primary-screen inline viewport.
    #[must_use]
    pub const fn is_inline(self) -> bool {
        matches!(self, Self::Inline)
    }
}

/// Ownership queue for complete live transcripts awaiting shutdown-time native
/// scrollback insertion. Live frames never enqueue row prefixes: wheel, click,
/// drag selection, and typing all keep one app-owned history until interaction
/// ends.
#[derive(Debug, Default)]
pub(crate) struct FinalizedTranscriptQueue {
    pending: VecDeque<Box<Transcript>>,
}

impl FinalizedTranscriptQueue {
    /// Move all currently-live blocks into one shutdown chunk.
    pub(crate) fn finalize(&mut self, live: &mut Transcript) {
        if live.is_empty() {
            return;
        }
        self.pending.push_back(Box::new(std::mem::take(live)));
    }

    /// Drain complete transcripts transactionally. An insertion failure restores
    /// the current transcript at the front so shutdown can retry without data
    /// loss or reordering.
    pub(crate) fn drain_with<E>(
        &mut self,
        mut emit: impl FnMut(&mut Transcript) -> Result<(), E>,
    ) -> Result<(), E> {
        while let Some(mut transcript) = self.pending.pop_front() {
            if let Err(error) = emit(transcript.as_mut()) {
                self.pending.push_front(transcript);
                return Err(error);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn len(&self) -> usize { self.pending.len() }
}

/// Render queued shutdown transcripts with the existing styled transcript path
/// and insert the resulting cells before an inline viewport.
pub(crate) fn insert_finalized<B: Backend>(
    queue: &mut FinalizedTranscriptQueue,
    terminal: &mut Terminal<B>,
    theme: &Theme,
    tick: u64,
    image_protocol: ImageProtocol,
) -> Result<(), B::Error> {
    // `insert_before` uses cached viewport geometry. Refresh it so shutdown
    // rendering and insertion use the same wrapping width.
    terminal.autoresize()?;
    let width = terminal.size()?.width;
    queue.drain_with(|transcript| {
        // Scrollback is rendered into a buffer sized to the content itself, so
        // bottom-aligning against breathing room would push the tail off-screen.
        transcript.set_bottom_aligned(false);
        let content_height = transcript.scrollback_height(width, theme, image_protocol);
        if content_height == 0 || width == 0 {
            return Ok(());
        }

        // Two extra rows keep transcript viewport math from adding a scrollbar;
        // only actual content rows are copied into native scrollback.
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
                copy_scrollback_row(source, destination, y, width);
            }
        })
    })
}

/// Copy one rendered row into the scrollback buffer, emptying the continuation
/// cell that follows a double-width grapheme.
///
/// A live frame reaches the terminal through `Buffer::diff`, which skips that
/// continuation cell and lets the terminal advance the cursor by the glyph's own
/// two columns. `insert_before` has no diff — it paints every cell it is handed.
/// ratatui fills the continuation with a *space*, so copying the row verbatim
/// printed a column the terminal had never reserved, and every CJK character on
/// a settled line pushed the rest of that line one column right. That is why
/// scrollback read as letter-spaced while the live viewport above it did not.
///
/// An empty symbol prints nothing and leaves the cursor where the wide glyph
/// already put it, which is exactly what `diff` achieves by skipping.
fn copy_scrollback_row(source: &Buffer, destination: &mut Buffer, y: u16, width: u16) {
    let mut continuation: u16 = 0;
    // ratatui `reset()`s the continuation cell, dropping the style with the
    // symbol, so it is carried over from the glyph that owns those columns.
    let mut carried = Style::default();
    for x in 0..width {
        let position = Position::new(x, y);
        let Some(cell) = source.cell(position).cloned() else {
            continue;
        };
        let Some(target) = destination.cell_mut(position) else {
            continue;
        };
        if continuation > 0 {
            continuation -= 1;
            target.set_style(carried);
            target.set_symbol("");
            continue;
        }
        continuation = u16::try_from(cell.symbol().width())
            .unwrap_or(1)
            .saturating_sub(1);
        carried = cell.style();
        *target = cell;
    }
}

#[cfg(test)]
mod tests {
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::{Color, Style};
    use runtime::message_stream::{BlockId, RenderBlock};

    use super::copy_scrollback_row;

    use super::FinalizedTranscriptQueue;
    use crate::tui::Transcript;

    fn user_block(id: u64, text: &str) -> RenderBlock {
        RenderBlock::UserMessage {
            id: BlockId(id),
            text: text.to_string(),
        }
    }

    /// An inline viewport does NOT follow the terminal when it grows.
    ///
    /// This pins the ratatui behavior the resize path exists to work around:
    /// `Viewport::Inline(height)` stores the height handed to
    /// `Terminal::with_options`, and `autoresize` re-reads that stored value, so
    /// a terminal the user drags taller leaves the live region at its launch
    /// height and the extra rows are never painted (the reported "전체 창을
    /// 사용하지 않아 / 고정된 크기"). `session::tui_loop::resync_inline_viewport`
    /// therefore rebuilds the whole `Terminal`; if a future ratatui gains a
    /// setter and this assertion starts failing, that rebuild can go away.
    #[test]
    fn ratatui_pins_an_inline_viewport_to_its_construction_height() {
        use ratatui::backend::TestBackend;
        use ratatui::{Terminal, TerminalOptions, Viewport};

        let mut terminal = Terminal::with_options(
            TestBackend::new(40, 10),
            TerminalOptions {
                viewport: Viewport::Inline(super::inline_viewport_height(10)),
            },
        )
        .expect("inline test terminal");
        assert_eq!(terminal.get_frame().area().height, 9);

        terminal.backend_mut().resize(40, 40);
        terminal.autoresize().expect("autoresize");

        assert_eq!(
            terminal.get_frame().area().height,
            9,
            "the viewport keeps its construction height, so 31 of the 40 rows stay dead"
        );
        assert_eq!(
            super::inline_viewport_height(40),
            39,
            "the height the viewport should have had"
        );
    }

    /// Inline is the default presentation, and every way of pinning a mode
    /// folds in a fixed order.
    ///
    /// The load-bearing case is `None`: an absent `tui.inlineMode` must resolve
    /// to inline. Treating "unset" as `false` would have pinned every
    /// un-configured session back to the alternate screen and erase the
    /// primary-screen interaction/history strategy.
    #[test]
    fn terminal_mode_defaults_to_inline_and_flags_win_over_settings() {
        use super::TerminalMode;

        assert_eq!(TerminalMode::default(), TerminalMode::Inline);
        assert_eq!(TerminalMode::resolve(false, false, None), TerminalMode::Inline);

        // An explicit setting still decides when no flag is given — someone who
        // pinned `false` back when fullscreen was the default keeps it.
        assert_eq!(
            TerminalMode::resolve(false, false, Some(false)),
            TerminalMode::Fullscreen
        );
        assert_eq!(TerminalMode::resolve(false, false, Some(true)), TerminalMode::Inline);

        // Flags outrank settings, and `--fullscreen` outranks `--inline`.
        assert_eq!(
            TerminalMode::resolve(false, true, Some(true)),
            TerminalMode::Fullscreen
        );
        assert_eq!(TerminalMode::resolve(true, false, Some(false)), TerminalMode::Inline);
        assert_eq!(TerminalMode::resolve(true, true, None), TerminalMode::Fullscreen);
    }

    /// The live region takes the window, minus one shutdown scrollback
    /// headroom row.
    #[test]
    fn inline_viewport_takes_the_window_minus_its_scroll_headroom() {
        use super::{INLINE_VIEWPORT_MIN_HEIGHT, inline_viewport_height};

        assert_eq!(inline_viewport_height(90), 89);
        assert_eq!(inline_viewport_height(24), 23);
        // Never taller than the terminal, and never zero.
        assert_eq!(inline_viewport_height(1), 1);
        // An unreadable terminal size falls back rather than asking for 0 rows.
        assert_eq!(inline_viewport_height(0), INLINE_VIEWPORT_MIN_HEIGHT);
    }

    /// Settled CJK text must reach scrollback with the same column layout the
    /// live viewport gives it.
    ///
    /// The continuation cell of a wide grapheme has to print nothing. ratatui
    /// leaves a space there, and because `insert_before` paints every cell,
    /// that space used to be emitted — shifting the rest of the line right by
    /// one column per CJK character, which read as broken letter-spacing.
    #[test]
    fn wide_graphemes_reach_scrollback_without_their_padding_space() {
        let area = Rect::new(0, 0, 12, 1);
        let mut source = Buffer::empty(area);
        source.set_string(0, 0, "한글ab", Style::new().fg(Color::Red));
        let mut destination = Buffer::empty(area);

        copy_scrollback_row(&source, &mut destination, 0, area.width);

        let symbols: Vec<&str> = (0..4)
            .map(|x| destination[(x, 0)].symbol())
            .collect();
        assert_eq!(
            symbols,
            vec!["한", "", "글", ""],
            "continuation cells must print nothing"
        );
        // ASCII after the wide run is untouched, and styling survives the copy.
        assert_eq!(destination[(4, 0)].symbol(), "a");
        assert_eq!(destination[(0, 0)].style().fg, Some(Color::Red));
        assert_eq!(
            destination[(1, 0)].style().fg,
            Some(Color::Red),
            "the continuation keeps the glyph's style so backgrounds stay whole"
        );
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
            .drain_with::<()>(|transcript| {
                emitted.push(transcript.blocks().len());
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

        assert_eq!(
            queue.drain_with(|_| Err("terminal unavailable")),
            Err("terminal unavailable")
        );
        assert_eq!(queue.len(), 1);

        let mut emitted = 0;
        queue
            .drain_with::<()>(|transcript| {
                emitted += transcript.blocks().len();
                Ok(())
            })
            .expect("retry sink");
        assert_eq!(emitted, 1);
        assert_eq!(queue.len(), 0);
    }
}
