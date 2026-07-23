//! Typed command output + the single render funnel (`W3`).
//!
//! Every slash-command handler returns a [`CommandOutput`] describing
//! *what* to show, never *how*. [`render`] is the one place that maps
//! that intent onto transcript [`RenderBlock`]s via [`push_report`] /
//! [`push_card`]. Keeping the projection in a single funnel means:
//!
//! - handlers stay pure-ish (no scattered `push_*` calls, easy to test),
//! - the headless path (W11) can serialize the same [`CommandOutput`]
//!   instead of re-deriving structure from rendered text,
//! - report styling changes live in exactly one function.
//!
//! Handlers that must mutate view state directly (open a modal, reset the
//! transcript) do so through [`DispatchCtx`](super::context::DispatchCtx)
//! and return [`CommandOutput::Quiet`].

use core_types::CardModel;
use runtime::message_stream::{BlockIdGen, SystemLevel};
use zo_cli::tui::App;
use zo_cli::tui::modals::{ReportTone, ReportViewerBlock};

use super::helpers_tui::{push_card, push_report};

/// One renderable unit of command output.
#[derive(Debug)]
pub(super) enum OutputBlock {
    /// A plain system report at the given severity.
    Report(SystemLevel, String),
    /// A structured rich card.
    Card(CardModel),
}

/// The structured result of dispatching one slash command.
///
/// Constructed by handlers, consumed exactly once by [`render`].
#[derive(Debug)]
pub(super) enum CommandOutput {
    /// Nothing to render — the handler already mutated view state
    /// (opened a modal, reseeded the transcript, …).
    Quiet,
    /// Tear down the TUI and exit the REPL.
    Exit,
    /// Render these blocks into the transcript, in order. The shape for
    /// action confirmations, usage hints, and errors — small, contextual,
    /// part of the conversation history.
    Blocks(Vec<OutputBlock>),
    /// Show these blocks in the centered report popup instead of the
    /// transcript. The shape for read-only reports (`/mcp`, `/doctor`, …):
    /// nothing is recorded — a report is re-derivable by re-running its
    /// command — so the conversation stays clean.
    Popup {
        /// Popup title (also the copy toast label), e.g. `"/mcp"`.
        title: String,
        /// Report content, same block model as the transcript path.
        blocks: Vec<OutputBlock>,
    },
}

impl CommandOutput {
    /// A single info-level report.
    pub(super) fn info(text: impl Into<String>) -> Self {
        Self::Blocks(vec![OutputBlock::Report(SystemLevel::Info, text.into())])
    }

    /// A single warning-level report.
    pub(super) fn warn(text: impl Into<String>) -> Self {
        Self::Blocks(vec![OutputBlock::Report(SystemLevel::Warn, text.into())])
    }

    /// A single error-level report.
    pub(super) fn error(text: impl Into<String>) -> Self {
        Self::Blocks(vec![OutputBlock::Report(SystemLevel::Error, text.into())])
    }

    /// A plain-text report shown in the centered popup.
    pub(super) fn popup(title: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Popup {
            title: title.into(),
            blocks: vec![OutputBlock::Report(SystemLevel::Info, text.into())],
        }
    }

    /// A rich card shown in the centered popup, titled by the card itself.
    pub(super) fn popup_card(card: CardModel) -> Self {
        Self::Popup {
            title: card.title.trim().to_string(),
            blocks: vec![OutputBlock::Card(card)],
        }
    }

    /// Append another report, enabling multi-block handlers (e.g. login)
    /// to chain without hand-building a `Vec`.
    #[must_use]
    pub(super) fn and_report(mut self, level: SystemLevel, text: impl Into<String>) -> Self {
        match &mut self {
            Self::Blocks(blocks) | Self::Popup { blocks, .. } => {
                blocks.push(OutputBlock::Report(level, text.into()));
            }
            Self::Quiet | Self::Exit => {}
        }
        self
    }
}

/// The single funnel: project a [`CommandOutput`] onto the transcript or the
/// centered report popup.
///
/// Returns `true` when the command requested process exit.
pub(super) fn render(app: &mut App, ids: &BlockIdGen, output: CommandOutput) -> bool {
    match output {
        CommandOutput::Quiet => false,
        CommandOutput::Exit => true,
        CommandOutput::Blocks(blocks) => {
            for block in blocks {
                match block {
                    OutputBlock::Report(level, text) => push_report(app, ids, level, text),
                    OutputBlock::Card(card) => push_card(app, ids, card),
                }
            }
            false
        }
        CommandOutput::Popup { title, blocks } => {
            let blocks = blocks.into_iter().map(popup_block).collect();
            app.open_report_modal(title, blocks);
            false
        }
    }
}

/// Map one dispatcher block onto the popup's mirror model.
fn popup_block(block: OutputBlock) -> ReportViewerBlock {
    match block {
        OutputBlock::Report(level, body) => ReportViewerBlock::Text {
            tone: match level {
                SystemLevel::Warn => ReportTone::Warn,
                SystemLevel::Error => ReportTone::Error,
                SystemLevel::Info => ReportTone::Info,
            },
            body,
        },
        OutputBlock::Card(card) => ReportViewerBlock::Card(card),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_warn_error_wrap_single_report() {
        for output in [
            CommandOutput::info("a"),
            CommandOutput::warn("b"),
            CommandOutput::error("c"),
        ] {
            match output {
                CommandOutput::Blocks(blocks) => assert_eq!(blocks.len(), 1),
                _ => panic!("expected Blocks"),
            }
        }
    }

    #[test]
    fn and_report_appends_in_order() {
        let output = CommandOutput::info("first").and_report(SystemLevel::Warn, "second");
        match output {
            CommandOutput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match (&blocks[0], &blocks[1]) {
                    (OutputBlock::Report(_, a), OutputBlock::Report(_, b)) => {
                        assert_eq!(a, "first");
                        assert_eq!(b, "second");
                    }
                    _ => panic!("expected two reports"),
                }
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn and_report_extends_popup_blocks_and_titles_from_card() {
        let output =
            CommandOutput::popup_card(CardModel::new(" /x ")).and_report(SystemLevel::Info, "y");
        match output {
            CommandOutput::Popup { title, blocks } => {
                assert_eq!(title, "/x", "popup title derives from the trimmed card title");
                assert_eq!(blocks.len(), 2);
            }
            _ => panic!("expected Popup"),
        }
    }
}
