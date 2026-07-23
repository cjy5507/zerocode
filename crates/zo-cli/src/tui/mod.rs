//! Ratatui-based TUI foundation (Phase 3, Lane L2).
//!
//! This module is the receiving side of the Phase-3 seam established in
//! Lane L1 (`runtime::message_stream`). It owns the `ratatui` terminal,
//! the three-region layout (transcript / input / HUD), an
//! [`AppMode`] state machine, and a [`Theme`] loaded from
//! `.zo/design/tokens.json`.
//!
//! ## Living standard (mirrors L1)
//!
//! Per `.zo/tasks/L1-message-stream.handoff.md`:
//!
//! 1. Module layout: `tui/{mod,app,layout,theme}.rs`; only neutral
//!    symbols re-exported here. No widgets for [`RenderBlock`] variants
//!    live in this lane — Lane L5 adds those under `tui/blocks/`.
//! 2. Errors: one `thiserror`-derived enum per module. [`TuiError`] is
//!    the cross-module catch-all with a generic `Adapter { source }`
//!    variant. No `anyhow` below the `tui/` boundary.
//! 3. Async traits (if any) use hand-rolled
//!    `fn … -> Pin<Box<dyn Future + Send + '_>>`. This lane currently
//!    has no trait, but [`App::run`] is an async method consistent with
//!    that style.
//! 4. Tests live at `crates/zo-cli/tests/tui_<scope>.rs` and
//!    are named `<area>_<scenario>` (e.g. `theme_loads_full_palette`).
//! 5. Every `pub` item carries a `///` doc comment whose first sentence
//!    is a one-line summary.
//!
//! ## Scope boundary (`code-rules.md` R1, R2, R9)
//!
//! * No Anthropic-specific terminology — the only currency flowing in
//!   is [`runtime::message_stream::RenderBlock`].
//! * No ANSI strings. All styling derives from [`Theme`] inside the
//!   widget layer.
//! * Channels are bounded (R8) and constructed by callers (Lane L7);
//!   this lane only consumes the [`AppEndpoint`]-shaped handles.

#![allow(clippy::module_name_repetitions)]

pub mod agent_manifests;
pub mod agent_session_filter;
pub mod ansi_spans;
pub mod app;
pub mod blocks;
pub mod cards;
pub mod command_history;
pub mod fuzzy;
pub mod glyphs;
pub mod history;
pub mod heat;
pub mod hud;
pub mod image_protocol;
pub mod inline;
pub mod input;
pub mod keybindings;
pub mod layout;
pub mod markdown;
pub mod mermaid_layout;
pub mod modals;
pub mod render_schedule;
pub mod sidebar;
pub mod spinner;
pub mod stale_binary;
pub mod startup;
pub mod stderr_redirect;
pub mod term;
pub mod text_metrics;
pub mod theme;
pub mod transcript;
pub mod watchdog;
pub mod workflow_progress;
pub mod workspace_status;

pub use app::{AgentCommand, App, AppMode};
pub use command_history::CommandHistory;
pub use history::{History, HistoryError, HistoryRecord};
pub use heat::{COOLING_SECS, HeatState, cooling_ramp_idx};
pub use inline::{INLINE_VIEWPORT_HEIGHT, TerminalMode};
pub use hud::{
    AgentTaskSummary, HudState, LspStatusItem, PermissionMode, SecurityPosture, TodoChecklistItem,
    TodoChecklistStatus,
};
pub use input::{InputCommand, InputWidget};
pub use layout::LayoutRegions;
pub use modals::{
    ChoicePickerModal, ModalResult, ModalSelection, ModelPickerEntry, ModelPickerModal,
    PermissionPickerModal,
};
pub use sidebar::{ChangedFile, FileStatus, GitStatusSnapshot, SidebarState};
pub use spinner::TurnActivity;
pub use stale_binary::StaleBinaryInfo;
pub use startup::{RecentSession, StartupAuthState, StartupScreen};
pub use theme::{Breakpoint, Theme};
pub use transcript::Transcript;

use std::io;

/// Error type surfaced by the TUI foundation.
///
/// Follows the L1 living-standard error pattern: one `thiserror`-derived
/// enum per module with a generic `Adapter` catch-all for wrapped
/// downstream failures.
#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    /// I/O failure from the terminal backend (crossterm / stdout).
    #[error("terminal io error: {0}")]
    Io(#[from] io::Error),

    /// Failed to read the design-tokens JSON file from disk.
    #[error("failed to read theme tokens at {path}: {source}")]
    ThemeRead {
        /// The path we tried to read.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },

    /// The tokens file was present but could not be parsed as the
    /// expected shape.
    #[error("failed to parse theme tokens: {0}")]
    ThemeParse(#[from] serde_json::Error),

    /// The command channel to the agent task was closed before the TUI
    /// could send a command (usually means the agent task exited).
    #[error("agent command channel closed")]
    CommandChannelClosed,

    /// Catch-all wrapper for downstream adapter errors (kept for
    /// symmetry with the L1 `StreamError::Adapter` pattern).
    #[error("{component}: {message}")]
    Adapter {
        /// Component that produced the failure (e.g. `"layout"`).
        component: &'static str,
        /// Human-readable description.
        message: String,
    },
}
