//! In-app modal widgets (Phase 3, Lane L6).
//!
//! The TUI hosts three modals on top of the transcript:
//!
//! * [`model_picker::ModelPickerModal`] — provider-grouped model list
//!   (components.md §6.1).
//! * [`permission_picker::PermissionPickerModal`] — mirrors
//!   `session.rs::prompt_permissions_picker` semantics in-app
//!   (components.md §6.2).
//! * [`choice_picker::ChoicePickerModal`] — a generic single-select
//!   list used for blocking yes/no or arbitrary prompts.
//!
//! All three share a single [`ModalResult`] currency so the host app
//! can dispatch without knowing which modal produced the event.
//!
//! ## Living standard (mirrors L1)
//!
//! 1. Module layout: `tui/modals/{mod,model_picker,permission_picker,choice_picker}.rs`.
//! 2. Errors: none surfaced — modals are stateful widgets; any failure
//!    bubbles up through [`super::TuiError`].
//! 3. No async — all three modals are synchronous widgets.
//! 4. Tests live at `crates/zo-cli/tests/tui_modals.rs`.
//! 5. Every `pub` item carries a `///` doc comment.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Scrollbar, ScrollbarOrientation, ScrollbarState};
use runtime::PermissionMode;
use runtime::message_stream::ActiveModel;

use super::cards::{CardFrame, SurfaceKind};
use super::glyphs;
use super::theme::Theme;

pub mod agents_viewer;
pub mod api_key;
pub mod choice_picker;
pub mod custom_provider;
pub mod deep_tier;
pub mod diff_viewer;
pub mod effort_picker;
pub mod file_picker;
pub mod model_picker;
pub mod permission_picker;
pub mod report_viewer;
pub mod rewind_viewer;
pub mod review;
pub mod remote_onboarding;
pub mod smart_settings;
pub mod team_inbox_viewer;
pub mod tool_toggle;
pub mod usage_dashboard;
pub mod user_question;
pub mod workflow_viewer;

pub use agents_viewer::{AgentsViewerAction, AgentsViewerModal};
pub use api_key::{ApiKeyConnectInfo, ApiKeyModal};
pub use choice_picker::ChoicePickerModal;
pub use custom_provider::{CustomProviderDraft, CustomProviderWizardModal};
pub use deep_tier::{DeepTierModal, DeepTierView};
pub use diff_viewer::{DiffViewerAction, DiffViewerModal};
pub use effort_picker::{EFFORT_STEPS, Effort, EffortPickerModal, EffortStep, effort_level_label};
pub use file_picker::FilePickerModal;
pub use model_picker::{ModelPickerEntry, ModelPickerModal};
pub use permission_picker::PermissionPickerModal;
pub use report_viewer::{ReportTone, ReportViewerBlock, ReportViewerModal};
pub use rewind_viewer::{RewindRow, RewindViewerAction, RewindViewerModal};
pub use review::ReviewModal;
pub use remote_onboarding::{RemoteOnboardingModal, RemoteOnboardingView, RemotePendingPair};
pub use smart_settings::{
    SmartSettingsCommit, SmartSettingsFreshness, SmartSettingsModal, SmartSettingsModel,
    SmartSettingsObservedRoute, SmartSettingsRecommendation, SmartSettingsTarget,
    SmartSettingsTargetKind, SmartSettingsUpdate, SmartSettingsView,
};
pub use team_inbox_viewer::{TeamInboxViewerAction, TeamInboxViewerModal};
pub use tool_toggle::{ToolToggleModal, ToolToggleRow};
pub use usage_dashboard::{UsageDashboardAction, UsageDashboardModal};
pub use user_question::UserQuestionModal;
pub use workflow_viewer::{
    WorkflowAgentRow, WorkflowPhaseRow, WorkflowView, WorkflowViewerAction, WorkflowViewerModal,
};

/// Values that a modal can hand back to the host app.
#[derive(Debug, Clone)]
pub enum ModalSelection {
    /// A model was picked from [`ModelPickerModal`].
    Model(ActiveModel),
    /// A permission mode was picked from [`PermissionPickerModal`].
    Permission(PermissionMode),
    /// A generic choice index + label was picked from
    /// [`ChoicePickerModal`].
    Choice {
        /// Zero-based index of the selected option.
        index: usize,
        /// Display label of the selected option.
        label: String,
    },
    /// Answer(s) from an `AskUserQuestion` modal. A single-select prompt hands
    /// back one value; a multi-select prompt hands back every checked option
    /// (plus any typed free-form text).
    QuestionAnswer(Vec<String>),
    /// API key submitted from the `/connect` adapter setup modal.
    ApiKey {
        /// Canonical provider id, e.g. `deepseek`.
        provider: String,
        /// Secret API key; never render or route through slash-command history.
        api_key: String,
    },
    /// Custom provider draft submitted from [`CustomProviderWizardModal`].
    CustomProvider(CustomProviderDraft),
    /// An effort level was confirmed in [`EffortPickerModal`].
    Effort {
        /// Canonical level label (e.g. `"smart"`).
        label: String,
        /// Thinking-token budget for the chosen level.
        budget: u32,
    },
    /// A runtime tool was toggled from [`ToolToggleModal`].
    ToolToggle {
        /// Canonical tool name.
        name: String,
        /// New enabled state.
        enabled: bool,
    },
    /// Smart Router settings were confirmed from [`SmartSettingsModal`].
    SmartSettings(SmartSettingsCommit),
    /// A `/tier` pool mutation requested by [`DeepTierModal`].
    DeepTier(commands::DeepTierAction),
    /// A Remote onboarding row was selected; the command is re-submitted so
    /// the existing session-level Remote lifecycle handler remains authoritative.
    RemoteCommand(String),
    /// The report popup's copy key: the popup's plain-text projection, routed
    /// to the host clipboard. The modal stays open so reading can continue.
    CopyText(String),
}

/// Outcome of a single `handle_key` call on a modal.
#[derive(Debug, Clone)]
pub enum ModalResult {
    /// The user confirmed a selection.
    Selected(ModalSelection),
    /// The user cancelled (Esc).
    Cancelled,
}

/// Where a modal overlay sits on screen. Each variant maps to one of the
/// `*_modal_rect` geometry strategies; a modal declares its own via
/// [`Modal::placement`], so `draw_modals` positions every slot modal through one
/// path instead of a per-mode arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModalPlacement {
    /// Anchored above the input, sized to its content (the list pickers).
    Anchored,
    /// The effort slider's banner strip.
    EffortBanner,
    /// A near-fullscreen viewer pane (diff / rewind / workflow / usage).
    Fullscreen,
    /// A centered command palette (the file picker).
    Palette,
    /// Screen-centered and content-sized via [`Modal::desired_size`] (the
    /// report popup): small reports get a compact card, long ones grow up to
    /// the size clamp instead of always paying a near-fullscreen pane.
    Centered,
}

/// A modal overlay owned by the App's single active-modal slot. This collapses
/// the ~11 near-identical per-modal key-dispatch and draw arms into one path:
/// the App calls [`Modal::handle_key`] and routes the [`ModalResult`] through
/// `apply_modal_outcome`, while `draw_modals` positions via
/// [`Modal::placement`]/[`Modal::desired_size`] and paints via [`Modal::draw`].
/// Free-text modals override [`Modal::cursor`] to anchor the IME composition
/// window at their text field.
pub trait Modal {
    /// Handle one key. `Some` acts on / closes the modal; `None` keeps it open.
    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult>;
    /// Paint the modal into its computed `area`.
    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme);
    /// Desired `(width, height)` for `area`, already clamped to fit the screen.
    fn desired_size(&self, area: Rect, theme: &Theme) -> (u16, u16);
    /// Where the overlay is positioned.
    fn placement(&self) -> ModalPlacement;
    /// Hardware cursor position for IME composition; only free-text modals set
    /// it, so the default parks no cursor.
    fn cursor(&self, _inner: Rect) -> Option<ratatui::layout::Position> {
        None
    }
    /// Scroll one mouse-wheel notch by `rows`; `up` scrolls toward the top.
    /// Default no-op — list modals override to move their highlight/viewport,
    /// which is how the App's wheel routing reaches a slot modal without a
    /// per-mode arm.
    fn scroll(&mut self, _up: bool, _rows: usize) {}
    /// Downcast hatch for the few modals the host must reach concretely *after*
    /// they are on the slot — the file picker's async scan lands rows via
    /// `set_items`, and the API-key modal receives a clipboard paste via
    /// `paste_text`. Every impl returns `self`; callers use
    /// [`App::active_modal_as`].
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl Modal for ChoicePickerModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        // Inherent method wins method resolution, so this is not recursive.
        ChoicePickerModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        ChoicePickerModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Anchored list picker. Byte-parity with the old `modal_size_for_mode`
        // Session/Login/ArgPick branches: width clamped to the area, height =
        // option rows + 6 (borders + blank spacer + key-hint footer), then the
        // shared (6, 18) clamp bounded by the available height.
        let width = area
            .width
            .clamp(36, 64)
            .min(area.width.saturating_sub(4).max(24));
        let content = u16::try_from(self.len())
            .unwrap_or(u16::MAX)
            .saturating_add(6);
        let height = content
            .clamp(6, 18)
            .min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    fn scroll(&mut self, up: bool, rows: usize) {
        if up {
            self.scroll_up(rows);
        } else {
            self.scroll_down(rows);
        }
    }
}

impl Modal for ModelPickerModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        ModelPickerModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        ModelPickerModal::draw(self, frame, area, theme);
    }

    fn cursor(&self, area: Rect) -> Option<Position> {
        self.cursor_position(area)
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Provider-grouped picker. Byte-parity with the old `modal_size_for_mode`
        // ModalModel branches: a WIDER width clamp (40..72) than the list
        // pickers, height = visual rows (options + group headers) + 6, then the
        // shared (6, 18) clamp bounded by the available height.
        let width = area
            .width
            .clamp(40, 72)
            .min(area.width.saturating_sub(4).max(24));
        let content = u16::try_from(self.visual_rows())
            .unwrap_or(u16::MAX)
            .saturating_add(6);
        let height = content
            .clamp(6, 18)
            .min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    fn scroll(&mut self, up: bool, rows: usize) {
        if up {
            self.scroll_up(rows);
        } else {
            self.scroll_down(rows);
        }
    }
}

impl Modal for PermissionPickerModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        PermissionPickerModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        PermissionPickerModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Fixed-height picker (the mode list + header + footer). Byte-parity
        // with the old `modal_size_for_mode` ModalPermissions arm: default
        // list-width clamp, a constant 11 content rows, shared (6, 18) clamp.
        let width = area
            .width
            .clamp(36, 64)
            .min(area.width.saturating_sub(4).max(24));
        let height = 11u16.clamp(6, 18).min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    fn scroll(&mut self, up: bool, _rows: usize) {
        // No dedicated scroll method; reuse arrow-key navigation (one row per
        // notch, matching the legacy wheel arm). The synthesized key only moves
        // the cursor, so its `ModalResult` is dropped.
        let code = if up { KeyCode::Up } else { KeyCode::Down };
        let _ = self.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
    }
}

impl Modal for EffortPickerModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        EffortPickerModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        EffortPickerModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // The effort banner is NOT the geometry authority for its own rect — it
        // uses [`ModalPlacement::EffortBanner`], so `draw_modals` positions it
        // via `effort_modal_rect` (confined to the transcript column). This
        // nominal size is only a fallback and is never consulted for the banner.
        (area.width, 12)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::EffortBanner
    }

    // No `scroll` override: the slider ignores the wheel, matching the legacy
    // `scroll_active_picker` no-op for effort/diff/rewind.
}

impl Modal for ToolToggleModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        ToolToggleModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        ToolToggleModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Non-authoritative: the near-fullscreen `/tools` toggle uses
        // [`ModalPlacement::Fullscreen`], so `draw_modals` sizes it via
        // `diff_modal_rect`. This fallback is never consulted for its rect.
        (area.width, area.height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Fullscreen
    }

    fn scroll(&mut self, up: bool, _rows: usize) {
        // No dedicated scroll method; reuse arrow-key navigation (matching the
        // legacy `ModalTools` wheel arm). The synthesized key only moves the
        // cursor, so its `ModalResult` is dropped.
        let code = if up { KeyCode::Up } else { KeyCode::Down };
        let _ = self.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
    }
}

impl Modal for ReportViewerModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        ReportViewerModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        ReportViewerModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, _area: Rect, _theme: &Theme) -> (u16, u16) {
        ReportViewerModal::desired_size(self)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Centered
    }

    fn scroll(&mut self, up: bool, rows: usize) {
        self.scroll_wheel(up, rows);
    }
}

impl Modal for ReviewModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        ReviewModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        ReviewModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        (area.width, area.height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Fullscreen
    }
}

impl Modal for RemoteOnboardingModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        RemoteOnboardingModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        RemoteOnboardingModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        let width = area
            .width
            .clamp(56, 84)
            .min(area.width.saturating_sub(4).max(30));
        let content = u16::try_from(self.content_rows())
            .unwrap_or(u16::MAX)
            .saturating_add(2);
        let height = content
            .clamp(12, 24)
            .min(area.height.saturating_sub(2).max(8));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    fn scroll(&mut self, up: bool, rows: usize) {
        for _ in 0..rows {
            if up {
                self.move_up();
            } else {
                self.move_down();
            }
        }
    }
}

impl Modal for DeepTierModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        DeepTierModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        DeepTierModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, theme: &Theme) -> (u16, u16) {
        DeepTierModal::desired_size(self, area, theme)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    fn scroll(&mut self, up: bool, rows: usize) {
        DeepTierModal::scroll(self, up, rows);
    }
}

impl Modal for SmartSettingsModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        SmartSettingsModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        SmartSettingsModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Non-authoritative: the `/smart` dashboard uses
        // [`ModalPlacement::Fullscreen`], sized via `diff_modal_rect`.
        (area.width, area.height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Fullscreen
    }

    // No `scroll` override: the dashboard ignored the wheel (legacy no-op).
}

impl Modal for UserQuestionModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        UserQuestionModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        UserQuestionModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, theme: &Theme) -> (u16, u16) {
        // Sizes from its real rendered rows (option descriptions, the free-form
        // row, footer, soft-wrap) so a `len()`-based guess cannot clip them.
        // Byte-parity with the old `modal_size_for_mode` ModalQuestion branch:
        // a wider width clamp (36..84) and `desired_rows + 2` (borders), then a
        // (6, 24) clamp bounded by the available height.
        let width = area
            .width
            .clamp(36, 84)
            .min(area.width.saturating_sub(4).max(24));
        let rows = self
            .desired_rows(theme, width.saturating_sub(2))
            .saturating_add(2);
        let height = rows.clamp(6, 24).min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    fn scroll(&mut self, up: bool, _rows: usize) {
        // No dedicated scroll method; reuse arrow-key navigation (matching the
        // legacy `ModalQuestion` wheel arm). The synthesized key only moves the
        // cursor, so its `ModalResult` is dropped.
        let code = if up { KeyCode::Up } else { KeyCode::Down };
        let _ = self.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
    }
}

impl Modal for CustomProviderWizardModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        CustomProviderWizardModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        CustomProviderWizardModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        let width = area
            .width
            .clamp(54, 86)
            .min(area.width.saturating_sub(4).max(36));
        let height = 17u16.clamp(10, 22).min(area.height.saturating_sub(2).max(10));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }
}

impl Modal for ApiKeyModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        ApiKeyModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        ApiKeyModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Fixed-height text-entry form. Byte-parity with the old
        // `modal_size_for_mode` ModalApiKey arm: default list-width clamp, a
        // constant 14 content rows, shared (6, 18) clamp.
        let width = area
            .width
            .clamp(36, 64)
            .min(area.width.saturating_sub(4).max(24));
        let height = 14u16.clamp(6, 18).min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Anchored
    }

    // No `scroll` override: the API-key form ignored the wheel (legacy no-op).
}

impl Modal for FilePickerModal {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        FilePickerModal::handle_key(self, key)
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        FilePickerModal::draw(self, frame, area, theme);
    }

    fn desired_size(&self, area: Rect, _theme: &Theme) -> (u16, u16) {
        // Non-authoritative: the fuzzy file picker uses
        // [`ModalPlacement::Palette`], so `draw_modals` sizes it via
        // `palette_modal_rect`. This fallback is never consulted for its rect.
        (area.width, area.height)
    }

    fn placement(&self) -> ModalPlacement {
        ModalPlacement::Palette
    }

    fn scroll(&mut self, up: bool, rows: usize) {
        if up {
            self.scroll_up(rows);
        } else {
            self.scroll_down(rows);
        }
    }
}

// ============================================================================
// Shared visual language for selection modals
//
// Every list-style modal (question / choice / model / permission) draws the
// cursor row with the same marker + accent emphasis and closes with the same
// key-hint footer, so the surfaces read as one family. `effort_picker` and
// `tool_toggle` already carry a footer; these helpers bring the rest in line.
// ============================================================================

/// Cursor marker for the highlighted row.
///
/// Routes through [`glyphs::modal_cursor`] so the Unicode chevron (`❯ `)
/// degrades to a one-cell ASCII `> ` under `NO_COLOR`/plain mode. Always two
/// display cells so selected and blank rows stay column-aligned. Callers pass
/// `!theme.no_color`.
#[must_use]
pub(super) fn cursor_marker(color: bool) -> &'static str {
    glyphs::modal_cursor(color)
}

/// Blank lead-in for non-selected rows, padded to the marker's cell width so
/// labels stay column-aligned. Mode-independent (two spaces).
#[must_use]
pub(super) fn blank_marker() -> &'static str {
    glyphs::modal_cursor_blank()
}

/// Emphasis style for the selected row: brand amber + bold. Under `NO_COLOR`
/// the accent resolves to `Reset`, so the bold weight still distinguishes it.
#[must_use]
pub(super) fn selected_style(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.palette.accent)
        .add_modifier(Modifier::BOLD)
}

/// One segment in a modal footer.
///
/// Footers are small status/hint rows, but several modals need a mix of plain
/// labels (for current mode/progress) and key/action pairs. Keeping the segment
/// renderer here makes separator spacing and key/body styling a single source of
/// truth instead of hand-assembled `Span` chains in each modal.
#[derive(Debug, Clone, Copy)]
pub(super) enum FooterSegment<'a> {
    /// A dim informational label, e.g. `split` or `rows 1-20/50`.
    Label(&'a str),
    /// A keyboard hint rendered as `key label`.
    Hint {
        key: &'a str,
        label: &'a str,
        key_style: Option<Style>,
    },
}

impl<'a> FooterSegment<'a> {
    /// Build a dim informational footer label.
    #[must_use]
    pub(super) fn label(text: &'a str) -> Self {
        Self::Label(text)
    }

    /// Build a key/action footer hint using the theme's key-hint style.
    #[must_use]
    pub(super) fn hint(key: &'a str, label: &'a str) -> Self {
        Self::Hint {
            key,
            label,
            key_style: None,
        }
    }

    /// Build a key/action footer hint with an overridden key style.
    ///
    /// Used for disabled/inert keys such as `/diff`'s split toggle on narrow
    /// terminals while preserving the shared separator and label styling.
    #[must_use]
    pub(super) fn hint_with_key_style(key: &'a str, label: &'a str, key_style: Style) -> Self {
        Self::Hint {
            key,
            label,
            key_style: Some(key_style),
        }
    }
}

/// Build a modal footer from pre-classified segments.
#[must_use]
pub(super) fn modal_footer(
    theme: &Theme,
    segments: &[FooterSegment<'_>],
    separator: &str,
) -> Line<'static> {
    let mut spans = Vec::with_capacity(segments.len() * 3);
    for (idx, segment) in segments.iter().enumerate() {
        if idx > 0 {
            spans.push(Span::styled(
                separator.to_string(),
                theme.typography.key_hint,
            ));
        }
        match *segment {
            FooterSegment::Label(label) => {
                spans.push(Span::styled(label.to_string(), theme.typography.dim));
            }
            FooterSegment::Hint {
                key,
                label,
                key_style,
            } => {
                let key_style = key_style.unwrap_or(theme.typography.key_hint);
                spans.push(Span::styled(format!("{key} "), key_style));
                spans.push(Span::styled(label.to_string(), theme.typography.dim));
            }
        }
    }
    Line::from(spans)
}

/// Build the shared key-hint footer line. Each `(key, label)` pair renders the
/// key in the dim key-hint style and the label in dim body, joined by ` · `,
/// mirroring `effort_picker::footer_line`.
#[must_use]
pub(super) fn key_hint_footer(theme: &Theme, hints: &[(&str, &str)]) -> Line<'static> {
    let segments = hints
        .iter()
        .map(|(key, label)| FooterSegment::hint(key, label))
        .collect::<Vec<_>>();
    modal_footer(theme, &segments, "  ·  ")
}

/// Build a compact key-hint footer with caller-chosen separator spacing.
#[must_use]
pub(super) fn key_hint_footer_with_separator(
    theme: &Theme,
    hints: &[(&str, &str)],
    separator: &str,
) -> Line<'static> {
    let segments = hints
        .iter()
        .map(|(key, label)| FooterSegment::hint(key, label))
        .collect::<Vec<_>>();
    modal_footer(theme, &segments, separator)
}

/// Render the standard modal frame and return its inner content rectangle.
///
/// Every simple modal uses the same role-based border, title, and code-surface
/// background. Centralizing that chrome keeps the visual contract in one place;
/// callers remain responsible only for their body layout (including any extra
/// inner margin they intentionally apply).
#[must_use]
pub(super) fn modal_frame(
    frame: &mut Frame<'_>,
    area: Rect,
    title: impl Into<String>,
    theme: &Theme,
) -> Rect {
    // Style the title `body` explicitly: an unstyled title would inherit the
    // frame's accent `border_style`, but these simple modals keep their plain
    // body-colored title — only the border gains the brand accent.
    CardFrame::new(SurfaceKind::Modal, theme)
        .title(Line::styled(title.into(), theme.typography.body))
        .render(frame, area)
}

/// Render a vertical scrollbar on the right edge of `area` for a scrollable
/// region of `content_rows` total rows currently at `scroll`.
///
/// No-op when the content fits the viewport (or the area is zero-height). The
/// thumb position and size are derived purely from `scroll`, `content_rows`,
/// and `area.height`: the track spans `max_scroll + 1` positions (one per
/// reachable scroll offset) and the thumb length tracks the viewport height,
/// so the geometry is identical across every modal that scrolls.
pub(crate) fn draw_scrollbar(
    frame: &mut Frame<'_>,
    area: Rect,
    scroll: u16,
    content_rows: usize,
    theme: &Theme,
) {
    if area.height == 0 || content_rows <= usize::from(area.height) {
        return;
    }
    let content_total = u16::try_from(content_rows).unwrap_or(u16::MAX);
    let viewport_h = area.height;
    let max_scroll = content_total.saturating_sub(viewport_h);
    let scroll = scroll.min(max_scroll);
    let scroll_positions = max_scroll.saturating_add(1);
    let mut state = ScrollbarState::new(usize::from(scroll_positions))
        .position(usize::from(scroll))
        .viewport_content_length(usize::from(viewport_h));
    let color = !theme.no_color;
    let arrow_style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::DIM);
    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(Some(glyphs::pick(
            color,
            glyphs::SCROLL_UP,
            glyphs::SCROLL_UP_NC,
        )))
        .end_symbol(Some(glyphs::pick(
            color,
            glyphs::SCROLL_DOWN,
            glyphs::SCROLL_DOWN_NC,
        )))
        .track_symbol(Some(glyphs::pick(
            color,
            glyphs::SCROLL_TRACK,
            glyphs::SCROLL_TRACK_NC,
        )))
        .thumb_symbol(glyphs::pick(
            color,
            glyphs::SCROLL_THUMB,
            glyphs::SCROLL_THUMB_NC,
        ))
        .begin_style(arrow_style)
        .end_style(arrow_style)
        .track_style(Style::new().fg(theme.palette.muted))
        .thumb_style(Style::new().fg(theme.palette.dim));
    frame.render_stateful_widget(scrollbar, area, &mut state);
}

#[cfg(test)]
mod scrollbar_tests {
    use super::*;
    use crate::tui::theme::Theme;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    /// Dump the cells of the scrollbar column (last column of `area`) as a
    /// string, one glyph per row, so the thumb position/size can be pinned.
    fn render_column(scroll: u16, content_rows: usize, height: u16) -> String {
        let theme = Theme::zo();
        let mut term = Terminal::new(TestBackend::new(4, height)).unwrap();
        term.draw(|frame| {
            let area = Rect::new(0, 0, 4, height);
            draw_scrollbar(frame, area, scroll, content_rows, &theme);
        })
        .unwrap();
        let buf = term.backend().buffer();
        (0..height)
            .map(|y| buf.cell((3, y)).unwrap().symbol())
            .collect::<String>()
    }

    #[test]
    fn modal_frame_returns_standard_inner_rect() {
        let theme = Theme::zo();
        let mut term = Terminal::new(TestBackend::new(10, 5)).unwrap();
        let mut inner = Rect::default();
        term.draw(|frame| {
            inner = modal_frame(frame, Rect::new(0, 0, 10, 5), "title", &theme);
        })
        .unwrap();

        assert_eq!(inner, Rect::new(1, 1, 8, 3));
    }

    #[test]
    fn no_scrollbar_when_content_fits() {
        // 5 rows of content in a 5-row viewport: nothing rendered.
        assert_eq!(render_column(0, 5, 5), "     ");
    }

    #[test]
    fn thumb_tracks_scroll_position() {
        // 20 rows in a 5-row viewport: arrows at top/bottom, thumb in the
        // track. At scroll 0 the thumb sits just below the up arrow; at the
        // bottom it sits just above the down arrow.
        let theme = Theme::zo();
        let up = glyphs::pick(!theme.no_color, glyphs::SCROLL_UP, glyphs::SCROLL_UP_NC);
        let down = glyphs::pick(
            !theme.no_color,
            glyphs::SCROLL_DOWN,
            glyphs::SCROLL_DOWN_NC,
        );
        let thumb = glyphs::pick(
            !theme.no_color,
            glyphs::SCROLL_THUMB,
            glyphs::SCROLL_THUMB_NC,
        );

        let top = render_column(0, 20, 5);
        let bottom = render_column(u16::MAX, 20, 5);

        // Arrows pin the ends in both states.
        assert!(top.starts_with(up), "top arrow: {top:?}");
        assert!(top.ends_with(down), "bottom arrow: {top:?}");

        // The thumb glyph appears, and its position moves down as we scroll.
        let top_thumb = top.find(thumb);
        let bottom_thumb = bottom.find(thumb);
        assert!(top_thumb.is_some(), "thumb present at top: {top:?}");
        assert!(bottom_thumb.is_some(), "thumb present at bottom: {bottom:?}");
        assert!(
            bottom_thumb > top_thumb,
            "thumb moves down as scroll increases: {top:?} -> {bottom:?}"
        );
    }
}
