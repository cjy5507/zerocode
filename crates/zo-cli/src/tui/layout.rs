//! Five-region layout calculator.
//!
//! Per the P1 redesign the TUI is a vertical stack of five regions:
//!
//! ```text
//! ┌─────────────────────────────┐
//! │          transcript         │  ← fills all remaining space
//! ├─────────────────────────────┤
//! │          rule_top           │  ← exactly 1 row, horizontal rule
//! ├─────────────────────────────┤
//! │          input box          │  ← 1..=10 rows (caller-driven)
//! ├─────────────────────────────┤
//! │          rule_bot           │  ← 0 rows (no extra separator chrome)
//! ├─────────────────────────────┤
//! │             HUD             │  ← 1 row (quiet session summary)
//! └─────────────────────────────┘
//! ```
//!
//! `rule_bot` is retained as a zero-height region so the struct shape and
//! tiling invariants stay stable. Whitespace and the HUD surface transition
//! separate the composer from session metadata without another full-width rule.
//!
//! The calculator is pure — given a `ratatui::layout::Rect`, the
//! currently-desired input row count, and the currently-desired HUD
//! row count, it returns five non-overlapping `Rect`s. Widget
//! rendering is done by callers.

use ratatui::layout::{Constraint, Direction, Layout, Rect};

use super::theme::{Breakpoint, Theme};
use super::TuiError;

/// Exact ASCII notice painted when a compact [`ViewportClass::TooSmall`] frame
/// cannot use the wide-but-short vertical degradation path. A genuinely
/// unusable size renders this line instead of partial chrome.
pub const TOO_SMALL_MESSAGE: &str = "Terminal too small - resize to continue";

/// Minimum transcript rows that must survive overlay pressure whenever the
/// viewport is usable (not [`ViewportClass::TooSmall`]). The transcript is the
/// conversation surface, so the *lower-priority* overlays (agent panel, queue)
/// collapse before it starves below this.
pub const MIN_READABLE_TRANSCRIPT_ROWS: u16 = 3;

/// Absolute floor the transcript is never squeezed below, even by the highest
/// keep-priority overlays (active search and the current running step). On a
/// short frame where the readable floor and the running step compete, the
/// running step keeps its single row while the transcript shrinks to this hard
/// minimum — matching "preserve the current running step as long as possible"
/// while still guaranteeing at least one conversation row.
pub const MIN_TRANSCRIPT_ROWS_HARD: u16 = 1;

/// Rows the composer chrome must reserve above the readable transcript for the
/// frame to stay usable. Not the absolute one-row region minimums (top rule +
/// input + bottom rule + HUD sum to only a few rows on paper); this is the
/// height at which the transcript, a real multi-line input, and the status HUD
/// all read at once without the transcript collapsing below
/// [`MIN_READABLE_TRANSCRIPT_ROWS`]. Empirically the lowest frame the TUI still
/// drives usefully is 11 rows tall (the height-constrained plan-dock frame), so
/// the composer reserve is that floor minus the transcript minimum. Kept as a
/// named constant, not a magic literal, so the intent and the `40x10`-unusable
/// / `80x24`-usable boundary are explicit.
const MIN_COMPOSER_ROWS: u16 = 8;

/// Smallest interior geometry that still hosts a readable transcript plus the
/// composer chrome. Below either bound the frame cannot show the conversation
/// and a HUD/input at once, so the viewport is [`ViewportClass::TooSmall`].
///
/// Width: enough for the [`TOO_SMALL_MESSAGE`] itself plus a small margin, and
/// wide enough for a usable one-line composer. Height: the minimum transcript
/// (`MIN_READABLE_TRANSCRIPT_ROWS`) plus the [`MIN_COMPOSER_ROWS`] reserve. This
/// is a readable-frame floor, not a hard `80x24` minimum: it lands at 11 rows,
/// making `40x10` unusable while `80x24`/`120x40`/`200x60` stay usable, and the
/// existing 11-row plan-dock frame remains valid.
const MIN_USABLE_WIDTH: u16 = 40;
const MIN_USABLE_HEIGHT: u16 = MIN_READABLE_TRANSCRIPT_ROWS + MIN_COMPOSER_ROWS;

/// Responsive viewport bucket for the whole frame.
///
/// This is the single classification the render path branches on. `Narrow` /
/// `Compact` / `Wide` mirror the theme's own width [`Breakpoint`]s (so a custom
/// tokens file with different thresholds flows through unchanged), while
/// `TooSmall` is decided purely from the minimum geometry a readable transcript
/// and composer need — it is *not* an 80x24 hard floor, only the point below
/// which the five-region stack cannot render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportClass {
    /// Below the minimum transcript/composer geometry. Sidebar chrome is
    /// suppressed; compact frames show [`TOO_SMALL_MESSAGE`], while wide-but-
    /// short frames can retain the single-row HUD.
    TooSmall,
    /// Usable but narrow (theme [`Breakpoint::Narrow`]). Sidebar suppressed.
    Narrow,
    /// Usable mid-width (theme [`Breakpoint::Compact`]). Sidebar suppressed.
    Compact,
    /// Wide enough for the optional right sidebar (theme [`Breakpoint::Wide`]).
    Wide,
}

impl ViewportClass {
    /// Classify a terminal `area` against a `theme`'s breakpoints.
    ///
    /// `TooSmall` is checked first from geometry (both dimensions must clear the
    /// minimum usable bounds); otherwise the width bucket is taken verbatim from
    /// [`Theme::for_width`] so classification and every other width-driven
    /// decision read one policy.
    #[must_use]
    pub fn classify(area: Rect, theme: &Theme) -> Self {
        if area.width < MIN_USABLE_WIDTH || area.height < MIN_USABLE_HEIGHT {
            return Self::TooSmall;
        }
        match theme.for_width(area.width) {
            Breakpoint::Narrow => Self::Narrow,
            Breakpoint::Compact => Self::Compact,
            Breakpoint::Wide => Self::Wide,
        }
    }

    /// Whether the optional sidebar is eligible at this class. Only `Wide`
    /// terminals host it, matching the wide-screen conversation-first layout.
    #[must_use]
    pub const fn sidebar_eligible(self) -> bool {
        matches!(self, Self::Wide)
    }

    /// Whether the frame is usable (anything but [`Self::TooSmall`]).
    #[must_use]
    pub const fn is_usable(self) -> bool {
        !matches!(self, Self::TooSmall)
    }

    /// Whether fullscreen rendering must replace the frame with the resize
    /// notice. A wide-but-short frame can still use the compact layout's
    /// single-row HUD; the notice is reserved for a `TooSmall` frame that also
    /// lacks enough width to make that vertical degradation useful.
    #[must_use]
    pub const fn requires_notice(self, area: Rect) -> bool {
        matches!(self, Self::TooSmall) && area.width <= MIN_USABLE_WIDTH
    }
}

/// Pure, `Copy` responsive layout decision for a frame: the viewport class plus
/// the resolved sidebar-visibility choice. Computed once per frame from the
/// area, theme, and user's sidebar toggle, then consumed by both measurement and
/// paint so the two never disagree about whether a sidebar exists or whether the
/// frame is [`ViewportClass::TooSmall`].
///
/// Geometry itself stays in [`LayoutRegions`]; this type only decides *policy*
/// (class + sidebar), keeping classification testable without a full frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutPlan {
    /// The viewport bucket for this frame.
    pub class: ViewportClass,
    /// Whether the sidebar is actually shown: the caller requested it, the
    /// viewport is [`ViewportClass::Wide`], and the remaining chat column keeps
    /// its minimum usable width. `false` for every unusable split.
    pub sidebar_visible: bool,
}

impl LayoutPlan {
    /// Build the plan for `area` under `theme`, honoring the caller's
    /// `sidebar_requested` toggle. This is the only place that decides sidebar
    /// visibility; downstream geometry trusts the resolved boolean.
    #[must_use]
    pub fn compute(area: Rect, theme: &Theme, sidebar_requested: bool) -> Self {
        let class = ViewportClass::classify(area, theme);
        Self {
            class,
            sidebar_visible: sidebar_requested
                && class.sidebar_eligible()
                && sidebar_split_fits(area.width),
        }
    }
}

/// Desired (uncapped) row reservations for the bottom overlay stack, in the
/// order they paint above the input: search bar, queue preview, Run Dock / todo
/// panel, and the pinned agent panel. Each field is what the corresponding
/// overlay *wants* before any transcript-protection pressure is applied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlayDemand {
    /// Active transcript search bar (1 row when in search mode, else 0).
    pub search: u16,
    /// Queue preview block (previews + gap rows), else 0.
    pub queue: u16,
    /// Run Dock / live todo panel (plan rows + current-step executor), else 0.
    pub run_dock: u16,
    /// Pinned live-agent panel, else 0.
    pub agent: u16,
}

/// Rows each overlay is actually granted after clamping the demand to the
/// overlay budget in collapse-priority order. The transcript keeps at least
/// [`MIN_READABLE_TRANSCRIPT_ROWS`] whenever the frame is usable, so overlays
/// give up rows before the conversation does.
///
/// Collapse order (sacrificed first → last): `queue`, then `run_dock`, then
/// `agent`; `search` (active search) and the transcript minimum are preserved
/// longest. This is the inverse of the reservation order below — the overlay
/// reserved *last* against the shrinking budget is the one squeezed first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlayPlan {
    /// Granted search-bar rows (never reduced while search is active and the
    /// frame is usable).
    pub search: u16,
    /// Granted queue-preview rows (first to collapse under pressure).
    pub queue: u16,
    /// Granted Run Dock / todo rows.
    pub run_dock: u16,
    /// Granted pinned-agent-panel rows.
    pub agent: u16,
}

impl OverlayPlan {
    /// Allocate `demand` within a `transcript_height`, keeping at least
    /// [`MIN_READABLE_TRANSCRIPT_ROWS`] for the transcript.
    ///
    /// The budget (`transcript_height − MIN_READABLE_TRANSCRIPT_ROWS`) is handed
    /// out in *keep-priority* order — `search`, then `run_dock` (the current
    /// running step), then `agent`, then `queue` — so when the budget runs out
    /// the queue is starved first and the running step / active search survive
    /// longest, exactly the documented collapse ladder. Every grant is `≤` its
    /// demand, so an overlay that wanted nothing still gets nothing.
    #[must_use]
    pub fn allocate(demand: OverlayDemand, transcript_height: u16) -> Self {
        // Two-tier transcript floor. The top-priority survivors — active search
        // and the current running step (Run Dock) — are only held back by the
        // absolute [`MIN_TRANSCRIPT_ROWS_HARD`] floor, so on a short frame the
        // running step still gets its row while the transcript shrinks to that
        // hard minimum. The lower-priority overlays (agent panel, then queue)
        // are held back by the full readable floor, so once the frame has room
        // they never push the conversation below [`MIN_READABLE_TRANSCRIPT_ROWS`].
        // This matches the collapse ladder: queue collapses first, then agent,
        // and the running step / active search survive longest.
        let hard_budget = transcript_height.saturating_sub(MIN_TRANSCRIPT_ROWS_HARD);
        let readable_budget = transcript_height.saturating_sub(MIN_READABLE_TRANSCRIPT_ROWS);

        // Survivors (active search, then the running step) draw from the hard
        // budget, in that keep-priority order.
        let search = demand.search.min(hard_budget);
        let run_dock = demand.run_dock.min(hard_budget - search);

        // Lower-priority overlays (agent panel, then queue) draw only from what
        // the readable budget has left after the survivors, so they collapse
        // before the conversation dips below the readable floor. Queue reserves
        // last, so it is the first overlay starved under pressure.
        let survivors = search + run_dock;
        let mut soft_budget = readable_budget.saturating_sub(survivors);
        let mut take_soft = |want: u16| -> u16 {
            let grant = want.min(soft_budget);
            soft_budget -= grant;
            grant
        };
        let agent = take_soft(demand.agent);
        let queue = take_soft(demand.queue);
        Self {
            search,
            queue,
            run_dock,
            agent,
        }
    }

    /// Total rows the overlay stack consumes.
    #[must_use]
    pub const fn total(self) -> u16 {
        self.search + self.queue + self.run_dock + self.agent
    }
}

/// Minimum rows the input box is allowed to occupy.
pub const INPUT_MIN_ROWS: u16 = 1;
/// Maximum rows the input box is allowed to occupy.
pub const INPUT_MAX_ROWS: u16 = 10;
/// Default HUD height for the native-terminal footer.
pub const HUD_ROWS: u16 = 1;
/// Maximum HUD height. A second row is granted only when a caller explicitly
/// asks for it (a narrow terminal that must move the workflow phase/percent off
/// the crowded single status line); the default stays one row.
pub const HUD_MAX_ROWS: u16 = 2;
/// Minimum HUD height — callers must always reserve at least this many rows.
pub const HUD_MIN_ROWS: u16 = 1;
/// Bottom rule row height. `0`: whitespace and the HUD surface transition
/// separate the input from session metadata without extra separator chrome.
pub const RULE_ROWS: u16 = 0;
/// Top indicator height. A single row hosting the spinner or the
/// horizontal rule — the spinner itself includes leading rule
/// characters so it doubles as a visual separator.
pub const RULE_TOP_ROWS: u16 = 1;

/// Minimum width of the sidebar panel to remain readable.
const SIDEBAR_MIN_WIDTH: u16 = 24;
/// Maximum width of the sidebar panel.
const SIDEBAR_MAX_WIDTH: u16 = 36;

/// Compute sidebar width as 20% of the available width (8:2 ratio),
/// clamped to readable bounds.
fn sidebar_width_for(total_width: u16) -> u16 {
    let target = total_width / 5;
    target.clamp(SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH)
}

/// Whether the target sidebar width leaves a minimally usable chat column.
/// This is consulted only by [`LayoutPlan`], keeping visibility policy in one
/// place before measurement and paint consume the resolved boolean.
fn sidebar_split_fits(area_width: u16) -> bool {
    let sidebar_width = sidebar_width_for(area_width);
    area_width >= sidebar_width.saturating_add(MIN_USABLE_WIDTH)
}

/// Resolve the sidebar width from the visibility decision already made by
/// [`LayoutPlan`]. Shared by [`LayoutRegions::compute_with_sidebar`] and
/// [`LayoutRegions::content_width`] so measurement and paint use the same width.
pub(crate) fn resolved_sidebar_width(area_width: u16, sidebar_visible: bool) -> u16 {
    if sidebar_visible {
        sidebar_width_for(area_width)
    } else {
        0
    }
}

/// The regions of the top-level TUI layout.
///
/// The base five vertical regions (transcript, `rule_top`, input, `rule_bot`,
/// hud) are always present. When the sidebar is visible, the transcript
/// row is split horizontally so that the sidebar occupies ~20% of the
/// width (8:2 ratio) and the transcript takes the remainder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutRegions {
    /// Sidebar panel on the right — zero-sized when hidden.
    pub sidebar: Rect,
    /// Width reserved for the sidebar (0 when hidden).
    pub sidebar_width: u16,
    /// Transcript area — fills all space above the top rule.
    pub transcript: Rect,
    /// Top horizontal rule between transcript and input.
    pub rule_top: Rect,
    /// Input box area — `INPUT_MIN_ROWS..=INPUT_MAX_ROWS` rows.
    pub input: Rect,
    /// Bottom horizontal rule between input and HUD.
    pub rule_bot: Rect,
    /// HUD status area — 1 row (compact status bar).
    pub hud: Rect,
}

impl LayoutRegions {
    /// Compute the regions for a given terminal area and a
    /// caller-supplied desired input row count. HUD defaults to
    /// [`HUD_ROWS`]; sidebar is hidden.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Adapter`] if the area is too small to host
    /// a single row.
    pub fn compute(area: Rect, input_rows: u16) -> Result<Self, TuiError> {
        Self::compute_with_hud(area, input_rows, HUD_ROWS)
    }

    /// Compute the regions with an explicit HUD row count (sidebar hidden).
    ///
    /// `input_rows` is clamped into `INPUT_MIN_ROWS..=INPUT_MAX_ROWS`;
    /// `hud_rows` is clamped into `HUD_MIN_ROWS..=HUD_MAX_ROWS`. On
    /// very small terminals the layout degrades gracefully: rules are
    /// dropped first, then the HUD shrinks, then the input collapses.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Adapter`] if the area height is zero.
    pub fn compute_with_hud(area: Rect, input_rows: u16, hud_rows: u16) -> Result<Self, TuiError> {
        Self::compute_with_sidebar(area, input_rows, hud_rows, false)
    }

    /// Compute the regions with an explicit HUD row count and optional
    /// sidebar panel.
    ///
    /// When `sidebar_visible` is `true`, the transcript row is split so the
    /// sidebar gets ~20% of the width and the transcript gets the remainder.
    /// The caller must pass the decision resolved by [`LayoutPlan`]; this
    /// geometry function does not apply a second visibility policy. The chat
    /// chrome (`rule_top`, input, `rule_bot`, hud) is narrowed to the same
    /// content column so the footer does not visually run underneath the right
    /// metadata panel.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Adapter`] if the area height is zero.
    pub fn compute_with_sidebar(
        area: Rect,
        input_rows: u16,
        hud_rows: u16,
        sidebar_visible: bool,
    ) -> Result<Self, TuiError> {
        if area.height == 0 {
            return Err(TuiError::Adapter {
                component: "layout",
                message: "terminal height is zero".to_string(),
            });
        }

        let clamped_input = input_rows.clamp(INPUT_MIN_ROWS, INPUT_MAX_ROWS);
        let clamped_hud = hud_rows.clamp(HUD_MIN_ROWS, HUD_MAX_ROWS);
        let rules_total = RULE_TOP_ROWS + RULE_ROWS;
        let chrome = clamped_input + clamped_hud + rules_total;

        let (transcript_rows, rule_top_rows, input_rows_final, rule_bot_rows, hud_rows_final) =
            if area.height > chrome {
                (
                    area.height - chrome,
                    RULE_TOP_ROWS,
                    clamped_input,
                    RULE_ROWS,
                    clamped_hud,
                )
            } else if area.height > clamped_input + clamped_hud {
                // Drop rules to make room — transcript gets 0.
                (
                    area.height - clamped_input - clamped_hud,
                    0,
                    clamped_input,
                    0,
                    clamped_hud,
                )
            } else if area.height > clamped_hud {
                // Shrink input to whatever is left.
                let input = (area.height - clamped_hud).min(clamped_input).max(1);
                let transcript = area.height - input - clamped_hud;
                (transcript, 0, input, 0, clamped_hud)
            } else if area.height > HUD_MIN_ROWS {
                // Degrade HUD toward its minimum.
                (0, 0, 0, 0, area.height.min(clamped_hud))
            } else {
                // Single-row terminal: HUD only, no rules, no input, no transcript.
                (0, 0, 0, 0, area.height)
            };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(transcript_rows),
                Constraint::Length(rule_top_rows),
                Constraint::Length(input_rows_final),
                Constraint::Length(rule_bot_rows),
                Constraint::Length(hud_rows_final),
            ])
            .split(area);

        // Determine sidebar geometry. The sidebar occupies the right
        // portion of the transcript row only. If the terminal is too
        // narrow for a useful split we suppress the sidebar even when
        // requested.
        let full_transcript = chunks[0];
        let computed_sidebar_w = resolved_sidebar_width(full_transcript.width, sidebar_visible);
        let (sidebar_rect, transcript_rect, sw) = if computed_sidebar_w > 0 {
            let tr = Rect::new(
                full_transcript.x,
                full_transcript.y,
                full_transcript.width - computed_sidebar_w,
                full_transcript.height,
            );
            let sb = Rect::new(
                full_transcript.x + full_transcript.width - computed_sidebar_w,
                full_transcript.y,
                computed_sidebar_w,
                full_transcript.height,
            );
            (sb, tr, computed_sidebar_w)
        } else {
            (
                Rect::new(full_transcript.x, full_transcript.y, 0, 0),
                full_transcript,
                0,
            )
        };

        // The chat column: whatever the sidebar leaves over, anchored to the
        // left edge and filling the full remaining width up to the sidebar (or
        // the terminal edge when the sidebar is hidden). Transcript, rules,
        // input, and HUD all share this exact column so the chat reads as one
        // aligned document (components.md §1.1) that tracks the terminal width
        // instead of stranding a dead gutter between the chat and the sidebar.
        let col_w = area.width.saturating_sub(sw);
        let column = |r: Rect| -> Rect { Rect::new(area.x, r.y, col_w, r.height) };

        Ok(Self {
            sidebar: sidebar_rect,
            sidebar_width: sw,
            transcript: column(transcript_rect),
            rule_top: column(chunks[1]),
            input: column(chunks[2]),
            rule_bot: column(chunks[3]),
            hud: column(chunks[4]),
        })
    }

    /// Width of the chat content column (input/rules/hud) for a given terminal
    /// width and sidebar visibility — the full width minus any resolved
    /// sidebar. Callers use this to size the input box *before* laying out so
    /// the input's soft-wrap row count can be computed against its real inner
    /// width; this MUST match the column produced by
    /// [`Self::compute_with_sidebar`] or the input wraps against a width it
    /// will not be drawn at.
    #[must_use]
    pub fn content_width(area_width: u16, sidebar_visible: bool) -> u16 {
        area_width.saturating_sub(resolved_sidebar_width(area_width, sidebar_visible))
    }

    /// `true` if the regions form the expected column layout for `area`:
    /// the five rows stack with no vertical gap, the sidebar (when present)
    /// hugs the right edge for the transcript row, and every chat region
    /// (transcript, rules, input, hud) shares one left-anchored content
    /// column filling the sidebar leftover (or the full width when the
    /// sidebar is hidden).
    #[must_use]
    pub fn tiles(&self, area: Rect) -> bool {
        // The transcript row height is the same whether or not the
        // sidebar is visible — the sidebar simply shares the row.
        let transcript_row_height = self.transcript.height;
        let total_height = transcript_row_height
            + self.rule_top.height
            + self.input.height
            + self.rule_bot.height
            + self.hud.height;
        if total_height != area.height {
            return false;
        }

        // Sidebar geometry: transcript-row height, pinned to the right edge.
        if self.sidebar_width > 0 {
            if self.sidebar.y != area.y || self.sidebar.height != transcript_row_height {
                return false;
            }
            if self.sidebar.width != self.sidebar_width
                || self.sidebar.x + self.sidebar.width != area.x + area.width
            {
                return false;
            }
        }

        if self.transcript.y != area.y {
            return false;
        }
        if self.rule_top.y != self.transcript.y + self.transcript.height {
            return false;
        }
        if self.input.y != self.rule_top.y + self.rule_top.height {
            return false;
        }
        if self.rule_bot.y != self.input.y + self.input.height {
            return false;
        }
        if self.hud.y != self.rule_bot.y + self.rule_bot.height {
            return false;
        }

        // One shared chat column for transcript + chrome: the full sidebar
        // leftover, left-anchored.
        let expected_w = area.width.saturating_sub(self.sidebar_width);
        let expected_x = area.x;
        [
            self.transcript,
            self.rule_top,
            self.input,
            self.rule_bot,
            self.hud,
        ]
        .iter()
        .all(|r| r.x == expected_x && r.width == expected_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidebar_hidden_gives_zero_width_sidebar_rect() {
        let area = Rect::new(0, 0, 120, 40);
        let regions =
            LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, false).expect("layout");
        assert_eq!(regions.sidebar.width, 0);
        assert_eq!(regions.sidebar.height, 0);
        assert_eq!(regions.sidebar_width, 0);
    }

    #[test]
    fn sidebar_visible_splits_transcript_area() {
        let area = Rect::new(0, 0, 120, 40);
        let regions = LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, true).expect("layout");
        let expected_sw = sidebar_width_for(area.width);

        assert_eq!(regions.sidebar_width, expected_sw);
        assert_eq!(regions.sidebar.width, expected_sw);
        assert_eq!(regions.transcript.x, 0);
        assert_eq!(regions.sidebar.x, area.width - expected_sw);
        assert_eq!(regions.sidebar.width + regions.transcript.width, area.width);
        assert_eq!(regions.sidebar.y, regions.transcript.y);
        assert_eq!(regions.sidebar.height, regions.transcript.height);
        assert!(regions.tiles(area), "sidebar-visible layout must tile");
    }

    #[test]
    fn sidebar_visible_narrows_chat_footer() {
        let area = Rect::new(0, 0, 120, 40);
        let regions = LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, true).expect("layout");

        let expected_chat_width = area.width - regions.sidebar_width;
        assert_eq!(regions.rule_top.width, expected_chat_width);
        assert_eq!(regions.input.width, expected_chat_width);
        assert_eq!(regions.rule_bot.width, expected_chat_width);
        assert_eq!(
            regions.hud.width, expected_chat_width,
            "HUD/footer should stay in the chat column instead of running below the sidebar"
        );
    }

    #[test]
    fn wide_terminal_fills_left_anchored_content_column() {
        let area = Rect::new(0, 0, 160, 40);
        let regions =
            LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, false).expect("layout");
        assert_eq!(
            regions.transcript.width, area.width,
            "no sidebar ⇒ the chat column fills the full terminal width"
        );
        assert_eq!(
            regions.transcript.x, 0,
            "column stays anchored to the left edge"
        );
        for r in [regions.rule_top, regions.input, regions.rule_bot, regions.hud] {
            assert_eq!(r.x, regions.transcript.x, "chrome shares the column x");
            assert_eq!(r.width, regions.transcript.width, "chrome shares the column width");
        }
        assert_eq!(
            LayoutRegions::content_width(area.width, false),
            regions.input.width,
            "pre-layout wrap width must match the drawn input width"
        );
        assert!(regions.tiles(area));
    }

    #[test]
    fn wide_terminal_with_sidebar_fills_column_up_to_the_sidebar() {
        let area = Rect::new(0, 0, 200, 40);
        let regions =
            LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, true).expect("layout");
        let sw = regions.sidebar_width;
        assert!(sw > 0, "200 cols hosts a sidebar");
        assert_eq!(
            regions.sidebar.x + regions.sidebar.width,
            area.width,
            "sidebar stays pinned to the right edge"
        );
        assert_eq!(
            regions.transcript.width,
            area.width - sw,
            "the chat column fills every cell up to the sidebar — no dead gutter"
        );
        assert_eq!(
            regions.transcript.x, 0,
            "column stays left-anchored; it now spans the whole leftover"
        );
        assert_eq!(
            LayoutRegions::content_width(area.width, true),
            regions.input.width
        );
        assert!(regions.tiles(area));
    }

    #[test]
    fn narrow_terminal_keeps_full_width_column() {
        let area = Rect::new(0, 0, 100, 30);
        let regions =
            LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, false).expect("layout");
        assert_eq!(regions.transcript.width, area.width);
        assert_eq!(regions.transcript.x, 0, "no gutters at any width");
        assert!(regions.tiles(area));
    }

    #[test]
    fn sidebar_hidden_matches_original_layout() {
        let area = Rect::new(0, 0, 100, 30);
        let original = LayoutRegions::compute_with_hud(area, 3, HUD_ROWS).expect("original");
        let with_sidebar =
            LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, false).expect("sidebar=false");

        assert_eq!(original.transcript, with_sidebar.transcript);
        assert_eq!(original.rule_top, with_sidebar.rule_top);
        assert_eq!(original.input, with_sidebar.input);
        assert_eq!(original.rule_bot, with_sidebar.rule_bot);
        assert_eq!(original.hud, with_sidebar.hud);
        assert!(with_sidebar.tiles(area));
    }

    // ── ViewportClass classification ────────────────────────────────────

    /// Classify against the default theme, whose breakpoints are
    /// `narrow ≤ 59`, `compact 60..=99`, `wide ≥ 100`.
    fn classify_default(w: u16, h: u16) -> ViewportClass {
        ViewportClass::classify(Rect::new(0, 0, w, h), &Theme::no_color())
    }

    #[test]
    fn viewport_class_maps_theme_breakpoints_when_usable() {
        // Narrow bucket (≤ 59 cols), tall enough to be usable.
        assert_eq!(classify_default(59, 24), ViewportClass::Narrow);
        // Narrow → Compact edge at the theme's narrow_max + 1.
        assert_eq!(classify_default(60, 24), ViewportClass::Compact);
        // Compact upper edge.
        assert_eq!(classify_default(99, 24), ViewportClass::Compact);
        // Compact → Wide edge at the theme's wide_min.
        assert_eq!(classify_default(100, 24), ViewportClass::Wide);
        assert_eq!(classify_default(200, 60), ViewportClass::Wide);
    }

    #[test]
    fn viewport_class_too_small_below_the_minimum_usable_geometry() {
        // Below the min usable width, regardless of height.
        assert_eq!(
            classify_default(MIN_USABLE_WIDTH - 1, 60),
            ViewportClass::TooSmall
        );
        // Below the min usable height, regardless of width.
        assert_eq!(
            classify_default(200, MIN_USABLE_HEIGHT - 1),
            ViewportClass::TooSmall
        );
        // The unusable example from the spec.
        assert_eq!(classify_default(40, 10), ViewportClass::TooSmall);
    }

    #[test]
    fn viewport_class_edges_at_the_exact_minimum_usable_bounds() {
        // Exactly at both minimums is usable (Narrow at 40 cols).
        assert_eq!(
            classify_default(MIN_USABLE_WIDTH, MIN_USABLE_HEIGHT),
            ViewportClass::Narrow
        );
        // One below either bound flips to TooSmall.
        assert_eq!(
            classify_default(MIN_USABLE_WIDTH, MIN_USABLE_HEIGHT - 1),
            ViewportClass::TooSmall
        );
        assert_eq!(
            classify_default(MIN_USABLE_WIDTH - 1, MIN_USABLE_HEIGHT),
            ViewportClass::TooSmall
        );
    }

    #[test]
    fn required_sizes_are_usable_and_40x10_is_too_small() {
        for (w, h) in [(80u16, 24u16), (120, 40), (200, 60)] {
            assert!(
                classify_default(w, h).is_usable(),
                "{w}x{h} must be usable"
            );
        }
        assert_eq!(classify_default(40, 10), ViewportClass::TooSmall);
    }

    #[test]
    fn sidebar_only_eligible_when_wide() {
        assert!(!ViewportClass::TooSmall.sidebar_eligible());
        assert!(!ViewportClass::Narrow.sidebar_eligible());
        assert!(!ViewportClass::Compact.sidebar_eligible());
        assert!(ViewportClass::Wide.sidebar_eligible());
    }

    #[test]
    fn layout_plan_shows_sidebar_only_when_requested_and_wide() {
        let theme = Theme::no_color();
        // Wide + requested ⇒ shown.
        let wide = LayoutPlan::compute(Rect::new(0, 0, 200, 60), &theme, true);
        assert_eq!(wide.class, ViewportClass::Wide);
        assert!(wide.sidebar_visible);
        // Wide but not requested ⇒ hidden.
        assert!(!LayoutPlan::compute(Rect::new(0, 0, 200, 60), &theme, false).sidebar_visible);
        // Requested but only Compact ⇒ hidden (no split below wide).
        let compact = LayoutPlan::compute(Rect::new(0, 0, 80, 24), &theme, true);
        assert_eq!(compact.class, ViewportClass::Compact);
        assert!(!compact.sidebar_visible);
        // TooSmall never shows a sidebar even if requested.
        assert!(!LayoutPlan::compute(Rect::new(0, 0, 40, 10), &theme, true).sidebar_visible);
    }

    #[test]
    fn layout_plan_keeps_a_useful_chat_column_with_custom_breakpoints() {
        let mut theme = Theme::no_color();
        theme.wide_min = 60;

        for width in [60, 63] {
            let plan = LayoutPlan::compute(Rect::new(0, 0, width, 24), &theme, true);
            assert_eq!(plan.class, ViewportClass::Wide);
            assert!(
                !plan.sidebar_visible,
                "{width} columns cannot preserve the minimum chat width"
            );
        }

        let boundary = LayoutPlan::compute(Rect::new(0, 0, 64, 24), &theme, true);
        assert_eq!(boundary.class, ViewportClass::Wide);
        assert!(boundary.sidebar_visible);
        assert_eq!(
            LayoutRegions::content_width(64, boundary.sidebar_visible),
            MIN_USABLE_WIDTH
        );
    }

    // ── Geometry at the required sizes ──────────────────────────────────

    /// Every region must sit within `area` and the five rows must tile it.
    fn assert_regions_in_bounds(regions: &LayoutRegions, area: Rect) {
        for r in [
            regions.transcript,
            regions.rule_top,
            regions.input,
            regions.rule_bot,
            regions.hud,
            regions.sidebar,
        ] {
            assert!(r.x >= area.x, "rect {r:?} left of area");
            assert!(r.y >= area.y, "rect {r:?} above area");
            assert!(r.x + r.width <= area.x + area.width, "rect {r:?} overflows right");
            assert!(r.y + r.height <= area.y + area.height, "rect {r:?} overflows bottom");
        }
    }

    #[test]
    fn geometry_in_bounds_and_tiles_at_required_sizes() {
        for (w, h, sidebar) in [
            (80u16, 24u16, false),
            (120, 40, false),
            (120, 40, true),
            (200, 60, false),
            (200, 60, true),
        ] {
            let area = Rect::new(0, 0, w, h);
            let regions =
                LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, sidebar).expect("layout");
            assert_regions_in_bounds(&regions, area);
            assert!(regions.tiles(area), "{w}x{h} sidebar={sidebar} must tile");
            // The transcript keeps a readable height at every usable size.
            assert!(
                regions.transcript.height >= MIN_READABLE_TRANSCRIPT_ROWS,
                "{w}x{h}: transcript {} below readable minimum",
                regions.transcript.height
            );
        }
    }

    #[test]
    fn wide_sidebar_preserved_at_120_and_200() {
        for w in [120u16, 200u16] {
            let area = Rect::new(0, 0, w, 40);
            let regions =
                LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, true).expect("layout");
            assert!(regions.sidebar_width > 0, "{w} cols should host a sidebar");
            assert_eq!(
                regions.sidebar.x + regions.sidebar.width,
                area.width,
                "sidebar pinned to the right edge at {w} cols"
            );
            // Transcript + sidebar exactly cover the width, no overlap/gutter.
            assert_eq!(regions.transcript.width + regions.sidebar_width, area.width);
        }
    }

    // ── OverlayPlan collapse priority + transcript minimum ──────────────

    #[test]
    fn overlay_plan_grants_everything_when_transcript_is_tall() {
        let demand = OverlayDemand {
            search: 1,
            queue: 6,
            run_dock: 5,
            agent: 4,
        };
        // 40-row transcript easily fits all 16 overlay rows.
        let plan = OverlayPlan::allocate(demand, 40);
        assert_eq!(plan.search, 1);
        assert_eq!(plan.queue, 6);
        assert_eq!(plan.run_dock, 5);
        assert_eq!(plan.agent, 4);
        // Transcript keeps well above the minimum.
        assert!(40 - plan.total() >= MIN_READABLE_TRANSCRIPT_ROWS);
    }

    #[test]
    fn overlay_plan_lower_tier_reserves_the_readable_minimum() {
        // Only the lower-priority overlays (agent, queue) are bounded by the
        // readable floor. With no survivors demanding rows, the transcript keeps
        // at least `MIN_READABLE_TRANSCRIPT_ROWS` at every height.
        let demand = OverlayDemand {
            search: 0,
            queue: 20,
            run_dock: 0,
            agent: 20,
        };
        for h in 0..=40u16 {
            let plan = OverlayPlan::allocate(demand, h);
            let kept = h.saturating_sub(plan.total());
            if h > MIN_READABLE_TRANSCRIPT_ROWS {
                assert!(
                    kept >= MIN_READABLE_TRANSCRIPT_ROWS,
                    "h={h}: kept={kept} starved the transcript below the readable floor"
                );
            } else {
                assert_eq!(plan.total(), 0, "h={h}: overlays must yield entirely");
            }
        }
    }

    #[test]
    fn overlay_plan_survivors_never_dip_below_the_hard_floor() {
        // Active search and the running step may squeeze the transcript below
        // the *readable* floor (so a short frame still shows the running step),
        // but never below the absolute hard floor: at least one conversation row
        // always survives while the frame is usable.
        let demand = OverlayDemand {
            search: 1,
            queue: 0,
            run_dock: 20,
            agent: 0,
        };
        for h in MIN_TRANSCRIPT_ROWS_HARD..=40u16 {
            let plan = OverlayPlan::allocate(demand, h);
            let kept = h.saturating_sub(plan.total());
            assert!(
                kept >= MIN_TRANSCRIPT_ROWS_HARD,
                "h={h}: kept={kept} starved the transcript below the hard floor"
            );
        }
    }

    #[test]
    fn overlay_plan_collapses_queue_first_then_run_dock_then_agent() {
        let demand = OverlayDemand {
            search: 1,
            queue: 4,
            run_dock: 3,
            agent: 2,
        };
        // Budget = h - 3. Shrink h and watch the ladder: search + the running
        // step survive longest, the queue starves first.
        //
        // h=13 ⇒ budget 10 ⇒ everything fits (1+4+3+2 = 10).
        let full = OverlayPlan::allocate(demand, 13);
        assert_eq!((full.search, full.queue, full.run_dock, full.agent), (1, 4, 3, 2));

        // h=12 ⇒ budget 9 ⇒ 1 row must yield, and it is the queue.
        let p12 = OverlayPlan::allocate(demand, 12);
        assert_eq!(p12.search, 1, "search preserved");
        assert_eq!(p12.run_dock, 3, "running step preserved");
        assert_eq!(p12.agent, 2, "agent preserved before queue fully yields");
        assert_eq!(p12.queue, 3, "queue is the first to shrink");

        // h=9 ⇒ budget 6 ⇒ queue fully gone (0), search+run_dock+agent kept.
        let p9 = OverlayPlan::allocate(demand, 9);
        assert_eq!(p9.queue, 0, "queue collapses entirely first");
        assert_eq!((p9.search, p9.run_dock, p9.agent), (1, 3, 2));

        // h=8 ⇒ budget 5 ⇒ queue gone, now the agent yields before run_dock.
        let p8 = OverlayPlan::allocate(demand, 8);
        assert_eq!(p8.queue, 0);
        assert_eq!(p8.run_dock, 3, "running step still preserved");
        assert_eq!(p8.agent, 1, "agent yields after the queue, before run_dock");

        // h=6 ⇒ queue + agent gone; search and the full running step survive
        // together (they draw from the hard budget, not the readable floor).
        let p6 = OverlayPlan::allocate(demand, 6);
        assert_eq!((p6.search, p6.queue, p6.agent), (1, 0, 0));
        assert_eq!(p6.run_dock, 3, "the running step survives longest with search");

        // h=4 ⇒ only search + the running step remain, the step shrinking to fit
        // the hard budget (4 − hard floor 1 = 3, minus the 1-row search = 2).
        let p4 = OverlayPlan::allocate(demand, 4);
        assert_eq!((p4.search, p4.queue, p4.agent), (1, 0, 0));
        assert_eq!(
            p4.run_dock, 2,
            "the running step is the last overlay to shrink, bounded by the hard floor"
        );
    }

    #[test]
    fn overlay_plan_search_survives_longest() {
        // Even with a huge queue and no room, the 1-row active search bar is
        // reserved first, so it is the last thing standing.
        let demand = OverlayDemand {
            search: 1,
            queue: 30,
            run_dock: 0,
            agent: 0,
        };
        let plan = OverlayPlan::allocate(demand, MIN_READABLE_TRANSCRIPT_ROWS + 1);
        assert_eq!(plan.search, 1);
        assert_eq!(plan.queue, 0);
    }
}
