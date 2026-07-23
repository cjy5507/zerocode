//! `/effort` interactive slider modal.
//!
//! Mirrors Claude Code's effort picker: a horizontal violet gradient
//! bar with seven stops — `low → medium → high → xhigh → max →
//! ultra → smart` — controlled by left/right arrows, confirmed with
//! Enter, dismissed with Esc.
//!
//! The numeric thinking budgets in [`EFFORT_STEPS`] are the single
//! source of truth for the slider; the slash dispatch maps them onto
//! `cli.thinking_budget` after [`ModalSelection::Effort`] fires.

// Slider geometry is bounded by the terminal width (a `u16`) and the
// fixed-size `EFFORT_STEPS` table, so the `usize`→`u16` and `usize`→`f32`
// casts used for layout can never lose information in practice.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::vec_init_then_push
)]
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};

use super::super::cards::{CardFrame, SurfaceKind};

use super::super::theme::Theme;
use super::{ModalResult, ModalSelection};

/// Canonical effort level. Single source of truth for the thinking
/// budget, the slash-command preset table ([`crate::tui`] re-exports it
/// for the dispatcher), and the slider stops in [`EFFORT_STEPS`].
///
/// `Smart` additionally injects a parallel-orchestration system
/// reminder at turn time (handled by the session layer), which is why
/// its description mentions agent fan-out rather than a raw budget —
/// and unlike every other level, its wire effort is a DYNAMIC BAND
/// (`[xhigh .. model ceiling]`, resolved per request by
/// `api::resolve_effort_band`) rather than one static tier: see
/// [`Self::level`] (returns the band floor) and [`Self::band_ceiling`]
/// (the band top).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    /// Extended thinking disabled.
    Off,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    /// Static top-tier pin, one rung above `Max`. Unlike `Smart`, this is a
    /// single named wire level with no per-request escalation and no
    /// orchestration hint — the "just always send the real top tier" preset.
    Ultra,
    /// Dynamic band `[xhigh .. model ceiling]` plus a parallel-agent
    /// orchestration hint. Formerly named `Ultracode`; still accepts
    /// `ultracode` as a legacy alias (including in persisted settings).
    Smart,
}

impl Effort {
    /// Every level in canonical order (`Off` first, `Smart` last).
    /// The slider in [`EFFORT_STEPS`] skips `Off`.
    pub const ALL: &'static [Effort] = &[
        Effort::Off,
        Effort::Low,
        Effort::Medium,
        Effort::High,
        Effort::Xhigh,
        Effort::Max,
        Effort::Ultra,
        Effort::Smart,
    ];

    /// Thinking-token budget. `0` disables extended thinking.
    #[must_use]
    pub const fn budget(self) -> u32 {
        match self {
            Effort::Off => 0,
            Effort::Low => 1_024,
            Effort::Medium => 4_096,
            Effort::High => 10_000,
            Effort::Xhigh => 16_000,
            Effort::Max => 24_000,
            // Between Max (24_000) and Smart (28_000) — a distinct value so
            // `from_budget`/`step_for_budget` reverse lookups can tell the two
            // apart. Matches the CLI ladder order (Max < Ultra < Smart), which
            // also mirrors `effort_rank`'s Max < Ultra ordering (`runtime_bridge.rs`).
            Effort::Ultra => 26_000,
            // Above Ultra's 26_000 so budget-derived paths (`effort_level_for_budget`,
            // which has no Ultra bucket and tops out at Max) tier Smart at least
            // as high as Max/Ultra. Smart's *named* wire tier is the band floor
            // (`Xhigh`, see `level()`) carried independently of this legacy
            // budget, so this value must still remain distinct from Ultra's
            // 26_000 for `from_budget`/`step_for_budget`.
            Effort::Smart => 28_000,
        }
    }

    /// The provider-neutral [`api::EffortLevel`] this level sends on the wire, or
    /// `None` for [`Effort::Off`] (no effort control — the backend default
    /// applies). This is the single source of truth for the CLI level → wire
    /// effort mapping, so a headless `ZO_EFFORT=max` reaches Anthropic as
    /// `output_config.effort="max"` and reaches GPT through the model-specific
    /// projection (`max` only on confirmed GPT-5.6 families, otherwise `xhigh`)
    /// instead of being re-derived from a thinking budget and landing a tier low.
    ///
    /// `Max` maps to `EffortLevel::Max`. `Ultra` maps to `EffortLevel::Ultra` as
    /// a STATIC pin (no band) — the wire always carries the real top tier
    /// (clamped per-provider same as before). `Smart` maps to
    /// `EffortLevel::Xhigh`, the dynamic band's FLOOR — see [`Self::band_ceiling`]
    /// for the band's top; callers building a wire request must set BOTH.
    #[must_use]
    // Xhigh and Smart deliberately return the identical `Some(L::Xhigh)`
    // today (Smart's floor happens to equal the static Xhigh tier) — kept as
    // separate arms rather than merged because they mean different things
    // (a static pin vs. a dynamic band's floor) and are free to diverge
    // independently later; do not let clippy quietly collapse them.
    #[allow(clippy::match_same_arms)]
    pub const fn level(self) -> Option<api::EffortLevel> {
        use api::EffortLevel as L;
        match self {
            Effort::Off => None,
            Effort::Low => Some(L::Low),
            Effort::Medium => Some(L::Medium),
            Effort::High => Some(L::High),
            Effort::Xhigh => Some(L::Xhigh),
            Effort::Max => Some(L::Max),
            Effort::Ultra => Some(L::Ultra),
            Effort::Smart => Some(L::Xhigh),
        }
    }

    /// The dynamic band's ceiling — `Some(EffortLevel::Ultra)` for
    /// [`Effort::Smart`] only, `None` for every static level (including
    /// [`Effort::Ultra`], which has no band). Callers building a
    /// [`api::MessageRequest`] set `effort: level()` and
    /// `effort_band_ceiling: band_ceiling()`; the wire backends resolve the
    /// band via `api::resolve_effort_band` per request. `None` here is what
    /// keeps every other preset's wire behavior byte-identical to before
    /// dynamic bands existed.
    #[must_use]
    pub const fn band_ceiling(self) -> Option<api::EffortLevel> {
        match self {
            Effort::Smart => Some(api::EffortLevel::Ultra),
            Effort::Off
            | Effort::Low
            | Effort::Medium
            | Effort::High
            | Effort::Xhigh
            | Effort::Max
            | Effort::Ultra => None,
        }
    }

    /// Primary name shown in banners and under the slider.
    #[must_use]
    pub const fn canonical(self) -> &'static str {
        match self {
            Effort::Off => "off",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Xhigh => "xhigh",
            Effort::Max => "max",
            Effort::Ultra => "ultra",
            Effort::Smart => "smart",
        }
    }

    /// Accepted aliases (matched case-insensitively at parse time).
    #[must_use]
    pub const fn aliases(self) -> &'static [&'static str] {
        match self {
            Effort::Off => &["none", "disable"],
            Effort::Medium => &["med"],
            // Legacy spellings — `ultra` itself moved to its own static level
            // (`Effort::Ultra`) and is deliberately NOT an alias here anymore;
            // `ultracode` MUST keep parsing (persisted settings/scripts) and
            // `smartcode`/`uc` are the new/short spellings for the same preset.
            Effort::Smart => &["smartcode", "ultracode", "uc"],
            Effort::Low | Effort::High | Effort::Xhigh | Effort::Max | Effort::Ultra => &[],
        }
    }

    /// Aliases worth advertising in user-facing level tables.
    #[must_use]
    pub const fn display_aliases(self) -> &'static [&'static str] {
        match self {
            Effort::Max => &[],
            other => other.aliases(),
        }
    }

    /// One-line hint for the levels table and the slider.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Effort::Off => "no extended thinking",
            Effort::Low => "quick answers, short reasoning",
            Effort::Medium => "balanced reasoning",
            Effort::High => "deep reasoning, longer turns",
            Effort::Xhigh => "extended thinking + bigger budget",
            Effort::Max => "maximum thinking budget",
            Effort::Ultra => "true top-tier reasoning, no orchestration",
            // Deliberately arrow-free: this string is always shown in the
            // full levels table (`format_effort_levels`), including when
            // rendering a status banner for a DIFFERENT level — a "→" here
            // would make any "no arrow" assertion about that banner false
            // regardless of which level is actually active.
            Effort::Smart => "dynamic top band, xhigh up to ceiling, + parallel agent orchestration",
        }
    }

    /// Resolve a level from a token (canonical name or alias),
    /// case-insensitively. Returns `None` for unrecognized input.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Effort> {
        Self::ALL.iter().copied().find(|level| {
            token.eq_ignore_ascii_case(level.canonical())
                || level
                    .aliases()
                    .iter()
                    .any(|alias| token.eq_ignore_ascii_case(alias))
        })
    }

    /// Resolve the level whose budget matches `budget` exactly. A `None`
    /// or zero budget maps to [`Effort::Off`]; any other unmatched value
    /// returns `None` (the caller treats it as a custom budget).
    #[must_use]
    pub fn from_budget(budget: Option<u32>) -> Option<Effort> {
        match budget {
            None | Some(0) => Some(Effort::Off),
            Some(value) => Self::ALL
                .iter()
                .copied()
                .find(|level| level.budget() == value),
        }
    }

    /// For [`Effort::Smart`] only: the dynamic band's floor/ceiling display
    /// labels as actually projected onto `model` via
    /// `api::effective_effort_for_model` — e.g. `("xhigh", "ultra")` on sol,
    /// `("xhigh", "max")` on fable, or a degenerate `("xhigh", "xhigh")` /
    /// `("high", "high")` where the model's ceiling collapses the whole band
    /// onto one rung. `None` for every other level (no band to show).
    ///
    /// Truth surfaces (`hud::effort_badge_label`, `/effort show`) use this
    /// instead of the single-tier clamp check every other preset gets, since
    /// Smart's `level()` is only the band FLOOR — showing just that would
    /// silently hide the escalation headroom the preset actually has.
    #[must_use]
    pub fn band_labels_for_model(self, model: &str) -> Option<(&'static str, &'static str)> {
        if self != Effort::Smart {
            return None;
        }
        let floor = self.level()?;
        let ceiling = self.band_ceiling()?;
        // Mirror the model-capability selection path in two steps: (1) resolve
        // the band the same way the backends do (`api::resolve_effort_band`,
        // which already caps against the model's internal ceiling — e.g. Max
        // for Luna); (2) run the result through the same model-specific
        // capability projection every other level's display uses
        // (`api::effective_effort_for_model`) — this is what applies e.g.
        // Anthropic's sonnet/haiku "no xhigh" clamp on top of the band pick,
        // so a floor of Xhigh on Sonnet correctly displays as "high".
        let no_signals = api::BandDifficulty::default();
        let max_signals = api::BandDifficulty {
            heavy_intent: true,
            large_context: true,
            long_ask: true,
        };
        let floor_resolved = api::resolve_effort_band(floor, ceiling, model, no_signals);
        let ceiling_resolved = api::resolve_effort_band(floor, ceiling, model, max_signals);
        let floor_label = effort_level_label(api::effective_effort_for_model(floor_resolved, model));
        let ceiling_label =
            effort_level_label(api::effective_effort_for_model(ceiling_resolved, model));
        Some((floor_label, ceiling_label))
    }
}

/// Generic lowercase display token for a provider-neutral `api::EffortLevel`,
/// used for band-range display (`Effort::band_labels_for_model`,
/// `hud::effort_badge_label`, `/effort show`). Unlike
/// [`api::EffortLevel::anthropic`], this does NOT clamp `Ultra` down to
/// `xhigh` — callers use it only on a level already projected through
/// `api::effective_effort_for_model`, so `Ultra` here names the internal Zo
/// tier the model exposes. The GPT serializer may encode it as wire `xhigh`.
#[must_use]
pub fn effort_level_label(level: api::EffortLevel) -> &'static str {
    use api::EffortLevel as L;
    match level {
        L::Low => "low",
        L::Medium => "medium",
        L::High => "high",
        L::Xhigh => "xhigh",
        L::Max => "max",
        L::Ultra => "ultra",
    }
}

/// One stop on the effort slider. Order in [`EFFORT_STEPS`] is the
/// left-to-right order shown in the gradient bar.
#[derive(Clone, Copy, Debug)]
pub struct EffortStep {
    /// Canonical label rendered under the gradient bar.
    pub label: &'static str,
    /// Thinking-token budget applied when this step is selected.
    pub budget: u32,
    /// Short one-line description shown beneath the selected step.
    pub hint: &'static str,
}

impl EffortStep {
    const fn from_effort(effort: Effort) -> Self {
        Self {
            label: effort.canonical(),
            budget: effort.budget(),
            hint: effort.description(),
        }
    }
}

/// Seven-stop effort scale, ordered from `Faster` (low budget) to
/// `Smarter` (full smart band). Derived from [`Effort`] so the slider,
/// the slash presets, and the budget map never drift apart. `Off` is
/// intentionally omitted — the slider's leftmost stop is `low`.
pub const EFFORT_STEPS: &[EffortStep] = &[
    EffortStep::from_effort(Effort::Low),
    EffortStep::from_effort(Effort::Medium),
    EffortStep::from_effort(Effort::High),
    EffortStep::from_effort(Effort::Xhigh),
    EffortStep::from_effort(Effort::Max),
    EffortStep::from_effort(Effort::Ultra),
    EffortStep::from_effort(Effort::Smart),
];

/// Resolve the slider position for an externally-set budget, snapping
/// to the numerically closest step's budget. Returns `0` below the
/// minimum. Step order is by capability (`smart` last) and now also by
/// budget — `ultra` (26000) sits above `max` (24000), and `smart` (28000)
/// sits above `ultra` — so a very large custom budget snaps to the
/// rightmost stop.
#[must_use]
pub fn step_for_budget(budget: Option<u32>) -> usize {
    let Some(value) = budget else {
        return 0;
    };
    EFFORT_STEPS
        .iter()
        .enumerate()
        .min_by_key(|(_, step)| step.budget.abs_diff(value))
        .map_or(0, |(i, _)| i)
}

/// Interactive effort-level slider modal.
#[derive(Debug, Clone)]
pub struct EffortPickerModal {
    cursor: usize,
}

impl Default for EffortPickerModal {
    fn default() -> Self {
        Self::new()
    }
}

impl EffortPickerModal {
    /// Construct a modal pre-positioned on the first step (`low`).
    #[must_use]
    pub const fn new() -> Self {
        Self { cursor: 0 }
    }

    /// Construct a modal pre-positioned on the step that best matches
    /// the supplied budget.
    #[must_use]
    pub fn with_budget(budget: Option<u32>) -> Self {
        Self {
            cursor: step_for_budget(budget),
        }
    }

    /// Current cursor index (0-based).
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Step under the cursor.
    #[must_use]
    pub fn current(&self) -> EffortStep {
        EFFORT_STEPS[self.cursor]
    }

    /// Move the cursor one step left, clamped at the first step.
    pub fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Move the cursor one step right, clamped at the last step.
    pub fn move_right(&mut self) {
        if self.cursor + 1 < EFFORT_STEPS.len() {
            self.cursor += 1;
        }
    }

    /// Jump to the first step.
    pub fn jump_to_first(&mut self) {
        self.cursor = 0;
    }

    /// Jump to the last step.
    pub fn jump_to_last(&mut self) {
        self.cursor = EFFORT_STEPS.len().saturating_sub(1);
    }

    /// Handle one key event. Returns `Some(...)` when the modal should
    /// close (selection or cancellation).
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        // Ctrl-C closes the modal without committing.
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(ModalResult::Cancelled);
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Left | KeyCode::Char('h') => {
                self.move_left();
                None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.move_right();
                None
            }
            KeyCode::Home => {
                self.jump_to_first();
                None
            }
            KeyCode::End => {
                self.jump_to_last();
                None
            }
            // Number shortcuts 1..7 jump directly to the matching step.
            KeyCode::Char(ch @ '1'..='7') => {
                let idx = (ch as u8 - b'1') as usize;
                if idx < EFFORT_STEPS.len() {
                    self.cursor = idx;
                }
                None
            }
            KeyCode::Enter => {
                let step = self.current();
                Some(ModalResult::Selected(ModalSelection::Effort {
                    label: step.label.to_owned(),
                    budget: step.budget,
                }))
            }
            _ => None,
        }
    }

    /// Build the renderable lines used by both [`Self::draw`] and the
    /// unit tests. The vertical layout is:
    ///
    /// ```text
    ///   ┌─────────────────────────────────────────────┐
    ///   │ Faster                              Smarter │   ← row 0
    ///   │ ░░░▒▒▒▓▓▓███▓▓▓▒▒▒░░░ ← gradient bar        │   ← row 1
    ///   │ low  medium high xhigh max ultra   smart    │   ← row 2
    ///   │                                  ▲          │   ← row 3 (caret)
    ///   │                              ultra          │   ← row 4 (current label)
    ///   │            true top-tier reasoning, no orchestration │ ← row 5 (hint)
    ///   │ ←/→ adjust · Enter confirm · Esc cancel     │   ← row 6
    ///   └─────────────────────────────────────────────┘
    /// ```
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme, width: u16) -> Vec<Line<'a>> {
        let bar_width = width.max(EFFORT_STEPS.len() as u16 * 4);
        let step_positions = step_centers(bar_width as usize, EFFORT_STEPS.len());

        let mut lines = Vec::new();
        lines.push(faster_smarter_line(bar_width as usize, theme));
        lines.push(gradient_bar_line(bar_width as usize, self.cursor, theme));
        lines.push(step_labels_line(
            &step_positions,
            bar_width as usize,
            self.cursor,
            theme,
        ));
        lines.push(caret_line(
            &step_positions,
            bar_width as usize,
            self.cursor,
            theme,
        ));
        lines.push(selected_label_line(self.current(), theme));
        lines.push(selected_hint_line(self.current(), theme));
        lines.push(Line::from(""));
        lines.push(footer_line(theme));
        lines
    }

    /// Draw the modal into `area`.
    ///
    /// A rounded border tinted with the zo accent, an accent title,
    /// and 2-col / 1-row interior padding keep the slider clear of the
    /// frame edge. The padded inner width is what [`Self::render_lines`]
    /// lays the gradient bar out against.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(" /effort ", theme.typography.heading_1))
            .padding(Padding::symmetric(2, 1))
            .render(frame, area);
        let lines = self.render_lines(theme, inner.width);
        let paragraph = Paragraph::new(lines).style(theme.typography.body);
        frame.render_widget(paragraph, inner);
    }
}

// ============================================================================
// Layout helpers (kept module-private to keep `EffortPickerModal` thin)
// ============================================================================

/// Pixel-perfect "Faster" left, "Smarter" right header row.
fn faster_smarter_line(width: usize, theme: &Theme) -> Line<'static> {
    let label_style = theme.typography.dim;
    let faster = "Faster";
    let smarter = "Smarter";
    let pad = width.saturating_sub(faster.len() + smarter.len());
    let spacer = " ".repeat(pad.max(1));
    Line::from(vec![
        Span::styled(faster, label_style),
        Span::raw(spacer),
        Span::styled(smarter, label_style),
    ])
}

/// Render the gradient bar as a row of filled blocks. The cell at the
/// cursor position is brightened so the user can see the active step
/// even on terminals that strip RGB colors.
fn gradient_bar_line(width: usize, cursor: usize, theme: &Theme) -> Line<'static> {
    let mut spans = Vec::with_capacity(width);
    let step_count = EFFORT_STEPS.len();
    let active_segment_start = (cursor * width) / step_count;
    let active_segment_end = ((cursor + 1) * width) / step_count;
    for x in 0..width {
        let in_active = x >= active_segment_start && x < active_segment_end;
        let t = if width > 1 {
            x as f32 / (width - 1) as f32
        } else {
            0.0
        };
        let color = zo_gradient(t, in_active, theme);
        let glyph = if in_active { '█' } else { '▓' };
        spans.push(Span::styled(glyph.to_string(), Style::default().fg(color)));
    }
    Line::from(spans)
}

/// Lay each step label centered under its slot in the gradient bar.
fn step_labels_line(
    centers: &[usize],
    width: usize,
    cursor: usize,
    theme: &Theme,
) -> Line<'static> {
    let mut row = vec![' '; width];
    for (idx, (center, step)) in centers.iter().zip(EFFORT_STEPS.iter()).enumerate() {
        let label = step.label;
        let start = center.saturating_sub(label.len() / 2);
        for (offset, ch) in label.chars().enumerate() {
            let pos = start + offset;
            if pos < row.len() {
                row[pos] = ch;
            }
        }
        // Reserve color hints in a follow-up styled span; we render the
        // raw row then re-color the active segment.
        let _ = idx;
    }
    // Two passes: first the underlying row, then we recolor the active
    // label by overlaying styled spans at its exact position.
    let raw: String = row.iter().collect();
    let active_step = EFFORT_STEPS[cursor];
    let active_label = active_step.label;
    let active_center = centers[cursor];
    let active_start = active_center.saturating_sub(active_label.len() / 2);
    let active_end = (active_start + active_label.len()).min(width);

    let mut spans = Vec::with_capacity(3);
    if active_start > 0 {
        spans.push(Span::styled(
            raw[..active_start].to_string(),
            theme.typography.dim,
        ));
    }
    spans.push(Span::styled(
        raw[active_start..active_end].to_string(),
        Style::default()
            .fg(theme.palette.bright)
            .add_modifier(Modifier::BOLD),
    ));
    if active_end < raw.len() {
        spans.push(Span::styled(
            raw[active_end..].to_string(),
            theme.typography.dim,
        ));
    }
    Line::from(spans)
}

/// Render a triangular caret directly under the selected step.
fn caret_line(centers: &[usize], width: usize, cursor: usize, theme: &Theme) -> Line<'static> {
    let mut row = vec![' '; width];
    if let Some(&center) = centers.get(cursor) {
        if center < row.len() {
            row[center] = '▲';
        }
    }
    let raw: String = row.iter().collect();
    Line::from(Span::styled(
        raw,
        Style::default()
            .fg(theme.palette.bright)
            .add_modifier(Modifier::BOLD),
    ))
}

fn selected_label_line(step: EffortStep, theme: &Theme) -> Line<'static> {
    let bright = Style::default()
        .fg(theme.palette.bright)
        .add_modifier(Modifier::BOLD);
    let budget = crate::util::format_thousands(step.budget);
    Line::from(vec![
        Span::styled(step.label.to_string(), bright),
        Span::styled(
            format!("   {budget} tokens"),
            Style::default().fg(theme.palette.dim),
        ),
    ])
}

fn selected_hint_line(step: EffortStep, theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(step.hint.to_string(), theme.typography.dim))
}

fn footer_line(theme: &Theme) -> Line<'static> {
    super::key_hint_footer(
        theme,
        &[("←/→", "adjust"), ("Enter", "confirm"), ("Esc", "cancel")],
    )
}

/// Compute the center column for each step inside a bar of `width`.
fn step_centers(width: usize, step_count: usize) -> Vec<usize> {
    if step_count == 0 || width == 0 {
        return Vec::new();
    }
    (0..step_count)
        .map(|idx| ((2 * idx + 1) * width) / (2 * step_count))
        .collect()
}

/// Zo ignition gradient sourced entirely from the theme's semantic heat ramp.
/// The active segment advances two stops toward the spark crest; neutral themes
/// already carry a reset-only ramp, so no color-mode branch is needed here.
fn zo_gradient(t: f32, active: bool, theme: &Theme) -> ratatui::style::Color {
    let ramp = &theme.heat().ignition;
    let last = ramp.len().saturating_sub(1);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = (t.clamp(0.0, 1.0) * last as f32) as usize;
    let index = if active {
        index.saturating_add(2).min(last)
    } else {
        index.min(last)
    };
    ramp[index]
}

/// `1234` → `1 234` thin-space grouping for the budget read-out.
#[cfg(test)]
mod tests {
    use super::{EFFORT_STEPS, EffortPickerModal, step_centers, step_for_budget};
    use crate::tui::Theme;
    use crate::tui::modals::{ModalResult, ModalSelection};
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    #[test]
    fn step_centers_distributes_evenly() {
        let centers = step_centers(60, EFFORT_STEPS.len());
        assert_eq!(centers.len(), EFFORT_STEPS.len());
        for window in centers.windows(2) {
            assert!(window[1] > window[0], "centers must be strictly increasing");
        }
    }

    #[test]
    fn left_right_clamps_at_ends() {
        let mut modal = EffortPickerModal::new();
        modal.move_left();
        assert_eq!(modal.cursor(), 0);
        for _ in 0..20 {
            modal.move_right();
        }
        assert_eq!(modal.cursor(), EFFORT_STEPS.len() - 1);
    }

    #[test]
    fn enter_emits_selected_effort_with_label_and_budget() {
        let mut modal = EffortPickerModal::new();
        modal.cursor = 6; // smart (last step)
        let result = modal.handle_key(press(KeyCode::Enter));
        match result {
            Some(ModalResult::Selected(ModalSelection::Effort { label, budget })) => {
                assert_eq!(label, "smart");
                assert_eq!(budget, 28_000);
            }
            other => panic!("expected Effort selection, got {other:?}"),
        }
    }

    #[test]
    fn enter_emits_ultra_with_its_own_budget() {
        let mut modal = EffortPickerModal::new();
        modal.cursor = 5; // ultra
        let result = modal.handle_key(press(KeyCode::Enter));
        match result {
            Some(ModalResult::Selected(ModalSelection::Effort { label, budget })) => {
                assert_eq!(label, "ultra");
                assert_eq!(budget, 26_000);
            }
            other => panic!("expected Effort selection, got {other:?}"),
        }
    }

    #[test]
    fn esc_cancels() {
        let mut modal = EffortPickerModal::new();
        let result = modal.handle_key(press(KeyCode::Esc));
        assert!(matches!(result, Some(ModalResult::Cancelled)));
    }

    #[test]
    fn number_shortcut_jumps_to_step() {
        let mut modal = EffortPickerModal::new();
        let _ = modal.handle_key(press(KeyCode::Char('4')));
        assert_eq!(modal.cursor(), 3); // '4' → index 3 (xhigh)
        assert_eq!(modal.current().label, "xhigh");
    }

    #[test]
    fn step_for_budget_snaps_to_closest() {
        assert_eq!(step_for_budget(None), 0);
        assert_eq!(step_for_budget(Some(0)), 0);
        assert_eq!(step_for_budget(Some(5_000)), 1); // closer to medium(4096)
        assert_eq!(step_for_budget(Some(22_000)), 4); // closer to max(24000)
        // Budgets are now monotonically increasing across all seven steps —
        // ultra's 26000 sits above max's 24000, and smart's 28000 sits above
        // ultra's — so a very large custom budget snaps to the rightmost stop
        // instead of falling back to max.
        assert_eq!(step_for_budget(Some(26_000)), 5); // exact ultra budget
        assert_eq!(step_for_budget(Some(28_000)), 6); // exact smart budget
        assert_eq!(step_for_budget(Some(50_000)), 6); // nearest preset is smart(28000)
    }

    #[test]
    fn step_for_budget_boundary_between_ultra_and_smart() {
        // Reverse-lookup boundary tests for the new Ultra rung (26000), which
        // sits exactly between Max (24000) and Smart (28000).
        //
        // 25000 is a dead tie in distance between max(24000, diff 1000) and
        // ultra(26000, diff 1000); `min_by_key` keeps the FIRST minimum in
        // iteration order, and max precedes ultra in EFFORT_STEPS, so the tie
        // resolves to max.
        assert_eq!(step_for_budget(Some(25_000)), 4); // tie -> first (max)
        assert_eq!(step_for_budget(Some(25_001)), 5); // ultra wins past the tie
        // 27000 is the symmetric tie one rung up: ultra(26000, diff 1000) vs
        // smart(28000, diff 1000) — ultra precedes smart, so it wins the tie.
        assert_eq!(step_for_budget(Some(27_000)), 5); // tie -> first (ultra)
        assert_eq!(step_for_budget(Some(27_001)), 6); // smart wins past the tie
    }

    #[test]
    fn render_lines_contains_caret_under_active_step() {
        let mut modal = EffortPickerModal::new();
        modal.cursor = 6; // smart (last step)
        let theme = Theme::zo();
        let lines = modal.render_lines(&theme, 60);
        let caret_row: String = lines[3]
            .spans
            .iter()
            .flat_map(|s| s.content.chars())
            .collect();
        assert!(
            caret_row.contains('▲'),
            "caret row missing triangle marker: {caret_row:?}"
        );
        // The caret should sit in the right half (smart = last step).
        let caret_pos = caret_row.find('▲').expect("triangle present");
        assert!(
            caret_pos > 30,
            "caret expected on right half, got col {caret_pos}"
        );
    }

    #[test]
    fn level_maps_each_effort_to_its_wire_tier() {
        use super::Effort;
        use api::EffortLevel as L;
        assert_eq!(Effort::Off.level(), None);
        assert_eq!(Effort::Low.level(), Some(L::Low));
        assert_eq!(Effort::Medium.level(), Some(L::Medium));
        assert_eq!(Effort::High.level(), Some(L::High));
        assert_eq!(Effort::Xhigh.level(), Some(L::Xhigh));
        // Max maps to EffortLevel::Max. Anthropic preserves it; GPT preserves
        // it only for confirmed max-capable families (currently GPT-5.6),
        // otherwise projecting to xhigh.
        assert_eq!(Effort::Max.level(), Some(L::Max));
        // Ultra is a STATIC pin on the real provider-neutral Ultra tier — no
        // band, no orchestration hint, just always the true top tier.
        assert_eq!(Effort::Ultra.level(), Some(L::Ultra));
        assert_eq!(Effort::Ultra.band_ceiling(), None);
        // Smart carries the DYNAMIC band's floor (Xhigh) as its named level,
        // plus a ceiling of Ultra — the wire backends resolve the concrete
        // per-request rung via `api::resolve_effort_band`.
        assert_eq!(Effort::Smart.level(), Some(L::Xhigh));
        assert_eq!(Effort::Smart.band_ceiling(), Some(L::Ultra));
        // No other level carries a band.
        for &effort in Effort::ALL {
            if effort != Effort::Smart {
                assert_eq!(
                    effort.band_ceiling(),
                    None,
                    "{effort:?} must not carry a dynamic band"
                );
            }
        }
    }

    #[test]
    fn effort_budget_round_trips_through_effort_level_for_budget() {
        use super::Effort;
        // The api crate's budget→level fallback (used at every wire-building site
        // that only carries a thinking budget) must be the exact inverse of these
        // preset budgets, or a headless `ZO_EFFORT=max` selects a tier or two
        // low before provider projection. If `Effort::budget()` or
        // `api::effort_level_for_budget` drift apart, this fails loudly instead
        // of silently under-clocking.
        //
        // `Ultra` and `Smart` are deliberate exceptions: both budgets (26_000,
        // 28_000) land in `effort_level_for_budget`'s open-ended `Max` bucket
        // (it has no Ultra bucket), which matches neither Ultra's own named
        // level (`Ultra`) nor Smart's (the band floor, `Xhigh`) — so both are
        // asserted directly instead of round-tripped through the budget map.
        for &effort in Effort::ALL {
            match effort {
                Effort::Off => assert_eq!(effort.budget(), 0, "only Off has no wire effort"),
                Effort::Ultra => {
                    assert_eq!(effort.level(), Some(api::EffortLevel::Ultra));
                    assert_eq!(effort.budget(), 26_000);
                }
                Effort::Smart => {
                    assert_eq!(effort.level(), Some(api::EffortLevel::Xhigh));
                    assert_eq!(effort.budget(), 28_000);
                }
                _ => {
                    let level = effort
                        .level()
                        .expect("every non-Off/Ultra/Smart level carries a wire effort");
                    assert_eq!(
                        api::effort_level_for_budget(effort.budget()),
                        level,
                        "{effort:?} (budget {}) must map back to its own wire tier",
                        effort.budget(),
                    );
                }
            }
        }
    }

    #[test]
    fn benchmark_effort_label_reaches_wire_faithfully() {
        use super::Effort;
        use api::EffortLevel as L;
        let matrix = [
            ("off", None),
            ("none", None),
            ("low", Some(L::Low)),
            ("medium", Some(L::Medium)),
            ("med", Some(L::Medium)),
            ("HIGH", Some(L::High)),
            ("xhigh", Some(L::Xhigh)),
            ("max", Some(L::Max)),
            // `ultra` is now its own static top-tier pin (second meaning
            // change for this token: P0 made it an alias for the old
            // Ultracode preset; it is now `Effort::Ultra` itself).
            ("ultra", Some(L::Ultra)),
            // `smart`/`smartcode`/`ultracode`/`uc` all resolve to Smart, whose
            // named level is the band FLOOR (Xhigh), not Ultra — the ceiling
            // lives in `band_ceiling()` and is resolved per-request on the wire.
            ("smart", Some(L::Xhigh)),
            ("smartcode", Some(L::Xhigh)),
            ("ultracode", Some(L::Xhigh)),
            ("uc", Some(L::Xhigh)),
        ];
        for (label, expected) in matrix {
            let effort = Effort::from_token(label)
                .unwrap_or_else(|| panic!("ZO_EFFORT={label:?} should parse"));
            assert_eq!(effort.level(), expected, "ZO_EFFORT={label:?}");
        }
        assert_eq!(Effort::Ultra.budget(), 26_000);
        assert_eq!(Effort::Smart.budget(), 28_000);
        assert_eq!(Effort::Max.budget(), 24_000);
    }

    #[test]
    fn ultracode_persisted_setting_still_parses_as_smart() {
        use super::Effort;
        // Persisted settings/scripts that wrote the old preset name must keep
        // working after the Ultracode -> Smart rename.
        assert_eq!(Effort::from_token("ultracode"), Some(Effort::Smart));
        assert_eq!(Effort::from_token("ULTRACODE"), Some(Effort::Smart));
    }

    #[test]
    fn band_labels_for_model_ranges_per_provider_ceiling() {
        use super::Effort;
        // Sol/terra: the full 3-rung internal band reaches Ultra.
        assert_eq!(
            Effort::Smart.band_labels_for_model("gpt-5.6-sol"),
            Some(("xhigh", "ultra"))
        );
        // Fable/Luna: the internal selection ceiling tops out at Max.
        assert_eq!(
            Effort::Smart.band_labels_for_model("claude-fable-5"),
            Some(("xhigh", "max"))
        );
        assert_eq!(
            Effort::Smart.band_labels_for_model("gpt-5.6-luna"),
            Some(("xhigh", "max"))
        );
        // Legacy GPT (xhigh-only ceiling): both ends collapse onto xhigh.
        assert_eq!(
            Effort::Smart.band_labels_for_model("gpt-5.5"),
            Some(("xhigh", "xhigh"))
        );
        // Gemini: caps hard at `high` (its own ceiling is below xhigh), so
        // both ends collapse onto high.
        assert_eq!(
            Effort::Smart.band_labels_for_model("gemini-3.5-flash"),
            Some(("high", "high"))
        );
        // Sonnet: no `xhigh` wire tier, but `max` IS in its supported set — so
        // the floor clamps to `high` while a heavy-signal turn still reaches
        // `max`, a genuine (not degenerate) [high..max] band.
        assert_eq!(
            Effort::Smart.band_labels_for_model("claude-sonnet-5"),
            Some(("high", "max"))
        );
        // Every other level has no band.
        assert_eq!(Effort::Max.band_labels_for_model("gpt-5.6-sol"), None);
        assert_eq!(Effort::Ultra.band_labels_for_model("gpt-5.6-sol"), None);
        assert_eq!(Effort::Off.band_labels_for_model("gpt-5.6-sol"), None);
    }
}
