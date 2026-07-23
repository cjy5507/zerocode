//! Theme loader for `.zo/design/tokens.json`.
//!
//! The design tokens JSON is the source of truth for colors,
//! typography, spacing, border usage, and breakpoints. This module
//! parses a subset of the schema into strongly-typed structs that the
//! widget layer (Lane L5) consumes via `&Theme`.
//!
//! Honors `code-rules.md`:
//! * R2 — never emit ANSI escapes; expose `ratatui::style::Style`.
//! * R9 — every renderable decision reads through `&Theme`.
//! * R10 — honor the `NO_COLOR` env via [`Theme::no_color`].

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::BorderType;
use serde::Deserialize;

use super::TuiError;

// ============================================================================
// Parsed shape of the on-disk tokens JSON (subset we consume in L2)
// ============================================================================

#[derive(Debug, Deserialize)]
struct TokensFile {
    color: TokensColor,
    // `spacing` and `breakpoint` carry sensible fallbacks, so a tokens file
    // may omit either section (or individual keys within them) without failing
    // to load — only `color` is genuinely required to define a theme.
    #[serde(default)]
    spacing: TokensSpacing,
    #[serde(default)]
    breakpoint: TokensBreakpoint,
    // Omitting `border_usage` yields no border overrides, matching the
    // "partial override" contract above rather than failing to load.
    #[serde(default)]
    border_usage: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TokensColor {
    primary: BTreeMap<String, TokenColorEntry>,
    secondary: BTreeMap<String, TokenColorEntry>,
    neutral: BTreeMap<String, TokenColorEntry>,
    semantic: BTreeMap<String, TokenColorEntry>,
}

#[derive(Debug, Deserialize)]
struct TokenColorEntry {
    /// 24-bit hex (`#RRGGBB`) — preferred when present so the on-disk
    /// tokens can carry the true-color Zo palette losslessly.
    #[serde(default)]
    hex: Option<String>,
    /// ANSI-256 index — used as the fallback for terminals/tokens that
    /// predate true-color support.
    #[serde(default)]
    ansi256: Option<u8>,
}

// A missing individual key falls back to the matching field here (see the
// container `#[serde(default)]`), so a partial `[spacing]` section is valid.
// Defaults mirror [`Spacing::fallback`].
#[derive(Debug, Deserialize)]
#[serde(default)]
struct TokensSpacing {
    row_gap: u16,
    block_gap: u16,
    card_padding_x: u16,
    card_padding_y: u16,
    indent: u16,
    gutter: u16,
    hud_sep: u16,
    modal_padding_x: u16,
    modal_padding_y: u16,
}

impl Default for TokensSpacing {
    fn default() -> Self {
        Self {
            row_gap: 1,
            block_gap: 1,
            card_padding_x: 1,
            card_padding_y: 0,
            indent: 2,
            gutter: 2,
            hud_sep: 1,
            modal_padding_x: 2,
            modal_padding_y: 1,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct TokensBreakpoint {
    narrow: TokensBreakpointRange,
    compact: TokensBreakpointRange,
    wide: TokensBreakpointRange,
}

#[derive(Debug, Deserialize, Default)]
struct TokensBreakpointRange {
    #[serde(default)]
    max: Option<u16>,
    #[serde(default)]
    min: Option<u16>,
}

// ============================================================================
// Public theme shape
// ============================================================================

/// Terminal-friendly palette resolved from the tokens file.
///
/// Every field is a `ratatui::style::Color`. Values default to
/// `Color::Reset` when the tokens file does not carry an `ansi256`
/// index for a given role; this keeps the TUI renderable on minimal
/// terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    /// Brand accent (prompt glyphs, focus underlines, primary borders).
    pub accent: Color,
    /// Dim variant of [`Palette::accent`] for inactive borders.
    pub accent_dim: Color,
    /// Info / model badge / cwd path.
    pub cyan: Color,
    /// Reasoning / MCP indicator.
    pub violet: Color,
    /// Git branch indicator.
    pub teal: Color,
    /// Default body foreground.
    pub fg: Color,
    /// Bold heading foreground.
    pub bright: Color,
    /// Dim body (hints, thinking, secondary HUD fields).
    pub dim: Color,
    /// Muted (borders, separators, collapsed previews).
    pub muted: Color,
    /// Faint (disabled, placeholders).
    pub faint: Color,
    /// Code block background.
    pub code_bg: Color,
    /// `semantic.success`.
    pub success: Color,
    /// `semantic.warn`.
    pub warn: Color,
    /// `semantic.error`.
    pub error: Color,
    /// `semantic.info`.
    pub info: Color,
}

impl Palette {
    /// Requantize every role into `tier`'s color space, leaving the palette's
    /// meaning intact. [`ColorTier::TrueColor`] is identity; `Ansi256`/`Ansi16`
    /// map each color down (see [`quantize_color`]); `NoColor` returns
    /// [`Self::neutral`]. Used by [`Theme::apply_color_tier`] so a resolved
    /// theme renders correctly on the running terminal without touching the
    /// canonical built-in palette.
    #[must_use]
    fn to_tier(self, tier: ColorTier) -> Self {
        match tier {
            ColorTier::TrueColor => self,
            ColorTier::NoColor => Self::neutral(),
            ColorTier::Ansi256 | ColorTier::Ansi16 => Self {
                accent: quantize_color(self.accent, tier),
                accent_dim: quantize_color(self.accent_dim, tier),
                cyan: quantize_color(self.cyan, tier),
                violet: quantize_color(self.violet, tier),
                teal: quantize_color(self.teal, tier),
                fg: quantize_color(self.fg, tier),
                bright: quantize_color(self.bright, tier),
                dim: quantize_color(self.dim, tier),
                muted: quantize_color(self.muted, tier),
                faint: quantize_color(self.faint, tier),
                code_bg: quantize_color(self.code_bg, tier),
                success: quantize_color(self.success, tier),
                warn: quantize_color(self.warn, tier),
                error: quantize_color(self.error, tier),
                info: quantize_color(self.info, tier),
            },
        }
    }

    /// All-neutral palette used when `NO_COLOR` is set or the tokens
    /// file is unreadable. Every color resolves to `Color::Reset`.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            accent: Color::Reset,
            accent_dim: Color::Reset,
            cyan: Color::Reset,
            violet: Color::Reset,
            teal: Color::Reset,
            fg: Color::Reset,
            bright: Color::Reset,
            dim: Color::Reset,
            muted: Color::Reset,
            faint: Color::Reset,
            code_bg: Color::Reset,
            success: Color::Reset,
            warn: Color::Reset,
            error: Color::Reset,
            info: Color::Reset,
        }
    }
}

/// Typographic roles mapped to `ratatui::style::Style`.
///
/// These are the semantic roles referenced by widgets: `body`, `dim`,
/// `bold`, `heading_*`, `placeholder`, `key_hint`. The widget layer
/// never constructs a `Style` directly — it always reads from here.
#[derive(Debug, Clone, Copy)]
pub struct Typography {
    /// Default body text.
    pub body: Style,
    /// Dim body (hints, thinking surface).
    pub dim: Style,
    /// Bold emphasis.
    pub bold: Style,
    /// Italic emphasis.
    pub italic: Style,
    /// `heading_1` — bold, accent.
    pub heading_1: Style,
    /// `heading_2` — bold, cyan.
    pub heading_2: Style,
    /// `heading_3` — bold + underlined, bright.
    pub heading_3: Style,
    /// Placeholder / ghost text.
    pub placeholder: Style,
    /// Dim "press ? for help" style key hints.
    pub key_hint: Style,
}

impl Typography {
    /// Build a typography table from a palette.
    #[must_use]
    pub const fn from_palette(palette: &Palette) -> Self {
        Self {
            body: Style::new().fg(palette.fg),
            dim: Style::new().fg(palette.dim).add_modifier(Modifier::DIM),
            bold: Style::new().fg(palette.bright).add_modifier(Modifier::BOLD),
            italic: Style::new().fg(palette.fg).add_modifier(Modifier::ITALIC),
            heading_1: Style::new().fg(palette.accent).add_modifier(Modifier::BOLD),
            heading_2: Style::new().fg(palette.cyan).add_modifier(Modifier::BOLD),
            heading_3: Style::new()
                .fg(palette.bright)
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED),
            placeholder: Style::new()
                .fg(palette.faint)
                .add_modifier(Modifier::DIM)
                .add_modifier(Modifier::ITALIC),
            key_hint: Style::new().fg(palette.dim).add_modifier(Modifier::DIM),
        }
    }
}

/// Spacing tokens, in terminal cells.
#[derive(Debug, Clone, Copy)]
pub struct Spacing {
    /// Vertical gap between rows of wrapped text.
    pub row_gap: u16,
    /// Vertical gap between adjacent blocks in the transcript.
    pub block_gap: u16,
    /// Horizontal padding inside cards.
    pub card_padding_x: u16,
    /// Vertical padding inside cards.
    pub card_padding_y: u16,
    /// Indent used for nested content.
    pub indent: u16,
    /// Gutter between parallel columns.
    pub gutter: u16,
    /// Separator gap between HUD fields.
    pub hud_sep: u16,
    /// Horizontal padding inside modal cards.
    pub modal_padding_x: u16,
    /// Vertical padding inside modal cards.
    pub modal_padding_y: u16,
}

impl Spacing {
    /// Sensible fallback spacing used when no tokens file is present.
    #[must_use]
    pub const fn fallback() -> Self {
        Self {
            row_gap: 1,
            block_gap: 1,
            card_padding_x: 1,
            card_padding_y: 0,
            indent: 2,
            gutter: 2,
            hud_sep: 1,
            modal_padding_x: 2,
            modal_padding_y: 1,
        }
    }
}

/// Responsive breakpoint (column width bucket).
///
/// Mirrors the `breakpoint` section of the tokens file. The thresholds
/// used by [`Theme::for_width`] are read from the same file, but
/// default to `narrow ≤ 59`, `compact 60..=99`, `wide ≥ 100` if the
/// tokens file is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Breakpoint {
    /// `cols ≤ narrow_max`.
    Narrow,
    /// `narrow_max < cols < wide_min`.
    Compact,
    /// `cols ≥ wide_min`.
    Wide,
}

/// Semantic message role, used to pick a consistent gutter/rail
/// color across the transcript, context grid, and cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The human operator.
    User,
    /// The model.
    Assistant,
    /// System / dispatcher notices.
    System,
    /// Tool calls and their results.
    Tool,
}

/// Callout (admonition) kind parsed from a leading blockquote token
/// such as `> [!NOTE]` or a `Note:` prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalloutKind {
    /// Neutral informational aside.
    Note,
    /// Helpful suggestion.
    Tip,
    /// Caution.
    Warning,
    /// High-priority callout.
    Important,
}

/// Syntax-highlighting role, collapsed from syntect's full scope taxonomy to
/// the small set zo actually colors. syntect resolves each code token to a
/// base16-ocean.dark RGB; classifying by scope into these roles instead lets
/// the color come from the zo [`Palette`] (via [`Theme::syntax_style`]) so a
/// code card reads as the same app as the rest of the TUI, degrades with the
/// terminal (256-color / `NO_COLOR`) for free, and stays within zo's own
/// structural hues — the "1 브랜드 + 5 계조 + semantic ramp" system, restrained
/// per `styles.md` "카드당 최대 3계조".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntaxRole {
    /// Identifiers, numbers, operators, punctuation, variables — body `fg`.
    Plain,
    /// Comments — recede to the quiet `muted` step (italic carries the rest).
    Comment,
    /// Keywords and storage (`fn`/`let`/`if`/`struct`/…) — `violet`.
    Keyword,
    /// String / char / regex literals — `teal`.
    Str,
    /// Declared names: `entity.name.*` + `support.*` (functions, types,
    /// classes, macros) — `cyan`.
    Name,
}

/// The color capability tier a terminal is treated as supporting.
///
/// Ordered from richest to poorest. The renderer never emits ANSI escapes
/// (code-rules R2) — this tier only decides which `ratatui::style::Color`
/// space a resolved palette is quantized into, so a truecolor palette still
/// looks right on a 256-color or 16-color terminal instead of being sent RGB
/// the terminal will mangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorTier {
    /// 24-bit RGB. Palettes render verbatim.
    TrueColor,
    /// xterm 256-color. `Rgb` colors quantize to the nearest ANSI-256 index;
    /// existing `Indexed` colors pass through.
    Ansi256,
    /// Legacy 16-color. Both `Rgb` and `Indexed` colors quantize to the 16
    /// base ANSI slots.
    Ansi16,
    /// `NO_COLOR` — no color at all. Every palette role collapses to
    /// `Color::Reset` (the neutral palette), so weight/underline carry meaning.
    NoColor,
}

impl ColorTier {
    /// Classify a terminal's color capability from the standard environment
    /// signals, conservatively. This is a **pure** function of its inputs —
    /// no global env reads — so it is directly unit-testable without mutating
    /// process state.
    ///
    /// Precedence, highest first:
    /// 1. `NO_COLOR` (non-empty) — always wins, per the `NO_COLOR` spec.
    /// 2. `TERM=dumb` → [`Self::NoColor`]. A dumb terminal has no color *or*
    ///    cursor-addressing capability, so the whole codebase's contract
    ///    (mirroring [`crate::tui::term`] / `TermProfile::plain`) is plain
    ///    ASCII output. This is checked *before* `COLORTERM`, so a stray
    ///    `COLORTERM=truecolor` on a dumb terminal cannot force color on.
    /// 3. `COLORTERM` of `truecolor`/`24bit`, a `TERM` carrying `direct` or
    ///    `truecolor`, a known native-truecolor terminal id (via `TERM` or
    ///    `TERM_PROGRAM`), or Windows Terminal's `WT_SESSION` signal →
    ///    [`Self::TrueColor`].
    /// 4. `TERM` carrying `256color` → [`Self::Ansi256`].
    /// 5. Any other non-empty / empty / unset `TERM` (bare `xterm`, `screen`,
    ///    `linux`, …) → [`Self::Ansi16`]. Unknown terminals are treated as
    ///    16-color rather than assumed to support 256/truecolor.
    #[must_use]
    pub fn detect(
        no_color: bool,
        colorterm: Option<&str>,
        term: Option<&str>,
        term_program: Option<&str>,
    ) -> Self {
        Self::detect_with_native_terminal(no_color, colorterm, term, term_program, false)
    }

    fn detect_with_native_terminal(
        no_color: bool,
        colorterm: Option<&str>,
        term: Option<&str>,
        term_program: Option<&str>,
        windows_terminal: bool,
    ) -> Self {
        if no_color {
            return Self::NoColor;
        }

        let term = term.unwrap_or("");
        let term_lower = term.to_ascii_lowercase();

        // A dumb terminal cannot render color or address the cursor; classify
        // it as NoColor before any COLORTERM/TERM_PROGRAM signal can lift it,
        // matching the plain-ASCII contract the rest of the TUI already honors.
        if term_lower == "dumb" {
            return Self::NoColor;
        }

        let colorterm = colorterm.unwrap_or("");
        if colorterm.eq_ignore_ascii_case("truecolor") || colorterm.eq_ignore_ascii_case("24bit") {
            return Self::TrueColor;
        }

        if term_lower.contains("direct") || term_lower.contains("truecolor") {
            return Self::TrueColor;
        }
        // A known native-truecolor emulator identified by either `TERM` (e.g.
        // `xterm-kitty`) or `TERM_PROGRAM` (e.g. `iTerm.app`, `WezTerm`) — real
        // iTerm2/WezTerm set `TERM=xterm-256color` and carry their identity only
        // in `TERM_PROGRAM`, so consulting `TERM` alone would miss them.
        let term_program_lower = term_program.unwrap_or("").to_ascii_lowercase();
        if windows_terminal
            || is_known_truecolor_term(&term_lower)
            || is_known_truecolor_term(&term_program_lower)
        {
            return Self::TrueColor;
        }
        if term_lower.contains("256color") {
            return Self::Ansi256;
        }

        // Unknown / bare / empty / unset TERM: degrade to the conservative
        // ANSI-16 floor rather than assume a richer tier.
        Self::Ansi16
    }
}

/// Known terminal identifiers that natively speak 24-bit color even when they
/// do not export `COLORTERM=truecolor`. Matched conservatively against a
/// lower-cased `TERM` *or* `TERM_PROGRAM` value; anything not on this list stays
/// on the tier its `TERM` implies (256color / ANSI-16). An empty input never
/// matches. Deliberately excludes non-truecolor programs like `Apple_Terminal`,
/// which are left to the `TERM` 256/16 classification.
fn is_known_truecolor_term(value_lower: &str) -> bool {
    const KNOWN: [&str; 6] = [
        "iterm", "kitty", "wezterm", "alacritty", "ghostty", "contour",
    ];
    if value_lower.is_empty() {
        return false;
    }
    KNOWN.iter().any(|id| value_lower.contains(id))
}

/// Border map keyed by role (e.g. `"input_box"` → `BorderType::Rounded`).
#[derive(Debug, Clone)]
pub struct BorderMap {
    roles: BTreeMap<String, BorderType>,
}

impl BorderMap {
    /// Look up the border type for a role. Returns `BorderType::Plain`
    /// if the role is unknown or the tokens file specified `"none"`.
    #[must_use]
    pub fn for_role(&self, role: &str) -> BorderType {
        self.roles.get(role).copied().unwrap_or(BorderType::Plain)
    }

    /// `true` if `for_role` would return a real border (vs. `"none"`).
    #[must_use]
    pub fn has_border(&self, role: &str) -> bool {
        self.roles.contains_key(role)
    }
}

/// Boot-time Cold Steel / Hot Core colors, precomputed once per theme.
#[derive(Debug, Clone, Copy)]
pub struct HeatTokens {
    /// Cold chrome protagonist.
    pub steel: Color,
    /// Cold secondary for HUD details and separators.
    pub steel_dim: Color,
    /// The zo's live coal; always the palette accent.
    pub ember: Color,
    /// Hottest gradient end and critical-context color.
    pub molten: Color,
    /// Crest highlight for spinners and verb waves.
    pub spark: Color,
    /// Cooling steps from ember to steel, with exact endpoints.
    pub ramp: [Color; 8],
    /// Turn-start ignition wave: ember through molten to the spark crest.
    pub ignition: [Color; 16],
    /// Hot input-rail fade, indexed bottom-up from molten to ember.
    pub rail_fade: [Color; 10],
    /// Hot HUD fill fade from ember to cold secondary steel.
    pub fill_fade: [Color; 12],
}

impl HeatTokens {
    const fn neutral() -> Self {
        Self {
            steel: Color::Reset,
            steel_dim: Color::Reset,
            ember: Color::Reset,
            molten: Color::Reset,
            spark: Color::Reset,
            ramp: [Color::Reset; 8],
            ignition: [Color::Reset; 16],
            rail_fade: [Color::Reset; 10],
            fill_fade: [Color::Reset; 12],
        }
    }

    fn from_palette(theme_name: &str, palette: &Palette, no_color: bool) -> Self {
        if no_color || color_to_rgb(palette.accent).is_none() {
            return Self::neutral();
        }

        let ember = palette.accent;
        let (steel, steel_dim, molten, spark) = if theme_name == "zo" {
            (
                Color::Rgb(0x7E, 0x96, 0xB8),
                Color::Rgb(0x5C, 0x64, 0x70),
                Color::Rgb(0xFF, 0x7A, 0x45),
                Color::Rgb(0xFF, 0xD9, 0xA0),
            )
        } else {
            let Some(molten) = blend_toward(palette.accent, palette.error, 0.35) else {
                return Self::neutral();
            };
            let Some(spark) = blend_toward(palette.accent, palette.bright, 0.45) else {
                return Self::neutral();
            };
            let Some(steel) = blend_toward(palette.dim, palette.info, 0.25) else {
                return Self::neutral();
            };
            let Some(steel_dim) = blend_toward(palette.muted, palette.info, 0.20) else {
                return Self::neutral();
            };
            (steel, steel_dim, molten, spark)
        };

        let ramp = std::array::from_fn(|index| match index {
            0 => ember,
            7 => steel,
            _ => {
                let step = f32::from(u8::try_from(index).expect("ramp index fits in u8"));
                blend_toward(ember, steel, step / 7.0).unwrap_or(Color::Reset)
            }
        });
        let ignition = std::array::from_fn(|index| match index {
            0 => ember,
            8 => molten,
            15 => spark,
            1..=7 => {
                let step = f32::from(u8::try_from(index).expect("ignition index fits in u8"));
                blend_toward(ember, molten, step / 8.0).unwrap_or(Color::Reset)
            }
            9..=14 => {
                let step =
                    f32::from(u8::try_from(index - 8).expect("ignition index fits in u8"));
                blend_toward(molten, spark, step / 7.0).unwrap_or(Color::Reset)
            }
            _ => unreachable!("ignition ramp has exactly 16 entries"),
        });
        let rail_fade = std::array::from_fn(|index| match index {
            0 => molten,
            9 => ember,
            _ => {
                let step = f32::from(u8::try_from(index).expect("rail index fits in u8"));
                blend_toward(molten, ember, step / 9.0).unwrap_or(Color::Reset)
            }
        });
        let fill_fade = std::array::from_fn(|index| {
            if index == 0 {
                ember
            } else if index == 11 {
                steel_dim
            } else {
                let step = f32::from(u8::try_from(index).expect("fill index fits in u8"));
                blend_toward(ember, steel_dim, step / 11.0).unwrap_or(Color::Reset)
            }
        });

        Self {
            steel,
            steel_dim,
            ember,
            molten,
            spark,
            ramp,
            ignition,
            rail_fade,
            fill_fade,
        }
    }

    /// Requantize every cached heat color into `tier`'s color space.
    ///
    /// The heat cache is built by *blending* palette colors, which can produce
    /// fresh `Rgb`/`Indexed` values even after the palette itself was
    /// quantized (e.g. the hard-coded `"zo"` steel/molten/spark RGB, or an
    /// indexed blend that routes through `rgb_to_ansi256`). Running each cached
    /// color back through [`quantize_color`] guarantees no heat color escapes
    /// its tier — Ansi16 in particular collapses every step to a named color,
    /// so a 16-color terminal never receives a 256-color heat sequence.
    /// `TrueColor` is identity; `NoColor` is already `Reset` from `from_palette`.
    #[must_use]
    fn to_tier(self, tier: ColorTier) -> Self {
        if matches!(tier, ColorTier::TrueColor | ColorTier::NoColor) {
            return self;
        }
        let q = |c: Color| quantize_color(c, tier);
        let qarr = |arr: [Color; 8]| arr.map(q);
        Self {
            steel: q(self.steel),
            steel_dim: q(self.steel_dim),
            ember: q(self.ember),
            molten: q(self.molten),
            spark: q(self.spark),
            ramp: qarr(self.ramp),
            ignition: self.ignition.map(q),
            rail_fade: self.rail_fade.map(q),
            fill_fade: self.fill_fade.map(q),
        }
    }

    /// Evenly spaced ember-to-molten colors for a character-based wordmark.
    ///
    /// A single-character wordmark stays at the ember origin. Neutral themes
    /// return `Color::Reset` for every stop.
    #[must_use]
    pub fn wordmark_gradient(&self, count: usize) -> Vec<Color> {
        if count == 0 {
            return Vec::new();
        }

        let last = count - 1;
        let steps = u16::try_from(last).unwrap_or(u16::MAX);
        (0..count)
            .map(|index| {
                if index == 0 {
                    self.ember
                } else if index == last {
                    self.molten
                } else {
                    let step = u16::try_from(index).unwrap_or(steps);
                    let t = f32::from(step) / f32::from(steps);
                    blend_toward(self.ember, self.molten, t).unwrap_or(Color::Reset)
                }
            })
            .collect()
    }
}

/// The full theme, consumed by widgets via `&Theme`.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Human-readable theme name (e.g. `"dark"`, `"light"`, `"no_color"`).
    pub name: String,
    /// Resolved color palette.
    pub palette: Palette,
    /// Typographic roles.
    pub typography: Typography,
    /// Spacing tokens in cells.
    pub spacing: Spacing,
    /// Border usage map.
    pub borders: BorderMap,
    /// `narrow` breakpoint upper bound (inclusive).
    pub narrow_max: u16,
    /// `wide` breakpoint lower bound (inclusive).
    pub wide_min: u16,
    /// `true` if this theme was constructed in `NO_COLOR` mode.
    pub no_color: bool,
    /// The color tier this theme has been applied for. Every *dynamically*
    /// derived color (heat blends, wordmark gradient, per-agent hues, diff /
    /// surface tints) is requantized into this tier so a lower-tier terminal
    /// never receives an out-of-tier `Rgb`/`Indexed` sequence. Defaults to
    /// [`ColorTier::TrueColor`] for a freshly constructed built-in palette
    /// (the canonical source is true-color / indexed and untouched until an
    /// apply path runs).
    tier: ColorTier,
    /// Boot-time derived Cold Steel / Hot Core color cache, already requantized
    /// into `tier`.
    heat: HeatTokens,
}

/// Eight per-agent identity hues (256-color indices) that tint a sub-agent's
/// name in a fan-out so agents are visually distinguishable at a glance. Chosen
/// in the cool/pink half of the wheel, clear of every built-in theme's status
/// colors (success/warn/error and the running teal all live in the green /
/// amber / red bands), so a per-agent tint never reads as a status — the
/// `agent_hues_avoid_every_builtin_status_color` test enforces this across all
/// themes. [`Theme::agent_color`] keys these off a stable hash of the agent id
/// (not render position), drops to `Reset` under `NO_COLOR`, and yields to the
/// body `fg` on light themes (these hues are tuned for dark backgrounds).
const AGENT_COLORS: [Color; 8] = [
    Color::Indexed(39),  // azure
    Color::Indexed(208), // orange
    Color::Indexed(141), // purple
    Color::Indexed(213), // pink
    Color::Indexed(45),  // cyan
    Color::Indexed(171), // magenta
    Color::Indexed(105), // periwinkle
    Color::Indexed(176), // orchid
];

impl Theme {
    /// Default `narrow_max` used when the tokens file is absent.
    pub const DEFAULT_NARROW_MAX: u16 = 59;
    /// Default `wide_min` used when the tokens file is absent.
    pub const DEFAULT_WIDE_MIN: u16 = 100;

    /// Load a theme from a tokens JSON file and apply the current terminal's
    /// color capability.
    pub fn load(path: &Path) -> Result<Self, TuiError> {
        Self::load_canonical(path).map(Self::for_current_terminal)
    }

    /// Load a theme and apply a detected background before terminal color-tier
    /// quantization. A missing background preserves [`Self::load`] behavior.
    pub fn load_for_terminal(
        path: &Path,
        terminal_background: Option<(u8, u8, u8)>,
    ) -> Result<Self, TuiError> {
        Self::load_canonical(path)
            .map(|theme| theme.for_current_terminal_with_background(terminal_background))
    }

    fn load_canonical(path: &Path) -> Result<Self, TuiError> {
        let raw = fs::read_to_string(path).map_err(|source| TuiError::ThemeRead {
            path: path.display().to_string(),
            source,
        })?;
        let parsed: TokensFile = serde_json::from_str(&raw)?;

        // The tokens file is a *partial override*: any key it omits (or
        // carries with no `hex`/`ansi256`) falls back to the Zo
        // default for that role, never to `Color::Reset`. This keeps the
        // on-disk file the source of truth while guaranteeing a complete,
        // on-brand palette even for sparse token files.
        let base = Self::zo().palette;
        let palette = Palette {
            accent: resolve_color(&parsed.color.primary, "accent", base.accent),
            accent_dim: resolve_color(&parsed.color.primary, "accent_dim", base.accent_dim),
            cyan: resolve_color(&parsed.color.secondary, "cyan", base.cyan),
            violet: resolve_color(&parsed.color.secondary, "violet", base.violet),
            teal: resolve_color(&parsed.color.secondary, "teal", base.teal),
            fg: resolve_color(&parsed.color.neutral, "fg", base.fg),
            bright: resolve_color(&parsed.color.neutral, "bright", base.bright),
            dim: resolve_color(&parsed.color.neutral, "dim", base.dim),
            muted: resolve_color(&parsed.color.neutral, "muted", base.muted),
            faint: resolve_color(&parsed.color.neutral, "faint", base.faint),
            code_bg: resolve_color(&parsed.color.neutral, "code_bg", base.code_bg),
            success: resolve_color(&parsed.color.semantic, "success", base.success),
            warn: resolve_color(&parsed.color.semantic, "warn", base.warn),
            error: resolve_color(&parsed.color.semantic, "error", base.error),
            info: resolve_color(&parsed.color.semantic, "info", base.info),
        };

        let spacing = Spacing {
            row_gap: parsed.spacing.row_gap,
            block_gap: parsed.spacing.block_gap,
            card_padding_x: parsed.spacing.card_padding_x,
            card_padding_y: parsed.spacing.card_padding_y,
            indent: parsed.spacing.indent,
            gutter: parsed.spacing.gutter,
            hud_sep: parsed.spacing.hud_sep,
            modal_padding_x: parsed.spacing.modal_padding_x,
            modal_padding_y: parsed.spacing.modal_padding_y,
        };

        // A tokens file may omit these; fall back to the same thresholds the
        // no-tokens theme uses rather than refusing to load.
        let narrow_max = parsed
            .breakpoint
            .narrow
            .max
            .unwrap_or(Self::DEFAULT_NARROW_MAX);
        let wide_min = parsed
            .breakpoint
            .wide
            .min
            .unwrap_or(Self::DEFAULT_WIDE_MIN);
        // `compact` is parsed for completeness but not stored directly.
        let _ = parsed.breakpoint.compact;

        let borders = BorderMap {
            roles: parsed
                .border_usage
                .into_iter()
                .filter_map(|(role, kind)| parse_border(&kind).map(|bt| (role, bt)))
                .collect(),
        };

        // Build the theme from the *raw* resolved palette, then apply the
        // terminal's color policy through the same apply path built-ins use —
        // so a custom token theme honors `NO_COLOR`, truecolor, ANSI-256, and
        // ANSI-16 exactly like a built-in, and low tiers requantize the palette
        // rather than only collapsing under `NO_COLOR`.
        Ok(Self::from_parts(
            "custom",
            palette,
            spacing,
            borders,
            narrow_max,
            wide_min,
            false,
        ))
    }

    /// Build a neutral theme with no palette and fallback spacing.
    ///
    /// Used when `NO_COLOR` is set or when the tokens file cannot be
    /// found and the TUI is bootstrapped from a test.
    #[must_use]
    pub fn no_color() -> Self {
        let palette = Palette::neutral();
        Self::from_parts(
            "no_color",
            palette,
            Spacing::fallback(),
            BorderMap {
                roles: default_border_roles(),
            },
            Self::DEFAULT_NARROW_MAX,
            Self::DEFAULT_WIDE_MIN,
            true,
        )
    }

    #[must_use]
    pub fn default_dark() -> Self {
        let palette = Palette {
            accent: Color::Indexed(111),
            accent_dim: Color::Indexed(67),
            cyan: Color::Indexed(117),
            violet: Color::Indexed(141),
            teal: Color::Indexed(109),
            fg: Color::Indexed(253),
            bright: Color::Indexed(255),
            dim: Color::Indexed(245),
            muted: Color::Indexed(240),
            faint: Color::Indexed(237),
            code_bg: Color::Indexed(234),
            success: Color::Indexed(114),
            warn: Color::Indexed(179),
            error: Color::Indexed(203),
            info: Color::Indexed(117),
        };
        Self::from_named_palette("dark", palette)
    }

    /// Built-in light theme with high-contrast colors for light
    /// terminal backgrounds.
    #[must_use]
    pub fn default_light() -> Self {
        let palette = Palette {
            accent: Color::Indexed(130),
            accent_dim: Color::Indexed(94),
            cyan: Color::Indexed(25),
            violet: Color::Indexed(91),
            teal: Color::Indexed(30),
            fg: Color::Indexed(235),
            bright: Color::Indexed(232),
            dim: Color::Indexed(243),
            muted: Color::Indexed(248),
            faint: Color::Indexed(252),
            code_bg: Color::Indexed(255),
            success: Color::Indexed(28),
            warn: Color::Indexed(136),
            error: Color::Indexed(160),
            info: Color::Indexed(25),
        };
        Self::from_named_palette("light", palette)
    }

    /// OpenCode-inspired neutral theme: low-chrome surfaces with one
    /// restrained accent for focus and active state.
    #[must_use]
    pub fn opencode() -> Self {
        let palette = Palette {
            // Inspired by OpenCode's orange primary and purple accent,
            // but shifted into a Zo-specific amber/blue pair so the
            // UI feels familiar without copying the source palette.
            accent: Color::Indexed(215),
            accent_dim: Color::Indexed(172),
            cyan: Color::Indexed(111),
            violet: Color::Indexed(141),
            teal: Color::Indexed(108),
            fg: Color::Indexed(252),
            bright: Color::Indexed(255),
            dim: Color::Indexed(245),
            muted: Color::Indexed(238),
            faint: Color::Indexed(236),
            code_bg: Color::Indexed(232),
            success: Color::Indexed(114),
            warn: Color::Indexed(179),
            error: Color::Indexed(167),
            info: Color::Indexed(117),
        };
        Self::from_named_palette("opencode", palette)
    }

    /// **Zo** — the default terminal theme.
    ///
    /// Cold Steel / Hot Core turns the zo metaphor into state: quiet steel
    /// chrome at rest, ember heat while a turn is active, and a deterministic
    /// cooling path back to stillness. The palette remains true-color for
    /// modern terminals; indexed built-ins below stay available as fallbacks.
    #[must_use]
    pub fn zo() -> Self {
        Self::from_named_palette(
            "zo",
            Palette {
                // Ember — the zo's live coal: user rails, active focus, primary badges.
                accent: Color::Rgb(0xF5, 0xA5, 0x24),
                // Dim amber for inactive focus/folded metadata.
                accent_dim: Color::Rgb(0xA8, 0x63, 0x1A),
                // Quiet info blue: links and rare informational callouts only.
                cyan: Color::Rgb(0x8A, 0xB4, 0xF8),
                // Muted reasoning violet, deliberately low-saturation.
                violet: Color::Rgb(0xA7, 0x8B, 0xFA),
                // Soft green-teal for branch/tool provenance.
                teal: Color::Rgb(0x6E, 0xC7, 0xA3),
                // Neutral body text on terminal default background.
                fg: Color::Rgb(0xD4, 0xD4, 0xD4),
                // Brightest emphasis.
                bright: Color::Rgb(0xF5, 0xF5, 0xF5),
                // Secondary labels and HUD details.
                dim: Color::Rgb(0xA3, 0xA3, 0xA3),
                // Muted dividers and folded previews.
                muted: Color::Rgb(0x73, 0x73, 0x73),
                // Hairline rules / inactive rails.
                faint: Color::Rgb(0x3F, 0x3F, 0x46),
                // Near-black code backdrop.
                code_bg: Color::Rgb(0x17, 0x17, 0x17),
                // Calm success green.
                success: Color::Rgb(0x86, 0xEF, 0xAC),
                // Caution gold — same warm family as the brand amber but
                // visibly softer, so warnings never read as the brand accent.
                warn: Color::Rgb(0xD7, 0xAF, 0x5F),
                // Clear but not neon red.
                error: Color::Rgb(0xF8, 0x71, 0x71),
                // Info stays cool and restrained.
                info: Color::Rgb(0x93, 0xC5, 0xFD),
            },
        )
    }

    fn from_named_palette(name: &str, palette: Palette) -> Self {
        Self::from_parts(
            name,
            palette,
            Spacing::fallback(),
            BorderMap {
                roles: default_border_roles(),
            },
            Self::DEFAULT_NARROW_MAX,
            Self::DEFAULT_WIDE_MIN,
            false,
        )
    }

    fn from_parts(
        name: &str,
        palette: Palette,
        spacing: Spacing,
        borders: BorderMap,
        narrow_max: u16,
        wide_min: u16,
        no_color: bool,
    ) -> Self {
        // Built-ins are constructed at their canonical (true-color / indexed)
        // fidelity; the apply path rebuilds via [`Self::with_tier`] when a
        // lower tier is selected.
        Self::with_tier(
            name,
            palette,
            spacing,
            borders,
            narrow_max,
            wide_min,
            no_color,
            if no_color {
                ColorTier::NoColor
            } else {
                ColorTier::TrueColor
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_tier(
        name: &str,
        palette: Palette,
        spacing: Spacing,
        borders: BorderMap,
        narrow_max: u16,
        wide_min: u16,
        no_color: bool,
        tier: ColorTier,
    ) -> Self {
        // The heat cache is derived from the (already tier-quantized) palette,
        // then any dynamically-blended step is requantized into `tier` so a
        // lower-tier terminal never receives an out-of-tier heat color.
        let heat = HeatTokens::from_palette(name, &palette, no_color).to_tier(tier);
        Self {
            name: name.to_string(),
            palette,
            typography: Typography::from_palette(&palette),
            spacing,
            borders,
            narrow_max,
            wide_min,
            no_color,
            tier,
            heat,
        }
    }

    #[must_use]
    pub fn tokyonight() -> Self {
        Self::from_named_palette(
            "tokyonight",
            Palette {
                accent: Color::Indexed(111),
                accent_dim: Color::Indexed(68),
                cyan: Color::Indexed(117),
                violet: Color::Indexed(141),
                teal: Color::Indexed(73),
                fg: Color::Indexed(189),
                bright: Color::Indexed(231),
                dim: Color::Indexed(103),
                muted: Color::Indexed(59),
                faint: Color::Indexed(238),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(120),
                warn: Color::Indexed(215),
                error: Color::Indexed(203),
                info: Color::Indexed(111),
            },
        )
    }

    #[must_use]
    pub fn catppuccin() -> Self {
        Self::from_named_palette(
            "catppuccin",
            Palette {
                accent: Color::Indexed(183),
                accent_dim: Color::Indexed(139),
                cyan: Color::Indexed(159),
                violet: Color::Indexed(183),
                teal: Color::Indexed(158),
                fg: Color::Indexed(253),
                bright: Color::Indexed(231),
                dim: Color::Indexed(250),
                muted: Color::Indexed(243),
                faint: Color::Indexed(238),
                code_bg: Color::Indexed(236),
                success: Color::Indexed(157),
                warn: Color::Indexed(223),
                error: Color::Indexed(210),
                info: Color::Indexed(117),
            },
        )
    }

    #[must_use]
    pub fn gruvbox() -> Self {
        Self::from_named_palette(
            "gruvbox",
            Palette {
                accent: Color::Indexed(214),
                accent_dim: Color::Indexed(172),
                cyan: Color::Indexed(109),
                violet: Color::Indexed(175),
                teal: Color::Indexed(142),
                fg: Color::Indexed(223),
                bright: Color::Indexed(229),
                dim: Color::Indexed(248),
                muted: Color::Indexed(245),
                faint: Color::Indexed(239),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(142),
                warn: Color::Indexed(179),
                error: Color::Indexed(167),
                info: Color::Indexed(109),
            },
        )
    }

    #[must_use]
    pub fn nord() -> Self {
        Self::from_named_palette(
            "nord",
            Palette {
                accent: Color::Indexed(110),
                accent_dim: Color::Indexed(67),
                cyan: Color::Indexed(110),
                violet: Color::Indexed(139),
                teal: Color::Indexed(108),
                fg: Color::Indexed(188),
                bright: Color::Indexed(231),
                dim: Color::Indexed(145),
                muted: Color::Indexed(60),
                faint: Color::Indexed(238),
                code_bg: Color::Indexed(236),
                success: Color::Indexed(108),
                warn: Color::Indexed(215),
                error: Color::Indexed(174),
                info: Color::Indexed(110),
            },
        )
    }

    #[must_use]
    pub fn everforest() -> Self {
        Self::from_named_palette(
            "everforest",
            Palette {
                accent: Color::Indexed(108),
                accent_dim: Color::Indexed(65),
                cyan: Color::Indexed(109),
                violet: Color::Indexed(175),
                teal: Color::Indexed(108),
                fg: Color::Indexed(223),
                bright: Color::Indexed(230),
                dim: Color::Indexed(248),
                muted: Color::Indexed(245),
                faint: Color::Indexed(239),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(108),
                warn: Color::Indexed(214),
                error: Color::Indexed(167),
                info: Color::Indexed(109),
            },
        )
    }

    #[must_use]
    pub fn kanagawa() -> Self {
        Self::from_named_palette(
            "kanagawa",
            Palette {
                accent: Color::Indexed(173),
                accent_dim: Color::Indexed(130),
                cyan: Color::Indexed(109),
                violet: Color::Indexed(176),
                teal: Color::Indexed(108),
                fg: Color::Indexed(223),
                bright: Color::Indexed(230),
                dim: Color::Indexed(249),
                muted: Color::Indexed(102),
                faint: Color::Indexed(238),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(108),
                warn: Color::Indexed(214),
                error: Color::Indexed(167),
                info: Color::Indexed(109),
            },
        )
    }

    #[must_use]
    pub fn dracula() -> Self {
        Self::from_named_palette(
            "dracula",
            Palette {
                accent: Color::Indexed(141),
                accent_dim: Color::Indexed(98),
                cyan: Color::Indexed(117),
                violet: Color::Indexed(141),
                teal: Color::Indexed(84),
                fg: Color::Indexed(253),
                bright: Color::Indexed(231),
                dim: Color::Indexed(248),
                muted: Color::Indexed(60),
                faint: Color::Indexed(238),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(84),
                warn: Color::Indexed(228),
                error: Color::Indexed(210),
                info: Color::Indexed(117),
            },
        )
    }

    /// **Darcula** — `JetBrains`' flagship dark scheme (`IntelliJ` IDEA's
    /// default) in 24-bit true color.
    ///
    /// Distinct from [`Self::dracula`] (the purple/pink "Dracula" theme):
    /// Darcula is the low-saturation `JetBrains` palette — a warm charcoal
    /// `#2B2B2B` editor surface, cool slate-blue `#A9B7C6` body text, and a
    /// *muted* keyword orange `#CC7832` (darker and less saturated than the
    /// Zo amber, so it reads as emphasis without glare). Diff add/remove
    /// use Darcula's own olive-green `#6A8759` and dusty-red `#CC666E` so a
    /// change reads as a calm band, not a stoplight.
    #[must_use]
    pub fn darcula() -> Self {
        Self::from_named_palette(
            "darcula",
            Palette {
                // Darcula keyword orange — the muted emphasis hue.
                accent: Color::Rgb(0xCC, 0x78, 0x32),
                // Dimmed/burnt variant for inactive borders.
                accent_dim: Color::Rgb(0x9E, 0x5C, 0x27),
                // Method-call / link blue.
                cyan: Color::Rgb(0x28, 0x7B, 0xDE),
                // Keyword/constant violet.
                violet: Color::Rgb(0x98, 0x76, 0xAA),
                // Doc-tag teal.
                teal: Color::Rgb(0x5E, 0x9C, 0x8F),
                // Slate-blue body text — the Darcula default foreground.
                fg: Color::Rgb(0xA9, 0xB7, 0xC6),
                // Brightest — headings / emphasis.
                bright: Color::Rgb(0xFF, 0xFF, 0xFF),
                // Secondary text.
                dim: Color::Rgb(0x80, 0x80, 0x80),
                // Tertiary labels / inactive UI.
                muted: Color::Rgb(0x60, 0x63, 0x66),
                // Hairline rules / borders — panel divider grey.
                faint: Color::Rgb(0x3C, 0x3F, 0x41),
                // Editor background — Darcula's signature warm charcoal.
                code_bg: Color::Rgb(0x2B, 0x2B, 0x2B),
                // Darcula string/diff-add olive green.
                success: Color::Rgb(0x6A, 0x87, 0x59),
                // Caution yellow (Darcula's `#FFC66D`).
                warn: Color::Rgb(0xFF, 0xC6, 0x6D),
                // Diff-remove dusty red.
                error: Color::Rgb(0xCC, 0x66, 0x6E),
                // Number/info blue (sibling of `cyan`).
                info: Color::Rgb(0x68, 0x97, 0xBB),
            },
        )
    }

    #[must_use]
    pub fn one_dark() -> Self {
        Self::from_named_palette(
            "one_dark",
            Palette {
                accent: Color::Indexed(75),
                accent_dim: Color::Indexed(32),
                cyan: Color::Indexed(38),
                violet: Color::Indexed(176),
                teal: Color::Indexed(114),
                fg: Color::Indexed(252),
                bright: Color::Indexed(231),
                dim: Color::Indexed(247),
                muted: Color::Indexed(59),
                faint: Color::Indexed(238),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(114),
                warn: Color::Indexed(180),
                error: Color::Indexed(204),
                info: Color::Indexed(75),
            },
        )
    }

    #[must_use]
    pub fn matrix() -> Self {
        Self::from_named_palette(
            "matrix",
            Palette {
                accent: Color::Indexed(46),
                accent_dim: Color::Indexed(34),
                cyan: Color::Indexed(48),
                violet: Color::Indexed(46),
                teal: Color::Indexed(41),
                fg: Color::Indexed(46),
                bright: Color::Indexed(82),
                dim: Color::Indexed(34),
                muted: Color::Indexed(22),
                faint: Color::Indexed(236),
                code_bg: Color::Indexed(232),
                success: Color::Indexed(46),
                warn: Color::Indexed(226),
                error: Color::Indexed(196),
                info: Color::Indexed(48),
            },
        )
    }

    #[must_use]
    pub fn ayu() -> Self {
        Self::from_named_palette(
            "ayu",
            Palette {
                accent: Color::Indexed(215),
                accent_dim: Color::Indexed(172),
                cyan: Color::Indexed(73),
                violet: Color::Indexed(176),
                teal: Color::Indexed(73),
                fg: Color::Indexed(188),
                bright: Color::Indexed(231),
                dim: Color::Indexed(246),
                muted: Color::Indexed(240),
                faint: Color::Indexed(237),
                code_bg: Color::Indexed(235),
                success: Color::Indexed(114),
                warn: Color::Indexed(179),
                error: Color::Indexed(203),
                info: Color::Indexed(73),
            },
        )
    }

    /// Look up a built-in theme by name. Returns `None` for unknown names.
    #[must_use]
    pub fn builtin(name: &str) -> Option<Self> {
        match name {
            "zo" => Some(Self::zo()),
            "dark" => Some(Self::default_dark()),
            "light" => Some(Self::default_light()),
            "opencode" => Some(Self::opencode()),
            "tokyonight" => Some(Self::tokyonight()),
            "catppuccin" => Some(Self::catppuccin()),
            "gruvbox" => Some(Self::gruvbox()),
            "nord" => Some(Self::nord()),
            "everforest" => Some(Self::everforest()),
            "kanagawa" => Some(Self::kanagawa()),
            "dracula" => Some(Self::dracula()),
            // Note the spelling: `darcula` is the IntelliJ scheme, a near-twin
            // of `dracula` above — both are intentional, distinct themes.
            "darcula" => Some(Self::darcula()),
            "one_dark" => Some(Self::one_dark()),
            "matrix" => Some(Self::matrix()),
            "ayu" => Some(Self::ayu()),
            "no_color" => Some(Self::no_color()),
            _ => None,
        }
    }

    /// List all available built-in theme names.
    #[must_use]
    pub fn builtin_names() -> &'static [&'static str] {
        &[
            "zo",
            "dark",
            "light",
            "opencode",
            "tokyonight",
            "catppuccin",
            "gruvbox",
            "nord",
            "everforest",
            "kanagawa",
            "dracula",
            "darcula",
            "one_dark",
            "matrix",
            "ayu",
        ]
    }

    /// Return a copy of this theme with its palette (and all derived typography
    /// / heat / surface values) requantized into `tier`'s color space.
    ///
    /// This is the **apply path**: it produces the theme the renderer should
    /// draw with on the running terminal, and is deliberately separate from the
    /// canonical [`Self::builtin`] / [`Self::zo`] constructors and the
    /// persistence path, so applying a terminal policy never mutates the
    /// original palette that tests and `.zo/design/tokens.json` depend on.
    ///
    /// [`ColorTier::TrueColor`] is an identity transform. `Ansi256`/`Ansi16`
    /// requantize the palette; a lower tier re-quantizes an already-indexed
    /// palette only where a color falls outside that tier's space.
    /// [`ColorTier::NoColor`] yields the neutral (`Color::Reset`) palette and
    /// flags the theme `no_color`, so every downstream accessor degrades for
    /// free — the same collapse [`Self::no_color`] produces.
    #[must_use]
    pub fn apply_color_tier(self, tier: ColorTier) -> Self {
        // TrueColor is identity; skip the rebuild so the canonical palette
        // (and its heat cache) is returned untouched.
        if tier == ColorTier::TrueColor {
            return self;
        }
        let no_color = tier == ColorTier::NoColor;
        let palette = self.palette.to_tier(tier);
        // `with_tier` records the tier on the theme and requantizes the heat
        // cache into it, so every dynamically-derived color (heat blends,
        // wordmark gradient, per-agent hues) also respects the tier.
        Self::with_tier(
            &self.name,
            palette,
            self.spacing,
            self.borders,
            self.narrow_max,
            self.wide_min,
            no_color,
            tier,
        )
    }

    /// Adapt this applied theme to a detected terminal background.
    ///
    /// `None` is an identity operation. A light terminal behind a dark theme
    /// adopts the built-in light *palette* while preserving this theme's
    /// spacing, borders, and breakpoints, so a `tokens.json`'s non-color
    /// overrides are not discarded. On a dark background, low-contrast
    /// foreground roles are brightened with WCAG relative luminance (hue and
    /// saturation preserved) until they clear their target *once rendered in
    /// `target_tier`* — the search measures the tier-quantized color, so an
    /// ANSI-256/16 terminal cannot round a just-passing true-color value back
    /// under the floor. The returned theme is true-color; the caller quantizes
    /// it into `target_tier`.
    #[must_use]
    pub fn apply_terminal_background(
        self,
        background: Option<(u8, u8, u8)>,
        target_tier: ColorTier,
    ) -> Self {
        let Some(background) = background else {
            return self;
        };
        if self.no_color {
            return self;
        }
        if is_light_background(background) {
            if self.has_light_background() {
                return self;
            }
            // Swap only the color palette; keep this theme's spacing, borders,
            // and breakpoints so a loaded tokens file's non-color overrides
            // survive. The caller applies the real color tier afterward.
            return Self::with_tier(
                "light",
                Self::default_light().palette,
                self.spacing,
                self.borders,
                self.narrow_max,
                self.wide_min,
                false,
                ColorTier::TrueColor,
            );
        }

        let mut palette = self.palette;
        palette.muted = ensure_contrast_on_dark(palette.muted, background, 4.5, target_tier);
        palette.dim = ensure_contrast_on_dark(palette.dim, background, 4.5, target_tier);
        palette.faint = ensure_contrast_on_dark(palette.faint, background, 2.0, target_tier);
        palette.accent = ensure_contrast_on_dark(palette.accent, background, 4.5, target_tier);
        palette.fg = ensure_contrast_on_dark(palette.fg, background, 4.5, target_tier);
        palette.bright = ensure_contrast_on_dark(palette.bright, background, 4.5, target_tier);
        if palette == self.palette {
            return self;
        }

        // Return a true-color theme; `for_current_terminal_with_background`
        // quantizes into `target_tier`, and the contrast search above already
        // guaranteed each raised role clears its target after that step.
        Self::with_tier(
            &self.name,
            palette,
            self.spacing,
            self.borders,
            self.narrow_max,
            self.wide_min,
            false,
            ColorTier::TrueColor,
        )
    }

    /// Apply the *current process* terminal color policy to this theme.
    ///
    /// Reads the environment once ([`ColorTier::detect`] stays pure — the env
    /// is sampled here), then defers to [`Self::apply_color_tier`]. This is the
    /// single wiring point the boot path and the live `/theme` switch both use
    /// so a theme is always drawn under the terminal's real capability.
    #[must_use]
    pub fn for_current_terminal(self) -> Self {
        self.apply_color_tier(current_color_tier())
    }

    /// Apply a detected background, then the current terminal's color tier.
    ///
    /// Keeping this order means contrast calculations always use the canonical
    /// palette rather than an already-coarsened ANSI-256 or ANSI-16 palette.
    #[must_use]
    pub fn for_current_terminal_with_background(
        self,
        terminal_background: Option<(u8, u8, u8)>,
    ) -> Self {
        // Compute the tier once: the contrast search inside
        // `apply_terminal_background` targets this tier so ANSI-256/16 rounding
        // cannot drop a raised role below its floor, and the same tier then
        // quantizes the whole theme.
        let tier = current_color_tier();
        self.apply_terminal_background(terminal_background, tier)
            .apply_color_tier(tier)
    }

    /// Classify a terminal width into a responsive breakpoint bucket.
    #[must_use]
    pub fn for_width(&self, cols: u16) -> Breakpoint {
        if cols <= self.narrow_max {
            Breakpoint::Narrow
        } else if cols >= self.wide_min {
            Breakpoint::Wide
        } else {
            Breakpoint::Compact
        }
    }

    // ========================================================================
    // Semantic style accessors — the single source of truth for every color
    // decision in cards, diffs, callouts, role rails, and metric gauges.
    //
    // Each accessor *derives* from the already-resolved `palette`, so:
    //   * `NO_COLOR` / neutral themes degrade to `Color::Reset` automatically,
    //   * disk-loaded token overrides flow through transitively, and
    //   * callers never branch on raw palette colors (code-rules R9).
    // Adding a role/callout/metric here restyles every consumer at once.
    // ========================================================================

    /// Boot-time derived Cold Steel / Hot Core tokens for chrome surfaces.
    #[must_use]
    pub fn heat(&self) -> &HeatTokens {
        &self.heat
    }

    /// Cooling HUD fill one visual step below the selected corner ramp color.
    #[must_use]
    pub(crate) fn cooling_fill_color(&self, ramp_idx: usize) -> Color {
        let ramp_idx = ramp_idx.min(self.heat.ramp.len() - 1);
        if ramp_idx == self.heat.ramp.len() - 1 {
            return self.heat.steel_dim;
        }
        blend_toward(self.heat.ramp[ramp_idx], self.heat.steel_dim, 0.30)
            .unwrap_or(self.heat.steel_dim)
    }

    /// Brand color for a message role's gutter/rail glyph.
    #[must_use]
    pub fn role_color(&self, role: Role) -> Color {
        match role {
            Role::User => self.palette.accent,
            Role::Assistant => self.palette.fg,
            Role::System => self.palette.info,
            Role::Tool => self.palette.teal,
        }
    }

    /// Identity color for a sub-agent, keyed off a *stable hash of its id* — not
    /// its render position — so the hue stays put as siblings finish and the
    /// pruned live panel and the retained inline tree agree on one color per
    /// agent. The eight [`AGENT_COLORS`] hues are distinct from every theme's
    /// status colors, so a tint reads as *which* agent, never as status. Under
    /// `NO_COLOR` it drops to `Reset` (the name keeps its BOLD weight); on a
    /// light theme it yields to the body `fg`, since the hues are tuned for dark
    /// backgrounds and would lose contrast on a light one.
    #[must_use]
    pub fn agent_color(&self, agent_id: &str) -> Color {
        if self.no_color {
            return Color::Reset;
        }
        if self.has_light_background() {
            return self.palette.fg;
        }
        // FNV-1a: deterministic across runs (unlike `DefaultHasher`) so an
        // agent's color is stable for its whole life, and well-spread across the
        // eight buckets for short ids like `agent-3`.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in agent_id.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let bucket = usize::try_from(hash % AGENT_COLORS.len() as u64).unwrap_or(0);
        // The eight identity hues are authored as 256-color indices; requantize
        // the chosen one into the applied tier so an Ansi16 theme yields a named
        // 16-color (not a 256-color `Indexed`) and an Ansi256 theme keeps the
        // index. TrueColor/NoColor are handled above / pass through unchanged.
        quantize_color(AGENT_COLORS[bucket], self.tier)
    }

    /// Whether the theme sits on a light background, inferred from the luminance
    /// of the code surface (`code_bg` tracks the terminal background: near-white
    /// on light themes, near-black on dark ones). Used to skip the dark-tuned
    /// per-agent hues so light themes keep their high-contrast body `fg`.
    /// Handles both `Indexed` and true-color `Rgb` code backgrounds, so a
    /// hex-authored light theme is classified correctly too.
    #[must_use]
    fn has_light_background(&self) -> bool {
        let Some((r, g, b)) = color_to_rgb(self.palette.code_bg) else {
            return false;
        };
        // Rec. 601 luma; the light theme's code_bg (255) lands ~238, every dark
        // theme's (232–236, or a dark hex) lands well under 50, so the split is
        // unambiguous.
        (u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114) / 1000 > 140
    }

    /// Bold style for a role gutter glyph / badge.
    #[must_use]
    pub fn role_style(&self, role: Role) -> Style {
        Style::new()
            .fg(self.role_color(role))
            .add_modifier(Modifier::BOLD)
    }

    /// Foreground style for an added diff line.
    #[must_use]
    pub fn diff_add_style(&self) -> Style {
        Style::new().fg(self.palette.success)
    }

    /// Foreground style for a removed diff line.
    #[must_use]
    pub fn diff_del_style(&self) -> Style {
        Style::new().fg(self.palette.error)
    }

    /// Foreground style for an unchanged (context) diff line.
    #[must_use]
    pub fn diff_context_style(&self) -> Style {
        Style::new().fg(self.palette.dim)
    }

    /// Foreground style for a syntax-highlighted code token of `role`.
    ///
    /// Colors are pulled from the zo [`Palette`] rather than syntect's
    /// base16 theme, so code highlighting stays on-brand and inherits the
    /// palette's own degrade: truecolor keeps RGB, an indexed palette yields
    /// `Color::Indexed`, and a `NO_COLOR` palette yields `Color::Reset` (a
    /// neutral, uniform code body). Only zo's structural hues are spent
    /// (`violet`/`teal`/`cyan`, at most three), keeping the code card inside
    /// the "카드당 최대 3계조" budget; chromatic status hues (`success`/`warn`/
    /// `error`) and brand `accent` are deliberately never used here.
    #[must_use]
    pub fn syntax_style(&self, role: SyntaxRole) -> Style {
        let color = match role {
            SyntaxRole::Plain => self.palette.fg,
            SyntaxRole::Comment => self.palette.muted,
            SyntaxRole::Keyword => self.palette.violet,
            SyntaxRole::Str => self.palette.teal,
            SyntaxRole::Name => self.palette.cyan,
        };
        let style = Style::new().fg(color);
        // Comments recede via both hue and weight; italic is the one modifier
        // that survives the scope-based path (syntect's per-theme font styles
        // no longer reach us). `NO_COLOR` keeps the italic so comments stay
        // distinguishable without color.
        if matches!(role, SyntaxRole::Comment) {
            style.add_modifier(Modifier::ITALIC)
        } else {
            style
        }
    }

    /// Semantic background band for an added diff line. Indexed palettes keep
    /// the result indexed; neutral palettes return `None`. True-color terminals
    /// use a restrained wash so syntax colors stay readable on the row.
    ///
    /// Indexed palettes deliberately do NOT blend: a strong blend quantized
    /// onto the coarse 6×6×6 cube landed on mid-tone fields ((95,135,95)
    /// green, (215,95,95) salmon) that sat at the same luminance as the row's
    /// own text — the live "diff 배경과 글씨가 구분 안 됨" report. They pick
    /// dedicated dark/pastel diff slots instead (the vim/GitHub-dark diff
    /// convention), so the body fg keeps its full contrast on the band.
    #[must_use]
    pub fn diff_add_bg(&self) -> Option<Color> {
        self.diff_band(self.palette.success, 0.18, 22, 194) // dark (0,95,0) · light (215,255,215)
    }

    /// Semantic background band for a removed diff line. See
    /// [`Self::diff_add_bg`].
    #[must_use]
    pub fn diff_del_bg(&self) -> Option<Color> {
        self.diff_band(self.palette.error, 0.18, 52, 224) // dark (95,0,0) · light (255,215,215)
    }

    /// Stronger background tint for the *changed words* within an added line
    /// (word-level intra-line emphasis, Claude-Code parity). One cube step
    /// brighter than [`Self::diff_add_bg`] so the actual edit pops out of the
    /// line-wide band while the text on it stays readable. Indexed palettes
    /// keep the result indexed; neutral palettes return `None`.
    #[must_use]
    pub fn diff_add_emphasis_bg(&self) -> Option<Color> {
        self.diff_band(self.palette.success, 0.34, 28, 157) // dark (0,135,0) · light (175,255,175)
    }

    /// Stronger background tint for the changed words within a removed line.
    /// See [`Self::diff_add_emphasis_bg`].
    #[must_use]
    pub fn diff_del_emphasis_bg(&self) -> Option<Color> {
        self.diff_band(self.palette.error, 0.34, 88, 217) // dark (135,0,0) · light (255,175,175)
    }

    /// One diff band color. True-color palettes blend the status hue over the
    /// code surface at `truecolor_strength` (small RGB deltas stay visible
    /// without stealing luminance from the text). Indexed palettes pick the
    /// dedicated `dark_slot`/`light_slot` by surface luminance — see
    /// [`Self::diff_add_bg`] for why blending is wrong there. Neutral
    /// (`NO_COLOR`) palettes return `None`.
    fn diff_band(
        &self,
        hue: Color,
        truecolor_strength: f32,
        dark_slot: u8,
        light_slot: u8,
    ) -> Option<Color> {
        if let (Color::Rgb(..), Color::Rgb(..)) = (hue, self.palette.code_bg) {
            tint_strength(hue, self.palette.code_bg, truecolor_strength)
        } else {
            // Reset/named surfaces (the neutral palette) carry no RGB to
            // classify — keep the no-wash contract.
            color_to_rgb(self.palette.code_bg)?;
            Some(Color::Indexed(if self.has_light_background() {
                light_slot
            } else {
                dark_slot
            }))
        }
    }

    /// Whisper-faint neutral background for a *context* diff line adjacent to
    /// a change (CC `diffAddedDimmed` equivalent, v3 §4): the unchanged lines
    /// hugging an add/remove band read as part of the edit's neighborhood, so
    /// the eye lands on the band with its surroundings pre-grouped. Much
    /// weaker than [`Self::diff_add_bg`] — a fg-blend at glass-surface
    /// strength, not a color statement. `None` on indexed / neutral palettes.
    #[must_use]
    pub fn diff_context_dimmed_bg(&self) -> Option<Color> {
        tint_strength(self.palette.fg, self.palette.code_bg, 0.05)
    }

    /// Background wash for a mouse-selected transcript block (left-drag
    /// block-selection). A faint accent tint over the code surface so the
    /// selection reads as highlighted without overpowering the block's own
    /// foreground. `None` on indexed / neutral palettes (no well-defined blend),
    /// where the caller falls back to a reversed wash.
    #[must_use]
    pub fn selection_bg(&self) -> Option<Color> {
        tint(self.palette.accent, self.palette.code_bg)
    }

    // ── Glassmorphism layer (v3 §10) ────────────────────────────────────
    //
    // Terminal cells have no alpha or backdrop blur, so "glass" is color
    // arithmetic on the theme's own fg↔code_bg axis (the same base every
    // other bg helper blends over): a translucent-looking surface is the
    // base nudged toward the foreground, a glass edge is a brighter blend
    // than the hairline, and the modal scrim mutes what is behind the
    // pane by pulling each cell's fg halfway to the base. All of these
    // return `None`/identity on the `NO_COLOR` neutral palette.

    /// Elevation-1 glass surface — inline cards (code/diff/agent result),
    /// the sidebar, and the HUD strip. A 6% fg blend over the surface
    /// base: barely lighter (dark themes) / darker (light themes) than
    /// the backdrop, which is exactly the frosted-pane read.
    #[must_use]
    pub fn surface1(&self) -> Option<Color> {
        tint_strength(self.palette.fg, self.palette.code_bg, 0.06)
    }

    /// Elevation-2 glass surface — overlays (modals, the pinned agent
    /// panel). One step brighter than [`Self::surface1`] so a floating
    /// pane reads *above* inline cards.
    #[must_use]
    pub fn surface2(&self) -> Option<Color> {
        tint_strength(self.palette.fg, self.palette.code_bg, 0.10)
    }

    /// Glass-edge border — the "light catching the pane edge" accent for
    /// glass surfaces only. Deliberately brighter than the `faint`
    /// hairline so a glass panel is distinguishable from a plain card.
    #[must_use]
    pub fn border_glass(&self) -> Option<Color> {
        tint_strength(self.palette.fg, self.palette.code_bg, 0.45)
    }

    /// Frosted-scrim remap for one backdrop cell's foreground while a
    /// modal is open: pull it halfway to the surface base so the content
    /// behind the pane stays visible but visibly muted (the terminal
    /// equivalent of backdrop blur; 50% sits inside the 40–60% modal
    /// legibility band). Identity on the neutral palette, so `NO_COLOR`
    /// keeps its plain fullscreen behavior.
    #[must_use]
    pub fn scrim_fg(&self, fg: Color) -> Color {
        tint_strength(fg, self.palette.code_bg, 0.5).unwrap_or(fg)
    }

    /// The concrete background every elevation-1 surface paints (code-card
    /// interiors, the sidebar panel, the HUD strip, card frames): the glass
    /// [`Self::surface1`] when the palette can blend, else the raw `code_bg`
    /// token (`NO_COLOR`/neutral). Single accessor so the whole elevation-1
    /// layer shifts together.
    #[must_use]
    pub fn code_surface(&self) -> Color {
        self.surface1().unwrap_or(self.palette.code_bg)
    }

    /// Faint background wash for a tool block's card, tinted by `base` (the
    /// outcome color — success / error / accent). Blends the outcome over the
    /// code surface at low strength so the card reads as a soft tint behind
    /// unchanged glyphs. `None` on neutral palettes (no well-defined blend),
    /// where the caller skips the wash — exactly like [`selection_bg`].
    ///
    /// [`selection_bg`]: Self::selection_bg
    #[must_use]
    pub fn tool_card_bg(&self, base: Color) -> Option<Color> {
        tint_strength(base, self.palette.code_bg, 0.15)
    }

    /// Style for a diff's file-path header line (`── path ── +N -M`).
    ///
    /// Routes the header color through `&Theme` (R9) instead of the widget
    /// hardcoding a palette slot. Uses quiet `cyan` so structural labels stay
    /// distinct from add/remove hues without competing with the primary accent.
    #[must_use]
    pub fn diff_file_header_style(&self) -> Style {
        Style::new()
            .fg(self.palette.cyan)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for a diff hunk header line (`@@ -a,b +c,d @@`).
    ///
    /// Distinct from the file header so the two header tiers read as a
    /// hierarchy: file (cyan) frames the change, hunk (violet) marks each
    /// region within it. Mirrors opencode's dedicated `diffHunkHeader`
    /// token. R9: all styling flows through `&Theme`.
    #[must_use]
    pub fn diff_hunk_header_style(&self) -> Style {
        Style::new()
            .fg(self.palette.violet)
            .add_modifier(Modifier::BOLD)
    }

    /// Accent color for a callout / admonition of `kind`.
    #[must_use]
    pub fn callout_color(&self, kind: CalloutKind) -> Color {
        match kind {
            CalloutKind::Note => self.palette.info,
            CalloutKind::Tip => self.palette.success,
            CalloutKind::Warning => self.palette.warn,
            // GitHub renders [!IMPORTANT] purple; violet also keeps the
            // brand accent out of body callouts.
            CalloutKind::Important => self.palette.violet,
        }
    }

    /// Bold label style for a callout header (e.g. `Note`, `Warning`).
    #[must_use]
    pub fn callout_style(&self, kind: CalloutKind) -> Style {
        Style::new()
            .fg(self.callout_color(kind))
            .add_modifier(Modifier::BOLD)
    }

    /// Header-row style for tables and metric dashboards.
    #[must_use]
    pub fn table_header_style(&self) -> Style {
        Style::new()
            .fg(self.palette.bright)
            .add_modifier(Modifier::BOLD)
    }

    /// Threshold color for a `0.0..=1.0` metric ratio:
    /// `ok < 0.7 ≤ warn < 0.9 ≤ crit`. The ratio is clamped, so an
    /// over-budget value can never trip a gauge panic downstream.
    #[must_use]
    pub fn metric_color(&self, ratio: f64) -> Color {
        let r = ratio.clamp(0.0, 1.0);
        if r >= 0.9 {
            self.palette.error
        } else if r >= 0.7 {
            self.palette.warn
        } else {
            self.palette.success
        }
    }

    /// Foreground style wrapper around [`Self::metric_color`].
    #[must_use]
    pub fn metric_style(&self, ratio: f64) -> Style {
        Style::new().fg(self.metric_color(ratio))
    }

    /// Color for a markdown heading of `level` (1–6).
    ///
    /// Headings carry hierarchy through brightness and glyph weight, not
    /// hue (OpenCode-style restraint): H1–H3 render in `bright` while the
    /// gutter glyph thins (█ ▌ ▎), and H4–H6 recede into neutral `dim`.
    /// Keeping the brand accent out of headings reserves it for the user
    /// rail, focus borders, and live/spinner moments, so body prose stays
    /// the protagonist. `NO_COLOR` degrades through [`Self::heading_style`].
    #[must_use]
    pub fn heading_color(&self, level: u8) -> Color {
        match level {
            1..=3 => self.palette.bright,
            _ => self.palette.dim,
        }
    }

    /// Bold style for a markdown heading of `level`. `NO_COLOR` themes drop
    /// the hue and lean on weight alone.
    #[must_use]
    pub fn heading_style(&self, level: u8) -> Style {
        if self.no_color {
            return Style::new().add_modifier(Modifier::BOLD);
        }
        Style::new()
            .fg(self.heading_color(level))
            .add_modifier(Modifier::BOLD)
    }
}

// ============================================================================
// Helpers
// ============================================================================

const LIGHT_BACKGROUND_LUMINANCE: f64 = 0.179;

fn is_light_background(background: (u8, u8, u8)) -> bool {
    relative_luminance(background) > LIGHT_BACKGROUND_LUMINANCE
}

fn contrast_ratio(a: (u8, u8, u8), b: (u8, u8, u8)) -> f64 {
    let (lighter, darker) = {
        let a = relative_luminance(a);
        let b = relative_luminance(b);
        if a >= b { (a, b) } else { (b, a) }
    };
    (lighter + 0.05) / (darker + 0.05)
}

fn relative_luminance((r, g, b): (u8, u8, u8)) -> f64 {
    let linear = |channel: u8| {
        let channel = f64::from(channel) / 255.0;
        if channel <= 0.04045 {
            channel / 12.92
        } else {
            ((channel + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
}

fn ensure_contrast_on_dark(
    color: Color,
    background: (u8, u8, u8),
    target: f64,
    tier: ColorTier,
) -> Color {
    let Some(rgb) = color_to_rgb(color) else {
        return color;
    };
    if rendered_contrast(rgb, background, tier) >= target {
        return color;
    }

    let (hue, saturation, lightness) = rgb_to_hsl(rgb);
    for step in 0_u16..=255 {
        let candidate_lightness = f64::from(step) / 255.0;
        if candidate_lightness < lightness {
            continue;
        }
        let candidate = hsl_to_rgb(hue, saturation, candidate_lightness);
        if rendered_contrast(candidate, background, tier) >= target {
            return Color::Rgb(candidate.0, candidate.1, candidate.2);
        }
    }
    Color::Rgb(255, 255, 255)
}

/// Contrast of `candidate` against `background` after `candidate` is quantized
/// into `tier` — i.e. the color the terminal will actually render. Measuring the
/// rendered color lets the search reject a true-color value that ANSI-256/16
/// rounding would pull back under the floor. ANSI-16 uses the xterm-default RGB
/// of the nearest named color ([`ansi16_model_rgb`]), matching exactly how
/// [`quantize_color`] snaps that tier.
fn rendered_contrast(candidate: (u8, u8, u8), background: (u8, u8, u8), tier: ColorTier) -> f64 {
    let (r, g, b) = candidate;
    let rendered = match tier {
        ColorTier::Ansi256 => ansi256_to_rgb(rgb_to_ansi256(r, g, b)),
        ColorTier::Ansi16 => ansi16_model_rgb(r, g, b),
        ColorTier::TrueColor | ColorTier::NoColor => candidate,
    };
    contrast_ratio(rendered, background)
}

fn rgb_to_hsl((r, g, b): (u8, u8, u8)) -> (f64, f64, f64) {
    let r = f64::from(r) / 255.0;
    let g = f64::from(g) / 255.0;
    let b = f64::from(b) / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let lightness = f64::midpoint(max, min);
    if delta == 0.0 {
        return (0.0, 0.0, lightness);
    }
    let saturation = delta / (1.0 - (2.0 * lightness - 1.0).abs());
    // `max` is exactly `r.max(g).max(b)`, so selecting the maximum channel with
    // `>=` is exact and avoids a float-equality comparison.
    let hue = if r >= g && r >= b {
        ((g - b) / delta).rem_euclid(6.0)
    } else if g >= b {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    } / 6.0;
    (hue, saturation, lightness)
}

fn hsl_to_rgb(hue: f64, saturation: f64, lightness: f64) -> (u8, u8, u8) {
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let hue_sector = hue * 6.0;
    let secondary = chroma * (1.0 - (hue_sector.rem_euclid(2.0) - 1.0).abs());
    let (r, g, b) = match hue_sector {
        value if value < 1.0 => (chroma, secondary, 0.0),
        value if value < 2.0 => (secondary, chroma, 0.0),
        value if value < 3.0 => (0.0, chroma, secondary),
        value if value < 4.0 => (0.0, secondary, chroma),
        value if value < 5.0 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };
    let offset = lightness - chroma / 2.0;
    let channel = |value: f64| -> u8 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            ((value + offset) * 255.0).round().clamp(0.0, 255.0) as u8
        }
    };
    (channel(r), channel(g), channel(b))
}

/// Blend `fg` ~10% over `bg` to produce a faint background tint for diff lines.
/// Returns `Some` for any blendable inputs — true-color `Rgb` (kept as RGB) or
/// indexed `Indexed` (blended via [`ansi256_to_rgb`] and returned in the indexed
/// space, roadmap ⑦). Only `Reset`/named colors (the `NO_COLOR` palette) have no
/// well-defined value and stay background-free (`None`).
fn tint(fg: Color, bg: Color) -> Option<Color> {
    tint_strength(fg, bg, 0.10)
}

/// Blend `fg` over `bg` at strength `t` (0.0 = pure `bg`, 1.0 = pure `fg`).
/// Returns `None` when either color has no concrete RGB value. True-color
/// inputs stay RGB; indexed inputs are quantized back to ANSI-256.
/// xterm 256-color index → RGB. 0-15 system colors, 16-231 the 6×6×6 cube,
/// 232-255 the 24-step grayscale ramp. This is what lets an *indexed* palette
/// blend like a truecolor one instead of falling back to no diff wash at all.
fn ansi256_to_rgb(i: u8) -> (u8, u8, u8) {
    const SYSTEM: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    match i {
        0..=15 => SYSTEM[i as usize],
        16..=231 => {
            let c = i - 16;
            let level = |v: u8| -> u8 { if v == 0 { 0 } else { 55 + 40 * v } };
            (level(c / 36), level((c / 6) % 6), level(c % 6))
        }
        232..=255 => {
            let v = 8 + 10 * (i - 232);
            (v, v, v)
        }
    }
}

/// RGB → nearest xterm 256-color index, the inverse of [`ansi256_to_rgb`], so a
/// blended indexed color can return to the indexed space — keeping indexed
/// palettes correct on terminals without truecolor and matching the theme's own
/// register.
fn rgb_to_ansi256(r: u8, g: u8, b: u8) -> u8 {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    // Nearest of the six cube levels by value midpoints.
    let cube_level = |v: u8| -> u8 {
        match v {
            0..=47 => 0,
            48..=114 => 1,
            115..=154 => 2,
            155..=194 => 3,
            195..=234 => 4,
            _ => 5,
        }
    };
    let sq = |x: u8, y: u8| -> u32 {
        let d = (i32::from(x) - i32::from(y)).unsigned_abs();
        d * d
    };

    // Cube candidate.
    let (cr, cg, cb) = (cube_level(r), cube_level(g), cube_level(b));
    let cube_idx = 16 + 36 * cr + 6 * cg + cb;
    let cube_d = sq(LEVELS[usize::from(cr)], r)
        + sq(LEVELS[usize::from(cg)], g)
        + sq(LEVELS[usize::from(cb)], b);

    // Grayscale-ramp candidate. The closest gray often beats the cube for the
    // dark, near-neutral washes a faint diff tint produces — without this, a
    // color like (47,62,47) snaps to the oversaturated cube green (0,95,0)
    // instead of the near-identical dark gray (48,48,48).
    #[allow(clippy::cast_possible_truncation)]
    let avg = ((u16::from(r) + u16::from(g) + u16::from(b)) / 3) as u8;
    let gray_idx = if avg < 8 {
        16
    } else if avg > 238 {
        231
    } else {
        232 + (avg - 8) / 10
    };
    let (gr, gg, gb) = ansi256_to_rgb(gray_idx);
    let gray_d = sq(gr, r) + sq(gg, g) + sq(gb, b);

    // Pick the visually closer of the two candidates.
    if cube_d <= gray_d {
        cube_idx
    } else {
        gray_idx
    }
}

/// The xterm-default 16-color palette used to snap true-color values onto a
/// 16-color terminal. `.0` is the target RGB — the color a 16-color terminal is
/// modeled as rendering — and `.1` is the ratatui *named* color that emits the
/// true `3x`/`4x` (`9x`/`10x`) SGR, never a `Color::Indexed`.
const BASE16: [((u8, u8, u8), Color); 16] = [
    ((0x00, 0x00, 0x00), Color::Black),
    ((0x80, 0x00, 0x00), Color::Red),
    ((0x00, 0x80, 0x00), Color::Green),
    ((0x80, 0x80, 0x00), Color::Yellow),
    ((0x00, 0x00, 0x80), Color::Blue),
    ((0x80, 0x00, 0x80), Color::Magenta),
    ((0x00, 0x80, 0x80), Color::Cyan),
    ((0xC0, 0xC0, 0xC0), Color::Gray),
    ((0x80, 0x80, 0x80), Color::DarkGray),
    ((0xFF, 0x00, 0x00), Color::LightRed),
    ((0x00, 0xFF, 0x00), Color::LightGreen),
    ((0xFF, 0xFF, 0x00), Color::LightYellow),
    ((0x00, 0x00, 0xFF), Color::LightBlue),
    ((0xFF, 0x00, 0xFF), Color::LightMagenta),
    ((0x00, 0xFF, 0xFF), Color::LightCyan),
    ((0xFF, 0xFF, 0xFF), Color::White),
];

/// Index into [`BASE16`] of the nearest slot to `(r, g, b)` by squared RGB
/// distance. Ties resolve to the earlier slot, matching the original scan.
fn nearest_base16(r: u8, g: u8, b: u8) -> usize {
    let sq = |x: u8, y: u8| -> u32 {
        let d = (i32::from(x) - i32::from(y)).unsigned_abs();
        d * d
    };
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (idx, &((cr, cg, cb), _)) in BASE16.iter().enumerate() {
        let d = sq(cr, r) + sq(cg, g) + sq(cb, b);
        if d < best_d {
            best_d = d;
            best = idx;
        }
    }
    best
}

/// Nearest of the 16 base ANSI colors for an RGB value, returned as a **named**
/// `ratatui::style::Color` (`Black`, `Red`, …, `White`) — never
/// `Color::Indexed`.
///
/// This distinction is load-bearing on a 16-color terminal: ratatui-crossterm
/// maps `Color::Indexed(i)` to `CrosstermColor::AnsiValue(i)`, which crossterm
/// emits as a `38;5;i` / `48;5;i` sequence — a *256-color* escape even for
/// `i < 16`, so a "16-color" quantization that returned `Indexed(<16)` would
/// still send 256-color sequences to a terminal that cannot decode them. The
/// named variants map to true `3x`/`4x` (and `9x`/`10x`) SGR codes, which is
/// the actual ANSI-16 wire format. Picking by squared RGB distance keeps a
/// brand hue on its closest primary (amber→yellow, steel-blue→bright-blue)
/// instead of collapsing everything to white/black.
fn rgb_to_ansi16(r: u8, g: u8, b: u8) -> Color {
    BASE16[nearest_base16(r, g, b)].1
}

/// The xterm-default RGB the nearest ANSI-16 named color models — the value a
/// 16-color terminal is treated as rendering. Lets contrast be measured against
/// what the terminal actually shows, not the pre-quantization true-color value.
fn ansi16_model_rgb(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    BASE16[nearest_base16(r, g, b)].0
}

/// The modeled RGB of an ANSI-16 *named* color (the inverse of the `.1` lookup
/// in [`BASE16`]), for asserting what a 16-color terminal renders. `None` for
/// any color not in the 16-slot table.
#[cfg(test)]
fn named_ansi16_rgb(color: Color) -> Option<(u8, u8, u8)> {
    BASE16
        .iter()
        .find(|&&(_, named)| named == color)
        .map(|&(rgb, _)| rgb)
}

/// Quantize one resolved palette color into `tier`'s color space.
///
/// `Reset`/named colors (the neutral palette, or an already-quantized ANSI-16
/// named color) carry no RGB and pass through untouched — they are already
/// tier-agnostic. The rest:
/// * `Ansi256` — **only** `Rgb` is quantized (via [`rgb_to_ansi256`]); an
///   existing `Color::Indexed(i)` is preserved *byte-for-byte*. An indexed
///   built-in palette is already a valid 256-color palette, and round-tripping
///   it through RGB would land it on a *different* index (the RGB→index mapping
///   is not the inverse of the index→RGB table). This honors the "existing
///   indexed palettes only requantize at a lower tier" contract.
/// * `Ansi16` — both `Rgb` and `Indexed` snap to the nearest of the 16 base
///   slots via [`rgb_to_ansi16`], which returns a *named* color (never
///   `Indexed`), so the wire format is a true 16-color SGR code.
///
/// `TrueColor`/`NoColor` never reach here (handled in [`Palette::to_tier`]).
fn quantize_color(c: Color, tier: ColorTier) -> Color {
    match tier {
        // Preserve an existing indexed color exactly; quantize RGB only.
        ColorTier::Ansi256 => match c {
            Color::Rgb(r, g, b) => Color::Indexed(rgb_to_ansi256(r, g, b)),
            other => other,
        },
        ColorTier::Ansi16 => match color_to_rgb(c) {
            Some((r, g, b)) => rgb_to_ansi16(r, g, b),
            None => c,
        },
        ColorTier::TrueColor | ColorTier::NoColor => c,
    }
}

/// A blendable color as RGB. `None` for `Reset`/named colors that have no
/// well-defined value, so the no-color path stays foreground-only there.
fn color_to_rgb(c: Color) -> Option<(u8, u8, u8)> {
    match c {
        Color::Rgb(r, g, b) => Some((r, g, b)),
        Color::Indexed(i) => Some(ansi256_to_rgb(i)),
        _ => None,
    }
}

/// Move `from` toward `to` by `t`, preserving the palette's color space.
fn blend_toward(from: Color, to: Color, t: f32) -> Option<Color> {
    tint_strength(to, from, t)
}

fn tint_strength(fg: Color, bg: Color, t: f32) -> Option<Color> {
    let (fr, fgg, fb) = color_to_rgb(fg)?;
    let (br, bgg, bb) = color_to_rgb(bg)?;
    let mix = |f: u8, b: u8| -> u8 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v = (f32::from(f) * t + f32::from(b) * (1.0 - t))
            .round()
            .clamp(0.0, 255.0) as u8;
        v
    };
    let (r, g, b) = (mix(fr, br), mix(fgg, bgg), mix(fb, bb));
    // Preserve the palette's color space: a truecolor palette keeps full RGB; an
    // indexed palette returns to indexed so the wash still renders without
    // truecolor support. `Reset`/named inputs already returned `None` above.
    if matches!((fg, bg), (Color::Rgb(..), Color::Rgb(..))) {
        Some(Color::Rgb(r, g, b))
    } else {
        Some(Color::Indexed(rgb_to_ansi256(r, g, b)))
    }
}

/// Resolve a single color role from the tokens table.
///
/// Precedence: `hex` (24-bit true color) → `ansi256` (indexed) →
/// `fallback`. The `fallback` is the role's Zo default, so a sparse
/// or partially-populated tokens file still yields a complete, on-brand
/// palette instead of degrading to `Color::Reset`.
fn resolve_color(table: &BTreeMap<String, TokenColorEntry>, key: &str, fallback: Color) -> Color {
    match table.get(key) {
        Some(entry) => entry
            .hex
            .as_deref()
            .and_then(parse_hex)
            .or_else(|| entry.ansi256.map(Color::Indexed))
            .unwrap_or(fallback),
        None => fallback,
    }
}

/// Parse a `#RRGGBB` hex string into [`Color::Rgb`]. Returns `None` for
/// malformed input (wrong length, non-hex digits, missing `#`) so the
/// caller can fall through to the next color source.
fn parse_hex(raw: &str) -> Option<Color> {
    let hex = raw.strip_prefix('#').unwrap_or(raw);
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some(Color::Rgb(r, g, b))
}

fn parse_border(kind: &str) -> Option<BorderType> {
    match kind {
        "single" | "plain" => Some(BorderType::Plain),
        "rounded" => Some(BorderType::Rounded),
        "double" => Some(BorderType::Double),
        "thick" => Some(BorderType::Thick),
        _ => None,
    }
}

fn default_border_roles() -> BTreeMap<String, BorderType> {
    let mut m = BTreeMap::new();
    // The input keeps one heavy left rail as its focused structural cue. The
    // HUD below stays text-first and uses whitespace instead of joining chrome.
    m.insert("input_box".to_string(), BorderType::Thick);
    m.insert("tool_call_card".to_string(), BorderType::Rounded);
    m.insert("tool_result_card".to_string(), BorderType::Rounded);
    m.insert("diff_card".to_string(), BorderType::Rounded);
    m.insert("permission_card".to_string(), BorderType::Thick);
    m.insert("modal".to_string(), BorderType::Rounded);
    m.insert("error_card".to_string(), BorderType::Thick);
    m
}

fn env_no_color() -> bool {
    env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty())
}

/// Sample the process environment and classify the current terminal's color
/// tier via the pure [`ColorTier::detect`]. The one impure boundary between the
/// env and the pure detector, kept tiny so the classification logic stays
/// testable without global env mutation.
fn current_color_tier() -> ColorTier {
    let colorterm = env::var("COLORTERM").ok();
    let term = env::var("TERM").ok();
    let term_program = env::var("TERM_PROGRAM").ok();
    let windows_terminal = env::var_os("WT_SESSION").is_some_and(|value| !value.is_empty());
    ColorTier::detect_with_native_terminal(
        env_no_color(),
        colorterm.as_deref(),
        term.as_deref(),
        term_program.as_deref(),
        windows_terminal,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ansi256_to_rgb, color_to_rgb, contrast_ratio, named_ansi16_rgb, rgb_to_ansi256,
        CalloutKind, Color, ColorTier, Role, Spacing, Theme,
    };

    fn palette_contrast(color: Color, background: (u8, u8, u8)) -> f64 {
        contrast_ratio(
            color_to_rgb(color).expect("test palette color has concrete RGB"),
            background,
        )
    }

    /// Contrast of a resolved palette color, resolving even the ANSI-16 *named*
    /// colors (which `color_to_rgb` leaves undefined) to their xterm-default
    /// model so a 16-color render is measurable.
    fn rendered_palette_contrast(color: Color, background: (u8, u8, u8)) -> f64 {
        let rgb = color_to_rgb(color)
            .or_else(|| named_ansi16_rgb(color))
            .expect("resolved palette color has a modeled RGB");
        contrast_ratio(rgb, background)
    }

    // ── Terminal-background adaptation (pure, live-terminal-free) ──

    #[test]
    fn missing_terminal_background_preserves_palette_exactly() {
        let original = Theme::zo();
        let adapted = Theme::zo().apply_terminal_background(None, ColorTier::TrueColor);

        assert_eq!((adapted.name, adapted.palette), (original.name, original.palette));
    }

    #[test]
    fn light_terminal_background_selects_builtin_light_theme() {
        let adapted =
            Theme::zo().apply_terminal_background(Some((245, 245, 245)), ColorTier::TrueColor);

        assert_eq!(adapted.name, "light");
    }

    /// A light terminal swaps the palette for legibility, but a loaded theme's
    /// non-color overrides (spacing, borders, breakpoints) must survive — the
    /// light path used to replace the whole theme with `default_light`, dropping
    /// a `tokens.json`'s spacing/breakpoint overrides.
    #[test]
    fn light_terminal_background_keeps_non_color_overrides() {
        let mut base = Theme::zo();
        base.narrow_max = 42;
        base.wide_min = 111;
        base.spacing.block_gap = 4;
        let adapted = base.apply_terminal_background(Some((245, 245, 245)), ColorTier::TrueColor);

        assert_eq!(adapted.name, "light", "light terminal adopts the light palette");
        assert_eq!(adapted.narrow_max, 42, "narrow_max override must survive");
        assert_eq!(adapted.wide_min, 111, "wide_min override must survive");
        assert_eq!(adapted.spacing.block_gap, 4, "spacing override must survive");
    }

    #[test]
    fn zo_dark_background_raises_muted_and_faint_contrast() {
        let background = (0x17, 0x17, 0x17);
        let adapted = Theme::zo().apply_terminal_background(Some(background), ColorTier::TrueColor);

        assert!(
            palette_contrast(adapted.palette.muted, background) >= 4.5
                && palette_contrast(adapted.palette.faint, background) >= 2.0
        );
    }

    #[test]
    fn different_dark_background_raises_muted_and_faint_contrast() {
        let background = (0x30, 0x30, 0x30);
        let adapted = Theme::zo().apply_terminal_background(Some(background), ColorTier::TrueColor);

        assert!(
            palette_contrast(adapted.palette.muted, background) >= 4.5
                && palette_contrast(adapted.palette.faint, background) >= 2.0
        );
    }

    /// The contrast floor must hold for the color the terminal actually renders:
    /// a role raised in true-color space and then quantized to ANSI-256 must
    /// still clear its target (regression: quantization pulled `muted` back
    /// under 4.5:1). Mirrors the real boot path — adapt, then quantize.
    #[test]
    fn dark_background_keeps_contrast_after_ansi256_quantization() {
        let background = (0x30, 0x30, 0x30);
        let adapted = Theme::zo()
            .apply_terminal_background(Some(background), ColorTier::Ansi256)
            .apply_color_tier(ColorTier::Ansi256);

        assert!(
            palette_contrast(adapted.palette.muted, background) >= 4.5,
            "muted {:?} -> {:.3}:1",
            adapted.palette.muted,
            palette_contrast(adapted.palette.muted, background)
        );
        assert!(
            palette_contrast(adapted.palette.dim, background) >= 4.5,
            "dim {:?}",
            adapted.palette.dim
        );
        assert!(
            palette_contrast(adapted.palette.faint, background) >= 2.0,
            "faint {:?}",
            adapted.palette.faint
        );
    }

    /// Same guarantee on a 16-color terminal: after ANSI-16 quantization each
    /// raised role becomes a *named* color, and the xterm-default RGB it models
    /// must still clear its floor — the pre-quantization candidate must not be
    /// measured as a stand-in (regression: a gray raised to pass in true-color
    /// snapped to `DarkGray` ≈ 3.3:1).
    #[test]
    fn dark_background_keeps_contrast_after_ansi16_quantization() {
        let background = (0x30, 0x30, 0x30);
        let adapted = Theme::zo()
            .apply_terminal_background(Some(background), ColorTier::Ansi16)
            .apply_color_tier(ColorTier::Ansi16);

        assert!(
            rendered_palette_contrast(adapted.palette.muted, background) >= 4.5,
            "muted {:?} -> {:.3}:1",
            adapted.palette.muted,
            rendered_palette_contrast(adapted.palette.muted, background)
        );
        assert!(
            rendered_palette_contrast(adapted.palette.dim, background) >= 4.5,
            "dim {:?}",
            adapted.palette.dim
        );
        assert!(
            rendered_palette_contrast(adapted.palette.faint, background) >= 2.0,
            "faint {:?}",
            adapted.palette.faint
        );
    }

    // ── Color-tier detection (pure, env-free) ──

    /// `NO_COLOR` is the highest-precedence signal: it wins over any
    /// `COLORTERM`/`TERM`/`TERM_PROGRAM` that would otherwise imply color.
    #[test]
    fn detect_no_color_overrides_every_other_signal() {
        assert_eq!(
            ColorTier::detect(true, Some("truecolor"), Some("xterm-kitty"), Some("iTerm.app")),
            ColorTier::NoColor
        );
        assert_eq!(
            ColorTier::detect(true, Some("24bit"), Some("xterm-256color"), None),
            ColorTier::NoColor
        );
        assert_eq!(ColorTier::detect(true, None, None, None), ColorTier::NoColor);
    }

    /// `TERM=dumb` classifies as `NoColor` — the plain-ASCII contract the rest
    /// of the TUI honors — and does so *before* `COLORTERM`, so a stray
    /// `COLORTERM=truecolor` on a dumb terminal cannot force color on.
    #[test]
    fn detect_term_dumb_is_no_color_over_colorterm() {
        assert_eq!(
            ColorTier::detect(false, None, Some("dumb"), None),
            ColorTier::NoColor
        );
        assert_eq!(
            ColorTier::detect(false, Some("truecolor"), Some("dumb"), Some("iTerm.app")),
            ColorTier::NoColor,
            "COLORTERM=truecolor must NOT override TERM=dumb"
        );
        // NO_COLOR still wins over everything, dumb included.
        assert_eq!(
            ColorTier::detect(true, Some("truecolor"), Some("dumb"), None),
            ColorTier::NoColor
        );
    }

    /// `COLORTERM=truecolor|24bit` (case-insensitive), a `TERM` carrying
    /// `direct`/`truecolor`, and a known native-truecolor terminal id (via
    /// `TERM`) all classify as truecolor.
    #[test]
    fn detect_truecolor_signals() {
        assert_eq!(
            ColorTier::detect(false, Some("truecolor"), Some("xterm-256color"), None),
            ColorTier::TrueColor
        );
        assert_eq!(
            ColorTier::detect(false, Some("24bit"), None, None),
            ColorTier::TrueColor
        );
        assert_eq!(
            ColorTier::detect(false, Some("TrueColor"), None, None),
            ColorTier::TrueColor,
            "COLORTERM match is case-insensitive"
        );
        assert_eq!(
            ColorTier::detect(false, None, Some("xterm-direct"), None),
            ColorTier::TrueColor
        );
        for term in [
            "xterm-kitty",
            "xterm-ghostty",
            "wezterm",
            "alacritty",
            "ghostty",
        ] {
            assert_eq!(
                ColorTier::detect(false, None, Some(term), None),
                ColorTier::TrueColor,
                "known truecolor TERM {term} must classify as TrueColor"
            );
        }
    }

    /// A real `iTerm2` / `WezTerm` sets `TERM=xterm-256color` and carries its
    /// identity only in `TERM_PROGRAM`; the `TERM_PROGRAM` allow-list must
    /// promote those to truecolor. An unrecognized program (`Apple_Terminal`)
    /// is left to the `TERM` classification instead.
    #[test]
    fn detect_truecolor_from_term_program_allowlist() {
        assert_eq!(
            ColorTier::detect(false, None, Some("xterm-256color"), Some("iTerm.app")),
            ColorTier::TrueColor
        );
        assert_eq!(
            ColorTier::detect(false, None, Some("xterm-256color"), Some("WezTerm")),
            ColorTier::TrueColor
        );
        // Unknown program falls back to the TERM tier (256color here).
        assert_eq!(
            ColorTier::detect(false, None, Some("xterm-256color"), Some("Apple_Terminal")),
            ColorTier::Ansi256,
            "unrecognized TERM_PROGRAM must defer to the TERM 256/16 classification"
        );
    }

    /// Windows Terminal advertises native truecolor through `WT_SESSION`
    /// rather than `TERM_PROGRAM`; the impure boundary passes that presence as
    /// a boolean while the detector stays pure and preserves stronger guards.
    #[test]
    fn detect_truecolor_from_windows_terminal_signal() {
        assert_eq!(
            ColorTier::detect_with_native_terminal(false, None, None, None, true),
            ColorTier::TrueColor
        );
        assert_eq!(
            ColorTier::detect_with_native_terminal(
                false,
                Some("truecolor"),
                Some("dumb"),
                None,
                true,
            ),
            ColorTier::NoColor,
            "TERM=dumb must remain plain even under WT_SESSION"
        );
        assert_eq!(
            ColorTier::detect_with_native_terminal(true, None, None, None, true),
            ColorTier::NoColor,
            "NO_COLOR must remain the highest-priority signal"
        );
    }

    /// `TERM` carrying `256color` (without a truecolor signal) is ANSI-256.
    #[test]
    fn detect_ansi256_from_term() {
        assert_eq!(
            ColorTier::detect(false, None, Some("xterm-256color"), None),
            ColorTier::Ansi256
        );
        assert_eq!(
            ColorTier::detect(false, Some(""), Some("screen-256color"), None),
            ColorTier::Ansi256
        );
    }

    /// Unknown / bare / empty / unset `TERM` conservatively degrades to the
    /// ANSI-16 floor rather than assuming a richer tier. (`dumb` is NOT here —
    /// it is `NoColor`, covered above.)
    #[test]
    fn detect_unknown_or_missing_term_degrades_to_ansi16() {
        for term in [Some("xterm"), Some("screen"), Some("linux"), Some(""), None] {
            assert_eq!(
                ColorTier::detect(false, None, term, None),
                ColorTier::Ansi16,
                "unknown/bare TERM {term:?} must degrade to ANSI-16"
            );
        }
    }

    // ── Tier application (explicit tier, no env mutation) ──

    /// `TrueColor` application is identity: the canonical Zo RGB palette is
    /// returned untouched, so nothing is quantized on a capable terminal.
    #[test]
    fn apply_truecolor_tier_is_identity() {
        let zo = Theme::zo();
        let applied = Theme::zo().apply_color_tier(ColorTier::TrueColor);
        assert_eq!(applied.palette.accent, zo.palette.accent);
        assert!(!applied.no_color);
        assert!(matches!(applied.palette.accent, Color::Rgb(..)));
    }

    fn palette_colors(theme: &Theme) -> [Color; 15] {
        let palette = &theme.palette;
        [
            palette.accent,
            palette.accent_dim,
            palette.cyan,
            palette.violet,
            palette.teal,
            palette.fg,
            palette.bright,
            palette.dim,
            palette.muted,
            palette.faint,
            palette.code_bg,
            palette.success,
            palette.warn,
            palette.error,
            palette.info,
        ]
    }

    fn is_ansi16_color(color: Color) -> bool {
        matches!(
            color,
            Color::Reset
                | Color::Black
                | Color::Red
                | Color::Green
                | Color::Yellow
                | Color::Blue
                | Color::Magenta
                | Color::Cyan
                | Color::Gray
                | Color::DarkGray
                | Color::LightRed
                | Color::LightGreen
                | Color::LightYellow
                | Color::LightBlue
                | Color::LightMagenta
                | Color::LightCyan
                | Color::White
        )
    }

    /// ANSI-256 application quantizes RGB into the indexed space without
    /// changing an existing indexed built-in palette. Derived heat, agent, and
    /// surface colors must not re-introduce truecolor.
    #[test]
    fn apply_ansi256_tier_quantizes_rgb_and_preserves_existing_indexes() {
        let applied = Theme::zo().apply_color_tier(ColorTier::Ansi256);
        assert!(!applied.no_color);
        let mut colors = palette_colors(&applied).to_vec();
        colors.extend(heat_colors(&applied));
        colors.extend(applied.heat().wordmark_gradient(12));
        colors.push(applied.agent_color("agent-3"));
        colors.extend([
            applied.diff_add_bg(),
            applied.diff_del_bg(),
            applied.surface2(),
            applied.tool_card_bg(applied.palette.success),
        ]
        .into_iter()
        .flatten());
        assert!(
            colors
                .into_iter()
                .all(|color| !matches!(color, Color::Rgb(..))),
            "ANSI-256 themes must not emit truecolor from derived accessors"
        );

        let indexed = Theme::default_dark();
        let expected = palette_colors(&indexed);
        let reapplied = indexed.apply_color_tier(ColorTier::Ansi256);
        assert_eq!(
            palette_colors(&reapplied),
            expected,
            "an existing indexed palette must be preserved byte-for-byte"
        );
    }

    /// ANSI-16 uses Ratatui's named colors, never `Color::Indexed`: even an
    /// index below 16 would make Crossterm emit a 256-color `38;5`/`48;5`
    /// sequence. Representative derived accessors are included so heat blends,
    /// agent hues, and card/diff surfaces cannot escape the tier later.
    #[test]
    fn apply_ansi16_tier_uses_only_named_colors_or_reset() {
        for source in [Theme::zo(), Theme::default_dark()] {
            let applied = source.apply_color_tier(ColorTier::Ansi16);
            assert!(!applied.no_color);
            let mut colors = palette_colors(&applied).to_vec();
            colors.extend(heat_colors(&applied));
            colors.extend(applied.heat().wordmark_gradient(12));
            colors.push(applied.agent_color("agent-3"));
            colors.push(applied.cooling_fill_color(0));
            colors.extend([
                applied.diff_add_bg(),
                applied.diff_del_bg(),
                applied.surface2(),
                applied.tool_card_bg(applied.palette.success),
            ]
            .into_iter()
            .flatten());
            for color in colors {
                assert!(
                    is_ansi16_color(color),
                    "ANSI-16 theme emitted an out-of-tier color: {color:?}"
                );
            }
        }
    }

    /// `NoColor` application yields the neutral palette and flags the theme
    /// `no_color`, so every downstream accessor degrades to `Color::Reset` —
    /// the same collapse `Theme::no_color()` produces, but reachable via the
    /// tier apply path (the boot-fallback `NO_COLOR` bug this fixes).
    #[test]
    fn apply_no_color_tier_neutralizes_palette() {
        let applied = Theme::zo().apply_color_tier(ColorTier::NoColor);
        assert!(applied.no_color);
        assert_eq!(applied.palette.accent, Color::Reset);
        assert_eq!(applied.palette.fg, Color::Reset);
        assert_eq!(applied.role_color(Role::User), Color::Reset);
        assert_eq!(applied.heading_color(1), Color::Reset);
    }

    /// The apply path must NOT mutate the canonical built-in palette:
    /// `Theme::builtin`/`Theme::zo` stay deterministic true-color sources that
    /// tests and `.zo/design/tokens.json` persistence depend on, regardless of
    /// what tier a *separate* applied copy was quantized into.
    #[test]
    fn builtin_palette_is_unchanged_by_applying_a_tier() {
        let before = Theme::zo().palette.accent;
        let _neutral = Theme::zo().apply_color_tier(ColorTier::NoColor);
        let _low = Theme::zo().apply_color_tier(ColorTier::Ansi16);
        assert_eq!(
            Theme::zo().palette.accent,
            before,
            "Theme::zo() must keep returning its canonical RGB accent"
        );
        assert!(matches!(before, Color::Rgb(0xF5, 0xA5, 0x24)));
    }

    #[test]
    fn metric_color_crosses_thresholds_and_clamps() {
        let t = Theme::zo();
        assert_eq!(t.metric_color(0.0), t.palette.success);
        assert_eq!(t.metric_color(0.69), t.palette.success);
        assert_eq!(t.metric_color(0.70), t.palette.warn);
        assert_eq!(t.metric_color(0.89), t.palette.warn);
        assert_eq!(t.metric_color(0.90), t.palette.error);
        // Over-budget ratios clamp instead of escaping the scale.
        assert_eq!(t.metric_color(2.5), t.palette.error);
        assert_eq!(t.metric_color(-1.0), t.palette.success);
    }

    #[test]
    fn role_color_matches_palette_roles() {
        let t = Theme::zo();
        assert_eq!(t.role_color(Role::User), t.palette.accent);
        assert_eq!(t.role_color(Role::Assistant), t.palette.fg);
        assert_eq!(t.role_color(Role::System), t.palette.info);
        assert_eq!(t.role_color(Role::Tool), t.palette.teal);
    }

    #[test]
    fn agent_hues_are_distinct_and_avoid_every_builtin_status_color() {
        let unique: std::collections::HashSet<_> = super::AGENT_COLORS.iter().collect();
        assert_eq!(unique.len(), 8, "the eight agent hues must be distinct");
        // No hue may equal ANY built-in theme's status colors — an earlier draft
        // used orange (214), which is byte-identical to `warn` in gruvbox /
        // everforest / kanagawa, so a name read as a warning there.
        for name in Theme::builtin_names() {
            let t = Theme::builtin(name).expect("builtin theme resolves");
            for status in [
                t.palette.success,
                t.palette.warn,
                t.palette.error,
                t.palette.teal,
            ] {
                assert!(
                    !super::AGENT_COLORS.contains(&status),
                    "agent hue {status:?} collides with a status color in theme {name}"
                );
            }
        }
    }

    #[test]
    fn agent_color_is_stable_per_id_and_spreads_across_hues() {
        let t = Theme::default_dark();
        // Stable: the same id always maps to the same hue, independent of any
        // render position (the whole point — siblings finishing must not recolor).
        assert_eq!(t.agent_color("agent-3"), t.agent_color("agent-3"));
        assert!(super::AGENT_COLORS.contains(&t.agent_color("agent-3")));
        // The hash spreads a fan-out across several hues (not a degenerate map).
        let hues: std::collections::HashSet<_> = (0..40)
            .map(|n| t.agent_color(&format!("agent-{n}")))
            .collect();
        assert!(hues.len() >= 4, "hashing must spread agents across hues");
    }

    #[test]
    fn agent_color_drops_to_reset_under_no_color() {
        let t = Theme::no_color();
        assert_eq!(t.agent_color("agent-1"), Color::Reset);
        assert_eq!(t.agent_color(""), Color::Reset);
    }

    #[test]
    fn agent_color_keeps_high_contrast_fg_on_light_theme() {
        let t = Theme::default_light();
        // The dark-tuned hues would lose contrast on a light background, so a
        // light theme keeps its high-contrast body fg (no legibility regression).
        assert_eq!(t.agent_color("agent-1"), t.palette.fg);
        assert_ne!(t.palette.fg, Color::Reset);
    }

    #[test]
    fn callout_colors_map_to_semantic_palette() {
        let t = Theme::zo();
        assert_eq!(t.callout_color(CalloutKind::Note), t.palette.info);
        assert_eq!(t.callout_color(CalloutKind::Tip), t.palette.success);
        assert_eq!(t.callout_color(CalloutKind::Warning), t.palette.warn);
        // Important is violet (GitHub convention), not the brand accent —
        // body callouts must never masquerade as the focus/user-rail color.
        assert_eq!(t.callout_color(CalloutKind::Important), t.palette.violet);
    }

    #[test]
    fn no_color_theme_degrades_every_accessor_to_reset() {
        let t = Theme::no_color();
        assert_eq!(t.metric_color(0.95), Color::Reset);
        assert_eq!(t.role_color(Role::Assistant), Color::Reset);
        assert_eq!(t.callout_color(CalloutKind::Warning), Color::Reset);
        // Neutral palette has no true-color blend → no diff background.
        assert_eq!(t.diff_add_bg(), None);
        assert_eq!(t.diff_del_bg(), None);
    }

    /// Every advertised theme name must resolve to a built-in theme. Guards
    /// against adding a `Theme::foo()` constructor (or a `builtin_names` entry)
    /// without wiring the other half — which would either leave a theme
    /// unreachable from `/theme` or advertise a name that fails to load.
    #[test]
    fn every_builtin_name_resolves() {
        for name in Theme::builtin_names() {
            assert!(
                Theme::builtin(name).is_some(),
                "advertised theme `{name}` does not resolve via Theme::builtin"
            );
        }
    }

    /// `darcula` (`IntelliJ`) and `dracula` (purple) are deliberate near-twins;
    /// both must resolve, both must be advertised, and they must be *different*
    /// palettes so a one-letter typo never silently yields the wrong theme.
    #[test]
    fn darcula_and_dracula_are_distinct_themes() {
        let darcula = Theme::builtin("darcula").expect("darcula resolves");
        let dracula = Theme::builtin("dracula").expect("dracula resolves");
        assert!(
            Theme::builtin_names().contains(&"darcula"),
            "darcula must be advertised in builtin_names"
        );
        assert_ne!(
            darcula.palette.accent, dracula.palette.accent,
            "the near-twins must not share an accent (proves they're distinct)"
        );
        // Darcula's signature warm-charcoal editor background.
        assert_eq!(darcula.palette.code_bg, Color::Rgb(0x2B, 0x2B, 0x2B));
    }

    #[test]
    fn diff_bg_tints_truecolor_as_rgb_and_indexed_as_indexed() {
        // zo is a true-color (Rgb) palette → an RGB tint is produced.
        let zo = Theme::zo();
        assert!(matches!(zo.diff_add_bg(), Some(Color::Rgb(..))));
        assert!(matches!(zo.diff_del_bg(), Some(Color::Rgb(..))));
        // dark is an indexed palette → roadmap ⑦: the wash is now restored, but
        // it stays in the *indexed* space (safe without truecolor) rather than
        // vanishing to None as before.
        let dark = Theme::default_dark();
        assert!(matches!(dark.diff_add_bg(), Some(Color::Indexed(..))));
        assert!(matches!(dark.diff_del_bg(), Some(Color::Indexed(..))));
        // no-color (Reset palette) has no blendable value → still foreground-only.
        let nc = Theme::no_color();
        assert_eq!(nc.diff_add_bg(), None);
    }

    #[test]
    fn truecolor_diff_bands_keep_the_surface_quiet_and_words_stronger() {
        let theme = Theme::zo();
        let base = as_rgb(theme.palette.code_bg);
        let distance = |a: (u8, u8, u8), b: (u8, u8, u8)| -> u32 {
            let d = |x: u8, y: u8| (i32::from(x) - i32::from(y)).unsigned_abs();
            d(a.0, b.0) * d(a.0, b.0)
                + d(a.1, b.1) * d(a.1, b.1)
                + d(a.2, b.2) * d(a.2, b.2)
        };

        for (semantic, line, word) in [
            (
                theme.palette.success,
                theme.diff_add_bg().expect("add band"),
                theme.diff_add_emphasis_bg().expect("add word emphasis"),
            ),
            (
                theme.palette.error,
                theme.diff_del_bg().expect("delete band"),
                theme.diff_del_emphasis_bg().expect("delete word emphasis"),
            ),
        ] {
            let semantic = as_rgb(semantic);
            let line = as_rgb(line);
            let word = as_rgb(word);
            assert!(
                distance(line, base) < distance(line, semantic),
                "the row band must sit closer to the code surface than the semantic color"
            );
            assert!(
                distance(line, base) < distance(word, base)
                    && distance(word, base) < distance(semantic, base),
                "changed words must form the middle emphasis step"
            );
            assert!(
                contrast(semantic, line) >= 4.5,
                "the semantic +/- marker must stay readable on its row band"
            );
            assert!(
                contrast(as_rgb(theme.palette.fg), word) >= 4.5,
                "body text must stay readable on changed-word emphasis"
            );
        }
    }

    #[test]
    fn tool_card_bg_tints_on_color_palettes_and_none_on_neutral() {
        // The tool-card wash blends the outcome color over the code surface: a
        // faint RGB tint on truecolor, an indexed tint on indexed palettes, and
        // None on neutral (NO_COLOR) where the caller skips the wash.
        let zo = Theme::zo();
        assert!(matches!(
            zo.tool_card_bg(zo.palette.success),
            Some(Color::Rgb(..))
        ));
        assert!(matches!(
            zo.tool_card_bg(zo.palette.error),
            Some(Color::Rgb(..))
        ));
        let dark = Theme::default_dark();
        assert!(matches!(
            dark.tool_card_bg(dark.palette.success),
            Some(Color::Indexed(..))
        ));
        let nc = Theme::no_color();
        assert_eq!(nc.tool_card_bg(nc.palette.success), None);
    }

    #[test]
    fn indexed_diff_bands_keep_semantics_and_word_emphasis() {
        // Editor-style review rows need distinct semantic bands even after RGB
        // blending is quantized back into the ANSI-256 palette. A weaker wash
        // collapses both colors onto the same grayscale slot.
        let dark = Theme::default_dark();
        let add_line = dark.diff_add_bg().expect("indexed add band");
        let del_line = dark.diff_del_bg().expect("indexed delete band");
        let add_word = dark.diff_add_emphasis_bg().expect("indexed add emphasis");
        let del_word = dark.diff_del_emphasis_bg().expect("indexed delete emphasis");
        assert_ne!(add_line, del_line, "add/delete bands must remain distinct");
        assert_ne!(add_line, add_word, "add word emphasis must remain visible");
        assert_ne!(del_line, del_word, "delete word emphasis must remain visible");
    }

    #[test]
    fn ansi256_rgb_conversions_are_sane() {
        // index→rgb spot checks across cube ends + grayscale ramp ends.
        assert_eq!(ansi256_to_rgb(16), (0, 0, 0));
        assert_eq!(ansi256_to_rgb(231), (255, 255, 255));
        assert_eq!(ansi256_to_rgb(196), (255, 0, 0)); // cube pure red
        assert_eq!(ansi256_to_rgb(232), (8, 8, 8));
        assert_eq!(ansi256_to_rgb(255), (238, 238, 238));
        // rgb→index: pure colors and black/white land on the cube.
        assert_eq!(rgb_to_ansi256(0, 0, 0), 16);
        assert_eq!(rgb_to_ansi256(255, 255, 255), 231);
        assert_eq!(rgb_to_ansi256(255, 0, 0), 196);
        // Non-gray cube indices (the typical blend output) round-trip exactly.
        // (Cube *grays* legitimately map to the closer grayscale ramp instead,
        // so a full 16..=255 round-trip is not an invariant — only that a blended
        // color lands on a real, nearby slot, which these representative non-gray
        // indices prove.)
        for i in [196u8, 46, 21, 226, 51, 201, 130, 67, 114, 203] {
            let (r, g, b) = ansi256_to_rgb(i);
            assert_eq!(rgb_to_ansi256(r, g, b), i, "non-gray cube index {i} round-trips");
        }
        // The grayscale ramp round-trips exactly (8+10k lands back on its step).
        for i in 232u8..=255 {
            let (r, g, b) = ansi256_to_rgb(i);
            assert_eq!(rgb_to_ansi256(r, g, b), i, "gray ramp index {i} round-trips");
        }
    }

    #[test]
    fn parse_hex_accepts_rrggbb_and_rejects_garbage() {
        use super::parse_hex;
        assert_eq!(parse_hex("#FF9D5C"), Some(Color::Rgb(0xFF, 0x9D, 0x5C)));
        // `#` is optional and parsing is case-insensitive.
        assert_eq!(parse_hex("1a1712"), Some(Color::Rgb(0x1A, 0x17, 0x12)));
        // Malformed input falls through to `None`.
        assert_eq!(parse_hex("#FFF"), None);
        assert_eq!(parse_hex("#GG0000"), None);
        assert_eq!(parse_hex(""), None);
    }

    #[test]
    fn resolve_color_precedence_hex_then_ansi_then_fallback() {
        use super::{TokenColorEntry, resolve_color};
        use std::collections::BTreeMap;

        let mut table: BTreeMap<String, TokenColorEntry> = BTreeMap::new();
        table.insert(
            "hex_wins".into(),
            TokenColorEntry {
                hex: Some("#102030".into()),
                ansi256: Some(42),
            },
        );
        table.insert(
            "ansi_only".into(),
            TokenColorEntry {
                hex: None,
                ansi256: Some(42),
            },
        );
        table.insert(
            "empty".into(),
            TokenColorEntry {
                hex: None,
                ansi256: None,
            },
        );

        let fb = Color::Rgb(1, 2, 3);
        // hex beats ansi256.
        assert_eq!(
            resolve_color(&table, "hex_wins", fb),
            Color::Rgb(0x10, 0x20, 0x30)
        );
        // ansi256 used when hex absent.
        assert_eq!(resolve_color(&table, "ansi_only", fb), Color::Indexed(42));
        // empty entry → fallback (not Reset).
        assert_eq!(resolve_color(&table, "empty", fb), fb);
        // missing key → fallback.
        assert_eq!(resolve_color(&table, "absent", fb), fb);
    }

    /// The live boot path loads `.zo/design/tokens.json`, which mirrors
    /// `Theme::zo()`. This guards the "appearance unchanged" invariant:
    /// a tokens file carrying the Zo hex values must reproduce the exact
    /// Zo palette so wiring `Theme::load` into boot causes no visual
    /// regression.
    #[test]
    fn zo_mirroring_tokens_reproduce_zo_palette() {
        use std::io::Write;

        let zo = Theme::zo();
        let hex = |c: Color| match c {
            Color::Rgb(r, g, b) => format!("#{r:02X}{g:02X}{b:02X}"),
            _ => unreachable!("zo palette is all true-color"),
        };
        let p = &zo.palette;
        let json = format!(
            r#"{{
              "color": {{
                "primary": {{ "accent": {{"hex":"{}"}}, "accent_dim": {{"hex":"{}"}} }},
                "secondary": {{ "cyan": {{"hex":"{}"}}, "violet": {{"hex":"{}"}}, "teal": {{"hex":"{}"}} }},
                "neutral": {{ "fg": {{"hex":"{}"}}, "bright": {{"hex":"{}"}}, "dim": {{"hex":"{}"}}, "muted": {{"hex":"{}"}}, "faint": {{"hex":"{}"}}, "code_bg": {{"hex":"{}"}} }},
                "semantic": {{ "success": {{"hex":"{}"}}, "warn": {{"hex":"{}"}}, "error": {{"hex":"{}"}}, "info": {{"hex":"{}"}} }}
              }},
              "spacing": {{ "row_gap":1, "block_gap":1, "card_padding_x":1, "card_padding_y":0, "indent":2, "gutter":2, "hud_sep":1, "modal_padding_x":2, "modal_padding_y":1 }},
              "breakpoint": {{ "narrow": {{"max":59}}, "compact": {{"min":60,"max":99}}, "wide": {{"min":100}} }},
              "border_usage": {{}}
            }}"#,
            hex(p.accent),
            hex(p.accent_dim),
            hex(p.cyan),
            hex(p.violet),
            hex(p.teal),
            hex(p.fg),
            hex(p.bright),
            hex(p.dim),
            hex(p.muted),
            hex(p.faint),
            hex(p.code_bg),
            hex(p.success),
            hex(p.warn),
            hex(p.error),
            hex(p.info),
        );

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("zo_tokens_{}.json", std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp).expect("write temp tokens");
            f.write_all(json.as_bytes()).expect("write tokens body");
        }
        let loaded = Theme::load_canonical(&tmp).expect("canonical tokens load");
        let _ = std::fs::remove_file(&tmp);

        // Terminal capability quantization is covered separately; this test
        // compares the canonical token palette before environment-dependent
        // ANSI/NO_COLOR projection.
        assert_eq!(loaded.palette.accent, zo.palette.accent);
        assert_eq!(loaded.palette.accent_dim, zo.palette.accent_dim);
        assert_eq!(loaded.palette.cyan, zo.palette.cyan);
        assert_eq!(loaded.palette.violet, zo.palette.violet);
        assert_eq!(loaded.palette.teal, zo.palette.teal);
        assert_eq!(loaded.palette.fg, zo.palette.fg);
        assert_eq!(loaded.palette.bright, zo.palette.bright);
        assert_eq!(loaded.palette.dim, zo.palette.dim);
        assert_eq!(loaded.palette.muted, zo.palette.muted);
        assert_eq!(loaded.palette.faint, zo.palette.faint);
        assert_eq!(loaded.palette.code_bg, zo.palette.code_bg);
        assert_eq!(loaded.palette.success, zo.palette.success);
        assert_eq!(loaded.palette.warn, zo.palette.warn);
        assert_eq!(loaded.palette.error, zo.palette.error);
        assert_eq!(loaded.palette.info, zo.palette.info);
    }

    /// The brand accent is reserved for the user rail, focus borders, and
    /// live/spinner moments; `warn` is a semantic status that appears inside
    /// body content. If a theme reuses the accent hue for `warn`, warnings
    /// masquerade as the brand and the "amber = focus" grammar collapses
    /// (the CC "orange mess" failure mode). Guards the zo/gruvbox/ayu
    /// collisions fixed in the v2 visual pass.
    #[test]
    fn warn_is_distinct_from_accent_in_every_builtin_theme() {
        for name in Theme::builtin_names() {
            let theme = Theme::builtin(name).expect("builtin theme");
            if theme.no_color {
                continue; // NO_COLOR collapses the palette to Reset by design.
            }
            assert_ne!(
                theme.palette.accent, theme.palette.warn,
                "theme `{name}`: warn must not reuse the brand accent"
            );
        }
    }

    /// A tokens file that carries only `color` — omitting the `spacing`,
    /// `breakpoint`, and `border_usage` sections entirely — must still load,
    /// falling back to the built-in spacing/breakpoint defaults. This guards
    /// the fix for the "dead required field" paradox: those sections had no
    /// (or one) live consumer yet were required to parse, so a sparse token
    /// file failed to load outright.
    #[test]
    fn sparse_tokens_omitting_spacing_and_breakpoint_load_with_defaults() {
        use std::io::Write;

        let json = r#"{
          "color": {
            "primary": {}, "secondary": {}, "neutral": {}, "semantic": {}
          }
        }"#;

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("zo_sparse_tokens_{}.json", std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp).expect("write temp tokens");
            f.write_all(json.as_bytes()).expect("write tokens body");
        }
        let loaded = Theme::load(&tmp).expect("sparse tokens must load");
        let _ = std::fs::remove_file(&tmp);

        // Omitted sections fall back to the same values the no-tokens theme uses.
        let fallback = Spacing::fallback();
        assert_eq!(loaded.spacing.block_gap, fallback.block_gap);
        assert_eq!(loaded.spacing.card_padding_x, fallback.card_padding_x);
        assert_eq!(loaded.narrow_max, Theme::DEFAULT_NARROW_MAX);
        assert_eq!(loaded.wide_min, Theme::DEFAULT_WIDE_MIN);
    }

    /// A `spacing` section that overrides only one key keeps its override while
    /// every omitted key falls back to the default (container `#[serde(default)]`).
    #[test]
    fn partial_spacing_section_defaults_omitted_keys() {
        use std::io::Write;

        let json = r#"{
          "color": {
            "primary": {}, "secondary": {}, "neutral": {}, "semantic": {}
          },
          "spacing": { "block_gap": 3 }
        }"#;

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("zo_partial_spacing_{}.json", std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp).expect("write temp tokens");
            f.write_all(json.as_bytes()).expect("write tokens body");
        }
        let loaded = Theme::load(&tmp).expect("partial spacing must load");
        let _ = std::fs::remove_file(&tmp);

        // The present key wins; omitted keys default.
        assert_eq!(loaded.spacing.block_gap, 3);
        assert_eq!(loaded.spacing.indent, Spacing::fallback().indent);
    }

    // ── Glassmorphism tokens (v3 §10) ──

    /// WCAG relative-luminance contrast between two RGB colors.
    fn contrast(a: (u8, u8, u8), b: (u8, u8, u8)) -> f64 {
        fn lum((r, g, b): (u8, u8, u8)) -> f64 {
            fn chan(v: u8) -> f64 {
                let s = f64::from(v) / 255.0;
                if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
            }
            0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)
        }
        let (la, lb) = (lum(a), lum(b));
        let (hi, lo) = if la > lb { (la, lb) } else { (lb, la) };
        (hi + 0.05) / (lo + 0.05)
    }

    fn as_rgb(c: Color) -> (u8, u8, u8) {
        match c {
            Color::Rgb(r, g, b) => (r, g, b),
            Color::Indexed(i) => ansi256_to_rgb(i),
            other => panic!("expected a concrete color, got {other:?}"),
        }
    }

    // ── Heat tokens (v4 H0) ──

    fn heat_colors(theme: &Theme) -> Vec<Color> {
        let heat = theme.heat();
        let mut colors = Vec::with_capacity(51);
        colors.extend([
            heat.steel,
            heat.steel_dim,
            heat.ember,
            heat.molten,
            heat.spark,
        ]);
        colors.extend(heat.ramp);
        colors.extend(heat.ignition);
        colors.extend(heat.rail_fade);
        colors.extend(heat.fill_fade);
        colors
    }

    fn component(rgb: (u8, u8, u8), channel: usize) -> u8 {
        match channel {
            0 => rgb.0,
            1 => rgb.1,
            _ => rgb.2,
        }
    }

    fn assert_monotonic_toward(colors: &[Color], target: Color, context: &str) {
        let target = as_rgb(target);
        let start = as_rgb(colors[0]);
        for channel in 0..3 {
            let target_component = component(target, channel);
            let increasing = target_component >= component(start, channel);
            for pair in colors.windows(2) {
                let before = component(as_rgb(pair[0]), channel);
                let after = component(as_rgb(pair[1]), channel);
                if increasing {
                    assert!(
                        after >= before,
                        "{context} heat channel {channel} must not decrease: {before} -> {after}"
                    );
                } else {
                    assert!(
                        after <= before,
                        "{context} heat channel {channel} must not increase: {before} -> {after}"
                    );
                }
                assert!(
                    after.abs_diff(target_component) <= before.abs_diff(target_component),
                    "{context} heat channel {channel} must move toward its target"
                );
            }
        }
    }

    #[test]
    fn every_colored_builtin_has_complete_heat_tokens() {
        for name in Theme::builtin_names() {
            let theme = Theme::builtin(name).expect("builtin theme resolves");
            let heat = theme.heat();
            assert_eq!(heat.ember, theme.palette.accent, "{name} ember");
            if matches!(theme.palette.accent, Color::Rgb(..)) {
                assert_ne!(
                    as_rgb(heat.molten),
                    as_rgb(heat.ember),
                    "{name} molten must differ"
                );
                assert_ne!(
                    as_rgb(heat.steel),
                    as_rgb(theme.palette.dim),
                    "{name} steel must differ"
                );
            }
            assert!(
                heat_colors(&theme).into_iter().all(|color| color != Color::Reset),
                "{name} must have no Reset heat token"
            );
        }
    }

    #[test]
    fn zo_heat_anchors_match_the_brand_palette() {
        let theme = Theme::zo();
        let heat = theme.heat();
        assert_eq!(heat.steel, Color::Rgb(0x7E, 0x96, 0xB8));
        assert_eq!(heat.steel_dim, Color::Rgb(0x5C, 0x64, 0x70));
        assert_eq!(heat.ember, Color::Rgb(0xF5, 0xA5, 0x24));
        assert_eq!(heat.molten, Color::Rgb(0xFF, 0x7A, 0x45));
        assert_eq!(heat.spark, Color::Rgb(0xFF, 0xD9, 0xA0));
    }

    #[test]
    fn heat_tokens_preserve_the_palette_color_space() {
        let theme = Theme::tokyonight();
        let heat = theme.heat();
        assert_eq!(
            heat.molten,
            super::blend_toward(theme.palette.accent, theme.palette.error, 0.35)
                .expect("concrete palette")
        );
        assert_eq!(
            heat.spark,
            super::blend_toward(theme.palette.accent, theme.palette.bright, 0.45)
                .expect("concrete palette")
        );
        assert_eq!(
            heat.steel,
            super::blend_toward(theme.palette.dim, theme.palette.info, 0.25)
                .expect("concrete palette")
        );
        assert_eq!(
            heat.steel_dim,
            super::blend_toward(theme.palette.muted, theme.palette.info, 0.20)
                .expect("concrete palette")
        );
        assert!(
            heat_colors(&theme)
                .into_iter()
                .all(|color| matches!(color, Color::Indexed(_))),
            "every TokyoNight heat token must remain ANSI-256"
        );
        assert!(
            heat_colors(&Theme::zo())
                .into_iter()
                .all(|color| matches!(color, Color::Rgb(..))),
            "every Zo heat token must remain truecolor"
        );
    }

    #[test]
    fn neutral_theme_heat_tokens_are_all_reset() {
        let heat = Theme::no_color();
        assert!(
            heat_colors(&heat)
                .into_iter()
                .all(|color| color == Color::Reset)
        );
        assert_eq!(heat.heat().wordmark_gradient(5), vec![Color::Reset; 5]);
    }

    #[test]
    fn heat_ramps_keep_exact_endpoints_and_monotonic_components() {
        for name in Theme::builtin_names() {
            let theme = Theme::builtin(name).expect("builtin theme resolves");
            let heat = theme.heat();
            assert_eq!(heat.ramp[0], heat.ember, "{name} hot endpoint");
            assert_eq!(heat.ramp[7], heat.steel, "{name} cold endpoint");
            if matches!(theme.palette.accent, Color::Rgb(..)) {
                assert_monotonic_toward(&heat.ramp, heat.steel, name);
            }
        }
    }

    #[test]
    fn ignition_ramp_lerps_through_all_three_anchors() {
        let theme = Theme::zo();
        let heat = theme.heat();

        assert_eq!(heat.ignition[0], heat.ember);
        assert_eq!(heat.ignition[8], heat.molten);
        assert_eq!(heat.ignition[15], heat.spark);
        assert_monotonic_toward(&heat.ignition[..=8], heat.molten, "zo ignition hot rise");
        assert_monotonic_toward(&heat.ignition[8..], heat.spark, "zo ignition crest rise");
    }

    #[test]
    fn wordmark_gradient_has_exact_endpoints_and_monotonic_components() {
        let theme = Theme::zo();
        let heat = theme.heat();
        let gradient = heat.wordmark_gradient(5);

        assert_eq!(gradient.len(), 5);
        assert_eq!(gradient[0], heat.ember);
        assert_eq!(gradient[4], heat.molten);
        assert_monotonic_toward(&gradient, heat.molten, "zo wordmark");
    }

    #[test]
    fn rail_fades_from_molten_bottom_to_ember_top() {
        for name in Theme::builtin_names() {
            let theme = Theme::builtin(name).expect("builtin theme resolves");
            let heat = theme.heat();
            assert_eq!(heat.rail_fade[0], heat.molten, "{name} rail bottom");
            assert_eq!(heat.rail_fade[9], heat.ember, "{name} rail top");
            if matches!(theme.palette.accent, Color::Rgb(..)) {
                assert_monotonic_toward(&heat.rail_fade, heat.ember, name);
            }
        }
    }

    #[test]
    fn fill_fade_runs_from_ember_to_steel_dim() {
        for name in Theme::builtin_names() {
            let theme = Theme::builtin(name).expect("builtin theme resolves");
            let heat = theme.heat();
            assert_eq!(heat.fill_fade[0], heat.ember, "{name} fill start");
            assert_eq!(heat.fill_fade[11], heat.steel_dim, "{name} fill end");
            if matches!(theme.palette.accent, Color::Rgb(..)) {
                assert_monotonic_toward(&heat.fill_fade, heat.steel_dim, name);
            }
        }
    }

    #[test]
    fn zo_heat_tokens_keep_chrome_contrast() {
        let theme = Theme::zo();
        let heat = theme.heat();
        let code_bg = as_rgb(theme.palette.code_bg);
        for (name, color) in [("spark", heat.spark), ("ember", heat.ember)] {
            let ratio = contrast(as_rgb(color), code_bg);
            assert!(ratio >= 4.5, "{name} contrast must be >= 4.5:1, got {ratio:.2}");
        }
        // Steel is non-body chrome, so the WCAG graphical-component floor is 3.0:1.
        let steel_ratio = contrast(as_rgb(heat.steel), code_bg);
        assert!(
            steel_ratio >= 3.0,
            "steel contrast must be >= 3.0:1, got {steel_ratio:.2}"
        );
    }

    /// Body text on the brightest glass surface must stay readable — the
    /// ui-ux-pro-max guardrail (>= 4.5:1) that keeps the frosted look from
    /// costing legibility, pinned for both truecolor themes.
    #[test]
    fn glass_surfaces_keep_body_text_contrast() {
        for theme in [Theme::zo(), Theme::default_dark()] {
            let surface = theme.surface2().expect("truecolor theme has surface2");
            let ratio = contrast(as_rgb(theme.palette.fg), as_rgb(surface));
            assert!(
                ratio >= 4.5,
                "body fg on surface2 must be >= 4.5:1, got {ratio:.2}"
            );
        }
    }

    /// The glass elevation ladder must be ordered (surface2 sits closer to the
    /// foreground than surface1, the edge brighter than both) and the whole
    /// layer must vanish on the `NO_COLOR` neutral palette.
    #[test]
    fn glass_tokens_are_ordered_and_vanish_without_color() {
        let t = Theme::zo();
        let base = as_rgb(t.palette.code_bg);
        let fg = as_rgb(t.palette.fg);
        let dist = |c: (u8, u8, u8), to: (u8, u8, u8)| -> u32 {
            let d = |x: u8, y: u8| (i32::from(x) - i32::from(y)).unsigned_abs();
            d(c.0, to.0) * d(c.0, to.0) + d(c.1, to.1) * d(c.1, to.1) + d(c.2, to.2) * d(c.2, to.2)
        };
        let s1 = as_rgb(t.surface1().expect("surface1"));
        let s2 = as_rgb(t.surface2().expect("surface2"));
        let edge = as_rgb(t.border_glass().expect("border_glass"));
        assert!(dist(s1, base) < dist(s2, base), "surface2 must sit above surface1");
        assert!(dist(s2, fg) > dist(edge, fg), "the glass edge must be brighter than surfaces");

        let plain = Theme::no_color();
        assert_eq!(plain.surface1(), None);
        assert_eq!(plain.surface2(), None);
        assert_eq!(plain.border_glass(), None);
        let kept = Color::Reset;
        assert_eq!(plain.scrim_fg(kept), kept, "scrim must be identity under NO_COLOR");
    }

    /// The scrim pulls an arbitrary cell fg toward the surface base — closer
    /// to the base than the original, but never all the way (content behind
    /// the modal stays visible).
    #[test]
    fn scrim_mutes_but_does_not_erase() {
        let t = Theme::zo();
        let base = as_rgb(t.palette.code_bg);
        let original = Color::Rgb(220, 220, 220);
        let muted = as_rgb(t.scrim_fg(original));
        let dist = |c: (u8, u8, u8), to: (u8, u8, u8)| -> u32 {
            let d = |x: u8, y: u8| (i32::from(x) - i32::from(y)).unsigned_abs();
            d(c.0, to.0) * d(c.0, to.0) + d(c.1, to.1) * d(c.1, to.1) + d(c.2, to.2) * d(c.2, to.2)
        };
        assert!(dist(muted, base) < dist(as_rgb(original), base), "scrim must move fg toward the base");
        assert_ne!(muted, base, "scrim must not erase the content entirely");
    }

    fn luma(color: Color) -> u32 {
        let (r, g, b) = super::color_to_rgb(color).expect("rgb-resolvable color");
        (u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114) / 1000
    }

    /// The live "diff 배경과 글씨가 구분 안 됨" report: indexed palettes must
    /// use the dedicated dark/pastel diff slots (never a mid-tone blend that
    /// quantizes to the text's own luminance), and every band must keep a
    /// wide luminance gap to the body fg so text stays readable on it.
    #[test]
    fn indexed_diff_bands_pick_dedicated_slots_and_keep_text_contrast() {
        let dark = Theme::default_dark();
        assert_eq!(dark.diff_add_bg(), Some(Color::Indexed(22)));
        assert_eq!(dark.diff_del_bg(), Some(Color::Indexed(52)));
        assert_eq!(dark.diff_add_emphasis_bg(), Some(Color::Indexed(28)));
        assert_eq!(dark.diff_del_emphasis_bg(), Some(Color::Indexed(88)));

        let light = Theme::default_light();
        assert_eq!(light.diff_add_bg(), Some(Color::Indexed(194)));
        assert_eq!(light.diff_del_bg(), Some(Color::Indexed(224)));
        assert_eq!(light.diff_add_emphasis_bg(), Some(Color::Indexed(157)));
        assert_eq!(light.diff_del_emphasis_bg(), Some(Color::Indexed(217)));

        // Contrast contract: body fg vs every band, both themes. The old
        // blended bands ((95,135,95), (215,95,95)) sat ~40 luma from the fg;
        // the dedicated slots keep a comfortable gap.
        for theme in [&dark, &light] {
            let fg_luma = luma(theme.palette.fg);
            for band in [
                theme.diff_add_bg(),
                theme.diff_del_bg(),
                theme.diff_add_emphasis_bg(),
                theme.diff_del_emphasis_bg(),
            ] {
                let band = band.expect("indexed palettes render diff bands");
                let gap = fg_luma.abs_diff(luma(band));
                assert!(
                    gap >= 100,
                    "{}: fg luma {fg_luma} vs band {band:?} luma {} — gap {gap} too small",
                    theme.name,
                    luma(band)
                );
            }
        }

        // NO_COLOR keeps the no-wash contract.
        assert_eq!(Theme::no_color().diff_add_bg(), None);
        assert_eq!(Theme::no_color().diff_del_emphasis_bg(), None);
    }
}
