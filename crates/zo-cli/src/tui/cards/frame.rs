//! `CardFrame` — the single source of truth for framed-surface *chrome*.
//!
//! A border is the signal of something special, so every framed surface (modal,
//! result card, hint popup, danger guard, inner pane) must share one brand
//! border recipe instead of each call site repeating
//! `Block::default().borders(ALL).border_type(..).border_style(accent)`. All
//! color flows through [`Theme`] so `NO_COLOR` degrades automatically, and the
//! borders are box-drawing glyphs (1 cell everywhere — ambiguous-width immune),
//! so a wide/CJK title can never widen the frame.
//!
//! ## Pi dialog idiom (2026-07-28)
//!
//! Dialog-role surfaces ([`SurfaceKind::Modal`], [`SurfaceKind::Popup`],
//! [`SurfaceKind::Panel`], [`SurfaceKind::Danger`]) no longer draw a closed box.
//! They are delimited by two full-width horizontal rules — a top rule that the
//! title rides (`─ Title ─────`) and a bottom rule — with **no side borders and
//! no corners**, which is Pi's dialog signature. Two consequences the call sites
//! rely on:
//!
//! * The inner rect is now the *full* width of the area (only 2 rows shorter),
//!   so content gains the two columns the side borders used to eat.
//! * The [`Theme`] border-role map still selects the rule glyph — with only
//!   horizontal edges, `Rounded`/`Plain` both render `─` and `Thick` renders `━`,
//!   so the danger guard keeps its heavier rule for free.
//!
//! [`SurfaceKind::Card`] is deliberately *not* a dialog: it is a transcript
//! content card that keeps its closed rounded box (and is pinned by the
//! transcript golden fixtures).
//!
//! This owns only the closed, `Rect`-rendered chrome. The inline streaming
//! code-fence rail ([`super::super::markdown::code_card_frame_lines`]) is a
//! deliberately different visual language (3-sided open muted rail, emitted as
//! append-only `Line`s for the streaming/settle parity contract) and stays
//! line-based — see that module. The two are separate leaves under `cards/` on
//! purpose.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding};

use super::super::theme::Theme;

/// The kind of framed surface, which picks the brand border recipe (border
/// type, border color, surface fill, default padding).
//
// `Card`/`Danger`/`Panel` land with their respective surface migrations so no
// variant is ever unconstructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    /// A full-screen overlay modal: Pi dialog rules (top + bottom, no sides) in
    /// `border` — `border_accent` when focused — over an elevation-2 fill.
    Modal,
    /// An input-anchored hint popup (`@mention` / slash commands): the same Pi
    /// dialog rules and elevation-2 fill as [`SurfaceKind::Modal`], typically
    /// carrying a bottom key-hint via [`CardFrame::title_bottom`].
    Popup,
    /// A warning guard (permission prompt / confirm): heavy `━` rules in bold
    /// `error` and no surface fill, its rule color set by the caller via
    /// [`CardFrame::border_style`] to track focus/severity.
    Danger,
    /// A transcript content card (status/report card, agent result): a rounded
    /// `accent_dim` border, no surface fill, with the caller supplying its
    /// horizontal padding. The border color may be overridden via
    /// [`CardFrame::border_style`] to track focus.
    Card,
    /// An inner pane nested inside a modal (a viewer column, a selection box):
    /// quiet `dim` rules and no fill, so nested chrome reads as subordinate to
    /// the modal's frame. The active pane switches to `border_accent` via
    /// [`CardFrame::focused`] (or an explicit [`CardFrame::border_style`]).
    Panel,
}

/// Resolved chrome recipe for one [`SurfaceKind`].
struct Recipe {
    /// Theme border-role key, which selects the rule glyph weight.
    role: &'static str,
    /// Which edges are drawn. Dialog roles draw only `TOP | BOTTOM`.
    borders: Borders,
    /// Rule (and rule-glyph) style.
    rule: Style,
    /// Surface fill, or `None` to leave the body's own background.
    fill: Option<Style>,
    /// Default inner padding.
    padding: Padding,
}

/// Builder for a framed chrome surface. The [`SurfaceKind`] supplies the brand
/// defaults; the caller supplies only content and any per-site override (title,
/// padding). [`CardFrame::block`] is the one place a chrome [`Block`] is built.
pub struct CardFrame<'a> {
    kind: SurfaceKind,
    theme: &'a Theme,
    title: Option<Line<'a>>,
    title_bottom: Option<Line<'a>>,
    padding: Option<Padding>,
    border_style: Option<Style>,
    focused: bool,
}

impl<'a> CardFrame<'a> {
    /// Start a frame of `kind` styled from `theme`.
    #[must_use]
    pub fn new(kind: SurfaceKind, theme: &'a Theme) -> Self {
        Self {
            kind,
            theme,
            title: None,
            title_bottom: None,
            padding: None,
            border_style: None,
            focused: false,
        }
    }

    /// Set the top title. The caller supplies a fully-styled [`Line`] (the frame
    /// does not impose a title style), so a plain-string title stays body-styled
    /// and a `heading_1`-styled title passes through unchanged.
    #[must_use]
    pub fn title(mut self, title: impl Into<Line<'a>>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set a bottom-edge title, drawn on the lower border — used by popups for a
    /// muted key-hint line. As with [`Self::title`], the caller supplies the full
    /// style.
    #[must_use]
    pub fn title_bottom(mut self, title: impl Into<Line<'a>>) -> Self {
        self.title_bottom = Some(title.into());
        self
    }

    /// Override the surface kind's default border color. Danger cards use this
    /// to track a focus/severity state (e.g. `error` when focused, `warn`
    /// otherwise); other surfaces keep their brand border and never call it.
    #[must_use]
    pub fn border_style(mut self, style: Style) -> Self {
        self.border_style = Some(style);
        self
    }

    /// Set whether the card holds keyboard focus. This directs the border
    /// style to use the brand accent color when focused.
    #[must_use]
    pub fn focused(mut self, focused: bool) -> Self {
        self.focused = focused;
        self
    }

    /// Override the surface kind's default inner padding.
    #[must_use]
    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = Some(padding);
        self
    }

    /// The two horizontal rules a Pi dialog is delimited by.
    const DIALOG_RULES: Borders = Borders::TOP.union(Borders::BOTTOM);

    /// Rule color for a dialog surface: the resting `border` blue, switching to
    /// the `border_accent` focus cyan while the surface holds focus.
    fn dialog_rule(&self) -> Style {
        if self.focused {
            Style::new().fg(self.theme.palette.border_accent)
        } else {
            Style::new().fg(self.theme.palette.border)
        }
    }

    /// The elevation-2 glass fill shared by the modal and popup dialogs.
    fn glass_fill(&self) -> Style {
        self.theme
            .typography
            .body
            .bg(self.theme.surface2().unwrap_or(self.theme.palette.code_bg))
    }

    /// Border/fill/padding recipe for this surface kind.
    fn recipe(&self) -> Recipe {
        let theme = self.theme;
        match self.kind {
            // A dialog: two full-width rules, no sides, over the elevation-2
            // glass fill that keeps body text legible above the transcript.
            SurfaceKind::Modal | SurfaceKind::Popup => Recipe {
                role: "modal",
                borders: Self::DIALOG_RULES,
                rule: self.dialog_rule(),
                fill: Some(self.glass_fill()),
                padding: Padding::ZERO,
            },
            // A danger guard: heavy `━` rules and NO fill. The default rule is
            // the strongest `error`; callers restyle it via `border_style` (a
            // permission prompt drops to `warn` while unfocused).
            SurfaceKind::Danger => Recipe {
                role: "permission_card",
                borders: Self::DIALOG_RULES,
                rule: Style::new()
                    .fg(theme.palette.error)
                    .add_modifier(Modifier::BOLD),
                fill: None,
                padding: Padding::ZERO,
            },
            // A transcript content card — NOT a dialog. It keeps its closed
            // rounded box: it sits inline in the scrollback where a pair of
            // bare rules would read as a page break rather than a card.
            SurfaceKind::Card => Recipe {
                role: "tool_call_card",
                borders: Borders::ALL,
                rule: if self.focused {
                    Style::new().fg(theme.palette.accent)
                } else {
                    Style::new().fg(theme.palette.accent_dim)
                },
                fill: None,
                padding: Padding::ZERO,
            },
            // A nested modal pane: quiet `dim` rules and no fill so it reads as
            // subordinate to the dialog that hosts it.
            SurfaceKind::Panel => Recipe {
                role: "modal",
                borders: Self::DIALOG_RULES,
                rule: if self.focused {
                    Style::new().fg(theme.palette.border_accent)
                } else {
                    theme.typography.dim
                },
                fill: None,
                padding: Padding::ZERO,
            },
        }
    }

    /// Compose the top-rule title: `─ Title ─────`.
    ///
    /// The leading `─ ` and the trailing space are drawn in the rule style so
    /// the title reads as an inlay in the rule rather than a floating label;
    /// the caller's own spans keep their styling untouched. Outer whitespace on
    /// the caller's title is trimmed so the inlay spacing is exactly one cell on
    /// each side no matter how the call site padded its string.
    fn rule_title(title: Line<'a>, rule: Style) -> Line<'a> {
        let mut spans: Vec<Span<'a>> = Vec::with_capacity(title.spans.len() + 2);
        spans.push(Span::styled("\u{2500} ", rule));
        let last = title.spans.len().saturating_sub(1);
        for (idx, span) in title.spans.into_iter().enumerate() {
            let mut content = span.content;
            if idx == 0 {
                content = match content {
                    std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(s.trim_start()),
                    std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(s.trim_start().to_string()),
                };
            }
            if idx == last {
                content = match content {
                    std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(s.trim_end()),
                    std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(s.trim_end().to_string()),
                };
            }
            if content.is_empty() {
                continue;
            }
            spans.push(Span::styled(content, span.style));
        }
        spans.push(Span::styled(" ", rule));
        let mut line = Line::from(spans);
        line.style = title.style;
        line.alignment = title.alignment;
        line
    }

    /// Build the chrome [`Block`]. The single point that constructs a framed
    /// border — every framed surface flows through here.
    #[must_use]
    pub fn block(self) -> Block<'a> {
        let recipe = self.recipe();
        let rule = self.border_style.unwrap_or(recipe.rule);
        let dialog = recipe.borders == Self::DIALOG_RULES;
        let mut block = Block::default()
            .borders(recipe.borders)
            .border_type(self.theme.borders.for_role(recipe.role))
            .border_style(rule)
            .padding(self.padding.unwrap_or(recipe.padding));
        if let Some(fill) = recipe.fill {
            block = block.style(fill);
        }
        if let Some(title) = self.title {
            // A dialog title rides the top rule as `─ Title ─────`; a card title
            // keeps sitting in the gap its corners already cut for it.
            block = block.title(if dialog {
                Self::rule_title(title, rule)
            } else {
                title
            });
        }
        if let Some(title_bottom) = self.title_bottom {
            block = block.title_bottom(if dialog {
                Self::rule_title(title_bottom, rule)
            } else {
                title_bottom
            });
        }
        block
    }

    /// Build the block, render it into `area`, and return the inner content
    /// `Rect` (the caller lays out its body there).
    pub fn render(self, frame: &mut Frame<'_>, area: Rect) -> Rect {
        let block = self.block();
        let inner = block.inner(area);
        frame.render_widget(block, area);
        inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::widgets::BorderType;

    fn theme() -> Theme {
        Theme::default_dark()
    }

    /// Read row `y` of `buf` as a `String` of glyphs.
    fn row(buf: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, y)).expect("cell").symbol())
            .collect()
    }

    #[test]
    fn modal_is_two_plain_rules_with_no_sides_over_the_glass_fill() {
        let theme = theme();

        let mut term = Terminal::new(TestBackend::new(20, 6)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Modal, &theme).block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buffer = term.backend().buffer();
        // Pi dialog: full-width top and bottom rules, nothing on the sides.
        assert_eq!(row(buffer, 0), "\u{2500}".repeat(20));
        assert_eq!(row(buffer, 5), "\u{2500}".repeat(20));
        assert_eq!(row(buffer, 2), " ".repeat(20), "no side borders");
        // Resting rules take the `border` role; the interior keeps the
        // elevation-2 glass fill so the body stays legible over the transcript.
        assert_eq!(buffer.content()[0].fg, theme.palette.border);
        assert_eq!(
            buffer.content()[usize::from(buffer.area.width) + 1].bg,
            theme.surface2().expect("glass surface")
        );
    }

    #[test]
    fn a_focused_dialog_switches_its_rules_to_the_focus_cyan() {
        let theme = theme();
        assert_ne!(theme.palette.border, theme.palette.border_accent);

        let mut term = Terminal::new(TestBackend::new(20, 4)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Modal, &theme)
                .focused(true)
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        assert_eq!(
            term.backend().buffer().content()[0].fg,
            theme.palette.border_accent
        );
    }

    #[test]
    fn a_dialog_title_rides_the_top_rule() {
        let theme = theme();

        let mut term = Terminal::new(TestBackend::new(20, 4)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Modal, &theme)
                .title(Line::styled("  Models  ", theme.typography.heading_1))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buf = term.backend().buffer();
        // `─ Title ─────`: one rule cell, one space, the (trimmed) title, one
        // space, then the rule runs to the right edge.
        assert_eq!(row(buf, 0), "\u{2500} Models \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
        assert_eq!(buf.content()[0].fg, theme.palette.border, "rule inlay");
        let title = buf
            .content()
            .iter()
            .find(|cell| cell.symbol() == "M")
            .expect("title on the rule");
        assert_eq!(title.fg, theme.palette.accent);
        assert!(title.modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn styled_title_keeps_its_color_over_the_accent_border() {
        let theme = theme();
        let title_fg = theme.palette.warn;
        assert_ne!(title_fg, theme.palette.accent, "test needs a distinct color");

        let mut term = Terminal::new(TestBackend::new(20, 4)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Modal, &theme)
                .title(Line::styled("TT", Style::new().fg(title_fg)))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buf = term.backend().buffer();
        let width = buf.area.width as usize;
        let title_cell = buf.content()[..width]
            .iter()
            .find(|cell| cell.symbol() == "T")
            .expect("title rendered on the top rule");
        // A title with an explicit fg is NOT overwritten by the rule style —
        // `modals::modal_frame` relies on this to control its own title color.
        assert_eq!(title_cell.fg, title_fg);
    }

    #[test]
    fn popup_is_a_glass_dialog_with_a_bottom_rule_title() {
        let theme = theme();

        let mut term = Terminal::new(TestBackend::new(24, 5)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Popup, &theme)
                .title(Line::styled(" hint ", theme.typography.body))
                .title_bottom(Line::styled("BB", Style::new().fg(theme.palette.dim)))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buf = term.backend().buffer();
        // No corners: the first cell is the rule inlay leading the title.
        assert_eq!(buf.content()[0].symbol(), "\u{2500}");
        assert_eq!(buf.content()[0].fg, theme.palette.border);
        assert!(row(buf, 0).starts_with("\u{2500} hint \u{2500}"));
        assert_eq!(
            buf.content()[usize::from(buf.area.width) + 1].bg,
            theme.surface2().expect("glass surface")
        );
        // The bottom title lands on the last row, tinted `dim` (a bottom title is
        // what separates a popup's key-hint from a modal's title-only chrome).
        let width = buf.area.width as usize;
        let height = buf.area.height as usize;
        let last_row = &buf.content()[(height - 1) * width..height * width];
        let hint = last_row
            .iter()
            .find(|cell| cell.symbol() == "B")
            .expect("bottom title rendered on the lower rule");
        assert_eq!(hint.fg, theme.palette.dim);
    }

    #[test]
    fn danger_uses_heavy_error_rules_that_callers_can_restyle() {
        let theme = theme();
        // The danger role stays thick so its rules read heavier than a dialog's.
        assert_eq!(theme.borders.for_role("permission_card"), BorderType::Thick);
        assert_ne!(theme.palette.error, theme.palette.warn, "test needs distinct colors");

        // Default danger rule is the strong `error` color, on the heavy glyph.
        let mut term = Terminal::new(TestBackend::new(20, 4)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Danger, &theme).block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buf = term.backend().buffer();
        assert_eq!(row(buf, 0), "\u{2501}".repeat(20)); // ━ full-width, no corners
        assert_eq!(row(buf, 1), " ".repeat(20), "no side borders");
        assert_eq!(buf.content()[0].fg, theme.palette.error);

        // A caller can restyle the border — a permission prompt drops to `warn`
        // while unfocused; the override wins over the recipe default.
        let mut term2 = Terminal::new(TestBackend::new(20, 4)).expect("backend");
        term2.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Danger, &theme)
                .border_style(Style::new().fg(theme.palette.warn))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        assert_eq!(term2.backend().buffer().content()[0].fg, theme.palette.warn);
    }

    #[test]
    fn card_has_a_rounded_accent_dim_border_and_no_surface_fill() {
        let theme = theme();
        assert_eq!(theme.borders.for_role("tool_call_card"), BorderType::Rounded);

        let mut term = Terminal::new(TestBackend::new(20, 5)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Card, &theme)
                .padding(Padding::horizontal(1))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buf = term.backend().buffer();
        let corner = &buf.content()[0];
        assert_eq!(corner.symbol(), "\u{256d}"); // ╭
        assert_eq!(corner.fg, theme.palette.accent_dim);
        // No surface fill: an interior cell keeps the default bg — a Card leaves
        // its fill to the body Paragraph, unlike a Modal which paints code_bg.
        let width = buf.area.width as usize;
        let interior = &buf.content()[width + 2]; // row 1, just inside border+pad
        assert_eq!(interior.bg, ratatui::style::Color::Reset);

        // The border tracks focus via an override (accent when focused).
        let mut term2 = Terminal::new(TestBackend::new(20, 5)).expect("backend");
        term2.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Card, &theme)
                .border_style(Style::new().fg(theme.palette.accent))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        assert_eq!(term2.backend().buffer().content()[0].fg, theme.palette.accent);
    }

    #[test]
    fn panel_uses_quiet_dim_rules_with_no_fill() {
        let theme = theme();
        let dim_fg = theme.typography.dim.fg.expect("dim style has a foreground");

        let mut term = Terminal::new(TestBackend::new(20, 5)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Panel, &theme)
                .title(Line::styled(" Phases ", theme.typography.dim))
                .block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buf = term.backend().buffer();
        assert!(row(buf, 0).starts_with("\u{2500} Phases \u{2500}"));
        assert_eq!(
            buf.content()[0].fg, dim_fg,
            "a nested panel reads with quiet dim rules"
        );
        // No surface fill: an interior cell keeps the default bg.
        let width = buf.area.width as usize;
        assert_eq!(buf.content()[width + 1].bg, ratatui::style::Color::Reset);
    }

    #[test]
    fn dialog_rules_give_content_the_full_width_and_cost_two_rows() {
        let theme = theme();
        let area = Rect::new(0, 0, 40, 12);
        let inner = CardFrame::new(SurfaceKind::Modal, &theme)
            .title("A wide title")
            .block()
            .inner(area);
        // Without side borders the body keeps every column; only the two rules
        // cost height. This is the +2 columns the modal bodies gained.
        assert_eq!(inner.width, area.width);
        assert_eq!(inner.x, area.x);
        assert_eq!(inner.height, area.height - 2);

        // The transcript Card is not a dialog and keeps its closed box.
        let card_inner = CardFrame::new(SurfaceKind::Card, &theme).block().inner(area);
        assert_eq!(card_inner.width, area.width - 2);
        assert_eq!(card_inner.height, area.height - 2);
    }

    #[test]
    fn card_frame_focused_builder_selects_accent_style() {
        let theme = theme();

        // Unfocused SurfaceKind::Card should use accent_dim.
        let mut term_unfocused = Terminal::new(TestBackend::new(20, 5)).expect("backend");
        term_unfocused
            .draw(|f| {
                let block = CardFrame::new(SurfaceKind::Card, &theme)
                    .focused(false)
                    .block();
                f.render_widget(block, f.area());
            })
            .expect("draw");
        assert_eq!(
            term_unfocused.backend().buffer().content()[0].fg,
            theme.palette.accent_dim
        );

        // Focused SurfaceKind::Card should automatically use accent.
        let mut term_focused = Terminal::new(TestBackend::new(20, 5)).expect("backend");
        term_focused
            .draw(|f| {
                let block = CardFrame::new(SurfaceKind::Card, &theme)
                    .focused(true)
                    .block();
                f.render_widget(block, f.area());
            })
            .expect("draw");
        assert_eq!(
            term_focused.backend().buffer().content()[0].fg,
            theme.palette.accent
        );

        // Focused SurfaceKind::Panel takes the dialog focus cyan.
        let mut term_panel_focused = Terminal::new(TestBackend::new(20, 5)).expect("backend");
        term_panel_focused
            .draw(|f| {
                let block = CardFrame::new(SurfaceKind::Panel, &theme)
                    .focused(true)
                    .block();
                f.render_widget(block, f.area());
            })
            .expect("draw");
        assert_eq!(
            term_panel_focused.backend().buffer().content()[0].fg,
            theme.palette.border_accent
        );
    }
}
