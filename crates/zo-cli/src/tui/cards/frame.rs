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
//! This owns only the closed, `Rect`-rendered chrome. The inline streaming
//! code-fence rail ([`super::super::markdown::code_card_frame_lines`]) is a
//! deliberately different visual language (3-sided open muted rail, emitted as
//! append-only `Line`s for the streaming/settle parity contract) and stays
//! line-based — see that module. The two are separate leaves under `cards/` on
//! purpose.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Padding};

use super::super::theme::Theme;

/// The kind of framed surface, which picks the brand border recipe (border
/// type, border color, surface fill, default padding).
//
// `Card`/`Danger`/`Panel` land with their respective surface migrations so no
// variant is ever unconstructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    /// A full-screen overlay modal: rounded glass edge on an elevation-2 fill.
    Modal,
    /// An input-anchored hint popup (`@mention` / slash commands): a rounded
    /// glass edge on an elevation-2 fill, typically carrying a bottom key-hint
    /// via [`CardFrame::title_bottom`].
    Popup,
    /// A thick-border warning guard (permission prompt / confirm): a `Thick`
    /// border and no surface fill, its border color set by the caller via
    /// [`CardFrame::border_style`] to track focus/severity.
    Danger,
    /// A transcript content card (status/report card, agent result): a rounded
    /// `accent_dim` border, no surface fill, with the caller supplying its
    /// horizontal padding. The border color may be overridden via
    /// [`CardFrame::border_style`] to track focus.
    Card,
    /// An inner pane nested inside a modal (a viewer column, a selection box): a
    /// rounded, quiet `dim` border and no fill, so nested chrome reads as
    /// subordinate to the modal's accent frame. A selection box overrides the
    /// border via [`CardFrame::border_style`] to highlight the active pane.
    Panel,
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

    /// Override the surface kind's default inner padding.
    #[must_use]
    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = Some(padding);
        self
    }

    /// Border/fill/padding recipe for this surface kind.
    fn recipe(&self) -> (&'static str, Style, Option<Style>, Padding) {
        let theme = self.theme;
        match self.kind {
            SurfaceKind::Modal => (
                "modal",
                Style::new().fg(
                    theme
                        .border_glass()
                        .unwrap_or(theme.palette.accent),
                ),
                Some(
                    theme
                        .typography
                        .body
                        .bg(theme.surface2().unwrap_or(theme.palette.code_bg)),
                ),
                Padding::ZERO,
            ),
            // A popup shares the modal's rounded glass edge and elevation-2
            // fill, while its title/content styling remains caller-owned.
            SurfaceKind::Popup => (
                "modal",
                Style::new().fg(
                    theme
                        .border_glass()
                        .unwrap_or(theme.palette.accent_dim),
                ),
                Some(
                    Style::new()
                        .bg(theme.surface2().unwrap_or(theme.palette.code_bg)),
                ),
                Padding::ZERO,
            ),
            // A danger guard: a thick border and NO fill. The default border is
            // the strongest `error`; callers restyle it via `border_style` (a
            // permission prompt drops to `warn` while unfocused).
            SurfaceKind::Danger => (
                "permission_card",
                Style::new()
                    .fg(theme.palette.error)
                    .add_modifier(Modifier::BOLD),
                None,
                Padding::ZERO,
            ),
            // A transcript content card: a rounded `accent_dim` border and no
            // surface fill (the body Paragraph carries its own style). The
            // caller supplies horizontal padding and may restyle the border to
            // track focus.
            SurfaceKind::Card => (
                "tool_call_card",
                Style::new().fg(theme.palette.accent_dim),
                None,
                Padding::ZERO,
            ),
            // A nested modal pane: a rounded, quiet `dim` border and no fill so
            // it reads as subordinate to the modal's accent frame.
            SurfaceKind::Panel => ("modal", theme.typography.dim, None, Padding::ZERO),
        }
    }

    /// Build the chrome [`Block`]. The single point that constructs a framed
    /// border — every framed surface flows through here.
    #[must_use]
    pub fn block(self) -> Block<'a> {
        let (role, default_border_style, fill, default_padding) = self.recipe();
        let mut block = Block::default()
            .borders(Borders::ALL)
            .border_type(self.theme.borders.for_role(role))
            .border_style(self.border_style.unwrap_or(default_border_style))
            .padding(self.padding.unwrap_or(default_padding));
        if let Some(fill) = fill {
            block = block.style(fill);
        }
        if let Some(title) = self.title {
            block = block.title(title);
        }
        if let Some(title_bottom) = self.title_bottom {
            block = block.title_bottom(title_bottom);
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

    #[test]
    fn modal_renders_rounded_glass_border_and_surface() {
        let theme = theme();
        // The "modal" role must round corners for the brand look to hold.
        assert_eq!(theme.borders.for_role("modal"), BorderType::Rounded);

        let mut term = Terminal::new(TestBackend::new(20, 6)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Modal, &theme).block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let buffer = term.backend().buffer();
        let corner = &buffer.content()[0];
        // Rounded top-left corner proves the "modal" role border type flows
        // through, and the glass tokens own both edge and elevation-2 fill.
        assert_eq!(corner.symbol(), "\u{256d}"); // ╭
        assert_eq!(corner.fg, theme.border_glass().expect("glass edge"));
        assert_eq!(
            buffer.content()[usize::from(buffer.area.width) + 1].bg,
            theme.surface2().expect("glass surface")
        );
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
            .expect("title rendered on the top border");
        // A title with an explicit fg is NOT overwritten by the accent
        // `border_style` — `modals::modal_frame` relies on this to keep its
        // simple modals' titles body-colored while the border gains accent.
        assert_eq!(title_cell.fg, title_fg);
    }

    #[test]
    fn popup_uses_a_rounded_glass_border_with_a_bottom_title() {
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
        // Rounded corner and interior both use the elevation-2 glass tokens.
        let corner = &buf.content()[0];
        assert_eq!(corner.symbol(), "\u{256d}"); // ╭
        assert_eq!(corner.fg, theme.border_glass().expect("glass edge"));
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
            .expect("bottom title rendered on the lower border");
        assert_eq!(hint.fg, theme.palette.dim);
    }

    #[test]
    fn danger_uses_a_thick_error_border_that_callers_can_restyle() {
        let theme = theme();
        // The danger role must be thick for the warning to read as urgent.
        assert_eq!(theme.borders.for_role("permission_card"), BorderType::Thick);
        assert_ne!(theme.palette.error, theme.palette.warn, "test needs distinct colors");

        // Default danger border is the strong `error` color, on a thick corner.
        let mut term = Terminal::new(TestBackend::new(20, 4)).expect("backend");
        term.draw(|f| {
            let block = CardFrame::new(SurfaceKind::Danger, &theme).block();
            f.render_widget(block, f.area());
        })
        .expect("draw");
        let corner = &term.backend().buffer().content()[0];
        assert_eq!(corner.symbol(), "\u{250f}"); // ┏ (thick)
        assert_eq!(corner.fg, theme.palette.error);

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
    fn panel_uses_a_rounded_dim_border_with_no_fill() {
        let theme = theme();
        assert_eq!(theme.borders.for_role("modal"), BorderType::Rounded);
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
        let corner = &buf.content()[0];
        assert_eq!(corner.symbol(), "\u{256d}"); // ╭
        assert_eq!(corner.fg, dim_fg, "a nested panel reads with a quiet dim border");
        // No surface fill: an interior cell keeps the default bg.
        let width = buf.area.width as usize;
        assert_eq!(buf.content()[width + 1].bg, ratatui::style::Color::Reset);
    }

    #[test]
    fn closed_box_borders_are_one_cell_each_side() {
        let theme = theme();
        let area = Rect::new(0, 0, 40, 12);
        let inner = CardFrame::new(SurfaceKind::Modal, &theme)
            .title("A wide title")
            .block()
            .inner(area);
        // Box-drawing borders are 1 cell on every side, so the inner shrinks by
        // exactly 2 in each dimension regardless of the title's width.
        assert_eq!(inner.width, area.width - 2);
        assert_eq!(inner.height, area.height - 2);
    }
}
