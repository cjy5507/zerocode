//! `AppMode::ModalConfirmRewind` card — a y/n confirmation shown before an
//! Esc-Esc rewind discards the latest turn's code edits and conversation.
//!
//! This guards the one genuinely destructive Esc path: an Esc that denies a
//! permission prompt and a follow-up Esc-Esc that rewinds the turn share the
//! same key, so a reflexive double-tap could otherwise wipe just-written
//! files. The widget is purely a visual renderer; the decision is resolved in
//! the key dispatch (`y` confirms, `n`/Esc cancels).
//!
//! See `code-rules.md` R2 (no ANSI), R9 (`&Theme` styling).

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::tui::cards::{CardFrame, SurfaceKind};
use crate::tui::theme::Theme;

/// Render the rewind confirmation card from pre-built body lines.
pub fn draw(frame: &mut Frame<'_>, area: Rect, lines: &[String], theme: &Theme) {
    let title = Line::from(vec![
        Span::raw(" "),
        Span::styled(
            "Confirm rewind",
            Style::new()
                .fg(theme.palette.warn)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ]);
    let body: Vec<Line> = lines.iter().map(|line| Line::from(line.as_str())).collect();
    let block = CardFrame::new(SurfaceKind::Danger, theme)
        .title(title)
        .block();
    frame.render_widget(
        Paragraph::new(body).block(block).wrap(Wrap { trim: false }),
        area,
    );
}
