//! Minimal ANSI-SGR → ratatui [`Span`] conversion for the custom status line.
//!
//! The TUI's widget contract forbids raw ANSI in rendered text (code-rules
//! R2), but a user's `statusLine.command` legitimately emits color codes
//! (Claude Code renders them). This parser converts the supported SGR subset
//! — reset, bold/dim/italic/underline, 16-color, 256-color (`38;5;n`) and
//! truecolor (`38;2;r;g;b`) foreground/background — into styled spans, and
//! silently drops every other escape sequence (cursor movement, OSC titles)
//! so stray control bytes can never corrupt the frame.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

/// Parse one line of command output into styled spans. Unknown escape
/// sequences are dropped; text outside escapes is passed through verbatim.
#[must_use]
pub fn ansi_spans(input: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut style = Style::default();
    let mut text = String::new();
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            // Drop raw C0 controls (other than the escapes handled below) so a
            // BEL or stray CR cannot reach the terminal through the widget.
            if !ch.is_control() || ch == '\t' {
                text.push(ch);
            }
            continue;
        }
        match chars.peek() {
            Some('[') => {
                chars.next();
                let mut params = String::new();
                let mut terminator = None;
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        terminator = Some(c);
                        break;
                    }
                    params.push(c);
                }
                if terminator == Some('m') {
                    if !text.is_empty() {
                        spans.push(Span::styled(std::mem::take(&mut text), style));
                    }
                    style = apply_sgr(style, &params);
                }
                // Any other CSI (cursor moves, erase, …) is consumed and dropped.
            }
            Some(']') => {
                // OSC: consume until BEL or ST (ESC \).
                chars.next();
                let mut prev_esc = false;
                for c in chars.by_ref() {
                    if c == '\u{7}' || (prev_esc && c == '\\') {
                        break;
                    }
                    prev_esc = c == '\u{1b}';
                }
            }
            _ => {
                // Lone ESC or two-byte escape — drop the next char too.
                chars.next();
            }
        }
    }
    if !text.is_empty() {
        spans.push(Span::styled(text, style));
    }
    spans
}

/// Apply one SGR parameter list (the `…` of `ESC[…m`) to `style`.
fn apply_sgr(mut style: Style, params: &str) -> Style {
    let mut codes = params
        .split([';', ':'])
        .map(|part| part.parse::<u16>().unwrap_or(0));
    while let Some(code) = codes.next() {
        style = match code {
            0 => Style::default(),
            1 => style.add_modifier(Modifier::BOLD),
            2 => style.add_modifier(Modifier::DIM),
            3 => style.add_modifier(Modifier::ITALIC),
            4 => style.add_modifier(Modifier::UNDERLINED),
            22 => style.remove_modifier(Modifier::BOLD | Modifier::DIM),
            23 => style.remove_modifier(Modifier::ITALIC),
            24 => style.remove_modifier(Modifier::UNDERLINED),
            30..=37 => style.fg(basic_color(code - 30, false)),
            39 => {
                let mut cleared = style;
                cleared.fg = None;
                cleared
            }
            40..=47 => style.bg(basic_color(code - 40, false)),
            49 => {
                let mut cleared = style;
                cleared.bg = None;
                cleared
            }
            90..=97 => style.fg(basic_color(code - 90, true)),
            100..=107 => style.bg(basic_color(code - 100, true)),
            38 | 48 => {
                let is_fg = code == 38;
                match codes.next() {
                    Some(5) => match codes.next() {
                        Some(n) => {
                            let color = Color::Indexed(u8::try_from(n).unwrap_or(7));
                            if is_fg {
                                style.fg(color)
                            } else {
                                style.bg(color)
                            }
                        }
                        None => style,
                    },
                    Some(2) => {
                        let (r, g, b) = (
                            codes.next().unwrap_or(0),
                            codes.next().unwrap_or(0),
                            codes.next().unwrap_or(0),
                        );
                        let color = Color::Rgb(
                            u8::try_from(r).unwrap_or(255),
                            u8::try_from(g).unwrap_or(255),
                            u8::try_from(b).unwrap_or(255),
                        );
                        if is_fg {
                            style.fg(color)
                        } else {
                            style.bg(color)
                        }
                    }
                    _ => style,
                }
            }
            _ => style,
        };
    }
    style
}

const fn basic_color(index: u16, bright: bool) -> Color {
    match (index, bright) {
        (0, false) => Color::Black,
        (1, false) => Color::Red,
        (2, false) => Color::Green,
        (3, false) => Color::Yellow,
        (4, false) => Color::Blue,
        (5, false) => Color::Magenta,
        (6, false) => Color::Cyan,
        (0, true) => Color::DarkGray,
        (1, true) => Color::LightRed,
        (2, true) => Color::LightGreen,
        (3, true) => Color::LightYellow,
        (4, true) => Color::LightBlue,
        (5, true) => Color::LightMagenta,
        (6, true) => Color::LightCyan,
        _ => Color::White,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(spans: &[Span<'_>]) -> String {
        spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn plain_text_passes_through_unstyled() {
        let spans = ansi_spans("main · $0.42");
        assert_eq!(text_of(&spans), "main · $0.42");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].style, Style::default());
    }

    #[test]
    fn sgr_colors_and_reset_split_spans() {
        let spans = ansi_spans("\u{1b}[32mgreen\u{1b}[0m plain \u{1b}[1;31mbold-red\u{1b}[m");
        assert_eq!(text_of(&spans), "green plain bold-red");
        assert_eq!(spans[0].style.fg, Some(Color::Green));
        assert_eq!(spans[1].style, Style::default());
        assert_eq!(spans[2].style.fg, Some(Color::Red));
        assert!(spans[2].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn extended_colors_parse() {
        let spans = ansi_spans("\u{1b}[38;5;208morange\u{1b}[0m\u{1b}[38;2;1;2;3mrgb\u{1b}[0m");
        assert_eq!(spans[0].style.fg, Some(Color::Indexed(208)));
        assert_eq!(spans[1].style.fg, Some(Color::Rgb(1, 2, 3)));
    }

    #[test]
    fn non_sgr_escapes_are_dropped_not_rendered() {
        // Cursor-move CSI, an OSC title, and a BEL must all vanish.
        let spans = ansi_spans("a\u{1b}[2Kb\u{1b}]0;title\u{7}c\u{7}d");
        assert_eq!(text_of(&spans), "abcd");
    }
}
