//! `RenderBlock::Image` widget — inline image or fallback badge.
//!
//! When a terminal image protocol is available, this widget reserves
//! space in the ratatui buffer and writes the raw escape sequence via
//! a post-render side-channel. When no protocol is available, it
//! renders a styled text badge showing the media type and byte count.
//!
//! See `code-rules.md` R2 (no ANSI in Spans), R9 (`&Theme` styling).

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::tui::image_protocol::ImageProtocol;
use crate::tui::theme::Theme;

use super::wrapped_rows;

/// Default height (in rows) reserved for an inline image placeholder.
const IMAGE_PLACEHOLDER_ROWS: u16 = 1;

/// Render an image block into `area`.
///
/// Shows a styled fallback badge with the media type and size.
/// Terminals that report inline-image support instead reserve a single
/// placeholder row. Actual inline-image escape output is not emitted;
/// TermProfile-gated rendering is planned for the streaming-v2 redesign.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    data: &[u8],
    media_type: &str,
    protocol: ImageProtocol,
    theme: &Theme,
    scroll_offset: u16,
) {
    let lines = rendered_lines(data, media_type, protocol, theme);
    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));
    frame.render_widget(para, area);
}

/// Estimate the display height of an image block.
pub(crate) fn estimate_rows(
    data: &[u8],
    media_type: &str,
    protocol: ImageProtocol,
    _theme: &Theme,
    width: u16,
) -> u16 {
    if protocol.is_supported() {
        // Reserve space for the inline image. Terminal protocols
        // handle sizing; we reserve a modest placeholder.
        IMAGE_PLACEHOLDER_ROWS
    } else {
        let lines = fallback_lines(data, media_type);
        wrapped_rows(&lines, width)
    }
}

fn rendered_lines<'a>(
    data: &[u8],
    media_type: &str,
    protocol: ImageProtocol,
    theme: &Theme,
) -> Vec<Line<'a>> {
    if protocol.is_supported() {
        // Placeholder line — the actual image is written via escape
        // sequences after the ratatui frame flush.
        vec![Line::from(Span::styled(
            format!("[image: {media_type}]"),
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::DIM),
        ))]
    } else {
        fallback_lines(data, media_type)
    }
}

fn fallback_lines(data: &[u8], media_type: &str) -> Vec<Line<'static>> {
    let size = humanize_bytes(data.len());
    vec![Line::from(vec![Span::styled(
        format!("[image: {media_type}, {size}]"),
        Style::new().add_modifier(Modifier::BOLD),
    )])]
}

/// Format a byte count as a human-readable string.
fn humanize_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        #[allow(clippy::cast_precision_loss)]
        let kib = bytes as f64 / 1024.0;
        format!("{kib:.1} KiB")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let mib = bytes as f64 / (1024.0 * 1024.0);
        format!("{mib:.1} MiB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_shows_media_type_and_size() {
        let lines = fallback_lines(b"fake png data!!", "image/png");
        assert_eq!(lines.len(), 1);
        let text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("image/png"));
        assert!(text.contains("15 B"));
    }

    #[test]
    fn humanize_bytes_ranges() {
        assert_eq!(humanize_bytes(0), "0 B");
        assert_eq!(humanize_bytes(512), "512 B");
        assert_eq!(humanize_bytes(1024), "1.0 KiB");
        assert_eq!(humanize_bytes(1024 * 1024), "1.0 MiB");
    }

    #[test]
    fn supported_protocol_renders_placeholder() {
        let protocol = ImageProtocol::ITerm2;
        let theme = crate::tui::theme::Theme::default_dark();
        let lines = rendered_lines(b"data", "image/png", protocol, &theme);
        assert_eq!(lines.len(), 1);
        let text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("[image: image/png]"));
    }

    #[test]
    fn unsupported_protocol_renders_fallback_with_size() {
        let protocol = ImageProtocol::None;
        let theme = crate::tui::theme::Theme::default_dark();
        let lines = rendered_lines(b"data", "image/png", protocol, &theme);
        let text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("4 B"));
    }
}
