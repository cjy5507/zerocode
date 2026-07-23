//! Terminal image protocol detection and rendering.
//!
//! Supports iTerm2 (OSC 1337) and Kitty graphics protocols. Falls
//! back to `None` when neither is detected. See `code-rules.md` R2:
//! escape sequences are written to the output buffer, not embedded in
//! ratatui `Span`s.
//!
// TODO(P3): Sixel fallback — DA1 detect + encoder (needs image decode: file bytes -> RGB888)

use std::env;

/// Detected terminal image protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
    /// iTerm2 inline image protocol (OSC 1337).
    ITerm2,
    /// Kitty graphics protocol.
    Kitty,
    /// No image support detected.
    None,
}

impl ImageProtocol {
    /// Detect the best available image protocol from environment.
    #[must_use]
    pub fn detect() -> Self {
        // Check TERM_PROGRAM first (most reliable).
        if let Ok(prog) = env::var("TERM_PROGRAM") {
            let lower = prog.to_lowercase();
            if lower.contains("iterm")
                || lower.contains("wezterm")
                || lower.contains("ghostty")
                || lower.contains("mintty")
            {
                return Self::ITerm2;
            }
        }
        // Check for Kitty.
        if let Ok(term) = env::var("TERM") {
            if term.contains("kitty") {
                return Self::Kitty;
            }
        }
        // Check LC_TERMINAL for iTerm2 (some SSH setups).
        if let Ok(lc) = env::var("LC_TERMINAL") {
            if lc.to_lowercase().contains("iterm") {
                return Self::ITerm2;
            }
        }
        Self::None
    }

    /// `true` when inline image rendering is supported.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_supported_reflects_protocol() {
        assert!(!ImageProtocol::None.is_supported());
        assert!(ImageProtocol::ITerm2.is_supported());
        assert!(ImageProtocol::Kitty.is_supported());
    }
}
