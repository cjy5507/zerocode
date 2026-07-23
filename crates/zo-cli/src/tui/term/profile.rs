//! Boot-time snapshot of terminal capabilities, derived from the environment.
//!
//! Every capability decision — inline-image protocol, tmux OSC-52 wrapping,
//! whether to push the Kitty keyboard flags, whether to bracket frames in CSI
//! ?2026 synchronized output — branches off ONE [`TermProfile`] instead of the
//! same knowledge being re-sniffed from `env::var(...)` in `tui_loop` (keyboard
//! blacklist), `live_cli_commands` (tmux), and `image_protocol` (image); those
//! consumers must not independently re-sniff the environment.
//!
//! The profile is a pure function of the environment, so it is computed once at
//! first use and cached process-wide (a terminal's identity does not change
//! under a running process). The classifiers below are pure over their inputs
//! so they stay unit-testable without mutating process env.

use std::sync::OnceLock;

use crate::tui::image_protocol::ImageProtocol;

/// A single, cached view of what the host terminal can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermProfile {
    /// Best available inline-image protocol (`None` when unsupported).
    pub image: ImageProtocol,
    /// `true` when running inside tmux, where an OSC 52 clipboard write must be
    /// wrapped in a DCS passthrough.
    pub in_tmux: bool,
    /// `true` when the Kitty keyboard protocol must be skipped even if the
    /// terminal reports support (`JediTerm`, or an explicit opt-out).
    pub kitty_keyboard_disabled: bool,
    /// `true` when the terminal is known to implement CSI ?2026 synchronized
    /// output correctly, so each frame may be bracketed in Begin/End.
    pub synchronized_output: bool,
}

impl TermProfile {
    /// The cached, process-wide profile, computed from the real environment on
    /// first call. Cheap to call repeatedly (returns a `Copy`).
    #[must_use]
    pub fn current() -> Self {
        static CACHE: OnceLock<TermProfile> = OnceLock::new();
        *CACHE.get_or_init(Self::detect)
    }

    /// Read the real process environment. Prefer [`TermProfile::current`] on hot
    /// paths so the env reads happen only once.
    #[must_use]
    fn detect() -> Self {
        Self {
            image: ImageProtocol::detect(),
            in_tmux: std::env::var_os("TMUX").is_some(),
            kitty_keyboard_disabled: keyboard_enhancement_disabled(
                std::env::var("TERMINAL_EMULATOR").ok().as_deref(),
                std::env::var_os("ZO_DISABLE_KITTY_KEYBOARD").is_some(),
            ),
            synchronized_output: synchronized_output_supported(
                std::env::var("TERM_PROGRAM").ok().as_deref(),
                std::env::var("TERM").ok().as_deref(),
                std::env::var("ZO_SYNC_OUTPUT").ok().as_deref(),
            ),
        }
    }
}

/// True when the Kitty keyboard protocol must be skipped on this terminal even
/// though `supports_keyboard_enhancement()` reports support.
///
/// `JetBrains`' embedded terminal (`JediTerm` — IntelliJ/PyCharm/etc.) answers the
/// enhancement-capability query positively but its Kitty implementation is
/// incomplete: once the flags are pushed, key presses arrive malformed and the
/// app stops receiving input entirely (the input box never echoes typed text),
/// and a literal `<` leaks on exit. Detect it via the `TERMINAL_EMULATOR` var
/// `JetBrains` sets, and honor an explicit opt-out for any other terminal with
/// the same defect. Skipping the push falls back to legacy Press-only key
/// reporting, which works correctly there. Pure over its inputs so it is unit
/// testable without mutating process env.
#[must_use]
fn keyboard_enhancement_disabled(terminal_emulator: Option<&str>, explicit_opt_out: bool) -> bool {
    explicit_opt_out || terminal_emulator == Some("JetBrains-JediTerm")
}

/// True when the terminal is known to implement CSI ?2026 synchronized output
/// correctly, so `App::draw_frame` may bracket each frame in Begin/End.
///
/// Conservative allowlist. The blanket veto that 2026 used to sit behind existed
/// because xterm.js's incorrect partial implementation (VS Code / Cursor
/// integrated terminals, older web embeds) locks into a stuck-synchronized state
/// during high-frequency streaming and freezes the screen. Every terminal on
/// this allowlist is a *native* (non-xterm.js) emulator that either implements
/// 2026 correctly or, on an old version, simply ignores the unknown DEC private
/// mode (a no-op) — so a wrong "enable" here cannot reproduce that freeze.
/// Unknown terminals fall back to the `BufWriter` single-flush that already
/// coalesces each frame (today's proven behavior). `ZO_SYNC_OUTPUT=1|0`
/// overrides the allowlist either way (opt in on an unlisted terminal, or opt
/// out if a listed one ever misbehaves).
#[must_use]
fn synchronized_output_supported(
    term_program: Option<&str>,
    term: Option<&str>,
    force: Option<&str>,
) -> bool {
    if let Some(force) = force {
        match force.trim() {
            "1" | "on" | "true" | "yes" => return true,
            "0" | "off" | "false" | "no" => return false,
            _ => {}
        }
    }
    // TERM_PROGRAM is the most reliable identity signal when present.
    let program = term_program.unwrap_or_default().to_ascii_lowercase();
    if program.contains("ghostty") // Ghostty / cmux (libghostty) — native 2026
        || program.contains("wezterm")
        || program.contains("iterm") // iTerm2 3.5+; older ignores the mode
    {
        return true;
    }
    // Fall back to $TERM for terminals that don't set TERM_PROGRAM (or strip it
    // over ssh): kitty (`xterm-kitty`), ghostty (`xterm-ghostty`), and
    // alacritty (also Zed's embedded `alacritty_terminal`).
    let term = term.unwrap_or_default().to_ascii_lowercase();
    term.contains("kitty") || term.contains("ghostty") || term.contains("alacritty")
}

/// True when the user asked the TUI to hold every spinner/dots/shimmer/caret at
/// a static frame (accessibility / low-power / high-latency link). Env-backed
/// and snapshotted once per process — a preference does not change under a
/// running process, mirroring [`TermProfile::current`] — so it is cheap to call
/// on hot render paths and needs no `RenderCache` key (it is constant for the
/// run). Every animation site gates its per-tick branch on this; when it is off
/// (the default) each site takes its existing branch, byte-identical to today.
#[must_use]
pub fn reduce_motion_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        reduce_motion_from_env(
            std::env::var("ZO_REDUCE_MOTION").ok().as_deref(),
            std::env::var_os("NO_MOTION").as_deref(),
        )
    })
}

/// Pure classifier over the two env inputs, unit-testable without mutating the
/// process environment (same shape as [`synchronized_output_supported`]).
/// `ZO_REDUCE_MOTION` is canonical and wins both directions; a non-empty
/// `NO_MOTION` is honored as a de-facto analog of `NO_COLOR`.
#[must_use]
fn reduce_motion_from_env(force: Option<&str>, no_motion: Option<&std::ffi::OsStr>) -> bool {
    if let Some(force) = force {
        match force.trim() {
            "1" | "on" | "true" | "yes" => return true,
            "0" | "off" | "false" | "no" => return false,
            _ => {}
        }
    }
    no_motion.is_some_and(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kitty_disabled_for_jetbrains_or_explicit_opt_out() {
        // JediTerm is blacklisted regardless of the opt-out flag.
        assert!(keyboard_enhancement_disabled(
            Some("JetBrains-JediTerm"),
            false
        ));
        // An explicit opt-out disables it on any terminal.
        assert!(keyboard_enhancement_disabled(Some("xterm-kitty"), true));
        assert!(keyboard_enhancement_disabled(None, true));
    }

    #[test]
    fn kitty_enabled_for_ordinary_terminals() {
        assert!(!keyboard_enhancement_disabled(Some("xterm-kitty"), false));
        assert!(!keyboard_enhancement_disabled(None, false));
    }

    #[test]
    fn sync_output_force_override_wins_both_ways() {
        // Force-on enables even an unlisted terminal.
        assert!(synchronized_output_supported(Some("vscode"), None, Some("1")));
        // Force-off disables even a listed one.
        assert!(!synchronized_output_supported(
            Some("ghostty"),
            None,
            Some("0")
        ));
    }

    #[test]
    fn sync_output_allowlists_native_terminals() {
        assert!(synchronized_output_supported(Some("ghostty"), None, None));
        assert!(synchronized_output_supported(Some("WezTerm"), None, None));
        assert!(synchronized_output_supported(
            Some("iTerm.app"),
            None,
            None
        ));
        assert!(synchronized_output_supported(None, Some("xterm-kitty"), None));
        assert!(synchronized_output_supported(
            None,
            Some("alacritty"),
            None
        ));
    }

    #[test]
    fn sync_output_disabled_for_xterm_js_and_unknown() {
        // VS Code / Cursor integrated terminals (xterm.js) — the freeze case.
        assert!(!synchronized_output_supported(Some("vscode"), None, None));
        assert!(!synchronized_output_supported(Some("Cursor"), None, None));
        // Unknown / bare terminals fall back to the safe BufWriter path.
        assert!(!synchronized_output_supported(None, Some("xterm-256color"), None));
        assert!(!synchronized_output_supported(None, None, None));
    }

    #[test]
    fn reduce_motion_off_by_default() {
        assert!(!reduce_motion_from_env(None, None));
    }

    #[test]
    fn reduce_motion_zo_var_wins_both_ways() {
        for on in ["1", "on", "true", "yes", " on "] {
            assert!(reduce_motion_from_env(Some(on), None), "{on:?}");
        }
        // ZO_REDUCE_MOTION=off overrides even a set NO_MOTION.
        assert!(!reduce_motion_from_env(
            Some("0"),
            Some(std::ffi::OsStr::new("1"))
        ));
    }

    #[test]
    fn reduce_motion_honors_non_empty_no_motion() {
        assert!(reduce_motion_from_env(None, Some(std::ffi::OsStr::new("1"))));
        // Empty NO_MOTION does not trigger it (matches NO_COLOR semantics).
        assert!(!reduce_motion_from_env(None, Some(std::ffi::OsStr::new(""))));
    }
}
