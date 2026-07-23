//! Best-effort live terminal background detection.
//!
//! This is deliberately separate from [`super::TermProfile`]: the profile is a
//! pure environment snapshot, while querying the background performs bounded
//! terminal I/O once during TUI boot.

use std::io::{self, IsTerminal};
use std::time::Duration;

use terminal_colorsaurus::{QueryOptions, background_color};

/// Maximum time the startup query may wait for a terminal response.
const QUERY_TIMEOUT: Duration = Duration::from_millis(150);

/// Query the terminal background once, returning 8-bit RGB on success.
///
/// Every unsupported, disabled, timed-out, or malformed response is treated as
/// absence. Callers must preserve their existing theme when this returns
/// `None`.
#[must_use]
pub fn detect_background() -> Option<(u8, u8, u8)> {
    if !background_query_allowed(
        io::stdin().is_terminal(),
        io::stdout().is_terminal(),
        std::env::var("TERM").ok().as_deref(),
        std::env::var_os("CI").is_some(),
        std::env::var_os("NO_COLOR").is_some(),
        std::env::var("ZO_NO_TERM_QUERY").ok().as_deref(),
    ) {
        return None;
    }

    let mut options = QueryOptions::default();
    options.timeout = QUERY_TIMEOUT;
    std::panic::catch_unwind(|| background_color(options))
        .ok()?
        .ok()
        .map(|color| color.scale_to_8bit())
}

// Independent terminal/env guard signals; folding them into a struct would add
// indirection without clarifying a small pure predicate.
#[expect(clippy::fn_params_excessive_bools)]
#[must_use]
fn background_query_allowed(
    stdin_is_tty: bool,
    stdout_is_tty: bool,
    term: Option<&str>,
    ci_is_set: bool,
    no_color_is_set: bool,
    opt_out: Option<&str>,
) -> bool {
    stdin_is_tty
        && stdout_is_tty
        && !term.is_some_and(|term| term.trim().eq_ignore_ascii_case("dumb"))
        && !ci_is_set
        && !no_color_is_set
        && opt_out != Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_allowed_for_interactive_terminal() {
        assert!(background_query_allowed(
            true,
            true,
            Some("xterm-256color"),
            false,
            false,
            None,
        ));
    }

    #[test]
    fn query_skipped_when_any_safety_guard_blocks_it() {
        let cases = [
            (false, true, Some("xterm"), false, false, None),
            (true, false, Some("xterm"), false, false, None),
            (true, true, Some("dumb"), false, false, None),
            (true, true, Some(" DUMB "), false, false, None),
            (true, true, Some("xterm"), true, false, None),
            (true, true, Some("xterm"), false, true, None),
            (true, true, Some("xterm"), false, false, Some("1")),
        ];

        assert!(cases.into_iter().all(
            |(stdin, stdout, term, ci, no_color, opt_out)| !background_query_allowed(
                stdin, stdout, term, ci, no_color, opt_out,
            )
        ));
    }

    #[test]
    fn opt_out_requires_exact_value_one() {
        assert!(background_query_allowed(
            true,
            true,
            Some("xterm"),
            false,
            false,
            Some("true"),
        ));
    }
}
