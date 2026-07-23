//! Centralized Nerd Font glyph constants for the TUI.
//!
//! All Unicode/Nerd Font icons used across the TUI surface are defined
//! here so they can be swapped to ASCII fallbacks under `NO_COLOR` in
//! one place. Each constant has a `_NC` (no-color) sibling.

/// Pick the appropriate glyph depending on whether color is available.
#[must_use]
pub const fn pick(color: bool, rich: &'static str, plain: &'static str) -> &'static str {
    if color { rich } else { plain }
}

/// Selection cursor for list-style modal rows (`❯ ` rich / `> ` plain).
///
/// The single source of truth for the shared modal cursor marker: every list
/// picker (choice / model / permission / question / …) reads it here so the
/// Unicode chevron degrades to a one-cell ASCII `>` under `NO_COLOR`/plain mode
/// in one place. Always two display cells (glyph + trailing space) so selected
/// and blank rows stay column-aligned; pair with [`modal_cursor_blank`].
#[must_use]
pub const fn modal_cursor(color: bool) -> &'static str {
    pick(color, "\u{276f} ", "> ")
}

/// Blank lead-in matching [`modal_cursor`]'s width for non-selected rows.
/// Two spaces in both modes, so labels line up under the cursor column.
#[must_use]
pub const fn modal_cursor_blank() -> &'static str {
    "  "
}

/// Filled cell of the inline `CardModel` gauge bar (`█` rich / `#` plain).
#[must_use]
pub const fn card_gauge_fill(color: bool) -> &'static str {
    pick(color, CARD_GAUGE_FILL, CARD_GAUGE_FILL_NC)
}

/// Empty cell of the inline `CardModel` gauge bar (`░` rich / `-` plain).
#[must_use]
pub const fn card_gauge_empty(color: bool) -> &'static str {
    pick(color, CARD_GAUGE_EMPTY, CARD_GAUGE_EMPTY_NC)
}

// ── Brand / App ──────────────────────────────────────────────────

pub const ZO_DIAMOND: &str = "\u{25c6}";
pub const ZO_DIAMOND_NC: &str = "*";

pub const ZO_SPARK: &str = "\u{2726}";
pub const ZO_SPARK_NC: &str = "+";

pub const ZO_SPARK_HOLLOW: &str = "\u{2727}";
pub const ZO_SPARK_HOLLOW_NC: &str = "+";

pub const ZO_RAIL: &str = "\u{2503}";
pub const ZO_RAIL_NC: &str = "|";

// ── HUD / Status Bar ────────────────────────────────────────────

pub const GIT_BRANCH: &str = "\u{e0a0}";
pub const GIT_BRANCH_NC: &str = "@";

pub const PERMISSION_LOCK: &str = "\u{f023}";
pub const PERMISSION_LOCK_NC: &str = "#";

// ── Queue ──────────────────────────────────────────────────────

pub const QUEUE_CLOCK: &str = "\u{25f7}"; // ◷
pub const QUEUE_CLOCK_NC: &str = "o";

// ── Smart Router ───────────────────────────────────────────────

pub const SMART_AUTO: &str = "\u{f085}"; // gears
pub const SMART_AUTO_NC: &str = "S";

pub const SMART_MODEL: &str = "\u{f0e7}"; // bolt (Nerd Font)
pub const SMART_MODEL_NC: &str = "M";

pub const SMART_FAST: &str = "\u{f0e7}"; // bolt (Nerd Font)
pub const SMART_FAST_NC: &str = "F";

pub const SMART_CODE: &str = "\u{f121}"; // code
pub const SMART_CODE_NC: &str = "C";

pub const SMART_VERIFY: &str = "\u{f00c}"; // check
pub const SMART_VERIFY_NC: &str = "V";

pub const SMART_RESEARCH: &str = "\u{f002}"; // search
pub const SMART_RESEARCH_NC: &str = "R";

pub const SMART_REVIEW: &str = "\u{f06e}"; // eye
pub const SMART_REVIEW_NC: &str = "Q";

pub const SMART_DESIGN: &str = "\u{f1fc}"; // paint brush
pub const SMART_DESIGN_NC: &str = "D";

pub const SMART_PIN: &str = "\u{f08d}"; // thumb-tack
pub const SMART_PIN_NC: &str = "P";

pub const SMART_FALLBACK: &str = "\u{f071}"; // warning triangle
pub const SMART_FALLBACK_NC: &str = "!";

// ── Tool Calls ──────────────────────────────────────────────────

pub const TOOL_DOT: &str = "\u{25cf}";
pub const TOOL_DOT_NC: &str = "*";

// ── Per-tool semantic icons ─────────────────────────────────────
//
// Each tool gets a distinct glyph so the transcript reads at a glance
// (run a shell vs. read a file vs. search the web). Inspired by
// opencode's per-tool icon set, shifted into Zo's glyph vocabulary.
//
// The table below is the single source of truth: `tool_icon` does a
// data-driven lookup over it instead of scattering `match` arms across
// the widget layer. Every rich glyph carries a 1-cell ASCII sibling so
// the icon stays aligned (always one display column) under `NO_COLOR`
// or a `TERM=dumb` terminal.

/// Fallback icon for any tool not present in [`TOOL_ICONS`].
pub const TOOL_GENERIC: &str = "\u{25cf}"; // ●
/// ASCII sibling of [`TOOL_GENERIC`].
pub const TOOL_GENERIC_NC: &str = "*";

/// `(canonical_name, rich_glyph, ascii_glyph)` for each known tool.
///
/// Names are matched case-insensitively by [`tool_icon`], so both
/// `"bash"` and `"Bash"` resolve to the same entry. MCP tools are not
/// listed here — the caller resolves their `@server` chip separately
/// and falls back to [`TOOL_GENERIC`] for the leading glyph.
pub const TOOL_ICONS: &[(&str, &str, &str)] = &[
    // Shell / process.
    ("bash", "\u{f489}", "$"), //  terminal
    // File reads / navigation.
    ("read", "\u{f06e}", ">"), //  eye
    ("glob", "\u{f002}", "#"), //  magnifier (find files)
    ("grep", "\u{f002}", "/"), //  magnifier (search content)
    ("list", "\u{f07b}", "~"), //  folder
    // File writes / edits.
    ("write", "\u{f040}", "<"),        //  pencil
    ("edit", "\u{f040}", "<"),         //  pencil
    ("multiedit", "\u{f040}", "<"),    //  pencil
    ("notebookedit", "\u{f040}", "<"), //  pencil
    // Web.
    ("webfetch", "\u{f0ac}", "%"),  //  globe
    ("websearch", "\u{f002}", "@"), //  magnifier (web)
    // Planning / agents.
    ("todowrite", "\u{f046}", "+"),       //  checklist
    ("task", "\u{f0e7}", "*"),            //  bolt
    ("agent", "\u{f0e7}", "*"),           //  bolt
    ("spawnmultiagent", "\u{f0e7}", "*"), //  bolt
    ("skill", "\u{f0eb}", "!"),           //  lightbulb
    ("sleep", "\u{f017}", "~"),           //  clock
];

/// Resolve the leading glyph for a tool by its canonical name.
///
/// Returns the rich Nerd Font glyph when `color` is `true`, else the
/// 1-cell ASCII sibling. Unknown tools fall back to [`TOOL_GENERIC`] /
/// [`TOOL_GENERIC_NC`]. The lookup is case-insensitive so `"bash"` and
/// `"Bash"` map to the same icon.
#[must_use]
pub fn tool_icon(name: &str, color: bool) -> &'static str {
    for (tool, rich, plain) in TOOL_ICONS {
        if name.eq_ignore_ascii_case(tool) {
            return pick(color, rich, plain);
        }
    }
    pick(color, TOOL_GENERIC, TOOL_GENERIC_NC)
}

// (Tool results no longer carry a leader glyph — the chain rail `│ ` groups
//  the output under its parent tool call. The old `╰─►` `result_leader` and
//  `└` `TOOL_RESULT_HOOK` were retired as redundant beside the rail.)

// ── Status Badges ───────────────────────────────────────────────

pub const CHECK: &str = "\u{2714}";
pub const CHECK_NC: &str = "v";

pub const CROSS: &str = "\u{2718}";
pub const CROSS_NC: &str = "x";

pub const WARN_TRIANGLE: &str = "\u{26a0}";
pub const WARN_TRIANGLE_NC: &str = "!";

pub const INFO_CIRCLE: &str = "\u{2139}";
pub const INFO_CIRCLE_NC: &str = "i";

pub const SPINNER_DOT: &str = "\u{25cf}";
pub const SPINNER_DOT_NC: &str = "*";

// ── Input / Prompt ──────────────────────────────────────────────

pub const PROMPT_CHEVRON: &str = "\u{276f}";
pub const PROMPT_CHEVRON_NC: &str = ">";

pub const PROMPT_ARROW: &str = "\u{f054}";
pub const PROMPT_ARROW_NC: &str = ">";

// ── Heavy rule chrome ──────────────────────────────────────────
// EAW-Ambiguous box-drawing class; target terminals force it narrow.

pub const ANVIL_CORNER: &str = "\u{2517}"; // ┗
pub const ANVIL_CORNER_NC: &str = "+";
pub const ANVIL_LINE: &str = "\u{2501}"; // ━
pub const ANVIL_LINE_NC: &str = "-";

// ── Messages ────────────────────────────────────────────────────

pub const USER_BAR: &str = "\u{258c}";
pub const USER_BAR_NC: &str = "|";

pub const ASSISTANT_DOT: &str = "\u{25c7}";
pub const ASSISTANT_DOT_NC: &str = "-";

// ── Sidebar ─────────────────────────────────────────────────────

pub const FILE_MODIFIED: &str = "M";
pub const FILE_ADDED: &str = "A";
pub const FILE_DELETED: &str = "D";

// ── Gauges / meters ─────────────────────────────────────────────
//
// Fill/empty glyphs for fixed-width utilization bars (sidebar `ctx` gauge,
// rate-limit windows, Fleet phase bars). Both glyphs are East-Asian **Neutral**
// (`▬` U+25AC and `░` U+2591 → `UnicodeWidthStr::width_cjk() == 1`), so a
// `ko_KR` wide-ambiguous tmux still renders them one column each and the
// fixed-width bar cannot double its filled run or overflow the sidebar.
//
// The earlier fill glyphs `■` (U+25A0) and `█` (U+2588) are East-Asian
// **Ambiguous** (`width_cjk() == 2`): under a wide-ambiguous locale each filled
// cell painted two columns, so the gauge grew past its budget and wrapped. See
// `styles.md` "Glyphs & width" and the `width_cjk`-guarded gauge test.
pub const GAUGE_FILL: &str = "\u{25ac}"; // ▬
pub const GAUGE_FILL_NC: &str = "#";
pub const GAUGE_EMPTY: &str = "\u{2591}"; // ░
pub const GAUGE_EMPTY_NC: &str = ".";

pub const GAUGE_HUD_FILL: &str = "\u{25b0}"; // ▰
pub const GAUGE_HUD_FILL_NC: &str = "#";
pub const GAUGE_HUD_EMPTY: &str = "\u{25b1}"; // ▱
pub const GAUGE_HUD_EMPTY_NC: &str = ".";

// The inline `CardModel` gauge (`█████░░░`) keeps the solid block bar in
// normal/color mode — distinct from the sidebar/HUD utilization bars above,
// which use EAW-Neutral glyphs. Under `NO_COLOR`/plain mode it falls back to
// one-cell ASCII (`#`/`-`) so the bar stays readable and column-aligned on a
// dumb terminal. Accessed through [`card_gauge_fill`] / [`card_gauge_empty`]
// so the modal/card path never hardcodes the block glyphs.
pub const CARD_GAUGE_FILL: &str = "\u{2588}"; // █
pub const CARD_GAUGE_FILL_NC: &str = "#";
pub const CARD_GAUGE_EMPTY: &str = "\u{2591}"; // ░
pub const CARD_GAUGE_EMPTY_NC: &str = "-";

// ── Scroll ──────────────────────────────────────────────────────

pub const SCROLL_UP: &str = "\u{25b2}";
pub const SCROLL_UP_NC: &str = "^";

pub const SCROLL_DOWN: &str = "\u{25bc}";
pub const SCROLL_DOWN_NC: &str = "v";

pub const SCROLL_THUMB: &str = "\u{2588}";
pub const SCROLL_TRACK: &str = "\u{2591}";
pub const SCROLL_THUMB_NC: &str = "#";
pub const SCROLL_TRACK_NC: &str = ".";

// ── Category headers (command palette) ─────────────────────────

pub const CAT_HEADER: &str = "\u{25c6}";
pub const CAT_HEADER_NC: &str = "*";
pub const CAT_CURSOR: &str = "\u{25b8}";
pub const CAT_CURSOR_NC: &str = ">";

/// Star marking the "★ Suggested" (frecency) group pinned at the top of the
/// command palette; degrades to `*` under `NO_COLOR`/`TERM=dumb`.
pub const SUGGESTED: &str = "\u{2605}";
/// ASCII sibling of [`SUGGESTED`].
pub const SUGGESTED_NC: &str = "*";

// ── Reasoning ───────────────────────────────────────────────────

pub const REASONING_RAIL: &str = "\u{254e}";
/// ASCII sibling — the prototype's mono reasoning rail is `:` (a dotted bar
/// reads as a quiet margin), not the heavier `|`. Mirrors `zo-theme.js`
/// `GLYPH_NC.reasoningRail`.
pub const REASONING_RAIL_NC: &str = ":";

// ── Expand/Collapse ─────────────────────────────────────────────

pub const CHEVRON_RIGHT: &str = "\u{25b8}";
pub const CHEVRON_RIGHT_NC: &str = ">";

pub const CHEVRON_DOWN: &str = "\u{25be}";
pub const CHEVRON_DOWN_NC: &str = "v";

// ── Separator ───────────────────────────────────────────────────

pub const HORIZONTAL_RULE: &str = "\u{2500}";
pub const HORIZONTAL_RULE_NC: &str = "-";

pub const LEFT_BORDER: &str = "\u{258e}";
pub const LEFT_BORDER_NC: &str = "|";

pub const VERTICAL_SEP: &str = "\u{2502}";
pub const VERTICAL_SEP_NC: &str = "|";

// ── Keyboard Hints ──────────────────────────────────────────────

pub const KEY_ENTER: &str = "\u{23ce}";
pub const KEY_ENTER_NC: &str = "enter";

pub const KEY_ESC: &str = "esc";

pub const KEY_TAB: &str = "tab";

#[cfg(test)]
mod tests {
    use super::*;

    /// 도구 아이콘 lookup 은 대소문자 무관 — `"bash"` 와 `"Bash"` 가 동일.
    #[test]
    fn tool_icon_is_case_insensitive() {
        assert_eq!(tool_icon("bash", true), tool_icon("Bash", true));
        assert_eq!(tool_icon("read", true), tool_icon("Read", true));
        assert_eq!(tool_icon("webfetch", true), tool_icon("WebFetch", true));
    }

    /// 알려진 도구는 generic 폴백과 다른 고유 글리프를 받는다.
    #[test]
    fn known_tools_get_distinct_icons() {
        assert_ne!(tool_icon("bash", true), TOOL_GENERIC);
        assert_ne!(tool_icon("read", true), TOOL_GENERIC);
        assert_ne!(tool_icon("WebFetch", true), TOOL_GENERIC);
        // 서로 다른 카테고리는 서로 다른 글리프.
        assert_ne!(tool_icon("bash", true), tool_icon("read", true));
    }

    /// 미등록 도구는 generic 폴백.
    #[test]
    fn unknown_tool_falls_back_to_generic() {
        assert_eq!(tool_icon("DefinitelyNotARealTool", true), TOOL_GENERIC);
        assert_eq!(tool_icon("DefinitelyNotARealTool", false), TOOL_GENERIC_NC);
    }

    /// 게이지 채움/빈칸 글리프는 wide-ambiguous(`ko_KR` tmux) 에서도 1칸이어야
    /// 한다 — `width_cjk()==1`(EAW=Neutral). 과거 `■`/`█`(EAW=Ambiguous, cjk=2)
    /// 로 회귀하면 이 어서션이 잡아낸다.
    #[test]
    fn gauge_glyphs_are_one_cell_under_wide_ambiguous() {
        use unicode_width::UnicodeWidthStr;
        for g in [GAUGE_FILL, GAUGE_EMPTY, GAUGE_FILL_NC, GAUGE_EMPTY_NC] {
            assert_eq!(
                UnicodeWidthStr::width_cjk(g),
                1,
                "gauge glyph {g:?} must stay 1 cell even where ambiguous width is wide"
            );
        }
    }

    /// H0 heat-HUD marks and their ASCII siblings remain one cell under a
    /// wide-ambiguous locale; heavy box drawing is intentionally exempt.
    #[test]
    fn heat_hud_glyphs_are_one_cell_under_wide_ambiguous() {
        use unicode_width::UnicodeWidthStr;
        for g in [
            GAUGE_HUD_FILL,
            GAUGE_HUD_EMPTY,
            QUEUE_CLOCK,
            GAUGE_HUD_FILL_NC,
            GAUGE_HUD_EMPTY_NC,
            QUEUE_CLOCK_NC,
        ] {
            assert_eq!(
                UnicodeWidthStr::width_cjk(g),
                1,
                "heat HUD glyph {g:?} must stay 1 cell even where ambiguous width is wide"
            );
        }
    }

    #[test]
    fn zo_spark_glyphs_are_one_cell_under_wide_ambiguous() {
        use unicode_width::UnicodeWidthStr;
        for g in [
            ZO_SPARK,
            ZO_SPARK_HOLLOW,
            ZO_SPARK_NC,
            ZO_SPARK_HOLLOW_NC,
        ] {
            assert_eq!(
                UnicodeWidthStr::width_cjk(g),
                1,
                "zo spark glyph {g:?} must stay 1 cell even where ambiguous width is wide"
            );
        }
    }

    /// Modal cursor: rich `❯ ` and plain `> ` are both exactly two display
    /// cells so selected/blank rows stay column-aligned, and the plain glyph is
    /// one-cell ASCII. The blank lead-in matches the width in both modes.
    #[test]
    fn modal_cursor_and_blank_are_two_cells_each_mode() {
        use unicode_width::UnicodeWidthStr;
        for color in [true, false] {
            assert_eq!(
                UnicodeWidthStr::width(modal_cursor(color)),
                2,
                "modal cursor must be two display cells (color={color})"
            );
        }
        assert_eq!(UnicodeWidthStr::width(modal_cursor_blank()), 2);
        // Plain cursor is a one-cell ASCII marker plus its trailing space.
        assert_eq!(modal_cursor(false), "> ");
        assert!(!modal_cursor(false).contains('\u{276f}'));
    }

    /// Card gauge cells are one display column in either mode, and the plain
    /// fallbacks are the required ASCII `#` / `-` (never the Unicode blocks).
    #[test]
    fn card_gauge_cells_are_one_cell_and_ascii_under_no_color() {
        use unicode_width::UnicodeWidthStr;
        for color in [true, false] {
            assert_eq!(UnicodeWidthStr::width(card_gauge_fill(color)), 1);
            assert_eq!(UnicodeWidthStr::width(card_gauge_empty(color)), 1);
        }
        assert_eq!(card_gauge_fill(false), "#");
        assert_eq!(card_gauge_empty(false), "-");
        assert_eq!(card_gauge_fill(true), "\u{2588}"); // █
        assert_eq!(card_gauge_empty(true), "\u{2591}"); // ░
    }

    /// `NO_COLOR(color=false)` 는 항상 1-cell ASCII sibling 을 반환해 정렬 보존.
    #[test]
    fn no_color_returns_single_cell_ascii() {
        for (tool, _rich, plain) in TOOL_ICONS {
            let got = tool_icon(tool, false);
            assert_eq!(got, *plain, "ascii sibling mismatch for {tool}");
            assert_eq!(
                got.chars().count(),
                1,
                "no-color icon for {tool} must be exactly one cell: {got:?}"
            );
        }
        assert_eq!(TOOL_GENERIC_NC.chars().count(), 1);
    }
}
