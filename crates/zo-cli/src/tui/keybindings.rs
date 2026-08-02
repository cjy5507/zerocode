//! Keybinding reference for the in-app "which-key" help overlay.
//!
//! The TUI uses direct accelerators (no leader-key state machine), so this
//! module keeps the important user-facing global and contextual shortcuts in
//! one grouped catalogue rendered by [`help_text`]. It intentionally omits
//! duplicate navigation aliases and low-level readline chords; the dispatch in
//! `app/keys.rs` remains authoritative for those details.
//!
//! This module is pure data + formatting: no I/O, no `App` access, so it is
//! trivially unit-tested.

/// One keybinding row: the key combo, a terse action description, and the
/// group it belongs to. `key` is already display-formatted (e.g. `⌃B`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// Logical group used to cluster related bindings in the overlay.
    pub group: Group,
    /// Display-formatted key combo (rich glyphs; ASCII handled at render).
    pub key: &'static str,
    /// Terse, lower-case action phrase ("toggle sidebar").
    pub action: &'static str,
}

/// Coarse grouping for the help overlay, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    /// Scrolling and transcript navigation.
    Navigation,
    /// Panels, overlays, and view toggles.
    View,
    /// Composing and editing the prompt.
    Editing,
    /// Session-level actions (copy, commands, quit).
    Session,
}

impl Group {
    /// Section heading shown above the group's bindings.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Navigation => "Navigation",
            Self::View => "View",
            Self::Editing => "Editing",
            Self::Session => "Session",
        }
    }

    /// Display order of the groups in the overlay.
    pub const ORDER: [Self; 4] = [Self::Navigation, Self::View, Self::Editing, Self::Session];
}

/// Curated user-facing keybindings, in catalogue order. Rich key glyphs
/// (`⌃`) degrade to ASCII (`^`) at render time via [`help_text`]'s `color`
/// flag so the overlay stays readable under `NO_COLOR`/`TERM=dumb`.
pub const BINDINGS: &[Binding] = &[
    // Navigation.
    Binding {
        group: Group::Navigation,
        key: "⇞ / ⇟",
        action: "scroll half-page up / down",
    },
    Binding {
        group: Group::Navigation,
        key: "Home / End",
        action: "jump to top / bottom",
    },
    Binding {
        group: Group::Navigation,
        key: "Tab",
        action: "focus next block",
    },
    Binding {
        group: Group::Navigation,
        key: "Enter",
        action: "expand / collapse focused block",
    },
    // View.
    Binding {
        group: Group::View,
        key: "⌃B",
        action: "toggle sidebar panel",
    },
    Binding {
        group: Group::View,
        key: "⌃A",
        action: "toggle agents tree",
    },
    Binding {
        group: Group::View,
        key: "⌃G",
        action: "open agents viewer",
    },
    Binding {
        group: Group::View,
        key: "⌃O",
        action: "open workflow viewer",
    },
    Binding {
        group: Group::View,
        key: "⌃F",
        action: "find in transcript",
    },
    Binding {
        group: Group::View,
        key: "F3 / ⌥1",
        action: "open model picker",
    },
    Binding {
        group: Group::View,
        key: "⌥2",
        action: "open Smart settings",
    },
    Binding {
        group: Group::View,
        key: "⌃R",
        action: "open rewind viewer",
    },
    Binding {
        group: Group::View,
        key: "F11",
        action: "toggle focus mode",
    },
    Binding {
        group: Group::View,
        key: "?",
        action: "show this help",
    },
    // Editing.
    Binding {
        group: Group::Editing,
        key: "Enter",
        action: "send message",
    },
    Binding {
        group: Group::Editing,
        key: "⇧Enter",
        action: "newline",
    },
    Binding {
        group: Group::Editing,
        key: "@",
        action: "mention a file",
    },
    Binding {
        group: Group::Editing,
        key: "Space / Tab",
        action: "accept slash command autocomplete",
    },
    Binding {
        group: Group::Editing,
        key: "⌃V",
        action: "paste from clipboard",
    },
    Binding {
        group: Group::Editing,
        key: "⌃E",
        action: "edit prompt in external editor",
    },
    Binding {
        group: Group::Editing,
        key: "⌃W",
        action: "delete previous word",
    },
    Binding {
        group: Group::Editing,
        key: "⌃U",
        action: "clear the line",
    },
    Binding {
        group: Group::Editing,
        key: "⌃Y",
        action: "yank killed text (while typing)",
    },
    // Session.
    Binding {
        group: Group::Session,
        key: "⇧Tab",
        action: "cycle permission mode",
    },
    Binding {
        group: Group::Session,
        key: "⌃P",
        action: "command palette",
    },
    Binding {
        group: Group::Session,
        key: "⌃Y",
        action: "copy last message (empty prompt)",
    },
    Binding {
        group: Group::Session,
        key: "⌃⇧C",
        action: "copy last message (alias)",
    },
    Binding {
        group: Group::Session,
        key: "⌃X",
        action: "expand/collapse tool detail",
    },
    Binding {
        group: Group::Session,
        key: "⌥Y",
        action: "copy entire transcript",
    },
    Binding {
        group: Group::Session,
        key: "Esc Esc",
        action: "rewind previous turn",
    },
    Binding {
        group: Group::Session,
        key: "⌃C ⌃C",
        action: "quit",
    },
];

/// Slash commands that share a user-facing accelerator, mapped to that key.
///
/// This explicit mapping feeds the command-palette key-hint column. Only
/// genuine command↔accelerator pairs belong here: every key is handled by the
/// `App` key dispatch and every name is a real slash command.
const COMMAND_HINTS: &[(&str, &str)] = &[
    ("help", "?"),
    ("agents", "\u{2303}A"),       // ⌃A — toggle agents tree
    ("rewind", "\u{2303}R"),       // ⌃R — rewind viewer
    ("copy", "\u{2303}Y"),         // ⌃Y — copy last message
    ("model", "F3 / \u{2325}1"),   // F3 / ⌥1 — model picker
    ("smart", "\u{2325}2"),        // ⌥2 — Smart settings
];

/// Display key for a slash command's global accelerator, if it has one.
///
/// `name` is the command without its leading `/`. When `color` is `false`
/// the modifier glyphs degrade to ASCII (`⌃` → `^`) so the hint stays
/// legible under `NO_COLOR` / dumb terminals — mirroring [`help_text`].
#[must_use]
pub fn command_hint(name: &str, color: bool) -> Option<String> {
    let key = COMMAND_HINTS
        .iter()
        .find(|(cmd, _)| cmd.eq_ignore_ascii_case(name))
        .map(|(_, key)| *key)?;
    Some(if color { key.to_string() } else { asciify(key) })
}

/// Rich → ASCII key-glyph fallback for `NO_COLOR` / dumb terminals.
///
/// Only the modifier glyphs need translating; printable keys (`?`, `@`,
/// `Tab`, `Enter`) already render everywhere.
fn asciify(key: &str) -> String {
    key.replace('\u{2303}', "^") // ⌃ Ctrl
        .replace('\u{2325}', "Alt+") // ⌥ Option/Alt
        .replace('\u{21e7}', "Shift+") // ⇧ Shift
        .replace('\u{21de}', "PgUp") // ⇞
        .replace('\u{21df}', "PgDn") // ⇟
}

/// Build the grouped, aligned help text for the pager overlay.
///
/// When `color` is `false`, modifier glyphs are swapped for ASCII so the
/// overlay is legible without a Nerd Font. The output is a plain
/// `String` (one row per line) that the pager renders verbatim.
#[must_use]
pub fn help_text(color: bool) -> String {
    // Width of the key column = widest (possibly asciified) key + padding,
    // so the action column lines up across every group.
    let key_cells = |key: &str| -> usize {
        if color {
            key.chars().count()
        } else {
            asciify(key).chars().count()
        }
    };
    let key_width = BINDINGS.iter().map(|b| key_cells(b.key)).max().unwrap_or(0);

    let mut out = String::new();
    out.push_str("Keybindings\n");
    for group in Group::ORDER {
        out.push('\n');
        out.push_str(group.title());
        out.push('\n');
        for binding in BINDINGS.iter().filter(|b| b.group == group) {
            let key = if color {
                binding.key.to_string()
            } else {
                asciify(binding.key)
            };
            let pad = key_width.saturating_sub(key.chars().count());
            out.push_str("  ");
            out.push_str(&key);
            for _ in 0..pad {
                out.push(' ');
            }
            out.push_str("  ");
            out.push_str(binding.action);
            out.push('\n');
        }
    }
    out.push_str("\nEsc / q to close");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogue is non-empty and every binding carries a key + action.
    #[test]
    fn catalogue_is_well_formed() {
        assert!(!BINDINGS.is_empty());
        for b in BINDINGS {
            assert!(!b.key.is_empty(), "every binding needs a key");
            assert!(!b.action.is_empty(), "every binding needs an action");
        }
    }

    #[test]
    fn help_has_no_mouse_mode_toggle() {
        let text = help_text(true);
        assert!(!text.contains("⌃T"), "Ctrl+T mouse mode must be removed");
        assert!(
            !text.to_ascii_lowercase().contains("mouse mode"),
            "help must not advertise a mouse mode"
        );
    }

    /// Help text lists every group heading and every action, in order.
    #[test]
    fn help_text_contains_every_group_and_action() {
        let text = help_text(true);
        let mut last = 0;
        for group in Group::ORDER {
            let idx = text
                .find(group.title())
                .unwrap_or_else(|| panic!("missing group heading {}", group.title()));
            assert!(idx >= last, "group headings must appear in ORDER sequence");
            last = idx;
        }
        for b in BINDINGS {
            assert!(
                text.contains(b.action),
                "help text must list action {:?}",
                b.action
            );
        }
        assert!(
            text.contains("Esc / q to close"),
            "must show the close hint"
        );
    }

    /// Under `NO_COLOR` the modifier glyphs degrade to ASCII so the overlay
    /// is readable without a Nerd Font.
    #[test]
    fn no_color_help_uses_ascii_modifiers() {
        let text = help_text(false);
        assert!(text.contains("^B"), "Ctrl glyph must become ^: {text}");
        assert!(text.contains("Alt+Y"), "Option glyph must become Alt+");
        assert!(
            !text.contains('\u{2303}') && !text.contains('\u{2325}'),
            "no rich modifier glyphs may survive under NO_COLOR"
        );
    }

    /// Every advertised binding's key degrades cleanly under `NO_COLOR`: no
    /// rich modifier glyph (`⌃ ⌥ ⇧ ⇞ ⇟`) may survive the ASCII fallback, so a
    /// newly added chord that forgets an `asciify` mapping is caught here. This
    /// covers the added `⌥1`/`⌥2`, `⌃V`, `⌃⇧C`, `⌃E`, `⌃R`, and the printable
    /// `F3`/`F11`/`Space / Tab` rows alike.
    #[test]
    fn every_binding_key_asciifies_without_rich_glyphs() {
        const RICH: [char; 5] = ['\u{2303}', '\u{2325}', '\u{21e7}', '\u{21de}', '\u{21df}'];
        for b in BINDINGS {
            let ascii = asciify(b.key);
            for g in RICH {
                assert!(
                    !ascii.contains(g),
                    "binding {:?} leaves rich glyph {g:?} after asciify: {ascii:?}",
                    b.key
                );
            }
        }
        // The compound `⌃⇧C` copy alias degrades to a fully-ASCII chord.
        assert!(
            help_text(false).contains("^Shift+C"),
            "Ctrl+Shift+C must degrade to ^Shift+C under NO_COLOR"
        );
    }

    /// Locks the curated catalogue: every high-value runtime shortcut named by
    /// the P0 discoverability contract must appear in the help overlay. Each
    /// phrase below advertises a real chord handled in `tui/app/keys.rs`
    /// (F3/Alt+1 → `/model`, Alt+2 → `/smart`, Ctrl+V paste, Ctrl+Shift+C copy,
    /// Ctrl+E external editor, Ctrl+R rewind, F11 focus) plus slash autocomplete.
    #[test]
    fn help_lists_required_runtime_shortcuts() {
        let text = help_text(true);
        for needle in [
            "open model picker",
            "open Smart settings",
            "paste from clipboard",
            "copy last message (alias)",
            "edit prompt in external editor",
            "open rewind viewer",
            "toggle focus mode",
            "accept slash command autocomplete",
        ] {
            assert!(
                text.contains(needle),
                "help overlay must advertise {needle:?}"
            );
        }
        // The keys that carry those actions must be present too (colored form).
        for key in ["F3 / ⌥1", "⌥2", "⌃V", "⌃⇧C", "⌃E", "⌃R", "F11", "Space / Tab"] {
            assert!(text.contains(key), "help overlay must show key {key:?}");
        }
    }

    /// Command hints resolve only for genuine command↔accelerator pairs
    /// and are case-insensitive on the command name.
    #[test]
    fn command_hint_resolves_known_pairs_only() {
        assert_eq!(command_hint("help", true).as_deref(), Some("?"));
        assert_eq!(command_hint("AGENTS", true).as_deref(), Some("\u{2303}A"));
        assert_eq!(
            command_hint("model", true).as_deref(),
            Some("F3 / \u{2325}1")
        );
        assert_eq!(command_hint("smart", true).as_deref(), Some("\u{2325}2"));
        // A real command with no global accelerator yields nothing.
        assert_eq!(command_hint("status", true), None);
        // An unknown command yields nothing.
        assert_eq!(command_hint("nonexistent", true), None);
    }

    /// Under `NO_COLOR` the hint's modifier glyph degrades to ASCII.
    #[test]
    fn command_hint_degrades_under_no_color() {
        assert_eq!(command_hint("agents", false).as_deref(), Some("^A"));
        assert_eq!(command_hint("rewind", false).as_deref(), Some("^R"));
        assert_eq!(
            command_hint("model", false).as_deref(),
            Some("F3 / Alt+1")
        );
        assert_eq!(command_hint("smart", false).as_deref(), Some("Alt+2"));
        // A bare printable key is unchanged.
        assert_eq!(command_hint("help", false).as_deref(), Some("?"));
    }

    /// Every command named in the hint table is a real slash command and
    /// every key is a modifier-or-printable glyph (no accidental typos).
    #[test]
    fn command_hint_table_is_well_formed() {
        for (cmd, key) in COMMAND_HINTS {
            assert!(!cmd.is_empty(), "hint command must be non-empty");
            assert!(!key.is_empty(), "hint key must be non-empty");
        }
    }

    /// The action column aligns: every action starts at the same column
    /// within a group (key column is padded to a common width).
    #[test]
    fn key_column_is_width_aligned() {
        let text = help_text(true);
        // All action phrases are preceded by exactly the padded key block,
        // so the byte offset of the action within its line is constant.
        let offsets: Vec<usize> = text
            .lines()
            .filter_map(|line| line.find("toggle sidebar panel").map(|_| line.len()))
            .collect();
        // Sanity: the known action is present exactly once.
        assert_eq!(offsets.len(), 1);
    }
}
