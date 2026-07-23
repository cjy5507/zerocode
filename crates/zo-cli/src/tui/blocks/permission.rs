//! `RenderBlock::PermissionPrompt` widget — thick-border warning card.
//!
//! Per `.zo/design/components.md` §5.5. Keyboard focus routing and
//! the actual `oneshot::Sender` resolution are owned by the event
//! loop (Lane L6/L8); this widget renders the choices as a navigable
//! list with a visible cursor so the most safety-critical modal matches
//! every other modal's `↑↓` + `Enter` affordance (the letter keys stay
//! as accelerators).
//!
//! See `code-rules.md` R2 (no ANSI), R9 (`&Theme` styling).

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use runtime::message_stream::{PermissionDecision, PermissionPrompt};

use crate::tui::cards::{CardFrame, SurfaceKind};
use crate::tui::theme::Theme;

use super::wrapped_rows;

/// Cursor marker for the highlighted choice. Mirrors `modals::CURSOR_MARKER`
/// so the permission prompt reads identically to every other list modal.
const CURSOR_MARKER: &str = "\u{276F} "; // ❯
/// Blank lead-in for non-selected rows, padded to the marker's cell width so
/// the choices stay column-aligned.
const BLANK_MARKER: &str = "  ";

/// The safe default focus for a fresh prompt: the first choice that *denies*.
///
/// The permission prompt is the one modal where an accidental confirm is
/// dangerous, so the cursor starts on a deny choice (preferring the
/// non-sticky [`PermissionDecision::Deny`] over `DenyAlways`). When no deny
/// choice exists at all the last choice is used, since the conventional
/// ordering puts the safest option last.
#[must_use]
pub fn default_selected_index(prompt: &PermissionPrompt) -> usize {
    if let Some(idx) = prompt
        .choices
        .iter()
        .position(|c| c.decision == PermissionDecision::Deny)
    {
        return idx;
    }
    if let Some(idx) = prompt
        .choices
        .iter()
        .position(|c| c.decision == PermissionDecision::DenyAlways)
    {
        return idx;
    }
    prompt.choices.len().saturating_sub(1)
}

/// The decision the focused choice resolves to, for `Enter`-confirm.
///
/// Returns `None` when `selected` is out of range (an empty prompt), so the
/// caller can no-op rather than confirm something that does not exist.
#[must_use]
pub fn decision_for_selected(
    prompt: &PermissionPrompt,
    selected: usize,
) -> Option<PermissionDecision> {
    prompt.choices.get(selected).map(|c| c.decision)
}

/// Move the cursor one row `up` (or down) within `len` choices, clamped at
/// both ends so navigation never wraps.
///
/// Shared by the `↑`/`↓` handlers so the safety-critical prompt always has a
/// valid focused choice and an `↓` at the bottom can never wrap up to an
/// allow option.
#[must_use]
pub fn move_selection(selected: usize, up: bool, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let max = len - 1;
    if up {
        selected.saturating_sub(1)
    } else {
        selected.saturating_add(1).min(max)
    }
}

/// Render the permission prompt card with `selected` highlighting the
/// currently-focused choice.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    prompt: &PermissionPrompt,
    theme: &Theme,
    focused: bool,
    selected: usize,
    scroll_offset: u16,
) {
    let border_style = if focused {
        Style::new()
            .fg(theme.palette.error)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(theme.palette.warn)
    };
    let title = Line::from(vec![
        Span::raw(" "),
        Span::styled(
            "Permission required",
            Style::new()
                .fg(theme.palette.warn)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ]);

    let lines = rendered_lines(prompt, theme, selected);
    let block = CardFrame::new(SurfaceKind::Danger, theme)
        .border_style(border_style)
        .title(title)
        .block();
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0)),
        area,
    );
}

/// Render a height-bounded permission prompt for the inline viewport.
///
/// All choices remain visible and use the same selection/accelerator contract;
/// only the verbose reason/audit body and blank spacer rows are collapsed.
pub fn draw_compact(
    frame: &mut Frame<'_>,
    area: Rect,
    prompt: &PermissionPrompt,
    theme: &Theme,
    selected: usize,
) {
    let border_style = Style::new()
        .fg(theme.palette.error)
        .add_modifier(Modifier::BOLD);
    let title = Line::from(Span::styled(
        " Permission required ",
        Style::new()
            .fg(theme.palette.warn)
            .add_modifier(Modifier::BOLD),
    ));
    let mut lines = Vec::with_capacity(prompt.choices.len().saturating_add(2));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{} · ", prompt.tool_name),
            Style::new()
                .fg(theme.palette.fg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(prompt.reasoning.clone(), Style::new().fg(theme.palette.dim)),
    ]));
    for (idx, choice) in prompt.choices.iter().enumerate() {
        let is_selected = idx == selected;
        let marker = if is_selected {
            CURSOR_MARKER
        } else {
            BLANK_MARKER
        };
        let style = if is_selected {
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(theme.palette.fg)
        };
        lines.push(Line::from(vec![
            Span::styled(marker, style),
            Span::styled(format!("[ {} ] {}", choice.key, choice.label), style),
        ]));
    }
    lines.push(Line::from(Span::styled(
        "↑↓ move · Enter confirm · Esc deny",
        Style::new().fg(theme.palette.dim),
    )));

    let block = CardFrame::new(SurfaceKind::Danger, theme)
        .border_style(border_style)
        .title(title)
        .block();
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

pub(crate) fn estimate_rows(
    prompt: &PermissionPrompt,
    theme: &Theme,
    selected: usize,
    width: u16,
) -> u16 {
    let inner_width = width.saturating_sub(2).max(1);
    wrapped_rows(&rendered_lines(prompt, theme, selected), inner_width).saturating_add(2)
}

fn rendered_lines<'a>(
    prompt: &'a PermissionPrompt,
    theme: &Theme,
    selected: usize,
) -> Vec<Line<'a>> {
    let mut lines: Vec<Line<'_>> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            "Tool: ",
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(prompt.tool_name.clone(), Style::new().fg(theme.palette.fg)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(
            "Reason: ",
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(prompt.reasoning.clone(), Style::new().fg(theme.palette.fg)),
    ]));
    if let Some(audit_hint) = &prompt.audit_hint {
        lines.push(Line::from(vec![
            Span::styled(
                "Audit: ",
                Style::new()
                    .fg(theme.palette.dim)
                    .add_modifier(Modifier::DIM),
            ),
            Span::styled(audit_hint.clone(), Style::new().fg(theme.palette.warn)),
        ]));
    }
    lines.push(Line::from(""));

    // One navigable row per choice: a visible cursor marks the focused choice
    // and the single-key accelerator stays inline as a `[ y ]` hint.
    for (idx, choice) in prompt.choices.iter().enumerate() {
        let is_selected = idx == selected;
        let marker = if is_selected {
            CURSOR_MARKER
        } else {
            BLANK_MARKER
        };
        let label_style = if is_selected {
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(theme.palette.fg)
        };
        lines.push(Line::from(vec![
            Span::styled(
                marker,
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[ {} ]", choice.key),
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(choice.label.clone(), label_style),
        ]));
    }

    // Footer: the navigation contract. The per-choice `[ y ]` hints above
    // double as the letter-key accelerators, so they need no restating here.
    lines.push(Line::from(""));
    lines.push(Line::from(vec![Span::styled(
        "\u{2191}\u{2193} move  \u{00b7}  Enter confirm  \u{00b7}  Esc deny",
        Style::new()
            .fg(theme.palette.dim)
            .add_modifier(Modifier::DIM),
    )]));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::message_stream::{BlockId, PermissionChoice, ToolCallId};
    use tokio::sync::oneshot;

    /// Build a prompt from `(key, label, decision)` triples. The oneshot
    /// receiver is dropped immediately — the renderer and default helper
    /// never send through it, so only a live sender is needed.
    fn prompt_with(choices: &[(char, &str, PermissionDecision)]) -> PermissionPrompt {
        let (tx, _rx) = oneshot::channel();
        PermissionPrompt {
            id: BlockId(1),
            tool_call_id: ToolCallId("tc-1".to_string()),
            tool_name: "bash".to_string(),
            reasoning: "run a shell command".to_string(),
            audit_hint: None,
            choices: choices
                .iter()
                .map(|(key, label, decision)| PermissionChoice {
                    key: *key,
                    label: (*label).to_string(),
                    decision: *decision,
                })
                .collect(),
            responder: tx,
        }
    }

    fn sample_prompt() -> PermissionPrompt {
        prompt_with(&[
            ('y', "Allow once", PermissionDecision::AllowOnce),
            ('a', "Allow always", PermissionDecision::AllowAlways),
            ('n', "Deny", PermissionDecision::Deny),
        ])
    }

    /// The cursor defaults to the (safe) Deny choice rather than an allow.
    #[test]
    fn default_focus_is_deny() {
        let prompt = sample_prompt();
        let idx = default_selected_index(&prompt);
        assert_eq!(
            prompt.choices[idx].decision,
            PermissionDecision::Deny,
            "fresh permission prompt must default to the safe Deny choice"
        );
    }

    /// With no plain Deny, the default falls back to a `DenyAlways` choice
    /// rather than landing on an allow.
    #[test]
    fn default_focus_prefers_any_deny_when_no_plain_deny() {
        let prompt = prompt_with(&[
            ('y', "Allow once", PermissionDecision::AllowOnce),
            ('d', "Deny always", PermissionDecision::DenyAlways),
        ]);
        let idx = default_selected_index(&prompt);
        assert_eq!(prompt.choices[idx].decision, PermissionDecision::DenyAlways);
    }

    /// `↑`/`↓` move the cursor and clamp at both ends (no wrap), and `Enter`
    /// on the focused row resolves that exact choice's decision — the full
    /// arrow-navigation contract the gater wires into the key handler.
    #[test]
    fn arrow_navigation_clamps_and_enter_resolves_focused_choice() {
        let prompt = sample_prompt();
        let len = prompt.choices.len();

        // Start on the safe default (Deny == last index here).
        let start = default_selected_index(&prompt);
        assert_eq!(start, len - 1);

        // ↓ at the bottom stays put (no wrap to the dangerous top).
        let down = move_selection(start, false, len);
        assert_eq!(down, len - 1, "down at the end must not wrap to an allow");

        // ↑ walks toward the top, one row at a time.
        let up1 = move_selection(start, true, len);
        assert_eq!(up1, len - 2);
        let up2 = move_selection(up1, true, len);
        assert_eq!(up2, 0);
        // ↑ at the top stays put.
        assert_eq!(move_selection(up2, true, len), 0);

        // Enter resolves whatever is focused.
        assert_eq!(
            decision_for_selected(&prompt, start),
            Some(PermissionDecision::Deny)
        );
        assert_eq!(
            decision_for_selected(&prompt, 0),
            Some(PermissionDecision::AllowOnce)
        );
        // Out-of-range focus (empty prompt) confirms nothing.
        assert_eq!(decision_for_selected(&prompt, len), None);
    }

    /// The cursor marker renders on exactly the selected choice row and the
    /// other choice rows carry the blank lead-in — i.e. moving `selected`
    /// moves the visible cursor. This drives the production renderer so the
    /// `↑↓`-navigation wiring has a visible target on every index.
    #[test]
    fn cursor_marks_only_the_selected_choice() {
        let prompt = sample_prompt();
        let theme = Theme::no_color();

        for selected in 0..prompt.choices.len() {
            let lines = rendered_lines(&prompt, &theme, selected);
            // The choice rows are the ones whose first span is a marker.
            let choice_rows: Vec<&Line<'_>> = lines
                .iter()
                .filter(|l| {
                    l.spans
                        .first()
                        .is_some_and(|s| s.content == CURSOR_MARKER || s.content == BLANK_MARKER)
                })
                .collect();
            assert_eq!(
                choice_rows.len(),
                prompt.choices.len(),
                "every choice must render one navigable row"
            );
            for (idx, row) in choice_rows.iter().enumerate() {
                let lead = row.spans.first().expect("row has a lead span");
                if idx == selected {
                    assert_eq!(
                        lead.content, CURSOR_MARKER,
                        "selected choice {idx} must show the cursor"
                    );
                } else {
                    assert_eq!(
                        lead.content, BLANK_MARKER,
                        "unselected choice {idx} must show the blank lead-in"
                    );
                }
            }
        }
    }
}
