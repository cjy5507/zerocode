//! Interactive `/tier` picker for the ordered Architect PLAN/VERIFY model pool.
//!
//! The modal owns only selection, input, confirmation, and rendering state.
//! Settings reads and writes stay in the session host, which feeds a fresh
//! [`DeepTierView`] back after every [`DeepTierAction`].

use commands::DeepTierAction;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};
use unicode_width::UnicodeWidthStr;

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, draw_scrollbar, key_hint_footer_fitted,
    selected_style,
};

/// Active ordered pool plus the source that supplied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepTierView {
    /// Active models in preference order.
    pub models: Vec<String>,
    /// Whether the merged pool came from explicit configuration.
    pub configured: bool,
}

/// An open inline editor: either appending a new entry or replacing one in
/// place. The rank being replaced is carried here (not re-derived from
/// `selected` at commit time) so scrolling the list mid-edit cannot retarget
/// the write.
#[derive(Debug, Clone)]
struct TierInput {
    /// `None` appends; `Some(rank)` is the 1-based position being replaced.
    replacing: Option<usize>,
    text: String,
}

/// List picker and inline editor for [`DeepTierView`].
#[derive(Debug, Clone)]
pub struct DeepTierModal {
    view: DeepTierView,
    /// `0..models.len()` selects a model; `models.len()` is the trailing
    /// "add model" row, which is what makes adding reachable from Enter alone.
    selected: usize,
    input: Option<TierInput>,
    confirming_reset: bool,
    feedback: Option<(String, bool)>,
}

impl DeepTierModal {
    #[must_use]
    pub fn new(view: DeepTierView) -> Self {
        Self {
            view,
            selected: 0,
            input: None,
            confirming_reset: false,
            feedback: None,
        }
    }

    /// Land the authoritative post-action snapshot and its existing text-command result.
    pub fn apply_update(&mut self, view: Option<DeepTierView>, result: Result<String, String>) {
        if let Some(view) = view {
            let selected_model = self.view.models.get(self.selected).cloned();
            self.view = view;
            self.selected = selected_model
                .as_deref()
                .and_then(|model| self.view.models.iter().position(|candidate| candidate == model))
                .unwrap_or_else(|| self.selected.min(self.last_row()));
        }
        self.input = None;
        self.confirming_reset = false;
        self.feedback = Some(match result {
            Ok(message) => (single_line(&message), false),
            Err(error) => (single_line(&error), true),
        });
    }

    pub fn paste_text(&mut self, text: &str) {
        if let Some(input) = self.input.as_mut() {
            input
                .text
                .extend(text.chars().filter(|ch| !ch.is_control()));
        }
    }

    /// Index of the trailing "add model" row — the last selectable row.
    fn last_row(&self) -> usize {
        self.view.models.len()
    }

    fn on_add_row(&self) -> bool {
        self.selected >= self.view.models.len()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if self.input.is_some() {
            return self.handle_input_key(key);
        }
        if self.confirming_reset {
            return self.handle_reset_confirmation(key);
        }
        // Reorder is checked before plain navigation so Shift+Up/Down is not
        // eaten by the `Up`/`Down` arms. Every mutation is reachable without a
        // letter key: with an IME composing, `a`/`d`/`K`/`J`/`r` never arrive as
        // ASCII, which left the pool editable by `Del` alone — reorder and add
        // looked simply absent. Enter, Delete, and Shift+arrows always arrive.
        if key.modifiers.contains(KeyModifiers::SHIFT) {
            match key.code {
                KeyCode::Up => return self.move_selected_up(),
                KeyCode::Down => return self.move_selected_down(),
                _ => {}
            }
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(ModalResult::Cancelled),
            KeyCode::Up | KeyCode::Char('k') => {
                self.select_up(1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.select_down(1);
                None
            }
            KeyCode::Home => {
                self.selected = 0;
                None
            }
            KeyCode::End => {
                self.selected = self.last_row();
                None
            }
            // Enter is the one key that does the obvious thing on whatever row
            // it lands: the "add model" row opens an empty editor, a model row
            // opens one prefilled with that model, so replacing a planner keeps
            // its rank instead of costing a remove + add + move.
            KeyCode::Enter => {
                self.open_editor();
                None
            }
            KeyCode::Char('a') if key.modifiers.is_empty() => {
                self.selected = self.last_row();
                self.open_editor();
                None
            }
            KeyCode::Char('d') if key.modifiers.is_empty() => self.remove_selected(),
            KeyCode::Delete | KeyCode::Backspace => self.remove_selected(),
            KeyCode::Char('K') => self.move_selected_up(),
            KeyCode::Char('J') => self.move_selected_down(),
            KeyCode::Char('r') if key.modifiers.is_empty() => {
                self.confirming_reset = true;
                self.feedback = None;
                None
            }
            _ => None,
        }
    }

    /// Open the inline editor for the selected row — empty on the "add model"
    /// row, prefilled with the model being replaced otherwise.
    fn open_editor(&mut self) {
        let replacing = (!self.on_add_row()).then_some(self.selected + 1);
        let text = self
            .view
            .models
            .get(self.selected)
            .cloned()
            .unwrap_or_default();
        self.input = Some(TierInput {
            replacing,
            text: if replacing.is_some() { text } else { String::new() },
        });
        self.feedback = None;
    }

    fn remove_selected(&mut self) -> Option<ModalResult> {
        if self.on_add_row() || self.view.models.is_empty() {
            return None;
        }
        self.feedback = None;
        Some(Self::selection(DeepTierAction::Remove {
            target: (self.selected + 1).to_string(),
        }))
    }

    fn move_selected_up(&mut self) -> Option<ModalResult> {
        if self.on_add_row() || self.selected == 0 {
            return None;
        }
        self.feedback = None;
        Some(Self::selection(DeepTierAction::Move {
            from: self.selected + 1,
            to: self.selected,
        }))
    }

    fn move_selected_down(&mut self) -> Option<ModalResult> {
        if self.selected + 1 >= self.view.models.len() {
            return None;
        }
        self.feedback = None;
        Some(Self::selection(DeepTierAction::Move {
            from: self.selected + 1,
            to: self.selected + 2,
        }))
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        let pool = self.view.models.clone();
        let input = self.input.as_mut()?;
        match key.code {
            KeyCode::Esc => {
                self.input = None;
                None
            }
            // Completion from the model catalog: the editor accepts any id, but
            // nobody keeps `gemini-3.6-flash` in their head, and a pool you can
            // only extend by typing an exact id is one most users never extend.
            KeyCode::Tab => {
                if let Some(candidate) = catalog_suggestions(&input.text, &pool, input.replacing)
                    .into_iter()
                    .next()
                {
                    input.text = candidate;
                }
                None
            }
            KeyCode::Enter => {
                let model = input.text.trim().to_string();
                if model.is_empty() {
                    return None;
                }
                let replacing = input.replacing;
                self.input = None;
                Some(Self::selection(match replacing {
                    Some(rank) => DeepTierAction::Set {
                        target: rank.to_string(),
                        model,
                    },
                    None => DeepTierAction::Add { model },
                }))
            }
            KeyCode::Backspace => {
                input.text.pop();
                None
            }
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                input.text.push(ch);
                None
            }
            _ => None,
        }
    }

    fn handle_reset_confirmation(&mut self, key: KeyEvent) -> Option<ModalResult> {
        match key.code {
            KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                self.confirming_reset = false;
                Some(Self::selection(DeepTierAction::Reset))
            }
            KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                self.confirming_reset = false;
                None
            }
            KeyCode::Char('q') => Some(ModalResult::Cancelled),
            _ => None,
        }
    }

    fn selection(action: DeepTierAction) -> ModalResult {
        ModalResult::Selected(ModalSelection::DeepTier(action))
    }

    fn select_up(&mut self, rows: usize) {
        self.selected = self.selected.saturating_sub(rows);
    }

    fn select_down(&mut self, rows: usize) {
        self.selected = self.selected.saturating_add(rows).min(self.last_row());
    }

    fn list_offset(&self, height: u16) -> u16 {
        let len = u16::try_from(self.row_count()).unwrap_or(u16::MAX);
        let max_offset = len.saturating_sub(height);
        let selected = u16::try_from(self.selected).unwrap_or(u16::MAX);
        selected
            .saturating_sub(height.saturating_sub(1))
            .min(max_offset)
    }

    /// Painted list rows: every model plus the trailing "add model" row.
    fn row_count(&self) -> usize {
        self.view.models.len().saturating_add(1)
    }

    #[must_use]
    fn content_rows(&self) -> usize {
        self.row_count().saturating_add(5)
    }

    #[must_use]
    pub fn desired_size(&self, area: Rect, theme: &Theme) -> (u16, u16) {
        let source = self.source_label();
        let row_width = self
            .view
            .models
            .iter()
            .enumerate()
            .map(|(index, model)| {
                let marker = cursor_marker(!theme.no_color);
                format!("{marker}{}. {model} ({source})", index + 1).width()
            })
            .chain(std::iter::once(
                format!("{}{ADD_ROW_LABEL}", cursor_marker(!theme.no_color)).width(),
            ))
            .max()
            .unwrap_or_default();
        // Measured unbounded on purpose: this is deciding how wide the modal
        // *wants* to be, so the footer must not be trimmed to a width that does
        // not exist yet. `draw` fits it to whatever the screen actually granted.
        let footer_width = normal_footer_lines(theme, u16::MAX)
            .iter()
            .map(line_width)
            .max()
            .unwrap_or_default();
        let content_width = row_width
            .max(footer_width)
            .max("Architect PLAN/VERIFY pool · first entry is preferred".width());
        let width = u16::try_from(content_width.saturating_add(4))
            .unwrap_or(u16::MAX)
            .clamp(64, 104)
            .min(area.width.saturating_sub(4).max(24));
        let content = u16::try_from(self.content_rows())
            .unwrap_or(u16::MAX)
            .saturating_add(2);
        let height = content
            .clamp(9, 24)
            .min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    pub fn scroll(&mut self, up: bool, rows: usize) {
        if up {
            self.select_up(rows);
        } else {
            self.select_down(rows);
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(super::modal_title(theme, "Deep-tier models"))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.width == 0 || inner.height < 5 {
            return;
        }
        let [header, list, action, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .areas(inner);

        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                vec![Line::from(Span::styled(
                    "Architect PLAN/VERIFY pool · first entry is preferred",
                    theme.typography.dim,
                ))],
                header.width,
            )),
            header,
        );

        let mut rows: Vec<Line<'static>> = self
            .view
            .models
            .iter()
            .enumerate()
            .map(|(index, model)| self.row_line(index, model, theme))
            .collect();
        rows.push(self.add_row_line(theme));
        let offset = self.list_offset(list.height);
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(rows, list.width)).scroll((offset, 0)),
            list,
        );
        draw_scrollbar(frame, list, offset, self.row_count(), theme);

        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                vec![self.action_line(theme)],
                action.width,
            )),
            action,
        );
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                self.footer_lines(theme, footer.width),
                footer.width,
            )),
            footer,
        );
    }

    fn row_line(&self, index: usize, model: &str, theme: &Theme) -> Line<'static> {
        let selected = index == self.selected;
        let marker = if selected {
            cursor_marker(!theme.no_color)
        } else {
            blank_marker()
        };
        let style = if selected { selected_style(theme) } else { theme.typography.body };
        Line::from(Span::styled(
            format!("{marker}{}. {model} ({})", index + 1, self.source_label()),
            style,
        ))
    }

    /// The trailing "add model" row. Rendered as part of the list (not as a
    /// footer hint) so adding is something the cursor can land on, which is how
    /// a reader discovers it exists.
    fn add_row_line(&self, theme: &Theme) -> Line<'static> {
        let selected = self.on_add_row();
        let marker = if selected {
            cursor_marker(!theme.no_color)
        } else {
            blank_marker()
        };
        let style = if selected {
            selected_style(theme)
        } else {
            theme.typography.dim
        };
        Line::from(Span::styled(format!("{marker}{ADD_ROW_LABEL}"), style))
    }

    fn action_line(&self, theme: &Theme) -> Line<'static> {
        if let Some(input) = self.input.as_ref() {
            let label = match input.replacing {
                Some(rank) => format!("replace #{rank} ❯ "),
                None => "add model ❯ ".to_string(),
            };
            return Line::from(vec![
                Span::styled(label, Style::new().fg(theme.palette.accent)),
                Span::styled(input.text.clone(), theme.typography.body),
                Span::styled("▌", Style::new().fg(theme.palette.accent)),
            ]);
        }
        if self.confirming_reset {
            return Line::from(Span::styled(
                "Reset to the built-in default?",
                Style::new().fg(theme.palette.warn),
            ));
        }
        if let Some((message, is_error)) = self.feedback.as_ref() {
            let color = if *is_error { theme.palette.warn } else { theme.palette.accent };
            return Line::from(Span::styled(message.clone(), Style::new().fg(color)));
        }
        Line::from(Span::styled(
            format!("{} active models · {}", self.view.models.len(), self.source_label()),
            theme.typography.dim,
        ))
    }

    /// `width` is the footer rect's width; hint rows are fitted to it because
    /// this renders without `Wrap`. The modal is *sized* from the unfitted footer
    /// (see `desired_size`), so trimming only happens when the screen clamped the
    /// modal below the width it asked for.
    fn footer_lines(&self, theme: &Theme, width: u16) -> Vec<Line<'static>> {
        if let Some(input) = self.input.as_ref() {
            let commit = if input.replacing.is_some() { "replace" } else { "add" };
            return vec![
                self.suggestion_line(input, theme),
                Line::default(),
                key_hint_footer_fitted(
                    theme,
                    &[
                        ("Enter", commit),
                        ("Tab", "complete"),
                        ("Esc", "cancel input"),
                    ],
                    width,
                ),
            ];
        }
        if self.confirming_reset {
            return vec![
                next_turn_line(theme),
                Line::default(),
                key_hint_footer_fitted(
                    theme,
                    &[("y/Enter", "confirm"), ("n/Esc", "cancel")],
                    width,
                ),
            ];
        }
        normal_footer_lines(theme, width)
    }

    /// The catalog candidates for what has been typed so far, in the footer's
    /// first row (which otherwise carries the "next turn" note — irrelevant
    /// while an edit is still open). Shows what `Tab` would take, plus how many
    /// other models match, so the editor teaches the catalog instead of
    /// demanding the reader already know it.
    fn suggestion_line(&self, input: &TierInput, theme: &Theme) -> Line<'static> {
        let matches = catalog_suggestions(&input.text, &self.view.models, input.replacing);
        let Some(first) = matches.first() else {
            return Line::from(Span::styled(
                "no catalog match · any model id is accepted",
                theme.typography.dim,
            ));
        };
        let mut spans = vec![
            Span::styled("Tab ", theme.typography.dim),
            Span::styled(first.clone(), Style::new().fg(theme.palette.accent)),
        ];
        if matches.len() > 1 {
            spans.push(Span::styled(
                format!(" · {} more match", matches.len() - 1),
                theme.typography.dim,
            ));
        }
        Line::from(spans)
    }

    fn source_label(&self) -> &'static str {
        if self.view.configured { "configured" } else { "built-in default" }
    }
}

/// Trailing list row that opens the editor in append mode.
const ADD_ROW_LABEL: &str = "+ add model…";

/// Catalog models offered for `typed`, best first, excluding entries already in
/// `pool` (a duplicate rank is refused by the writer anyway) — except the one
/// being replaced, which must stay offerable so re-opening its editor and
/// pressing Tab is not a dead end.
///
/// An empty query offers the orchestration-ranked models: the pool is the
/// PLAN/VERIFY roster, so the reasoning-first models are the useful default,
/// not the alphabetical head of the catalog.
fn catalog_suggestions(typed: &str, pool: &[String], replacing: Option<usize>) -> Vec<String> {
    let query = typed.trim().to_ascii_lowercase();
    let kept = replacing
        .and_then(|rank| pool.get(rank.saturating_sub(1)))
        .map(|model| model.to_ascii_lowercase());
    let mut ranked: Vec<(u8, &str)> = api::provider_catalog()
        .iter()
        .filter(|entry| {
            let canonical = entry.canonical_model_id.to_ascii_lowercase();
            kept.as_ref() == Some(&canonical)
                || !pool
                    .iter()
                    .any(|model| model.eq_ignore_ascii_case(entry.canonical_model_id))
        })
        .filter(|entry| {
            query.is_empty()
                || entry.canonical_model_id.to_ascii_lowercase().contains(&query)
                || entry.alias.to_ascii_lowercase().contains(&query)
        })
        .map(|entry| {
            // Orchestrators first (that is what this pool is for), then the
            // rest of the catalog in registry order.
            (
                entry.orchestration_rank.unwrap_or(u8::MAX),
                entry.canonical_model_id,
            )
        })
        .collect();
    ranked.sort_by_key(|(rank, _)| *rank);
    let mut seen = Vec::new();
    for (_, model) in ranked {
        if !seen.iter().any(|kept: &String| kept == model) {
            seen.push(model.to_string());
        }
    }
    seen
}

fn normal_footer_lines(theme: &Theme, width: u16) -> Vec<Line<'static>> {
    vec![
        next_turn_line(theme),
        Line::default(),
        // Keys that survive an IME: the letter shortcuts still work, but the
        // footer advertises the arrow/Enter/Del set, because those are the ones
        // that arrive while Hangul (or any other) composition is active.
        key_hint_footer_fitted(
            theme,
            &[
                ("↑↓", "select"),
                ("Enter", "edit"),
                ("shift+↑↓", "reorder"),
                ("Del", "remove"),
                ("r", "reset"),
                ("Esc", "close"),
            ],
            width,
        ),
    ]
}

fn next_turn_line(theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        "changes apply from the next turn",
        theme.typography.dim,
    ))
}

fn single_line(value: &str) -> String {
    value.replace(['\r', '\n'], "  ·  ")
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| span.content.as_ref().width())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn view(models: &[&str], configured: bool) -> DeepTierView {
        DeepTierView {
            models: models.iter().map(|model| (*model).to_string()).collect(),
            configured,
        }
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn shifted(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::SHIFT)
    }

    fn dump(modal: &DeepTierModal, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| modal.draw(frame, frame.area(), &Theme::zo()))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|row| {
                (0..width)
                    .map(|column| buffer[(column, row)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }


    #[test]
    fn selection_clamps_at_pool_bounds_including_the_add_row() {
        let mut modal = DeepTierModal::new(view(&["architect-a", "architect-b"], true));
        modal.handle_key(press(KeyCode::Up));
        assert_eq!(modal.selected, 0);
        modal.handle_key(press(KeyCode::Char('j')));
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected, 2, "the add row is the last selectable row");
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected, 2, "and selection stops there");
        modal.handle_key(press(KeyCode::Char('k')));
        assert_eq!(modal.selected, 1);
    }

    #[test]
    fn add_input_commits_and_escape_cancels_only_the_input() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        modal.handle_key(press(KeyCode::Char('a')));
        modal.handle_key(press(KeyCode::Char('x')));
        assert!(modal.input.is_some());
        assert!(modal.handle_key(press(KeyCode::Esc)).is_none());
        assert!(modal.input.is_none());

        modal.handle_key(press(KeyCode::Char('a')));
        for ch in "new-model".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert!(matches!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Add { model }
            ))) if model == "new-model"
        ));
    }

    #[test]
    fn remove_last_refusal_is_surfaced_inline() {
        let pool = view(&["only-architect"], true);
        let mut modal = DeepTierModal::new(pool.clone());
        assert!(matches!(
            modal.handle_key(press(KeyCode::Char('d'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Remove { target }
            ))) if target == "1"
        ));
        modal.apply_update(
            Some(pool),
            Err("Cannot remove the last deep-tier model".to_string()),
        );
        let rendered = dump(&modal, 96, 12);
        assert!(rendered.contains("Cannot remove the last deep-tier model"), "{rendered}");
    }

    #[test]
    fn uppercase_jk_emit_ordered_move_actions() {
        let mut modal = DeepTierModal::new(view(&["a", "b", "c"], true));
        modal.handle_key(press(KeyCode::Down));
        assert!(matches!(
            modal.handle_key(shifted(KeyCode::Char('K'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Move { from: 2, to: 1 }
            )))
        ));
        assert!(matches!(
            modal.handle_key(shifted(KeyCode::Char('J'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Move { from: 2, to: 3 }
            )))
        ));
    }

    #[test]
    fn reset_requires_inline_confirmation() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        assert!(modal.handle_key(press(KeyCode::Char('r'))).is_none());
        assert!(modal.confirming_reset);
        assert!(modal.handle_key(press(KeyCode::Char('n'))).is_none());
        assert!(!modal.confirming_reset);

        modal.handle_key(press(KeyCode::Char('r')));
        assert!(matches!(
            modal.handle_key(press(KeyCode::Char('y'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Reset
            )))
        ));
    }

    #[test]
    fn render_dump_shows_named_pool_source_selection_and_keys() {
        let modal = DeepTierModal::new(view(
            &["claude-architect", "gpt-verifier", "gemini-reviewer"],
            true,
        ));
        let rendered = dump(&modal, 104, 14);
        for expected in [
            "Deep-tier models",
            "1. claude-architect (configured)",
            "2. gpt-verifier (configured)",
            "3. gemini-reviewer (configured)",
            ADD_ROW_LABEL,
            // Every advertised key survives an IME: arrows, Enter, Del.
            "↑↓ select",
            "Enter edit",
            "shift+↑↓ reorder",
            "Del remove",
            "changes apply from the next turn",
        ] {
            assert!(rendered.contains(expected), "missing {expected:?} in:\n{rendered}");
        }
    }

    /// The bug this modal shipped with: with Hangul composition active none of
    /// `a` / `d` / `K` / `J` / `r` reach the modal as ASCII, so `Delete` was the
    /// only mutation that worked and the pool read as remove-only. Every
    /// mutation must therefore be reachable without a letter key.
    #[test]
    fn every_mutation_is_reachable_without_a_letter_key() {
        let mut modal = DeepTierModal::new(view(&["a", "b", "c"], true));

        // Reorder: shift+arrows.
        modal.handle_key(press(KeyCode::Down));
        assert!(matches!(
            modal.handle_key(shifted(KeyCode::Up)),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Move { from: 2, to: 1 }
            )))
        ));
        assert!(matches!(
            modal.handle_key(shifted(KeyCode::Down)),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Move { from: 2, to: 3 }
            )))
        ));

        // Remove: Delete or Backspace.
        assert!(matches!(
            modal.handle_key(press(KeyCode::Backspace)),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Remove { target }
            ))) if target == "2"
        ));

        // Add: End lands on the add row, Enter opens an empty editor.
        modal.handle_key(press(KeyCode::End));
        modal.handle_key(press(KeyCode::Enter));
        assert_eq!(modal.input.as_ref().map(|input| input.replacing), Some(None));
    }

    #[test]
    fn enter_on_a_model_row_replaces_it_in_place() {
        let mut modal = DeepTierModal::new(view(&["architect-a", "architect-b"], true));
        modal.handle_key(press(KeyCode::Down));
        modal.handle_key(press(KeyCode::Enter));
        let input = modal.input.as_ref().expect("editor opens prefilled");
        assert_eq!(input.replacing, Some(2));
        assert_eq!(input.text, "architect-b", "prefilled with the model it replaces");

        for _ in 0.."architect-b".len() {
            modal.handle_key(press(KeyCode::Backspace));
        }
        for ch in "architect-c".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert!(matches!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Set { target, model }
            ))) if target == "2" && model == "architect-c"
        ));
    }

    #[test]
    fn the_add_row_has_nothing_to_remove_or_reorder() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        modal.handle_key(press(KeyCode::End));
        assert!(modal.on_add_row());
        assert!(modal.handle_key(press(KeyCode::Delete)).is_none());
        assert!(modal.handle_key(shifted(KeyCode::Up)).is_none());
        assert!(modal.handle_key(shifted(KeyCode::Down)).is_none());
    }

    #[test]
    fn tab_completes_the_editor_from_the_model_catalog() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        modal.handle_key(press(KeyCode::End));
        modal.handle_key(press(KeyCode::Enter));
        for ch in "opus".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        modal.handle_key(press(KeyCode::Tab));
        let completed = modal.input.as_ref().expect("editor open").text.clone();
        assert!(
            completed.contains("opus") && completed != "opus",
            "Tab should land a catalog id, got {completed:?}"
        );
    }

    #[test]
    fn catalog_suggestions_skip_models_already_pooled_but_keep_the_replaced_one() {
        let pooled = api::provider_catalog()
            .iter()
            .find(|entry| entry.orchestration_rank.is_some())
            .expect("catalog has an orchestrator")
            .canonical_model_id
            .to_string();

        let pool = vec![pooled.clone()];
        assert!(
            !catalog_suggestions("", &pool, None).contains(&pooled),
            "an already-pooled model is not offered for append"
        );
        assert!(
            catalog_suggestions("", &pool, Some(1)).contains(&pooled),
            "but re-opening its own editor must still offer it"
        );
    }

    #[test]
    fn the_editor_footer_shows_what_tab_would_take() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        modal.handle_key(press(KeyCode::End));
        modal.handle_key(press(KeyCode::Enter));
        for ch in "opus".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        let rendered = dump(&modal, 104, 14);
        assert!(rendered.contains("add model ❯ opus"), "{rendered}");
        assert!(rendered.contains("Tab "), "{rendered}");
        assert!(rendered.contains("Tab complete"), "{rendered}");
    }
}
