//! Generic single-select choice modal (Phase 3, Lane L6).
//!
//! Backs arbitrary in-app yes/no prompts, the `/resume` session list, slash
//! arg-pickers, and the `/login` · `/connect` provider list. See
//! `.zo/design/components.md` §6 for the visual language.
//!
//! A row is three columns — label, description, status badge — laid out by the
//! modal from structured [`ChoiceRow`] data. Callers used to hand-pad a single
//! string (`"Claude   —  Anthropic OAuth"`), which only lines up for the exact
//! labels it was counted against and cannot be styled per column.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, key_hint_footer_fitted,
    row_detail_style, selected_style,
};
use crate::tui::fuzzy;

/// Cursor rows a PageUp/PageDown jumps through. Matches the page stride used by
/// the other selection-list modals (see `tool_toggle::page_down`).
const PAGE_STRIDE: usize = 8;

/// Status shown at the end of a choice row.
///
/// Deliberately an enum of meanings rather than caller-supplied text: the theme
/// owns how each state looks, and the vocabulary stays consistent across every
/// list that adopts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceBadge {
    /// A credential for this entry is already stored on disk.
    ///
    /// Deliberately *not* "connected": nothing here contacts the provider or
    /// checks expiry, so the badge claims only what it can actually see.
    Saved,
    /// Choosing this row will prompt for an API key.
    NeedsKey,
    /// Runs on this machine — discovered locally, no account needed.
    Local,
    /// The entry currently in effect.
    Current,
}

impl ChoiceBadge {
    /// Short lowercase text rendered inside the badge brackets.
    ///
    /// ASCII by construction, so the badge needs no `NO_COLOR` fallback glyph.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Saved => "saved",
            Self::NeedsKey => "needs key",
            Self::Local => "local",
            Self::Current => "current",
        }
    }
}

/// One selectable row.
#[derive(Debug, Clone)]
pub struct ChoiceRow {
    /// User-visible option text, and what the host receives on selection —
    /// several callers re-dispatch it verbatim as `/<command> <label>`, so it
    /// must stay exactly what the row displays.
    pub label: String,
    /// One-line explanation shown beside the label in the muted column.
    pub description: Option<String>,
    /// Status badge closing the row.
    pub badge: Option<ChoiceBadge>,
    /// Section this row belongs to. Headers render only when the visible rows
    /// span more than one section, so a single-group list stays a plain list.
    pub group: Option<String>,
}

impl ChoiceRow {
    /// A bare row carrying only its label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
            badge: None,
            group: None,
        }
    }

    /// Attach the muted one-line description.
    #[must_use]
    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Attach the trailing status badge.
    #[must_use]
    pub fn with_badge(mut self, badge: ChoiceBadge) -> Self {
        self.badge = Some(badge);
        self
    }

    /// Place the row in a named section.
    #[must_use]
    pub fn in_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }
}

/// Generic single-select list modal.
#[derive(Debug, Clone)]
pub struct ChoicePickerModal {
    title: String,
    rows: Vec<ChoiceRow>,
    /// Live type-ahead query, matched as a fuzzy subsequence against a row's
    /// label, description, and group.
    query: String,
    /// Indices into `rows` surviving `query`: sections contiguous in
    /// first-appearance order, registry order preserved inside each section.
    ///
    /// Selection reports the ORIGINAL index taken from here. The host indexes
    /// parallel side-lists with it (`login_provider_ids`, `session_ids` in
    /// `app/mod.rs`), so handing back a filtered position would open a
    /// different provider — or resume a different session — than the one the
    /// user was looking at.
    filtered: Vec<usize>,
    /// Cursor position *within `filtered`*.
    cursor: usize,
}

impl ChoicePickerModal {
    /// Construct a modal with `title` and a list of plain option labels.
    #[must_use]
    pub fn new(title: impl Into<String>, options: Vec<String>) -> Self {
        Self::with_rows(title, options.into_iter().map(ChoiceRow::new).collect())
    }

    /// Construct a modal from structured rows (descriptions, badges, groups).
    #[must_use]
    pub fn with_rows(title: impl Into<String>, rows: Vec<ChoiceRow>) -> Self {
        let mut modal = Self {
            title: title.into(),
            rows,
            query: String::new(),
            filtered: Vec::new(),
            cursor: 0,
        };
        // Through `refilter` even with an empty query, so the initial view is
        // section-ordered by the same rule every later view uses.
        modal.refilter();
        modal
    }

    /// Title displayed in the modal border.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Current cursor index within the visible (filtered) list.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// The live type-ahead query string.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Number of currently visible options (after type-ahead filtering).
    #[must_use]
    pub fn len(&self) -> usize {
        self.filtered.len()
    }

    /// `true` if nothing is currently visible — an empty registry, or a query
    /// that matched no row.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filtered.is_empty()
    }

    /// The `rows` index the cursor points at, if anything is visible.
    fn selected_row_index(&self) -> Option<usize> {
        self.filtered.get(self.cursor).copied()
    }

    /// Re-derive `filtered` against the current `query`, clamping the cursor
    /// back into range. An empty query shows everything.
    fn refilter(&mut self) {
        let needle = self.query.to_lowercase();
        self.filtered = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                if needle.is_empty() {
                    return true;
                }
                if fuzzy::is_subsequence(&row.label.to_lowercase(), &needle) {
                    return true;
                }
                if row
                    .description
                    .as_ref()
                    .is_some_and(|text| fuzzy::is_subsequence(&text.to_lowercase(), &needle))
                {
                    return true;
                }
                row.group
                    .as_ref()
                    .is_some_and(|text| fuzzy::is_subsequence(&text.to_lowercase(), &needle))
            })
            .map(|(index, _)| index)
            .collect();
        // Keep each section contiguous, ordered by where it first appears.
        // Rows keep their registry order inside a section, so only whole
        // sections move. Without this a caller that appends a row belonging to
        // an earlier section — easy, since rows and their parallel id list are
        // built by pushing — reopens that section further down, printing the
        // same heading twice.
        let mut sections: Vec<&str> = Vec::new();
        let mut ranked: Vec<(usize, usize)> = Vec::with_capacity(self.filtered.len());
        for &index in &self.filtered {
            let group = self.rows[index].group.as_deref().unwrap_or_default();
            let rank = sections.iter().position(|seen| *seen == group).unwrap_or_else(|| {
                sections.push(group);
                sections.len() - 1
            });
            ranked.push((rank, index));
        }
        ranked.sort_by_key(|(rank, _)| *rank);
        self.filtered = ranked.into_iter().map(|(_, index)| index).collect();
        if self.cursor >= self.filtered.len() {
            self.cursor = self.filtered.len().saturating_sub(1);
        }
    }

    /// Move cursor down by one, wrapping.
    pub fn move_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + 1) % self.filtered.len();
    }

    /// Move cursor up by one, wrapping.
    pub fn move_up(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        if self.cursor == 0 {
            self.cursor = self.filtered.len() - 1;
        } else {
            self.cursor -= 1;
        }
    }

    /// Move the cursor down by a page, clamping at the last option.
    pub fn page_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + PAGE_STRIDE).min(self.filtered.len() - 1);
    }

    /// Move the cursor up by a page, clamping at the first option.
    pub fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(PAGE_STRIDE);
    }

    /// Jump the cursor to the first option.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Jump the cursor to the last option.
    pub fn move_end(&mut self) {
        self.cursor = self.filtered.len().saturating_sub(1);
    }

    /// Move the cursor down by `count` rows, clamping at the end. Used by the
    /// host's mouse-wheel routing (which owns the app-level dispatch).
    pub fn scroll_down(&mut self, count: usize) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + count).min(self.filtered.len() - 1);
    }

    /// Move the cursor up by `count` rows, clamping at the top. Used by the
    /// host's mouse-wheel routing.
    pub fn scroll_up(&mut self, count: usize) {
        self.cursor = self.cursor.saturating_sub(count);
    }

    /// Insert terminal paste or IME-committed text into the type-ahead query.
    pub fn paste_text(&mut self, text: &str) {
        self.query
            .extend(text.chars().filter(|ch| !ch.is_control()));
        self.refilter();
    }

    /// Handle a single key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
                None
            }
            KeyCode::PageUp => {
                self.page_up();
                None
            }
            KeyCode::PageDown => {
                self.page_down();
                None
            }
            KeyCode::Home => {
                self.move_home();
                None
            }
            KeyCode::End => {
                self.move_end();
                None
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.refilter();
                None
            }
            // Type-ahead. A query that matches nothing leaves the list empty,
            // and Enter there cancels rather than guessing a row.
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.query.push(ch);
                self.refilter();
                None
            }
            KeyCode::Enter => {
                let Some(index) = self.selected_row_index() else {
                    return Some(ModalResult::Cancelled);
                };
                Some(ModalResult::Selected(ModalSelection::Choice {
                    index,
                    label: self.rows[index].label.clone(),
                }))
            }
            _ => None,
        }
    }

    /// Number of distinct groups among the visible rows.
    fn visible_group_count(&self) -> usize {
        let mut seen: Vec<&str> = Vec::new();
        for &index in &self.filtered {
            if let Some(group) = self.rows[index].group.as_deref() {
                if !seen.contains(&group) {
                    seen.push(group);
                }
            }
        }
        seen.len()
    }

    /// Display width to pad labels to, so descriptions start on one column.
    fn label_column(&self) -> usize {
        self.filtered
            .iter()
            .filter(|&&index| self.rows[index].description.is_some())
            .map(|&index| UnicodeWidthStr::width(self.rows[index].label.as_str()))
            .max()
            .unwrap_or(0)
    }

    /// Body rows tagged with the filtered position they belong to (`None` for
    /// chrome such as group headers), so the viewport can keep the selected row
    /// on screen and the wash can find it.
    fn body_rows<'a>(&'a self, theme: &Theme) -> Vec<(Option<usize>, Line<'a>)> {
        let mut lines: Vec<(Option<usize>, Line<'a>)> = Vec::new();
        if !self.query.is_empty() {
            lines.push((
                None,
                Line::from(Span::styled(
                    format!("filter: {}", self.query),
                    row_detail_style(theme, false),
                )),
            ));
        }
        if self.filtered.is_empty() {
            lines.push((
                None,
                Line::from(Span::styled(
                    if self.rows.is_empty() {
                        "no options".to_string()
                    } else {
                        "no match — Backspace to widen".to_string()
                    },
                    row_detail_style(theme, false),
                )),
            ));
            return lines;
        }

        let show_groups = self.visible_group_count() > 1;
        let label_column = self.label_column();
        let mut last_group: Option<&str> = None;
        for (position, &index) in self.filtered.iter().enumerate() {
            let row = &self.rows[index];
            let group = row.group.as_deref();
            if show_groups && group.is_some() && group != last_group {
                lines.push((
                    None,
                    Line::from(Span::styled(
                        group.unwrap_or_default(),
                        row_detail_style(theme, false),
                    )),
                ));
            }
            last_group = group;

            let selected = position == self.cursor;
            let marker = if selected {
                cursor_marker(!theme.no_color)
            } else {
                blank_marker()
            };
            let label_style = if selected {
                selected_style(theme)
            } else {
                theme.typography.body
            };
            let mut spans = vec![
                Span::styled(marker, label_style),
                Span::styled(row.label.as_str(), label_style),
            ];
            if let Some(description) = row.description.as_deref() {
                let pad = label_column
                    .saturating_sub(UnicodeWidthStr::width(row.label.as_str()))
                    .saturating_add(2);
                spans.push(Span::styled(" ".repeat(pad), label_style));
                spans.push(Span::styled(description, row_detail_style(theme, selected)));
            }
            if let Some(badge) = row.badge {
                spans.push(Span::styled("  ", label_style));
                spans.push(Span::styled(
                    format!("[{}]", badge.label()),
                    row_detail_style(theme, selected),
                ));
            }
            lines.push((Some(position), Line::from(spans)));
        }
        lines
    }

    /// Rows the modal wants, including the blank spacer and key-hint footer.
    #[must_use]
    pub fn visual_rows(&self) -> usize {
        self.body_rows(&Theme::no_color()).len() + 2
    }

    /// Build the rendered lines used by both [`Self::draw`] and tests.
    ///
    /// `width` is the content width the rows will be painted into; the key-hint
    /// footer is fitted to it because `draw` renders without `Wrap` and would
    /// otherwise have the row cut mid-word at the rect edge.
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme, width: u16) -> Vec<Line<'a>> {
        let mut lines: Vec<Line<'a>> = self
            .body_rows(theme)
            .into_iter()
            .map(|(_, line)| line)
            .collect();
        lines.push(Line::from(""));
        lines.push(self.footer(theme, width));
        lines
    }

    /// Key hints. The filter hint appears only once a query is active, so an
    /// untouched list keeps its original three-hint footer.
    fn footer(&self, theme: &Theme, width: u16) -> Line<'static> {
        if self.query.is_empty() {
            key_hint_footer_fitted(
                theme,
                &[("↑↓", "move"), ("Enter", "confirm"), ("Esc", "cancel")],
                width,
            )
        } else {
            key_hint_footer_fitted(
                theme,
                &[
                    ("↑↓", "move"),
                    ("Enter", "confirm"),
                    ("Backspace", "widen"),
                    ("Esc", "cancel"),
                ],
                width,
            )
        }
    }

    /// [`Self::render_lines`] windowed to `height` rows with the selected row
    /// kept on screen.
    ///
    /// Without this the body was painted from its first row and anything past
    /// the pane bottom simply vanished — with descriptions and group headers a
    /// `/connect` list is tall enough for that to hide real options.
    #[must_use]
    fn render_lines_fitted<'a>(&'a self, theme: &Theme, width: u16, height: u16) -> Vec<Line<'a>> {
        let footer_rows = 2usize;
        let body = self.body_rows(theme);
        let budget = usize::from(height).saturating_sub(footer_rows).max(1);
        let mut lines: Vec<Line<'a>> = if body.len() <= budget {
            body.into_iter().map(|(_, line)| line).collect()
        } else {
            let cursor_line = body
                .iter()
                .position(|(position, _)| *position == Some(self.cursor))
                .unwrap_or(0);
            let start = (cursor_line + 1).saturating_sub(budget).min(body.len() - budget);
            body.into_iter()
                .skip(start)
                .take(budget)
                .map(|(_, line)| line)
                .collect()
        };
        lines.push(Line::from(""));
        lines.push(self.footer(theme, width));
        lines
    }

    /// Draw the modal into `area` using `theme`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = super::modal_frame(frame, area, self.title.clone(), theme);
        // Carry the cursor row's wash out to the column edge (Pi paints
        // selection as an edge-to-edge band, not a highlight round the label).
        let cursor_marker = cursor_marker(!theme.no_color);
        let lines = self
            .render_lines_fitted(theme, inner.width, inner.height)
            .into_iter()
            .map(|line| {
                let is_cursor_row = line
                    .spans
                    .first()
                    .is_some_and(|span| span.content == cursor_marker);
                if is_cursor_row {
                    super::wash_row(line, inner.width, theme)
                } else {
                    line
                }
            })
            .collect::<Vec<_>>();
        // Fitted: this body renders without `wrap`, so an over-wide row would be
        // cut mid-glyph by `LineTruncator` instead of losing its tail visibly.
        let paragraph =
            Paragraph::new(super::fit_body_rows(lines, inner.width)).style(theme.typography.body);
        frame.render_widget(paragraph, inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEventState, KeyModifiers};

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn modal_with(count: usize) -> ChoicePickerModal {
        let options: Vec<String> = (0..count).map(|i| format!("option {i}")).collect();
        ChoicePickerModal::new("pick", options)
    }

    #[test]
    fn page_down_advances_by_a_page_and_clamps() {
        let mut modal = modal_with(20);
        assert_eq!(modal.cursor(), 0);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), PAGE_STRIDE);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), PAGE_STRIDE * 2);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), 19, "PageDown clamps at the last option");
    }

    #[test]
    fn home_and_end_jump_to_bounds() {
        let mut modal = modal_with(10);
        modal.handle_key(press(KeyCode::End));
        assert_eq!(modal.cursor(), 9);
        modal.handle_key(press(KeyCode::PageUp));
        assert_eq!(modal.cursor(), 9 - PAGE_STRIDE);
        modal.handle_key(press(KeyCode::Home));
        assert_eq!(modal.cursor(), 0);
    }

    /// Flatten the modal's rendered rows into one string for glyph inspection.
    fn rendered_text(modal: &ChoicePickerModal, theme: &Theme) -> String {
        modal
            .render_lines(theme, 80)
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect()
    }

    /// Normal/color mode keeps Pi's Unicode selection arrow `→`; the ASCII
    /// fallback never leaks into the rich render.
    #[test]
    fn selection_cursor_keeps_unicode_in_color_mode() {
        let modal = modal_with(3);
        let text = rendered_text(&modal, &Theme::zo());
        assert!(
            text.contains('\u{2192}'),
            "color mode keeps the → cursor: {text:?}"
        );
    }

    /// Plain/`NO_COLOR` mode swaps the arrow for a one-cell ASCII `>` and
    /// never emits the Unicode arrow.
    #[test]
    fn selection_cursor_uses_ascii_fallback_under_no_color() {
        let modal = modal_with(3);
        let text = rendered_text(&modal, &Theme::no_color());
        assert!(text.contains('>'), "plain cursor is one-cell '>': {text:?}");
        assert!(
            !text.contains('\u{2192}'),
            "no Unicode arrow under NO_COLOR: {text:?}"
        );
    }

    fn provider_rows() -> Vec<ChoiceRow> {
        vec![
            ChoiceRow::new("Claude")
                .describe("Anthropic OAuth")
                .with_badge(ChoiceBadge::Saved)
                .in_group("Sign in"),
            ChoiceRow::new("ChatGPT")
                .describe("OpenAI subscription")
                .in_group("Sign in"),
            ChoiceRow::new("DeepSeek")
                .describe("cloud models")
                .with_badge(ChoiceBadge::NeedsKey)
                .in_group("API key"),
            ChoiceRow::new("Ollama")
                .describe("auto-discovered")
                .with_badge(ChoiceBadge::Local)
                .in_group("On this machine"),
        ]
    }

    fn type_query(modal: &mut ChoicePickerModal, text: &str) -> Option<ModalResult> {
        let mut last = None;
        for ch in text.chars() {
            last = modal.handle_key(press(KeyCode::Char(ch)));
        }
        last
    }

    /// The host indexes `login_provider_ids` / `session_ids` with the returned
    /// index, so a filtered list must still report the ORIGINAL position —
    /// otherwise picking the only visible row opens a different provider.
    #[test]
    fn a_filtered_selection_reports_the_original_index() {
        let mut modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        type_query(&mut modal, "ollama");
        assert_eq!(modal.len(), 1, "only one row survives the query");

        let result = modal.handle_key(press(KeyCode::Enter));
        match result {
            Some(ModalResult::Selected(ModalSelection::Choice { index, label })) => {
                assert_eq!(index, 3, "index addresses the unfiltered registry");
                assert_eq!(label, "Ollama", "label is the row's own text");
            }
            other => panic!("expected a selection, got {other:?}"),
        }
    }

    /// A query that matches nothing must cancel rather than confirm whatever
    /// row the cursor happens to sit on.
    #[test]
    fn enter_on_an_empty_filter_cancels_instead_of_guessing() {
        let mut modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        type_query(&mut modal, "zzzz");
        assert!(modal.is_empty());
        assert!(matches!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(ModalResult::Cancelled)
        ));
    }

    #[test]
    fn backspace_widens_the_filter() {
        let mut modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        type_query(&mut modal, "ollama");
        assert_eq!(modal.len(), 1);
        for _ in 0.."ollama".len() {
            modal.handle_key(press(KeyCode::Backspace));
        }
        assert_eq!(modal.len(), 4, "clearing the query restores every row");
        assert!(modal.query().is_empty());
    }

    /// Descriptions, badges and section headers are the point of the structured
    /// row — they must actually reach the render.
    #[test]
    fn rows_render_description_badge_and_group_headers() {
        let modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        let text = rendered_text(&modal, &Theme::zo());
        assert!(text.contains("Anthropic OAuth"), "description: {text:?}");
        assert!(text.contains("[saved]"), "badge: {text:?}");
        assert!(text.contains("[needs key]"), "badge: {text:?}");
        assert!(text.contains("Sign in"), "group header: {text:?}");
        assert!(text.contains("On this machine"), "group header: {text:?}");
    }

    /// A single-group list stays a plain list — headers appear only when they
    /// actually separate something.
    #[test]
    fn a_single_group_renders_no_header() {
        let rows = vec![
            ChoiceRow::new("one").in_group("Only"),
            ChoiceRow::new("two").in_group("Only"),
        ];
        let modal = ChoicePickerModal::with_rows("pick", rows);
        let text = rendered_text(&modal, &Theme::zo());
        assert!(!text.contains("Only"), "no lone header: {text:?}");
    }

    /// Badges are ASCII, so the plain render carries no Unicode decoration.
    #[test]
    fn badges_stay_ascii_under_no_color() {
        let modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        let text = rendered_text(&modal, &Theme::no_color());
        assert!(text.contains("[saved]"), "badge still shown: {text:?}");
        assert!(
            text.is_ascii() || !text.contains('\u{2192}'),
            "no Unicode cursor under NO_COLOR: {text:?}"
        );
    }

    /// Height is budgeted from the rows actually painted — descriptions ride
    /// the label row, but group headers and the filter line are extra rows, and
    /// sizing from the option count alone would clip them.
    #[test]
    fn visual_rows_counts_headers_and_the_filter_line() {
        let mut modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        // 4 options + 3 group headers + blank + footer.
        assert_eq!(modal.visual_rows(), 9);
        type_query(&mut modal, "o");
        assert!(
            modal.visual_rows() > modal.len() + 2,
            "an active filter adds its own row"
        );
    }

    /// A body taller than the pane scrolls with the selection instead of
    /// painting from row zero and dropping the rest.
    #[test]
    fn a_cramped_pane_keeps_the_selection_visible() {
        let mut modal = modal_with(20);
        modal.handle_key(press(KeyCode::End));
        let theme = Theme::zo();
        let lines = modal.render_lines_fitted(&theme, 40, 8);
        assert!(lines.len() <= 8, "never spends more rows than granted");
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(
            text.contains("option 19"),
            "the selected row stays on screen: {text:?}"
        );
    }

    /// Paint the real widget and read the terminal cells back, so the column
    /// layout is checked as the user sees it rather than as a span list.
    #[test]
    fn a_painted_provider_list_aligns_its_columns() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let modal = ChoicePickerModal::with_rows("Connect", provider_rows());
        let theme = Theme::zo();
        let (width, height) = (52u16, 14u16);
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|frame| modal.draw(frame, Rect::new(0, 0, width, height), &theme))
            .unwrap();
        let buf = term.backend().buffer();
        let rows: Vec<String> = (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buf.cell((x, y)).unwrap().symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        println!("{}", rows.join("\n"));

        let claude = rows
            .iter()
            .find(|row| row.contains("Claude"))
            .expect("Claude row painted");
        let chatgpt = rows
            .iter()
            .find(|row| row.contains("ChatGPT"))
            .expect("ChatGPT row painted");
        // Descriptions start on one column even though the labels differ in
        // width — the whole point of dropping the hand-padded label strings.
        // Measured in display cells, not bytes: the cursor arrow is one column
        // but three bytes, so byte offsets would disagree on identical layout.
        let description_column = |row: &str, needle: &str| {
            let byte = row.find(needle).expect("description painted");
            UnicodeWidthStr::width(&row[..byte])
        };
        assert_eq!(
            description_column(claude, "Anthropic OAuth"),
            description_column(chatgpt, "OpenAI subscription"),
            "description column must line up:\n{claude}\n{chatgpt}"
        );
        assert!(
            claude.trim_end().ends_with("[saved]"),
            "the badge closes the row: {claude}"
        );
    }

    /// A section is one contiguous block with one heading. A caller that
    /// appends a row belonging to an earlier section — easy to do, since the
    /// rows and their parallel id list are built by pushing — must not reopen
    /// that section further down: the same heading printed twice reads as two
    /// different sections, and the reader cannot tell which is which.
    #[test]
    fn a_section_is_rendered_once_even_when_its_rows_are_appended_late() {
        let rows = vec![
            ChoiceRow::new("alpha").in_group("First"),
            ChoiceRow::new("beta").in_group("Second"),
            ChoiceRow::new("gamma").in_group("First"),
        ];
        let modal = ChoicePickerModal::with_rows("pick", rows);
        let rendered: Vec<String> = modal
            .render_lines(&Theme::zo(), 60)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect();

        assert_eq!(
            rendered.iter().filter(|row| row.trim() == "First").count(),
            1,
            "one heading per section: {rendered:?}"
        );
        // The late row joins its own section rather than being stranded under
        // whichever heading happened to precede it.
        let first = rendered
            .iter()
            .position(|row| row.trim() == "First")
            .expect("First heading");
        let second = rendered
            .iter()
            .position(|row| row.trim() == "Second")
            .expect("Second heading");
        let gamma = rendered
            .iter()
            .position(|row| row.contains("gamma"))
            .expect("gamma row");
        assert!(
            first < gamma && gamma < second,
            "gamma sits under First: {rendered:?}"
        );
    }

    /// Ctrl-chords are host shortcuts, not filter input.
    #[test]
    fn control_chords_do_not_enter_the_filter() {
        let mut modal = modal_with(3);
        modal.handle_key(KeyEvent {
            code: KeyCode::Char('c'),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });
        assert!(modal.query().is_empty());
        assert_eq!(modal.len(), 3);
    }
}
