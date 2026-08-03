//! In-app modal for `AskUserQuestion` prompts.
//!
//! Claude-Code-parity surface: an optional topic chip in the title, options
//! with one-line dim descriptions, and an always-available free-form row —
//! the model's options are suggestions, never a cage.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use super::super::cards::{CardFrame, SurfaceKind};
use runtime::message_stream::UserQuestionPrompt;
use unicode_width::UnicodeWidthStr;

use super::super::theme::Theme;
use super::{
    FooterSegment, ModalResult, ModalSelection, blank_marker, cursor_marker,
    key_hint_footer_reflowing, key_hint_footer_with_separator, modal_footer, selected_style,
};

/// Narrowest inner width that still fits an option list and a preview pane
/// side by side. Below it the modal keeps its single column and stacks the
/// preview under the list — a squeezed two-column layout would clip the mockup
/// it exists to show.
const MIN_SPLIT_WIDTH: u16 = 72;

/// Columns reserved for the option list when the layout splits.
const OPTIONS_PANE_WIDTH: u16 = 38;

/// One decoded option row.
#[derive(Debug, Clone)]
struct OptionRow {
    label: String,
    description: Option<String>,
    /// Monospace mockup/snippet shown beside the list while this row is
    /// focused. Rendered verbatim — never wrapped — because the artifacts it
    /// carries (ASCII layouts, code) are column-significant.
    preview: Option<String>,
}

/// Modal state for a blocking user question.
#[derive(Debug, Clone)]
pub struct UserQuestionModal {
    question: String,
    header: Option<String>,
    options: Vec<OptionRow>,
    /// Cursor over `options.len() + 1` rows — the last row is the free-form
    /// "Other" entry (when options exist; with no options the modal is
    /// free-form only).
    cursor: usize,
    answer: String,
    /// When `true`, options render as `[x]`/`[ ]` checkboxes and Space toggles
    /// them; the user confirms several at once. When `false` (the default) the
    /// modal is a single-select radio and Enter returns the highlighted row.
    multi_select: bool,
    /// Per-option checked state, parallel to `options`. Only consulted in
    /// multi-select mode; a single-select prompt leaves it all-false.
    checked: Vec<bool>,
}

impl UserQuestionModal {
    /// Construct a modal from a render-block prompt.
    #[must_use]
    pub fn from_prompt(prompt: &UserQuestionPrompt) -> Self {
        let options: Vec<OptionRow> = prompt
            .options
            .iter()
            .map(|opt| OptionRow {
                label: decode_unicode_escapes(&opt.label),
                description: opt
                    .description
                    .as_deref()
                    .map(decode_unicode_escapes)
                    .filter(|d| !d.trim().is_empty()),
                preview: opt
                    .preview
                    .as_deref()
                    .map(decode_unicode_escapes)
                    .filter(|preview| !preview.trim().is_empty()),
            })
            .collect();
        let checked = vec![false; options.len()];
        Self {
            question: decode_unicode_escapes(&prompt.question),
            header: prompt
                .header
                .as_deref()
                .map(decode_unicode_escapes)
                .filter(|h| !h.trim().is_empty()),
            options,
            cursor: 0,
            answer: String::new(),
            // Multi-select only makes sense with a fixed choice list; a
            // free-form-only prompt stays single-answer.
            multi_select: prompt.multi_select && !prompt.options.is_empty(),
            checked,
        }
    }

    /// Insert clipboard or IME-committed text into the free-form answer.
    pub fn paste_text(&mut self, text: &str) {
        self.cursor = self.options.len();
        self.answer
            .extend(text.chars().filter(|ch| !ch.is_control()));
    }

    /// Number of fixed options.
    #[must_use]
    pub fn len(&self) -> usize {
        self.options.len()
    }

    /// Returns `true` if there are no fixed options.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// Total selectable rows: every option plus the trailing free-form row.
    fn row_count(&self) -> usize {
        if self.options.is_empty() {
            0
        } else {
            self.options.len() + 1
        }
    }

    /// Number of display rows the modal content needs at `inner_width`
    /// (the area inside the borders), counting soft-wrap. The caller adds
    /// the 2 border rows. Descriptions and the free-form row are real rows —
    /// sizing from `len()` alone clips them.
    #[must_use]
    pub fn desired_rows(&self, theme: &Theme, inner_width: u16) -> u16 {
        let w = usize::from(inner_width.max(1));
        let rows: usize = self
            .render_lines(theme)
            .iter()
            .map(|line| {
                let cells: usize = line
                    .spans
                    .iter()
                    .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                    .sum();
                cells.div_ceil(w).max(1)
            })
            .sum();
        u16::try_from(rows).unwrap_or(u16::MAX)
    }

    /// `true` while the cursor rests on the free-form row.
    fn on_freeform_row(&self) -> bool {
        !self.options.is_empty() && self.cursor == self.options.len()
    }

    fn move_down(&mut self) {
        let rows = self.row_count();
        if rows == 0 {
            return;
        }
        self.cursor = (self.cursor + 1) % rows;
    }

    fn move_up(&mut self) {
        let rows = self.row_count();
        if rows == 0 {
            return;
        }
        self.cursor = self.cursor.checked_sub(1).unwrap_or(rows - 1);
    }

    fn selected_answer(&self) -> Option<String> {
        if self.options.is_empty() || self.on_freeform_row() {
            let typed = self.answer.trim();
            if typed.is_empty() {
                None
            } else {
                Some(typed.to_string())
            }
        } else {
            self.options.get(self.cursor).map(|opt| opt.label.clone())
        }
    }

    /// Handle one key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        // Multi-select is a distinct interaction (Space toggles, Enter confirms
        // the whole set); route it separately so the single-select path below
        // stays byte-for-byte unchanged.
        if self.multi_select {
            return self.handle_key_multi(key);
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Enter => self.selected_answer().map(|answer| {
                ModalResult::Selected(ModalSelection::QuestionAnswer(vec![answer]))
            }),
            KeyCode::Up if !self.options.is_empty() => {
                self.move_up();
                None
            }
            KeyCode::Down if !self.options.is_empty() => {
                self.move_down();
                None
            }
            KeyCode::Backspace => {
                if self.options.is_empty() || self.on_freeform_row() {
                    self.answer.pop();
                }
                None
            }
            KeyCode::Char(ch) if self.options.is_empty() => {
                self.answer.push(ch);
                None
            }
            KeyCode::Char(ch) => {
                // Digits pick an option directly; the free-form row's own
                // number only moves the cursor there (it needs typed text).
                if !self.on_freeform_row() || self.answer.is_empty() {
                    if let Some(index) = ch
                        .to_digit(10)
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|n| (1..=self.options.len()).contains(n))
                    {
                        let answer = self.options[index - 1].label.clone();
                        return Some(ModalResult::Selected(ModalSelection::QuestionAnswer(
                            vec![answer],
                        )));
                    }
                    if ch
                        .to_digit(10)
                        .and_then(|n| usize::try_from(n).ok())
                        .is_some_and(|n| n == self.options.len() + 1)
                    {
                        self.cursor = self.options.len();
                        return None;
                    }
                }
                // Any other character jumps to the free-form row and starts
                // typing — the Claude Code "Other" reflex.
                self.cursor = self.options.len();
                self.answer.push(ch);
                None
            }
            _ => None,
        }
    }

    /// Multi-select key handling: Space (or a digit) toggles the option under
    /// the cursor, Enter confirms every checked label plus any typed free-form
    /// text, and typing a non-digit jumps to the free-form row — mirroring the
    /// single-select "Other" reflex. Enter with nothing chosen is a no-op, so
    /// the user cannot confirm an empty set by reflex.
    fn handle_key_multi(&mut self, key: KeyEvent) -> Option<ModalResult> {
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Enter => {
                let answers = self.collect_multi_answers();
                if answers.is_empty() {
                    None
                } else {
                    Some(ModalResult::Selected(ModalSelection::QuestionAnswer(
                        answers,
                    )))
                }
            }
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
                None
            }
            // Space toggles the checkbox on an option row. On the free-form row
            // it is ordinary typed text, so fall through to the char handler.
            KeyCode::Char(' ') if !self.on_freeform_row() => {
                self.toggle_current();
                None
            }
            KeyCode::Backspace => {
                if self.on_freeform_row() {
                    self.answer.pop();
                }
                None
            }
            KeyCode::Char(ch) => {
                // A digit toggles the matching option (moving the cursor there);
                // the free-form row's own number just parks the cursor on it.
                if !self.on_freeform_row() || self.answer.is_empty() {
                    if let Some(index) = ch
                        .to_digit(10)
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|n| (1..=self.options.len()).contains(n))
                    {
                        self.cursor = index - 1;
                        self.toggle_current();
                        return None;
                    }
                    if ch
                        .to_digit(10)
                        .and_then(|n| usize::try_from(n).ok())
                        .is_some_and(|n| n == self.options.len() + 1)
                    {
                        self.cursor = self.options.len();
                        return None;
                    }
                }
                // Any other character jumps to the free-form row and types.
                self.cursor = self.options.len();
                self.answer.push(ch);
                None
            }
            _ => None,
        }
    }

    /// How many options are currently checked (multi-select only).
    fn checked_count(&self) -> usize {
        self.checked.iter().filter(|checked| **checked).count()
    }

    /// Flip the checkbox for the option under the cursor (multi-select only).
    fn toggle_current(&mut self) {
        if let Some(slot) = self.checked.get_mut(self.cursor) {
            *slot = !*slot;
        }
    }

    /// Collect every checked option label, in display order, plus any typed
    /// free-form text as a trailing answer.
    fn collect_multi_answers(&self) -> Vec<String> {
        let mut answers: Vec<String> = self
            .options
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.checked.get(*idx).copied().unwrap_or(false))
            .map(|(_, opt)| opt.label.clone())
            .collect();
        let typed = self.answer.trim();
        if !typed.is_empty() {
            answers.push(typed.to_string());
        }
        answers
    }

    /// Title line: `◆ Question` plus the optional dim topic chip.
    fn title_line(&self, theme: &Theme) -> Line<'static> {
        let mut spans = vec![Span::styled(
            " \u{25c6} Question ".to_string(),
            theme.typography.heading_1,
        )];
        if let Some(header) = &self.header {
            spans.push(Span::styled(
                format!("\u{00b7} {header} "),
                theme.typography.dim,
            ));
        }
        Line::from(spans)
    }

    /// Key hints for an options-backed question.
    fn options_footer(&self, theme: &Theme) -> Line<'static> {
        // The split layout gives the list a narrow column, where the full hint
        // row folds onto two lines. Drop the digit-shortcut hint there (the
        // shortcut still works) so the footer stays one clean line.
        if self.has_previews() && !self.multi_select {
            return key_hint_footer_with_separator(
                theme,
                &[("↑↓", "move"), ("Enter", "select"), ("Esc", "cancel")],
                " · ",
            );
        }
        if self.multi_select {
            // Enter with nothing checked is a no-op, so the tally is the only
            // thing telling the reader why the modal is not closing — and with
            // the list windowed, checked rows can sit off-screen. It is a
            // status label, not a key hint, so it takes the dim `Label` segment
            // rather than being faked as a hint with an empty action.
            let checked = format!("{} selected", self.checked_count());
            modal_footer(
                theme,
                &[
                    FooterSegment::hint("↑↓", "move"),
                    FooterSegment::hint("Space", "toggle"),
                    FooterSegment::hint("Enter", "confirm"),
                    FooterSegment::hint("Esc", "cancel"),
                    FooterSegment::label(checked.as_str()),
                ],
                " · ",
            )
        } else {
            let pick = format!("1–{}", self.options.len() + 1);
            key_hint_footer_reflowing(
                theme,
                &[
                    ("↑↓", "move"),
                    (pick.as_str(), "select"),
                    ("Enter", "confirm"),
                    ("Esc", "cancel"),
                ],
            )
        }
    }

    /// Build rendered lines for tests and drawing.
    ///
    /// Layout: a blank lead-in, the question, a gap, then the option rows —
    /// `❯ N. label` with a dim description line under each — the trailing
    /// free-form row, and the shared key-hint footer.
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let mut lines = Vec::new();
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            self.question.clone(),
            theme.typography.body,
        )));
        lines.push(Line::from(""));

        if self.options.is_empty() {
            let marker = cursor_marker(!theme.no_color);
            let shown = if self.answer.is_empty() {
                marker.to_string()
            } else {
                format!("{marker}{}", self.answer)
            };
            lines.push(Line::from(Span::styled(shown, selected_style(theme))));
            lines.push(Line::from(""));
            lines.push(key_hint_footer_reflowing(
                theme,
                &[("Enter", "confirm"), ("Esc", "cancel")],
            ));
            return lines;
        }

        lines.extend(self.option_region(theme).into_iter().map(|(_, line)| line));
        lines.push(Line::from(""));
        lines.push(self.options_footer(theme));

        lines
    }

    /// The scrollable part of the modal: one entry per rendered line, tagged
    /// with the cursor row it belongs to (`options.len()` is the free-form row).
    ///
    /// Tagging is what lets the viewport window keep a selected option *and its
    /// description* together — a window that split them would show a dangling
    /// description under an option the user cannot see.
    fn option_region<'a>(&'a self, theme: &Theme) -> Vec<(usize, Line<'a>)> {
        let mut lines: Vec<(usize, Line<'a>)> = Vec::with_capacity(self.options.len() * 2 + 1);
        for (idx, option) in self.options.iter().enumerate() {
            let selected = idx == self.cursor;
            let marker = if selected {
                cursor_marker(!theme.no_color)
            } else {
                blank_marker()
            };
            let style = if selected {
                selected_style(theme)
            } else {
                theme.typography.body
            };
            // Multi-select prefixes each option with a `[x]`/`[ ]` checkbox;
            // single-select shows nothing extra so its rows are unchanged.
            let checkbox = if self.multi_select {
                if self.checked.get(idx).copied().unwrap_or(false) {
                    "[x] "
                } else {
                    "[ ] "
                }
            } else {
                ""
            };
            lines.push((
                idx,
                Line::from(Span::styled(
                    format!("{marker}{checkbox}{}. {}", idx + 1, option.label),
                    style,
                )),
            ));
            if let Some(description) = &option.description {
                // Description column: marker + optional checkbox + "N. " deep.
                let indent = " ".repeat(blank_marker().len() + checkbox.len() + 3);
                lines.push((
                    idx,
                    Line::from(Span::styled(
                        format!("{indent}{description}"),
                        theme.typography.dim,
                    )),
                ));
            }
        }

        // Trailing free-form row — always available, so the model's options
        // never cage the user.
        let freeform_selected = self.on_freeform_row();
        let marker = if freeform_selected {
            cursor_marker(!theme.no_color)
        } else {
            blank_marker()
        };
        let style = if freeform_selected {
            selected_style(theme)
        } else {
            theme.typography.dim
        };
        // In multi-select, pad where a checkbox would sit so the row number
        // aligns with the option rows above it.
        let gap = if self.multi_select { "    " } else { "" };
        let freeform = if freeform_selected && !self.answer.is_empty() {
            format!(
                "{marker}{gap}{}. Other: {}",
                self.options.len() + 1,
                self.answer
            )
        } else {
            format!("{marker}{gap}{}. Other…", self.options.len() + 1)
        };
        lines.push((self.options.len(), Line::from(Span::styled(freeform, style))));
        lines
    }

    /// Rows `line` occupies once soft-wrapped into `width` cells — the exact
    /// arithmetic [`Self::desired_rows`] sums, so what the modal measures is
    /// what it paints.
    fn wrapped_rows(line: &Line<'_>, width: u16) -> u16 {
        let width = usize::from(width.max(1));
        let cells: usize = line
            .spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum();
        u16::try_from(cells.div_ceil(width).max(1)).unwrap_or(u16::MAX)
    }

    /// The option region windowed to `budget` rows around the cursor, with a
    /// dim `↑/↓ N more` marker for whatever the window hides.
    ///
    /// Without this the list was simply painted from its first row and anything
    /// past the pane bottom vanished — a four-option question showed two, with
    /// no scrollbar, no marker, and no way to tell the rest existed.
    fn windowed_option_lines<'a>(
        &'a self,
        theme: &Theme,
        width: u16,
        budget: u16,
    ) -> Vec<Line<'a>> {
        let tagged = self.option_region(theme);
        if budget == 0 {
            return Vec::new();
        }
        let rows = |line: &Line<'_>| Self::wrapped_rows(line, width);
        let total: u16 = tagged
            .iter()
            .map(|(_, line)| rows(line))
            .fold(0u16, u16::saturating_add);
        if total <= budget {
            return tagged.into_iter().map(|(_, line)| line).collect();
        }

        // Two rows are held back for the markers. Spending both up front (even
        // when only one end ends up hidden) keeps the window arithmetic single
        // pass — one unused row in a modal beats a second measuring round.
        let window_budget = budget.saturating_sub(2).max(1);
        // The selection is the invariant: its rows go in first, then context
        // grows outward until the budget is spent.
        let selected_first = tagged
            .iter()
            .position(|(row, _)| *row == self.cursor)
            .unwrap_or(0);
        let selected_last = tagged
            .iter()
            .rposition(|(row, _)| *row == self.cursor)
            .unwrap_or(selected_first);
        let mut used: u16 = tagged[selected_first..=selected_last]
            .iter()
            .map(|(_, line)| rows(line))
            .fold(0u16, u16::saturating_add);
        let (mut start, mut end) = (selected_first, selected_last);
        loop {
            let mut grew = false;
            if start > 0 {
                let cost = rows(&tagged[start - 1].1);
                if used.saturating_add(cost) <= window_budget {
                    used += cost;
                    start -= 1;
                    grew = true;
                }
            }
            if end + 1 < tagged.len() {
                let cost = rows(&tagged[end + 1].1);
                if used.saturating_add(cost) <= window_budget {
                    used += cost;
                    end += 1;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }

        // Count hidden *options*, not hidden lines: "2 more" must mean two more
        // things to choose, not two more rows of text.
        let distinct = |slice: &[(usize, Line<'a>)]| {
            slice
                .iter()
                .map(|(row, _)| *row)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        let hidden_above = distinct(&tagged[..start]);
        let hidden_below = distinct(&tagged[end + 1..]);
        let dim = theme.typography.dim;
        let mut out: Vec<Line<'a>> = Vec::with_capacity(end - start + 3);
        if hidden_above > 0 {
            out.push(Line::from(Span::styled(
                format!("  \u{2191} {hidden_above} more"),
                dim,
            )));
        }
        out.extend(
            tagged
                .into_iter()
                .skip(start)
                .take(end + 1 - start)
                .map(|(_, line)| line),
        );
        if hidden_below > 0 {
            out.push(Line::from(Span::styled(
                format!("  \u{2193} {hidden_below} more"),
                dim,
            )));
        }

        // Final guard. The selected option goes into the window whatever it
        // costs — it must be visible — so a tall wrapped option in a cramped
        // pane can still push the total over. Trim from the end rather than let
        // the `Paragraph` clip it, because the paragraph's clip is exactly the
        // silent disappearance this window exists to prevent. One line always
        // survives, so the selection never vanishes entirely.
        let mut total = 0u16;
        let mut keep = 0usize;
        for line in &out {
            let cost = rows(line);
            if keep > 0 && total.saturating_add(cost) > budget {
                break;
            }
            total = total.saturating_add(cost);
            keep += 1;
        }
        out.truncate(keep);
        out
    }

    /// [`Self::render_lines`] fitted to a pane `height` rows tall: the question
    /// and the key hints stay pinned, and only the option region scrolls.
    #[must_use]
    pub fn render_lines_fitted<'a>(
        &'a self,
        theme: &Theme,
        width: u16,
        height: u16,
    ) -> Vec<Line<'a>> {
        if self.options.is_empty() {
            return self.render_lines(theme);
        }
        let head = vec![
            Line::from(""),
            Line::from(Span::styled(self.question.clone(), theme.typography.body)),
            Line::from(""),
        ];
        let foot = vec![Line::from(""), self.options_footer(theme)];
        let chrome: u16 = head
            .iter()
            .chain(foot.iter())
            .map(|line| Self::wrapped_rows(line, width))
            .fold(0u16, u16::saturating_add);
        let budget = height.saturating_sub(chrome);
        let mut lines = head;
        lines.extend(self.windowed_option_lines(theme, width, budget));
        lines.extend(foot);
        lines
    }

    /// Width the option list is actually painted into: its own column when the
    /// side-by-side layout is active, the whole dialog otherwise. Measuring at
    /// the full modal width while painting into a 38-column column is what let
    /// wrapped Korean labels push later options off the bottom unseen.
    fn list_paint_width(&self, inner_width: u16) -> u16 {
        if self.has_previews() && inner_width >= MIN_SPLIT_WIDTH {
            OPTIONS_PANE_WIDTH
        } else {
            inner_width
        }
    }

    /// Size the dialog wants inside `area`.
    ///
    /// A Pi dialog's inner rect is the *full* width of its area (only the two
    /// rules cost rows), so rows are measured at the paint width — not at
    /// `width - 2`, which is the side-border arithmetic this frame no longer
    /// has.
    #[must_use]
    pub fn desired_size(&self, area: Rect, theme: &Theme) -> (u16, u16) {
        let (min_width, max_width) = if self.has_previews() {
            (36, self.split_width_request())
        } else {
            (36, 84)
        };
        let width = area
            .width
            .clamp(min_width, max_width)
            .min(area.width.saturating_sub(4).max(24));
        let rows = self
            .desired_rows(theme, self.list_paint_width(width))
            .saturating_add(2)
            .max(self.preview_rows_request());
        // Grow into whatever the terminal has. The old fixed 30-row ceiling
        // clipped a tall question on a roomy screen, which is precisely when
        // there was space to show all of it.
        let height = rows
            .max(6)
            .min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    /// Whether any option carries a preview, which is what turns the modal into
    /// the side-by-side compare layout.
    #[must_use]
    pub fn has_previews(&self) -> bool {
        self.options
            .iter()
            .any(|option| option.preview.is_some())
    }

    /// The focused option's preview, if it has one. The free-form row has none,
    /// so the pane empties rather than showing a stale neighbour's mockup.
    fn focused_preview(&self) -> Option<&str> {
        self.options.get(self.cursor)?.preview.as_deref()
    }

    /// Widest preview line across every option, used to size the pane so the
    /// mockups do not jump width as the cursor moves.
    fn widest_preview(&self) -> u16 {
        let widest = self
            .options
            .iter()
            .filter_map(|option| option.preview.as_deref())
            .flat_map(str::lines)
            .map(UnicodeWidthStr::width)
            .max()
            .unwrap_or(0);
        u16::try_from(widest).unwrap_or(u16::MAX)
    }

    /// Total modal width the side-by-side layout wants: the option column, the
    /// gutter, the widest mockup, plus both sets of borders. Clamped so one
    /// runaway preview line cannot demand the whole terminal.
    #[must_use]
    pub fn split_width_request(&self) -> u16 {
        OPTIONS_PANE_WIDTH
            .saturating_add(1)
            .saturating_add(self.widest_preview())
            .saturating_add(4)
            .clamp(MIN_SPLIT_WIDTH, 140)
    }

    /// Rows the preview pane needs, including its own border and the modal's.
    #[must_use]
    pub fn preview_rows_request(&self) -> u16 {
        if !self.has_previews() {
            return 0;
        }
        self.tallest_preview().saturating_add(4)
    }

    /// Tallest preview across every option, in rows.
    fn tallest_preview(&self) -> u16 {
        let tallest = self
            .options
            .iter()
            .filter_map(|option| option.preview.as_deref())
            .map(|preview| preview.lines().count())
            .max()
            .unwrap_or(0);
        u16::try_from(tallest).unwrap_or(u16::MAX)
    }

    /// Preview body lines, clipped rather than wrapped: an ASCII mockup that
    /// soft-wraps stops being a mockup.
    #[must_use]
    pub fn preview_lines(&self, theme: &Theme, width: u16, height: u16) -> Vec<Line<'static>> {
        let Some(preview) = self.focused_preview() else {
            return vec![Line::from(Span::styled(
                "No preview for this option.".to_string(),
                theme.typography.dim,
            ))];
        };
        let width = usize::from(width.max(1));
        let height = usize::from(height.max(1));
        let all: Vec<&str> = preview.lines().collect();
        let truncated_rows = all.len() > height;
        let visible = if truncated_rows { height - 1 } else { all.len() };
        let mut lines: Vec<Line<'static>> = all
            .iter()
            .take(visible)
            .map(|row| {
                Line::from(Span::styled(
                    clip_to_width(row, width),
                    theme.typography.body,
                ))
            })
            .collect();
        if truncated_rows {
            // The overflow marker is content too — clip it to the same pane or
            // it becomes the one row that spills past the border.
            lines.push(Line::from(Span::styled(
                clip_to_width(&format!("… {} more line(s)", all.len() - visible), width),
                theme.typography.dim,
            )));
        }
        lines
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(self.title_line(theme))
            .render(frame, area);
        let body_style = theme.typography.body.bg(theme.code_surface());

        // Side-by-side only when there is something to compare and room to show
        // it; otherwise the plain single column is still the better read.
        if self.has_previews() && inner.width >= MIN_SPLIT_WIDTH {
            let panes = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(OPTIONS_PANE_WIDTH),
                    Constraint::Length(1),
                    Constraint::Min(20),
                ])
                .split(inner);
            frame.render_widget(
                Paragraph::new(super::wrap_body_rows(
                    &self.render_lines_fitted(theme, panes[0].width, panes[0].height),
                    panes[0].width,
                    false,
                ))
                .style(body_style),
                panes[0],
            );
            let preview_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(theme.typography.dim)
                .title(Span::styled(" preview ".to_string(), theme.typography.dim));
            let preview_inner = preview_block.inner(panes[2]);
            frame.render_widget(preview_block, panes[2]);
            frame.render_widget(
                Paragraph::new(self.preview_lines(theme, preview_inner.width, preview_inner.height))
                    .style(body_style),
                preview_inner,
            );
            return;
        }

        // Wrapped here rather than by the `Paragraph`: the question and its
        // options are the user's own words, and ratatui's wrapper drops a
        // double-width glyph that lands on the row boundary.
        frame.render_widget(
            Paragraph::new(super::wrap_body_rows(
                &self.render_lines_fitted(theme, inner.width, inner.height),
                inner.width,
                false,
            ))
            .style(body_style),
            inner,
        );
    }
}

/// Truncate `text` to `width` display columns, appending `…` when it does not
/// fit. Column-aware so a CJK glyph is never split down the middle.
fn clip_to_width(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_string();
    }
    let budget = width.saturating_sub(1);
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let cell = UnicodeWidthStr::width(ch.to_string().as_str());
        if used + cell > budget {
            break;
        }
        out.push(ch);
        used += cell;
    }
    out.push('\u{2026}');
    out
}

/// Decode literal `\uXXXX` escape sequences — including surrogate pairs — that
/// a model occasionally emits when it over-escapes a JSON string field. The
/// modal renders the result so the prompt shows the glyphs the model intended
/// rather than raw escape text. Plain text and malformed sequences pass through
/// untouched, so a question that legitimately contains a backslash-u sequence
/// is never mangled.
fn decode_unicode_escapes(input: &str) -> String {
    if !input.contains("\\u") {
        return input.to_string();
    }
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < chars.len() {
        if let Some(high) = hex_escape_at(&chars, i) {
            if (0xD800..=0xDBFF).contains(&high) {
                // High surrogate: pair it with the following low surrogate.
                if let Some(low) = hex_escape_at(&chars, i + 6) {
                    if (0xDC00..=0xDFFF).contains(&low) {
                        let cp = 0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                            i += 12;
                            continue;
                        }
                    }
                }
                // Lone / invalid surrogate — fall through and keep it verbatim.
            } else if let Some(ch) = char::from_u32(high) {
                out.push(ch);
                i += 6;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// If `chars[at..]` starts with a `\uXXXX` escape, return its 16-bit code unit.
fn hex_escape_at(chars: &[char], at: usize) -> Option<u32> {
    if at + 6 > chars.len() || chars[at] != '\\' || chars[at + 1] != 'u' {
        return None;
    }
    let mut value = 0u32;
    for offset in 0..4 {
        value = value * 16 + chars[at + 2 + offset].to_digit(16)?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    use runtime::message_stream::{BlockId, QuestionOption};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn rich_modal() -> UserQuestionModal {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let prompt = UserQuestionPrompt {
            id: BlockId(1),
            question: "어떤 인증 방식을 쓸까요?".to_string(),
            header: Some("Auth method".to_string()),
            options: vec![
                QuestionOption {
                    label: "OAuth".to_string(),
                    description: Some("브라우저 로그인·자동 갱신".to_string()),
                    preview: None,
                },
                QuestionOption::plain("API Key"),
            ],
            multi_select: false,
            responder,
        };
        UserQuestionModal::from_prompt(&prompt)
    }

    /// A three-option multi-select fixture (no descriptions) for checkbox tests.
    fn multi_modal() -> UserQuestionModal {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let prompt = UserQuestionPrompt {
            id: BlockId(10),
            question: "Which languages?".to_string(),
            header: Some("Langs".to_string()),
            options: vec![
                QuestionOption::plain("Rust"),
                QuestionOption::plain("Go"),
                QuestionOption::plain("Zig"),
            ],
            multi_select: true,
            responder,
        };
        UserQuestionModal::from_prompt(&prompt)
    }

    fn flat(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn decodes_over_escaped_hangul() {
        // Assemble the escape at runtime: a literal backslash-u pair in source
        // would be decoded by the editor tooling before the test could run.
        let bs = '\\';
        let input = format!("{bs}uac80{bs}uc99d{bs}uc774");
        assert_eq!(decode_unicode_escapes(&input), "검증이");
    }

    #[test]
    fn leaves_plain_text_untouched() {
        assert_eq!(decode_unicode_escapes("검증이 끝난"), "검증이 끝난");
        assert_eq!(decode_unicode_escapes("hello world"), "hello world");
    }

    #[test]
    fn preserves_malformed_escapes() {
        let bs = '\\';
        let bad_hex = format!("{bs}uZZZZ");
        let too_short = format!("{bs}uac8");
        assert_eq!(decode_unicode_escapes(&bad_hex), bad_hex);
        assert_eq!(decode_unicode_escapes(&too_short), too_short);
    }

    #[test]
    fn decodes_surrogate_pair_emoji() {
        let bs = '\\';
        let input = format!("{bs}uD83D{bs}uDE00");
        assert_eq!(decode_unicode_escapes(&input), "😀");
    }

    #[test]
    fn render_lines_marks_cursor_descriptions_and_freeform_row() {
        let theme = Theme::no_color();
        let modal = rich_modal();
        let joined = flat(&modal.render_lines(&theme));
        // `NO_COLOR`/plain mode: the selection cursor degrades to the one-cell
        // ASCII `>` and never leaks the Unicode chevron. Rich `❯` output is
        // covered by the glyphs unit tests.
        assert!(
            joined.contains("> 1. OAuth"),
            "cursor row missing: {joined}"
        );
        assert!(
            !joined.contains('\u{276f}'),
            "plain mode must not leak the Unicode chevron: {joined}"
        );
        assert!(
            joined.contains("브라우저 로그인·자동 갱신"),
            "description missing: {joined}"
        );
        assert!(
            joined.contains("  2. API Key"),
            "non-cursor row missing: {joined}"
        );
        assert!(
            joined.contains("3. Other…"),
            "free-form row missing: {joined}"
        );
        assert!(
            joined.contains("confirm") && joined.contains("cancel"),
            "footer missing: {joined}"
        );
    }

    /// The modal chrome — title, free-form row, and the key-hint footer — must
    /// read in English. The surrounding product surface is English-primary, so
    /// a stray Hangul glyph in any chrome line is a coherence regression.
    /// (Model-supplied question/option text can be any language; this fixture
    /// uses English content so only the chrome is under test.)
    #[test]
    fn footer_renders_english_no_hangul() {
        let theme = Theme::no_color();
        let (responder, _rx) = tokio::sync::oneshot::channel();
        // Options present: full move/select/confirm/cancel footer + "Other" row.
        let rich = UserQuestionModal::from_prompt(&UserQuestionPrompt {
            id: BlockId(3),
            question: "Which auth method?".to_string(),
            header: Some("Auth".to_string()),
            options: vec![QuestionOption::plain("OAuth"), QuestionOption::plain("Key")],
            multi_select: false,
            responder,
        });
        let rich_text = flat(&rich.render_lines(&theme));
        assert!(rich_text.contains("move") && rich_text.contains("select"));
        assert!(rich_text.contains("confirm") && rich_text.contains("cancel"));
        assert!(rich_text.contains("Other"), "free-form row missing");

        let (responder, _rx) = tokio::sync::oneshot::channel();
        // No options: free-form-only confirm/cancel footer.
        let freeform = UserQuestionModal::from_prompt(&UserQuestionPrompt {
            id: BlockId(4),
            question: "Name?".to_string(),
            header: None,
            options: Vec::new(),
            multi_select: false,
            responder,
        });
        let freeform_text = flat(&freeform.render_lines(&theme));
        assert!(freeform_text.contains("confirm") && freeform_text.contains("cancel"));

        let title: String = rich
            .title_line(&theme)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        for rendered in [&rich_text, &freeform_text, &title] {
            assert!(
                !rendered.chars().any(is_hangul),
                "modal chrome must be Hangul-free: {rendered:?}"
            );
        }
    }

    /// `true` for any Hangul syllable / Jamo code point.
    fn is_hangul(ch: char) -> bool {
        matches!(ch, '\u{AC00}'..='\u{D7A3}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}')
    }

    #[test]
    fn title_line_carries_header_chip() {
        let theme = Theme::no_color();
        let modal = rich_modal();
        let title: String = modal
            .title_line(&theme)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(title.contains("Question"), "title base missing: {title}");
        assert!(
            title.contains("Auth method"),
            "header chip missing: {title}"
        );
    }

    #[test]
    fn typing_jumps_to_freeform_row_and_enter_returns_typed_text() {
        let mut modal = rich_modal();
        // A non-digit char while an option is selected jumps to free-form.
        assert!(modal.handle_key(key(KeyCode::Char('f'))).is_none());
        assert!(modal.on_freeform_row(), "typing must jump to free-form row");
        assert!(modal.handle_key(key(KeyCode::Char('g'))).is_none());
        let result = modal.handle_key(key(KeyCode::Enter));
        match result {
            Some(ModalResult::Selected(ModalSelection::QuestionAnswer(answers))) => {
                assert_eq!(answers, vec!["fg".to_string()]);
            }
            other => panic!("expected typed answer, got {other:?}"),
        }
    }

    #[test]
    fn digit_submits_option_and_freeform_digit_only_moves_cursor() {
        let mut modal = rich_modal();
        // Digit for the free-form row (3) moves the cursor without submitting.
        assert!(modal.handle_key(key(KeyCode::Char('3'))).is_none());
        assert!(modal.on_freeform_row());
        // With an empty buffer, Enter on the free-form row is a no-op.
        assert!(modal.handle_key(key(KeyCode::Enter)).is_none());
        // Digit for a real option submits its label immediately.
        let mut second = rich_modal();
        match second.handle_key(key(KeyCode::Char('2'))) {
            Some(ModalResult::Selected(ModalSelection::QuestionAnswer(answers))) => {
                assert_eq!(answers, vec!["API Key".to_string()]);
            }
            other => panic!("expected option submit, got {other:?}"),
        }
    }

    #[test]
    fn desired_rows_counts_descriptions_freeform_and_wrap() {
        let theme = Theme::no_color();
        let modal = rich_modal();
        // Wide: blank+question+blank + (1.OAuth + desc) + (2.API Key) +
        // freeform + blank + footer = 9 rows minimum.
        let wide = modal.desired_rows(&theme, 80);
        assert!(wide >= 9, "all rows must be counted, got {wide}");
        // The old len()-based guess (len + 7 = 9 content rows incl. borders)
        // clipped the free-form row and footer; the measured count at the
        // real render width must exceed the option count by the full chrome.
        let narrow = modal.desired_rows(&theme, 24);
        assert!(
            narrow > wide,
            "soft-wrap at narrow width must add rows ({narrow} > {wide})"
        );
    }

    #[test]
    fn arrows_wrap_across_options_and_freeform_row() {
        let mut modal = rich_modal();
        modal.handle_key(key(KeyCode::Up));
        assert!(modal.on_freeform_row(), "up from first wraps to free-form");
        modal.handle_key(key(KeyCode::Down));
        assert_eq!(modal.cursor, 0, "down from free-form wraps to first");
    }

    #[test]
    fn freeform_only_prompt_still_accepts_typed_answer() {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let prompt = UserQuestionPrompt {
            id: BlockId(2),
            question: "이름은?".to_string(),
            header: None,
            options: Vec::new(),
            multi_select: false,
            responder,
        };
        let mut modal = UserQuestionModal::from_prompt(&prompt);
        modal.handle_key(key(KeyCode::Char('a')));
        match modal.handle_key(key(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::QuestionAnswer(answers))) => {
                assert_eq!(answers, vec!["a".to_string()]);
            }
            other => panic!("expected typed answer, got {other:?}"),
        }
    }

    #[test]
    fn multi_select_renders_checkboxes_and_space_toggle_footer() {
        let theme = Theme::no_color();
        let modal = multi_modal();
        let joined = flat(&modal.render_lines(&theme));
        // Every option carries an (initially empty) checkbox and the cursor row
        // still shows the caret. Under `NO_COLOR`/plain mode the caret is the
        // one-cell ASCII `>`; the Unicode chevron must not leak.
        assert!(
            joined.contains("> [ ] 1. Rust"),
            "cursor checkbox row missing: {joined}"
        );
        assert!(
            !joined.contains('\u{276f}'),
            "plain mode must not leak the Unicode chevron: {joined}"
        );
        assert!(
            joined.contains("[ ] 2. Go"),
            "second checkbox row missing: {joined}"
        );
        // The footer advertises Space to toggle, not a numeric pick range.
        assert!(
            joined.contains("Space") && joined.contains("toggle"),
            "multi-select footer missing Space/toggle: {joined}"
        );
        assert!(
            joined.contains("confirm") && joined.contains("cancel"),
            "footer missing: {joined}"
        );
    }

    #[test]
    fn multi_select_space_toggles_and_enter_returns_all_checked() {
        let mut modal = multi_modal();
        // Toggle the first option (cursor starts there), move down twice, toggle
        // the third — Rust + Zig are checked, Go is not.
        assert!(modal.handle_key(key(KeyCode::Char(' '))).is_none());
        modal.handle_key(key(KeyCode::Down));
        modal.handle_key(key(KeyCode::Down));
        assert!(modal.handle_key(key(KeyCode::Char(' '))).is_none());
        let checked = flat(&modal.render_lines(&Theme::no_color()));
        assert!(checked.contains("[x] 1. Rust"), "Rust must be checked: {checked}");
        assert!(checked.contains("[x] 3. Zig"), "Zig must be checked: {checked}");
        assert!(checked.contains("[ ] 2. Go"), "Go must stay unchecked: {checked}");
        match modal.handle_key(key(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::QuestionAnswer(answers))) => {
                assert_eq!(answers, vec!["Rust".to_string(), "Zig".to_string()]);
            }
            other => panic!("expected multi answers, got {other:?}"),
        }
    }

    #[test]
    fn multi_select_enter_with_nothing_checked_is_noop() {
        // Reflexive Enter must not confirm an empty set — the user has to pick.
        let mut modal = multi_modal();
        assert!(
            modal.handle_key(key(KeyCode::Enter)).is_none(),
            "empty multi-select confirm must be a no-op"
        );
    }

    #[test]
    fn multi_select_digit_toggles_option() {
        // In multi-select a digit toggles (rather than immediately submitting)
        // and parks the cursor on that option.
        let mut modal = multi_modal();
        assert!(modal.handle_key(key(KeyCode::Char('2'))).is_none());
        assert_eq!(modal.cursor, 1, "digit moves the cursor to the option");
        match modal.handle_key(key(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::QuestionAnswer(answers))) => {
                assert_eq!(answers, vec!["Go".to_string()]);
            }
            other => panic!("expected toggled option, got {other:?}"),
        }
    }

    #[test]
    fn multi_select_combines_checked_options_with_freeform_text() {
        let mut modal = multi_modal();
        // Check Rust, then type into the free-form "Other" row.
        assert!(modal.handle_key(key(KeyCode::Char(' '))).is_none());
        for ch in ['O', 'C', 'a', 'm', 'l'] {
            modal.handle_key(key(KeyCode::Char(ch)));
        }
        assert!(modal.on_freeform_row(), "typing jumps to the free-form row");
        match modal.handle_key(key(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::QuestionAnswer(answers))) => {
                assert_eq!(answers, vec!["Rust".to_string(), "OCaml".to_string()]);
            }
            other => panic!("expected checked + typed answers, got {other:?}"),
        }
    }

    const SIDEBAR_MOCKUP: &str = "┌ nav ┬ body ┐\n│     │      │\n└─────┴──────┘";
    const TOPBAR_MOCKUP: &str = "┌ topbar ─────┐\n│ body        │\n└─────────────┘";

    fn preview_modal() -> UserQuestionModal {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let prompt = UserQuestionPrompt {
            id: BlockId(7),
            question: "레이아웃을 어떻게 잡을까요?".to_string(),
            header: Some("Layout".to_string()),
            options: vec![
                QuestionOption::plain("사이드바").with_preview(SIDEBAR_MOCKUP),
                QuestionOption::plain("상단바").with_preview(TOPBAR_MOCKUP),
            ],
            multi_select: false,
            responder,
        };
        UserQuestionModal::from_prompt(&prompt)
    }

    fn preview_text(modal: &UserQuestionModal, width: u16, height: u16) -> String {
        modal
            .preview_lines(&Theme::default_dark(), width, height)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Moving the cursor swaps the pane, which is the entire point of the
    /// side-by-side layout: the alternatives are compared by looking.
    #[test]
    fn the_preview_pane_tracks_the_focused_option() {
        let mut modal = preview_modal();
        assert!(modal.has_previews());
        assert!(preview_text(&modal, 40, 10).contains("nav"));
        modal.handle_key(key(KeyCode::Down));
        let shown = preview_text(&modal, 40, 10);
        assert!(shown.contains("topbar"), "{shown}");
        assert!(!shown.contains("nav"), "{shown}");
    }

    /// The free-form row has no artifact to show, so the pane says so instead
    /// of leaving the previous option's mockup up as if it still applied.
    #[test]
    fn the_freeform_row_clears_the_preview_pane() {
        let mut modal = preview_modal();
        modal.handle_key(key(KeyCode::Down));
        modal.handle_key(key(KeyCode::Down));
        assert!(modal.on_freeform_row());
        assert!(preview_text(&modal, 40, 10).contains("No preview"));
    }

    /// Mockups are column-significant, so a pane too small clips them (with a
    /// count of what was cut) rather than soft-wrapping them into nonsense.
    #[test]
    fn a_small_pane_clips_instead_of_wrapping() {
        let modal = preview_modal();
        // Every row — including the overflow marker itself — stays inside the
        // pane, so nothing spills over the border.
        let narrow = preview_text(&modal, 8, 2);
        assert!(narrow.contains('\u{2026}'), "clipped marker: {narrow}");
        for row in narrow.lines() {
            assert!(
                UnicodeWidthStr::width(row) <= 8,
                "row overflows the pane: {row:?}"
            );
        }
        // A short pane says how much it cut rather than silently dropping it.
        let short = preview_text(&modal, 40, 2);
        assert!(short.contains("… 2 more line(s)"), "{short}");
    }

    /// A question with no previews must keep the original single-column modal —
    /// the split layout is opt-in per question, not a global restyle.
    #[test]
    fn questions_without_previews_keep_the_single_column_layout() {
        let modal = rich_modal();
        assert!(!modal.has_previews());
        assert_eq!(modal.preview_rows_request(), 0);
    }

    /// The modal asks for a width that actually fits the widest mockup, so a
    /// preview is not born clipped.
    #[test]
    fn the_width_request_covers_the_widest_mockup() {
        let modal = preview_modal();
        let widest = TOPBAR_MOCKUP
            .lines()
            .map(UnicodeWidthStr::width)
            .max()
            .expect("mockup rows");
        let requested = modal.split_width_request();
        assert!(requested >= MIN_SPLIT_WIDTH);
        assert!(
            usize::from(requested) >= usize::from(OPTIONS_PANE_WIDTH) + widest,
            "requested {requested} must seat the {widest}-column mockup beside the list"
        );
    }

    /// An empty preview string must not flip the modal into the split layout to
    /// display nothing.
    #[test]
    fn a_blank_preview_is_treated_as_absent() {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let prompt = UserQuestionPrompt {
            id: BlockId(8),
            question: "Pick".to_string(),
            header: None,
            options: vec![QuestionOption::plain("a").with_preview("   \n  ")],
            multi_select: false,
            responder,
        };
        assert!(!UserQuestionModal::from_prompt(&prompt).has_previews());
    }
}

#[cfg(test)]
mod clipping_tests {
    use super::*;
    use crate::tui::theme::Theme;
    use runtime::message_stream::{BlockId, QuestionOption, UserQuestionPrompt};

    /// Four options with long Korean labels *and* previews — the shape from the
    /// screenshot, where only options 1 and 2 ever appeared.
    fn four_korean_options() -> UserQuestionModal {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let option = |n: char| QuestionOption {
            label: format!("제안 {n} — 사이드바를 콘텐츠 높이에 맞춰 접는 방식"),
            description: Some(format!(
                "제안 {n}의 설명: 카드 상하 룰로 경계를 닫고 남는 열은 트랜스크립트에 돌려준다"
            )),
            preview: Some(format!("┌── 제안 {n} ──┐\n│ mockup row │\n└────────────┘")),
        };
        let prompt = UserQuestionPrompt {
            id: BlockId(77),
            question: "어떤 사이드바 경계 방식을 쓸까요?".to_string(),
            header: Some("Sidebar".to_string()),
            options: vec![option('A'), option('B'), option('C'), option('D')],
            multi_select: false,
            responder,
        };
        UserQuestionModal::from_prompt(&prompt)
    }

    fn flatten(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Rows must be measured at the width they are painted into. The split
    /// layout paints the list into a 38-column column while `desired_size` used
    /// to measure at the full modal width, so four wrapped Korean options were
    /// counted as if they fit in four rows and the modal was sized to clip them.
    #[test]
    fn desired_size_measures_the_option_column_not_the_whole_dialog() {
        let theme = Theme::no_color();
        let modal = four_korean_options();
        let area = Rect::new(0, 0, 200, 60);
        let (width, height) = modal.desired_size(area, &theme);
        assert!(width >= MIN_SPLIT_WIDTH, "the preview layout asks for its width");

        let rows_at_column = modal.desired_rows(&theme, OPTIONS_PANE_WIDTH);
        let rows_at_dialog = modal.desired_rows(&theme, width);
        assert!(
            rows_at_column > rows_at_dialog,
            "wrapped Korean labels must cost more rows in the narrow column \
             ({rows_at_column} vs {rows_at_dialog}) — that difference is the clip"
        );
        assert!(
            height >= rows_at_column,
            "the modal must be tall enough for the rows it will paint \
             (height {height} < {rows_at_column} rows)"
        );
    }

    /// Given room, every option is on screen — no window, no marker.
    #[test]
    fn roomy_terminal_shows_all_four_options_without_scrolling() {
        let theme = Theme::no_color();
        let modal = four_korean_options();
        let area = Rect::new(0, 0, 200, 60);
        let (width, height) = modal.desired_size(area, &theme);
        let rendered = flatten(&modal.render_lines_fitted(
            &theme,
            modal.list_paint_width(width),
            height,
        ));
        for n in ['A', 'B', 'C', 'D'] {
            assert!(
                rendered.contains(&format!("제안 {n}")),
                "option {n} must be visible with room to spare:\n{rendered}"
            );
        }
        assert!(
            !rendered.contains("more"),
            "nothing is hidden, so no clip marker:\n{rendered}"
        );
    }

    /// When the height genuinely cannot fit them, the list scrolls with the
    /// selection and says how much it is hiding — it never silently drops rows.
    #[test]
    fn cramped_terminal_keeps_the_selection_visible_and_marks_the_clip() {
        let theme = Theme::no_color();
        let mut modal = four_korean_options();
        let (width, height) = (OPTIONS_PANE_WIDTH, 14u16);

        for expected in ['A', 'B', 'C', 'D'] {
            let rendered = flatten(&modal.render_lines_fitted(&theme, width, height));
            assert!(
                rendered.contains(&format!("제안 {expected}")),
                "the selected option must always be on screen, {expected} was not:\n{rendered}"
            );
            assert!(
                rendered.contains("more"),
                "a clipped list must show its clip affordance:\n{rendered}"
            );
            // The question and the key hints stay pinned around the window.
            assert!(rendered.contains("어떤 사이드바"), "question stays pinned");
            assert!(rendered.contains("Esc"), "key hints stay pinned");
            modal.handle_key(KeyEvent::new(KeyCode::Down, crossterm::event::KeyModifiers::NONE));
        }
    }

    /// The window never spends more rows than it was given.
    ///
    /// From 12 rows up: below that the pinned chrome (question + key hints)
    /// alone leaves less than one wrapped option, and showing a clipped
    /// selection beats showing none.
    #[test]
    fn fitted_lines_never_exceed_the_pane_height() {
        let theme = Theme::no_color();
        let modal = four_korean_options();
        for height in 12..=24u16 {
            let lines = modal.render_lines_fitted(&theme, OPTIONS_PANE_WIDTH, height);
            let rows: u16 = lines
                .iter()
                .map(|line| UserQuestionModal::wrapped_rows(line, OPTIONS_PANE_WIDTH))
                .fold(0u16, u16::saturating_add);
            assert!(
                rows <= height,
                "fitted render used {rows} rows in a {height}-row pane"
            );
        }
    }
}
