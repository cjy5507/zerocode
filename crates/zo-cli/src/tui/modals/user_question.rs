//! In-app modal for `AskUserQuestion` prompts.
//!
//! Claude-Code-parity surface: an optional topic chip in the title, options
//! with one-line dim descriptions, and an always-available free-form row —
//! the model's options are suggestions, never a cage.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::super::cards::{CardFrame, SurfaceKind};
use runtime::message_stream::UserQuestionPrompt;
use unicode_width::UnicodeWidthStr;

use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, key_hint_footer, selected_style,
};

/// One decoded option row.
#[derive(Debug, Clone)]
struct OptionRow {
    label: String,
    description: Option<String>,
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

    /// `true` if this prompt expects free-form text only.
    #[must_use]
    pub fn is_freeform(&self) -> bool {
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
        if self.multi_select {
            key_hint_footer(
                theme,
                &[
                    ("↑↓", "move"),
                    ("Space", "toggle"),
                    ("Enter", "confirm"),
                    ("Esc", "cancel"),
                ],
            )
        } else {
            let pick = format!("1–{}", self.options.len() + 1);
            key_hint_footer(
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
            lines.push(key_hint_footer(
                theme,
                &[("Enter", "confirm"), ("Esc", "cancel")],
            ));
            return lines;
        }

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
            lines.push(Line::from(Span::styled(
                format!("{marker}{checkbox}{}. {}", idx + 1, option.label),
                style,
            )));
            if let Some(description) = &option.description {
                // Description column: marker + optional checkbox + "N. " deep.
                let indent = " ".repeat(blank_marker().len() + checkbox.len() + 3);
                lines.push(Line::from(Span::styled(
                    format!("{indent}{description}"),
                    theme.typography.dim,
                )));
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
        lines.push(Line::from(Span::styled(freeform, style)));

        lines.push(Line::from(""));
        lines.push(self.options_footer(theme));

        lines
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(self.title_line(theme))
            .render(frame, area);
        let paragraph = Paragraph::new(self.render_lines(theme))
            .style(theme.typography.body.bg(theme.code_surface()))
            .wrap(Wrap { trim: false });
        frame.render_widget(paragraph, inner);
    }
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
}
