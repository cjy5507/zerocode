use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::theme::Theme;
use super::{ModalResult, ModalSelection, key_hint_footer};
use crate::tui::fuzzy;

/// Rows visible in the picker window; also the PageUp/PageDown stride so a
/// page key advances by roughly one screen of results.
const MAX_VISIBLE: usize = 12;

/// Fuzzy file picker modal triggered by `@` in the input widget.
#[derive(Debug, Clone)]
pub struct FilePickerModal {
    query: String,
    items: Vec<String>,
    filtered: Vec<usize>,
    cursor: usize,
    /// `true` while the background workspace scan is still running. The
    /// item list is empty until [`FilePickerModal::set_items`] lands the
    /// result, so the modal shows a "scanning…" hint instead of blocking
    /// the UI thread on a large repo.
    loading: bool,
}

impl FilePickerModal {
    #[must_use]
    pub fn new(items: Vec<String>) -> Self {
        let filtered: Vec<usize> = (0..items.len()).collect();
        Self {
            query: String::new(),
            items,
            filtered,
            cursor: 0,
            loading: false,
        }
    }

    /// Mark the picker as awaiting (or done awaiting) its background scan.
    pub fn set_loading(&mut self, loading: bool) {
        self.loading = loading;
    }

    /// Land the result of the background workspace scan, preserving any
    /// query the user typed while it ran. Re-applies the current filter so
    /// the visible list reflects both the new items and the live query.
    pub fn set_items(&mut self, items: Vec<String>) {
        self.items = items;
        self.loading = false;
        self.cursor = 0;
        self.refilter();
    }

    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.filtered.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filtered.is_empty()
    }

    fn refilter(&mut self) {
        // Converge on the shared fuzzy SSOT (`tui::fuzzy`) so the file picker
        // ranks subsequences with the same rule as the slash surfaces.
        let q = self.query.to_lowercase();
        self.filtered = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                if q.is_empty() {
                    return true;
                }
                fuzzy::is_subsequence(&item.to_lowercase(), &q)
            })
            .map(|(i, _)| i)
            .collect();
        if self.cursor >= self.filtered.len() {
            self.cursor = self.filtered.len().saturating_sub(1);
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Enter => {
                if self.filtered.is_empty() {
                    return Some(ModalResult::Cancelled);
                }
                let idx = self.filtered[self.cursor];
                let label = self.items[idx].clone();
                Some(ModalResult::Selected(ModalSelection::Choice {
                    index: idx,
                    label,
                }))
            }
            KeyCode::Up => {
                if !self.filtered.is_empty() && self.cursor > 0 {
                    self.cursor -= 1;
                }
                None
            }
            KeyCode::Down => {
                if self.cursor + 1 < self.filtered.len() {
                    self.cursor += 1;
                }
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
            KeyCode::Char(ch) => {
                self.query.push(ch);
                self.refilter();
                None
            }
            _ => None,
        }
    }

    /// Move the cursor down by a page, clamping at the last match.
    pub fn page_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + MAX_VISIBLE).min(self.filtered.len() - 1);
    }

    /// Move the cursor up by a page, clamping at the first match.
    pub fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(MAX_VISIBLE);
    }

    /// Jump the cursor to the first match.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Jump the cursor to the last match.
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

    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let max_visible = MAX_VISIBLE;
        let start = self.cursor.saturating_sub(max_visible / 2);

        let mut lines = vec![Line::from(Span::styled(
            format!("  Search: {}_", self.query),
            theme.typography.heading_2,
        ))];

        for (vi, &idx) in self
            .filtered
            .iter()
            .skip(start)
            .take(max_visible)
            .enumerate()
        {
            let actual_pos = start + vi;
            let marker = if actual_pos == self.cursor {
                "▶ "
            } else {
                "  "
            };
            let text = format!("{marker}{}", self.items[idx]);
            let style = if actual_pos == self.cursor {
                theme.typography.bold.add_modifier(Modifier::REVERSED)
            } else {
                theme.typography.body
            };
            lines.push(Line::from(Span::styled(text, style)));
        }

        if self.filtered.is_empty() {
            let empty_hint = if self.loading {
                "  Scanning workspace\u{2026}"
            } else {
                "  No matches"
            };
            lines.push(Line::from(Span::styled(empty_hint, theme.typography.dim)));
        }

        lines
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let title = if self.loading {
            " @ File Reference (scanning\u{2026}) ".to_string()
        } else {
            format!(" @ File Reference ({} files) ", self.filtered.len())
        };
        let inner = super::modal_frame(frame, area, title, theme);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let [list_area, _spacer, footer_area] = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(inner);
        frame.render_widget(
            Paragraph::new(self.render_lines(theme)).style(theme.typography.body),
            list_area,
        );
        frame.render_widget(
            Paragraph::new(key_hint_footer(
                theme,
                &[
                    ("↑↓", "move"),
                    ("PgUp/PgDn", "page"),
                    ("Enter", "insert"),
                    ("Esc", "cancel"),
                ],
            )),
            footer_area,
        );
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

    #[test]
    fn fuzzy_filter_narrows_results() {
        let mut modal = FilePickerModal::new(vec![
            "src/main.rs".into(),
            "src/lib.rs".into(),
            "Cargo.toml".into(),
        ]);
        assert_eq!(modal.len(), 3);

        modal.handle_key(press(KeyCode::Char('m')));
        modal.handle_key(press(KeyCode::Char('a')));
        assert!(modal.len() <= 2);
    }

    #[test]
    fn enter_selects_current() {
        let mut modal = FilePickerModal::new(vec!["a.rs".into(), "b.rs".into()]);
        modal.handle_key(press(KeyCode::Down));
        let result = modal.handle_key(press(KeyCode::Enter));
        match result {
            Some(ModalResult::Selected(ModalSelection::Choice { label, .. })) => {
                assert_eq!(label, "b.rs");
            }
            _ => panic!("expected selection"),
        }
    }

    #[test]
    fn esc_cancels() {
        let mut modal = FilePickerModal::new(vec!["a.rs".into()]);
        assert!(matches!(
            modal.handle_key(press(KeyCode::Esc)),
            Some(ModalResult::Cancelled)
        ));
    }

    #[test]
    fn set_items_lands_scan_result_and_clears_loading() {
        // A picker opened empty + loading (as the async path does) shows no
        // entries until the background scan lands its result.
        let mut modal = FilePickerModal::new(Vec::new());
        modal.set_loading(true);
        assert_eq!(modal.len(), 0);

        modal.set_items(vec!["src/main.rs".into(), "Cargo.toml".into()]);
        assert_eq!(modal.len(), 2);
    }

    #[test]
    fn page_down_advances_by_a_page_and_clamps() {
        // 30 files, MAX_VISIBLE = 12 page stride: PageDown jumps a page; a
        // second clamps at the final row (file picker cursor is non-wrapping).
        let items: Vec<String> = (0..30).map(|i| format!("file{i}.rs")).collect();
        let mut modal = FilePickerModal::new(items);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor, MAX_VISIBLE);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor, MAX_VISIBLE * 2);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor, 29, "PageDown clamps at the last file");
        modal.handle_key(press(KeyCode::Home));
        assert_eq!(modal.cursor, 0);
        modal.handle_key(press(KeyCode::End));
        assert_eq!(modal.cursor, 29);
    }

    #[test]
    fn fuzzy_filter_uses_shared_ssot() {
        // Converged onto `tui::fuzzy`: a gapped subsequence still matches.
        let mut modal = FilePickerModal::new(vec![
            "src/main.rs".into(),
            "src/lib.rs".into(),
            "Cargo.toml".into(),
        ]);
        for ch in "srlib".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        // "srlib" is a subsequence of "src/lib.rs" only.
        assert_eq!(modal.len(), 1);
    }

    #[test]
    fn set_items_preserves_typed_query() {
        // The user can type while the scan runs; landing the result must
        // re-apply that live query rather than show every file.
        let mut modal = FilePickerModal::new(Vec::new());
        modal.set_loading(true);
        modal.handle_key(press(KeyCode::Char('c')));

        modal.set_items(vec!["src/main.rs".into(), "Cargo.toml".into()]);
        // Query "c" matches "Cargo.toml" (and "src/main.rs" has no 'c'),
        // so the filter narrows even though items arrived after typing.
        assert_eq!(modal.query(), "c");
        assert!(modal.len() <= 2);
    }
}
