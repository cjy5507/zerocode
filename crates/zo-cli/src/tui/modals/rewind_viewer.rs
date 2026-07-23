//! Interactive snapshot rewind viewer — Zo's safety-UX differentiator.
//!
//! Where `/diff` reviews the working tree against `HEAD`, this viewer walks the
//! per-turn [`SnapshotStack`](runtime::git_snapshot::SnapshotStack) timeline: a
//! list of turn checkpoints (newest first), the selected turn's diff in the
//! body, and `Enter` to rewind the worktree to *any* earlier snapshot in one
//! step — not just the previous one. Read-only until you confirm a rewind.
//!
//! The host (`tui_loop`) owns the `SnapshotStack`, so it precomputes the rows
//! ([`RewindRow`]) at open time and feeds them in; navigation and rendering are
//! then self-contained. Diff text is parsed by
//! [`super::diff_viewer::parse_unified_diff`] and rendered by
//! [`crate::tui::blocks::diff`], reusing the proven `/diff` machinery.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};

use super::super::cards::{CardFrame, SurfaceKind};
use runtime::message_stream::DiffView;

use super::super::theme::Theme;

/// One turn checkpoint in the timeline, precomputed by the host from the
/// snapshot stack so the modal needs no further git access.
#[derive(Debug, Clone)]
pub struct RewindRow {
    /// Position in the underlying stack (0 = baseline). Carried through to
    /// [`RewindViewerAction::RewindTo`] so the host rewinds the right snapshot.
    pub index: usize,
    /// The turn this snapshot was captured after.
    pub turn_number: usize,
    /// Whether this is the live worktree state (cannot be rewound *to*).
    pub is_current: bool,
    /// Total lines added by this turn.
    pub added: usize,
    /// Total lines removed by this turn.
    pub removed: usize,
    /// Number of files this turn touched.
    pub file_count: usize,
    /// Per-file diffs this turn introduced, for the body pane.
    pub views: Vec<DiffView>,
}

/// Outcome of a single key handled by [`RewindViewerModal`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewindViewerAction {
    /// `Esc`/`q`/`Ctrl+C` — close without rewinding.
    Close,
    /// `Enter` on an earlier snapshot — rewind the worktree to stack index `n`.
    RewindTo(usize),
}

/// Interactive snapshot-timeline rewind viewer.
#[derive(Debug, Clone)]
pub struct RewindViewerModal {
    /// Rows ordered newest-first (row 0 = current worktree state).
    rows: Vec<RewindRow>,
    selected: usize,
    scroll: u16,
    show_diff: bool,
}

impl RewindViewerModal {
    /// Build a viewer over `rows`, which must already be ordered newest-first
    /// (the current snapshot at index 0). The diff pane starts visible.
    #[must_use]
    pub fn new(rows: Vec<RewindRow>) -> Self {
        Self {
            rows,
            selected: 0,
            scroll: 0,
            show_diff: true,
        }
    }

    /// `true` when there is nothing meaningful to rewind through (fewer than
    /// two snapshots). The host shows a "nothing to rewind" note instead.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.len() < 2
    }

    /// Index of the highlighted timeline row.
    #[must_use]
    pub const fn selected(&self) -> usize {
        self.selected
    }

    /// `true` when the diff body pane is shown.
    #[must_use]
    pub const fn shows_diff(&self) -> bool {
        self.show_diff
    }

    /// The highlighted row, if any.
    #[must_use]
    pub fn selected_row(&self) -> Option<&RewindRow> {
        self.rows.get(self.selected)
    }

    fn select_older(&mut self) {
        if self.selected + 1 < self.rows.len() {
            self.selected += 1;
            self.scroll = 0;
        }
    }

    fn select_newer(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.scroll = 0;
        }
    }

    /// Handle one key. Returns `Some(Close)` to dismiss or `Some(RewindTo)` to
    /// request a rewind; `None` while navigating.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<RewindViewerAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(RewindViewerAction::Close);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return Some(RewindViewerAction::Close),
            // Timeline navigation (primary axis).
            KeyCode::Up | KeyCode::Char('k') => self.select_newer(),
            KeyCode::Down | KeyCode::Char('j') => self.select_older(),
            KeyCode::Home | KeyCode::Char('g') => {
                self.selected = 0;
                self.scroll = 0;
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.selected = self.rows.len().saturating_sub(1);
                self.scroll = 0;
            }
            // Diff-body scroll (secondary axis).
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            // `d` toggles the diff body.
            KeyCode::Char('d') => {
                self.show_diff = !self.show_diff;
                self.scroll = 0;
            }
            // `Enter` rewinds to the selected snapshot — but never to the
            // current state (row 0), which is a no-op.
            KeyCode::Enter => {
                if let Some(row) = self.selected_row() {
                    if !row.is_current {
                        return Some(RewindViewerAction::RewindTo(row.index));
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(" rewind ", theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }

        // Timeline list height: enough for every row, but never more than half
        // the body when the diff pane is showing.
        let list_rows = u16::try_from(self.rows.len()).unwrap_or(u16::MAX).max(1);
        let [list_area, body_area, footer_area] = if self.show_diff {
            let list_height = list_rows
                .min(inner.height.saturating_sub(2).max(1) / 2)
                .max(1);
            Layout::vertical([
                Constraint::Length(list_height),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .areas(inner)
        } else {
            Layout::vertical([
                Constraint::Min(1),
                Constraint::Length(0),
                Constraint::Length(1),
            ])
            .areas(inner)
        };

        frame.render_widget(Paragraph::new(self.timeline_lines(theme)), list_area);

        if self.show_diff && body_area.height > 0 {
            let body_lines = self
                .selected_row()
                .map(|row| diff_body_lines(row, theme))
                .unwrap_or_default();
            frame.render_widget(
                Paragraph::new(body_lines).scroll((self.scroll, 0)),
                body_area,
            );
        }

        frame.render_widget(Paragraph::new(footer_line(theme)), footer_area);
    }

    /// One styled line per timeline row, newest first, with a `▸` caret on the
    /// selection.
    fn timeline_lines(&self, theme: &Theme) -> Vec<Line<'static>> {
        self.rows
            .iter()
            .enumerate()
            .map(|(row_index, row)| self.timeline_row_line(row_index, row, theme))
            .collect()
    }

    fn timeline_row_line(&self, row_index: usize, row: &RewindRow, theme: &Theme) -> Line<'static> {
        let is_selected = row_index == self.selected;
        let caret = if is_selected { "▸ " } else { "  " };
        let label_style = if is_selected {
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            theme.typography.body
        };
        let tag = if row.is_current {
            "  now"
        } else if row.index == 0 {
            "  start"
        } else {
            ""
        };

        let mut spans = vec![
            Span::styled(caret.to_string(), label_style),
            Span::styled(format!("turn {}", row.turn_number), label_style),
            Span::styled(tag.to_string(), theme.typography.dim),
        ];
        if row.added > 0 || row.removed > 0 {
            spans.push(Span::raw("   "));
            spans.push(Span::styled(
                format!("+{}", row.added),
                theme.diff_add_style(),
            ));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(
                format!("-{}", row.removed),
                theme.diff_del_style(),
            ));
            spans.push(Span::styled(
                format!("  {} file(s)", row.file_count),
                theme.typography.dim,
            ));
        } else {
            spans.push(Span::styled("   —".to_string(), theme.typography.dim));
        }
        Line::from(spans)
    }
}

/// Render the selected row's per-file diffs as a flat, scrollable line list,
/// reusing the `/diff` renderer.
fn diff_body_lines<'a>(row: &'a RewindRow, theme: &Theme) -> Vec<Line<'a>> {
    if row.views.is_empty() {
        let note = if row.index == 0 {
            "baseline checkpoint — nothing was changed yet"
        } else {
            "no file changes in this turn"
        };
        return vec![Line::from(Span::styled(
            note.to_string(),
            theme.typography.dim,
        ))];
    }
    let mut lines = Vec::new();
    for view in &row.views {
        lines.extend(crate::tui::blocks::diff::lines(view, theme, true));
        lines.push(Line::from(""));
    }
    lines
}

fn footer_line(theme: &Theme) -> Line<'static> {
    super::key_hint_footer(
        theme,
        &[
            ("↑/↓", "turn"),
            ("d", "diff"),
            ("Enter", "rewind here"),
            ("Esc", "close"),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::{RewindRow, RewindViewerAction, RewindViewerModal};
    use crate::tui::theme::Theme;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn row(index: usize, turn: usize, is_current: bool, added: usize) -> RewindRow {
        RewindRow {
            index,
            turn_number: turn,
            is_current,
            added,
            removed: 0,
            file_count: usize::from(added > 0),
            views: Vec::new(),
        }
    }

    /// Three snapshots, newest-first: current(turn 2) · turn 1 · baseline(turn 0).
    fn sample() -> RewindViewerModal {
        RewindViewerModal::new(vec![
            row(2, 2, true, 5),
            row(1, 1, false, 12),
            row(0, 0, false, 0),
        ])
    }

    #[test]
    fn empty_when_fewer_than_two_snapshots() {
        assert!(RewindViewerModal::new(vec![row(0, 0, true, 0)]).is_empty());
        assert!(!sample().is_empty());
    }

    #[test]
    fn down_up_navigate_and_clamp() {
        let mut modal = sample();
        assert_eq!(modal.selected(), 0);
        modal.handle_key(press(KeyCode::Up)); // already newest, clamps
        assert_eq!(modal.selected(), 0);
        modal.handle_key(press(KeyCode::Down));
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected(), 2);
        modal.handle_key(press(KeyCode::Down)); // clamp at oldest
        assert_eq!(modal.selected(), 2);
    }

    #[test]
    fn enter_on_current_does_not_rewind() {
        let mut modal = sample();
        assert_eq!(modal.handle_key(press(KeyCode::Enter)), None);
    }

    #[test]
    fn enter_on_earlier_snapshot_emits_its_stack_index() {
        let mut modal = sample();
        modal.handle_key(press(KeyCode::Down)); // select turn 1 (stack index 1)
        assert_eq!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(RewindViewerAction::RewindTo(1))
        );
        // Down once more selects the baseline (stack index 0).
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(RewindViewerAction::RewindTo(0))
        );
    }

    #[test]
    fn d_toggles_diff_and_esc_closes() {
        let mut modal = sample();
        assert!(modal.shows_diff());
        modal.handle_key(press(KeyCode::Char('d')));
        assert!(!modal.shows_diff());
        assert_eq!(
            modal.handle_key(press(KeyCode::Esc)),
            Some(RewindViewerAction::Close)
        );
    }

    #[test]
    fn draws_without_panic_in_both_modes() {
        let theme = Theme::zo();
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).expect("backend");

        let modal = sample();
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 80, 24), &theme))
            .expect("draw with diff pane");

        let mut collapsed = sample();
        collapsed.handle_key(press(KeyCode::Char('d')));
        term.draw(|f| collapsed.draw(f, Rect::new(0, 0, 80, 24), &theme))
            .expect("draw without diff pane");
    }

    #[test]
    fn timeline_dump_shows_turns_and_caret() {
        let theme = Theme::zo();
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).expect("backend");
        let modal = sample();
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 80, 24), &theme))
            .expect("draw");

        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        assert!(dump.contains("turn 2"), "newest turn row present");
        assert!(dump.contains("now"), "current snapshot tagged");
        assert!(dump.contains("start"), "baseline tagged");
        assert!(dump.contains('▸'), "selection caret present");
    }
}
