//! Hunk-level human/agent attribution review modal for `/hunks`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};
use tools::{
    AttributionLineKind, AttributionOrigin, AttributionStatus, AttributedHunk,
    HunkAttributionLedger, ToolContext,
};

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::theme::Theme;
use super::ModalResult;

pub struct ReviewModal {
    context: ToolContext,
    ledger: HunkAttributionLedger,
    order: Vec<usize>,
    selected: usize,
    body_scroll: u16,
    error: Option<String>,
}

impl ReviewModal {
    #[must_use]
    pub fn new(context: ToolContext, ledger: HunkAttributionLedger) -> Self {
        let mut modal = Self {
            context,
            ledger,
            order: Vec::new(),
            selected: 0,
            body_scroll: 0,
            error: None,
        };
        modal.rebuild_order();
        modal
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    #[must_use]
    pub const fn ledger(&self) -> &HunkAttributionLedger {
        &self.ledger
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(ModalResult::Cancelled);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(ModalResult::Cancelled),
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = self.selected.saturating_sub(1);
                self.body_scroll = 0;
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.order.len() {
                    self.selected += 1;
                }
                self.body_scroll = 0;
                None
            }
            KeyCode::PageUp => {
                self.body_scroll = self.body_scroll.saturating_sub(10);
                None
            }
            KeyCode::PageDown => {
                self.body_scroll = self.body_scroll.saturating_add(10);
                None
            }
            KeyCode::Char('a') => {
                self.accept_selected();
                None
            }
            KeyCode::Char('A') => {
                self.accept_selected_file();
                None
            }
            KeyCode::Char('r') => {
                self.reject_selected();
                None
            }
            _ => None,
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let title = format!(
            " /hunks attribution · {} file(s) · {} hunk(s) ",
            self.file_count(),
            self.order.len()
        );
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(title, theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }

        let error_height = u16::from(self.error.is_some());
        let list_height = if self.order.is_empty() {
            2
        } else {
            u16::try_from(self.order.len().saturating_add(self.file_count()))
                .unwrap_or(u16::MAX)
                .min(inner.height.saturating_sub(error_height + 2) / 2)
                .max(2)
        };
        let [list_area, error_area, body_area, footer_area] = Layout::vertical([
            Constraint::Length(list_height),
            Constraint::Length(error_height),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        let (list_lines, selected_row) = self.list_lines(theme);
        let list_scroll = selected_row
            .saturating_sub(usize::from(list_area.height).saturating_sub(1))
            .try_into()
            .unwrap_or(u16::MAX);
        frame.render_widget(
            Paragraph::new(list_lines).scroll((list_scroll, 0)),
            list_area,
        );

        if let Some(error) = &self.error {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!("! {error}"),
                    Style::new().fg(theme.palette.error),
                ))),
                error_area,
            );
        }
        frame.render_widget(
            Paragraph::new(self.hunk_body_lines(theme)).scroll((self.body_scroll, 0)),
            body_area,
        );
        frame.render_widget(
            Paragraph::new(super::key_hint_footer(
                theme,
                &[
                    ("j/k", "hunk"),
                    ("a", "accept"),
                    ("r", "reject"),
                    ("A", "accept file"),
                    ("Esc", "close"),
                ],
            )),
            footer_area,
        );
    }

    fn rebuild_order(&mut self) {
        self.order = (0..self.ledger.hunks.len()).collect();
        self.order.sort_by(|left, right| {
            self.ledger.hunks[*left]
                .path
                .cmp(&self.ledger.hunks[*right].path)
                .then_with(|| left.cmp(right))
        });
        self.selected = self.selected.min(self.order.len().saturating_sub(1));
    }

    fn selected_ledger_index(&self) -> Option<usize> {
        self.order.get(self.selected).copied()
    }

    fn selected_hunk(&self) -> Option<&AttributedHunk> {
        self.selected_ledger_index()
            .and_then(|index| self.ledger.hunks.get(index))
    }

    fn accept_selected(&mut self) {
        let Some(index) = self.selected_ledger_index() else {
            return;
        };
        match self.context.accept_workspace_hunk(index) {
            Ok(ledger) => {
                self.ledger = ledger;
                self.error = None;
            }
            Err(error) => self.error = Some(error.to_string()),
        }
    }

    fn accept_selected_file(&mut self) {
        let Some(path) = self.selected_hunk().map(|hunk| hunk.path.clone()) else {
            return;
        };
        self.ledger = self.context.accept_workspace_file_hunks(&path);
        self.error = None;
    }

    fn reject_selected(&mut self) {
        let Some(index) = self.selected_ledger_index() else {
            return;
        };
        match self.context.reject_workspace_hunk(index) {
            Ok(ledger) => {
                self.ledger = ledger;
                self.error = None;
            }
            Err(error) => {
                self.error = Some(error.to_string());
                self.ledger = self.context.current_workspace_hunk_attribution();
            }
        }
        self.body_scroll = 0;
    }

    fn file_count(&self) -> usize {
        self.order
            .iter()
            .map(|index| &self.ledger.hunks[*index].path)
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn list_lines<'a>(&'a self, theme: &Theme) -> (Vec<Line<'a>>, usize) {
        if self.order.is_empty() {
            return (
                vec![Line::from(Span::styled(
                    "No attributed checkpoint hunks yet.",
                    theme.typography.dim,
                ))],
                0,
            );
        }

        let mut by_path = BTreeMap::<&Path, Vec<(usize, usize)>>::new();
        for (position, index) in self.order.iter().copied().enumerate() {
            by_path
                .entry(&self.ledger.hunks[index].path)
                .or_default()
                .push((position, index));
        }
        let mut lines = Vec::new();
        let mut selected_row = 0;
        for (path, entries) in by_path {
            lines.push(Line::from(Span::styled(
                path.display().to_string(),
                theme.typography.bold,
            )));
            for (position, index) in entries {
                let hunk = &self.ledger.hunks[index];
                let selected = position == self.selected;
                if selected {
                    selected_row = lines.len();
                }
                let style = if selected {
                    Style::new()
                        .fg(theme.palette.accent)
                        .add_modifier(Modifier::BOLD)
                } else {
                    theme.typography.body
                };
                lines.push(Line::from(vec![
                    Span::styled(if selected { "> " } else { "  " }, style),
                    Span::styled(origin_label(hunk.origin), origin_style(hunk.origin, theme)),
                    Span::raw(" "),
                    Span::styled(
                        format!(
                            "@@ -{},{} +{},{} @@",
                            hunk.old_start, hunk.old_lines, hunk.new_start, hunk.new_lines
                        ),
                        style,
                    ),
                    Span::raw("  "),
                    Span::styled(status_label(hunk.status), status_style(hunk.status, theme)),
                ]));
            }
        }
        (lines, selected_row)
    }

    fn hunk_body_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let Some(hunk) = self.selected_hunk() else {
            return vec![Line::from(Span::styled(
                "Run an agent file edit before opening /hunks.",
                theme.typography.dim,
            ))];
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(origin_label(hunk.origin), origin_style(hunk.origin, theme)),
            Span::raw("  "),
            Span::styled(status_label(hunk.status), status_style(hunk.status, theme)),
            Span::raw("  "),
            Span::styled(
                format!(
                    "@@ -{},{} +{},{} @@",
                    hunk.old_start, hunk.old_lines, hunk.new_start, hunk.new_lines
                ),
                theme.typography.dim,
            ),
        ])];
        for line in &hunk.lines {
            let (prefix, style) = match line.kind {
                AttributionLineKind::Context => (" ", theme.typography.body),
                AttributionLineKind::Removed => ("-", theme.diff_del_style()),
                AttributionLineKind::Added => ("+", theme.diff_add_style()),
            };
            lines.push(Line::from(Span::styled(
                format!("{prefix}{}", line.text),
                style,
            )));
        }
        lines
    }
}

fn origin_label(origin: AttributionOrigin) -> String {
    match origin {
        AttributionOrigin::Agent { turn_index } => format!("[agent t{turn_index}]"),
        AttributionOrigin::Human => "[human]".to_string(),
    }
}

fn origin_style(origin: AttributionOrigin, theme: &Theme) -> Style {
    match origin {
        AttributionOrigin::Agent { .. } => Style::new().fg(theme.palette.info),
        AttributionOrigin::Human => Style::new().fg(theme.palette.violet),
    }
}

const fn status_label(status: AttributionStatus) -> &'static str {
    match status {
        AttributionStatus::Pending => "pending",
        AttributionStatus::Accepted => "accepted",
        AttributionStatus::Rejected => "rejected",
        AttributionStatus::Stale => "stale",
    }
}

fn status_style(status: AttributionStatus, theme: &Theme) -> Style {
    match status {
        AttributionStatus::Pending => theme.typography.dim,
        AttributionStatus::Accepted => Style::new().fg(theme.palette.success),
        AttributionStatus::Rejected => Style::new().fg(theme.palette.error),
        AttributionStatus::Stale => Style::new().fg(theme.palette.warn),
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use tools::{
        AttributionLine, AttributionLineKind, AttributionOrigin, AttributionStatus,
        AttributedHunk, HunkAttributionLedger, ToolContext,
    };

    use super::ReviewModal;
    use crate::tui::theme::Theme;

    fn line(kind: AttributionLineKind, text: &str) -> AttributionLine {
        AttributionLine {
            kind,
            text: text.to_string(),
        }
    }

    fn sample() -> ReviewModal {
        let agent = AttributedHunk::new(
            "src/lib.rs",
            10,
            3,
            10,
            3,
            vec![
                line(AttributionLineKind::Context, "fn run() {"),
                line(AttributionLineKind::Removed, "    old();"),
                line(AttributionLineKind::Added, "    new();"),
                line(AttributionLineKind::Context, "}"),
            ],
            AttributionOrigin::Agent { turn_index: 4 },
        );
        let mut human = AttributedHunk::new(
            "README.md",
            2,
            1,
            2,
            2,
            vec![
                line(AttributionLineKind::Context, "Intro"),
                line(AttributionLineKind::Added, "Human note"),
            ],
            AttributionOrigin::Human,
        );
        human.status = AttributionStatus::Accepted;
        ReviewModal::new(
            ToolContext::new(),
            HunkAttributionLedger::from_hunks(vec![agent, human]),
        )
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn dump(term: &Terminal<TestBackend>) -> String {
        let buffer = term.backend().buffer();
        let mut output = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                output.push_str(buffer[(x, y)].symbol());
            }
            output.push('\n');
        }
        output
    }

    #[test]
    fn arrows_and_jk_navigate_hunks() {
        let mut modal = sample();
        assert_eq!(modal.selected_ledger_index(), Some(1));
        modal.handle_key(press(KeyCode::Char('j')));
        assert_eq!(modal.selected_ledger_index(), Some(0));
        modal.handle_key(press(KeyCode::Up));
        assert_eq!(modal.selected_ledger_index(), Some(1));
    }

    #[test]
    fn hunks_render_dump_shows_files_origins_status_diff_and_footer() {
        let theme = Theme::no_color();
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).expect("backend");
        let modal = sample();
        terminal
            .draw(|frame| modal.draw(frame, Rect::new(0, 0, 100, 24), &theme))
            .expect("draw");
        let rendered = dump(&terminal);

        assert!(rendered.contains("/hunks attribution"), "{rendered}");
        assert!(rendered.contains("README.md"), "{rendered}");
        assert!(rendered.contains("src/lib.rs"), "{rendered}");
        assert!(rendered.contains("[agent t4]"), "{rendered}");
        assert!(rendered.contains("[human]"), "{rendered}");
        assert!(rendered.contains("accepted"), "{rendered}");
        assert!(rendered.contains("+Human note"), "{rendered}");
        assert!(rendered.contains("a accept"), "{rendered}");
        assert!(rendered.contains("r reject"), "{rendered}");
        assert!(rendered.contains("A accept file"), "{rendered}");
    }
}
