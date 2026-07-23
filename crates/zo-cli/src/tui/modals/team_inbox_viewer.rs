//! `TeamInbox` viewer modal — list/detail over session-scoped inbox updates.
//!
//! The modal is deliberately fed by a snapshot from the host instead of reading
//! the store itself, so tests can inject rows and the app controls refresh/ack
//! side effects at the runtime boundary.

use std::time::{SystemTime, UNIX_EPOCH};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph, Wrap};
use runtime::{TeamInboxSnapshot, TeamInboxSnapshotRow};

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::theme::Theme;
use super::draw_scrollbar;
use super::workflow_viewer::visible_offset;

/// Outcome of a key press the `TeamInbox` modal handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeamInboxViewerAction {
    /// Close the viewer and return to Normal.
    Close,
    /// Re-read the snapshot from the runtime store.
    Refresh,
    /// Ack the selected update id.
    Ack(String),
    /// Insert a safe summary reference into the composer.
    Include(String),
}

/// List/detail viewer over a [`TeamInboxSnapshot`].
pub struct TeamInboxViewerModal {
    snapshot: TeamInboxSnapshot,
    selected: usize,
    list_scroll: u16,
    detail_scroll: u16,
}

impl TeamInboxViewerModal {
    #[must_use]
    pub fn new(snapshot: TeamInboxSnapshot) -> Self {
        Self { snapshot, selected: 0, list_scroll: 0, detail_scroll: 0 }
    }

    /// Feed a fresh snapshot, preserving selection by update id across reorder.
    pub fn refresh(&mut self, snapshot: TeamInboxSnapshot) {
        let selected_id = self.selected_row().map(|row| row.id.clone());
        self.snapshot = snapshot;
        if let Some(idx) = selected_id
            .as_deref()
            .and_then(|id| self.snapshot.rows.iter().position(|row| row.id == id))
        {
            self.selected = idx;
        } else {
            self.selected = self.selected.min(self.snapshot.rows.len().saturating_sub(1));
            self.detail_scroll = 0;
        }
    }

    #[must_use]
    pub fn selected_row(&self) -> Option<&TeamInboxSnapshotRow> {
        self.snapshot.rows.get(self.selected)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshot.rows.is_empty()
    }

    fn select_prev(&mut self, step: usize) {
        self.selected = self.selected.saturating_sub(step);
        self.detail_scroll = 0;
    }

    fn select_next(&mut self, step: usize) {
        let max = self.snapshot.rows.len().saturating_sub(1);
        self.selected = self.selected.saturating_add(step).min(max);
        self.detail_scroll = 0;
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<TeamInboxViewerAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(TeamInboxViewerAction::Close);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(TeamInboxViewerAction::Close),
            KeyCode::Char('r') if key.modifiers.is_empty() => Some(TeamInboxViewerAction::Refresh),
            KeyCode::Char('a') if key.modifiers.is_empty() => self
                .selected_row()
                .filter(|row| !is_terminal(row))
                .map(|row| TeamInboxViewerAction::Ack(row.id.clone())),
            KeyCode::Enter => self.selected_row().map(|row| {
                TeamInboxViewerAction::Include(format!(
                    "[TeamInbox {}/{}] {}",
                    row.channel,
                    row.id,
                    single_line(&row.summary)
                ))
            }),
            KeyCode::Up | KeyCode::Char('k') => { self.select_prev(1); None }
            KeyCode::Down | KeyCode::Char('j') => { self.select_next(1); None }
            KeyCode::Home | KeyCode::Char('g') => {
                self.selected = 0;
                self.list_scroll = 0;
                self.detail_scroll = 0;
                None
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.selected = self.snapshot.rows.len().saturating_sub(1);
                self.detail_scroll = 0;
                None
            }
            KeyCode::PageUp => { self.detail_scroll = self.detail_scroll.saturating_sub(10); None }
            KeyCode::PageDown => { self.detail_scroll = self.detail_scroll.saturating_add(10); None }
            _ => None,
        }
    }

    pub fn scroll_list(&mut self, up: bool, rows: u16) {
        if up { self.select_prev(usize::from(rows)); } else { self.select_next(usize::from(rows)); }
    }

    fn list_offset(&self, height: u16) -> u16 {
        let max_scroll = u16::try_from(self.snapshot.rows.len())
            .unwrap_or(u16::MAX)
            .saturating_sub(height);
        let selected = u16::try_from(self.selected).unwrap_or(u16::MAX);
        visible_offset(self.list_scroll.min(max_scroll), selected, height).min(max_scroll)
    }

    fn layout(area: Rect, theme: &Theme) -> Option<TeamInboxViewerLayout> {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::raw(" Team Inbox "))
            .padding(Padding::symmetric(1, 0))
            .block()
            .inner(area);
        if inner.height < 4 || inner.width == 0 { return None; }
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ]).areas(inner);
        let (list_area, detail_area) = if body.width >= 120 {
            let [list, detail] = Layout::horizontal([Constraint::Length(66), Constraint::Min(24)]).areas(body);
            (list, detail)
        } else {
            let [list, detail] = Layout::vertical([Constraint::Percentage(50), Constraint::Min(4)]).areas(body);
            (list, detail)
        };
        let list_inner = CardFrame::new(SurfaceKind::Panel, theme).block().inner(list_area);
        let detail_inner = CardFrame::new(SurfaceKind::Panel, theme).block().inner(detail_area);
        Some(TeamInboxViewerLayout { header, list_area, list_inner, detail_area, detail_inner, footer })
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(" Team Inbox ", theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 { return; }
        let Some(regions) = Self::layout(area, theme) else { return; };
        frame.render_widget(Paragraph::new(self.header_line(theme)), regions.header);
        self.draw_list(frame, regions.list_area, regions.list_inner, theme);
        self.draw_detail(frame, regions.detail_area, regions.detail_inner, theme);
        frame.render_widget(Paragraph::new(footer_line(theme)), regions.footer);
    }

    fn header_line(&self, theme: &Theme) -> Line<'static> {
        let channels = if self.snapshot.joined_channels.is_empty() {
            "none".to_string()
        } else {
            self.snapshot.joined_channels.join(", ")
        };
        Line::from(vec![
            Span::styled(format!("{} unread", self.snapshot.unread), Style::new().fg(theme.palette.accent)),
            Span::styled(format!("  ·  {} updates  ·  joined: {channels}", self.snapshot.rows.len()), theme.typography.dim),
        ])
    }

    fn draw_list(&self, frame: &mut Frame<'_>, area: Rect, inner: Rect, theme: &Theme) {
        let title = format!(" updates · {}/{} ", self.selected.saturating_add(1).min(self.snapshot.rows.len()), self.snapshot.rows.len());
        CardFrame::new(SurfaceKind::Panel, theme).title(Line::styled(title, theme.typography.dim)).render(frame, area);
        if inner.height == 0 || inner.width == 0 { return; }
        if self.snapshot.rows.is_empty() {
            frame.render_widget(Paragraph::new(Line::from(Span::styled("no TeamInbox updates for this session", theme.typography.dim))), inner);
            return;
        }
        let lines = self.snapshot.rows.iter().enumerate().map(|(idx, row)| row_line(row, idx == self.selected, theme)).collect::<Vec<_>>();
        let offset = self.list_offset(inner.height);
        frame.render_widget(Paragraph::new(lines).scroll((offset, 0)), inner);
        draw_scrollbar(frame, inner, offset, self.snapshot.rows.len(), theme);
    }

    fn draw_detail(&self, frame: &mut Frame<'_>, area: Rect, inner: Rect, theme: &Theme) {
        let title = self.selected_row().map_or_else(|| " details ".to_string(), |row| format!(" details · {} ", row.id));
        CardFrame::new(SurfaceKind::Panel, theme).title(Line::styled(title, theme.typography.dim)).render(frame, area);
        if inner.height == 0 || inner.width == 0 { return; }
        let Some(row) = self.selected_row() else {
            frame.render_widget(Paragraph::new(Line::from(Span::styled("select an update", theme.typography.dim))), inner);
            return;
        };
        let lines = detail_lines(row, theme);
        let max_scroll = u16::try_from(lines.len()).unwrap_or(u16::MAX).saturating_sub(inner.height.max(1));
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((self.detail_scroll.min(max_scroll), 0)), inner);
    }
}

struct TeamInboxViewerLayout {
    header: Rect,
    list_area: Rect,
    list_inner: Rect,
    detail_area: Rect,
    detail_inner: Rect,
    footer: Rect,
}

fn row_line(row: &TeamInboxSnapshotRow, selected: bool, theme: &Theme) -> Line<'static> {
    let style = if selected { Style::new().fg(theme.palette.accent) } else { theme.typography.body };
    let retries = if row.retry_count > 0 { format!(" retry×{}", row.retry_count) } else { String::new() };
    Line::from(vec![Span::styled(
        format!(
            "{}  {:<6} #{:<10} {:<12} {:>8}  {}{}",
            state_label(row),
            row.priority,
            row.channel,
            row.source,
            age_label(row.created_at_unix),
            single_line(&row.summary),
            retries,
        ),
        style,
    )])
}

fn detail_lines(row: &TeamInboxSnapshotRow, theme: &Theme) -> Vec<Line<'static>> {
    let dim = theme.typography.dim;
    let body = theme.typography.body;
    [
        ("state", state_label(row)),
        ("priority", row.priority.clone()),
        ("channel", row.channel.clone()),
        ("source", row.source.clone()),
        ("age", age_label(row.created_at_unix)),
        ("created_at_unix", row.created_at_unix.to_string()),
        ("retry_count", row.retry_count.to_string()),
        ("task_id", row.task_id.clone().unwrap_or_else(|| "—".to_string())),
        ("status", row.status.clone().unwrap_or_else(|| "—".to_string())),
        ("id", row.id.clone()),
        ("seq", row.seq.to_string()),
        ("summary", single_line(&row.summary)),
    ]
    .into_iter()
    .map(|(key, value)| Line::from(vec![Span::styled(format!("{key:<16}"), dim), Span::styled(value, body)]))
    .collect()
}

fn state_label(row: &TeamInboxSnapshotRow) -> String {
    match row.delivery_state.as_deref() {
        Some("failed") => "✗ failed".to_string(),
        Some("stale") => "⚠ stale".to_string(),
        Some("injected") => "○ injected".to_string(),
        Some("acked") => "✓ acked".to_string(),
        _ => "● unread".to_string(),
    }
}

fn is_terminal(row: &TeamInboxSnapshotRow) -> bool {
    matches!(row.delivery_state.as_deref(), Some("acked" | "stale"))
}

fn single_line(value: &str) -> String {
    value.replace(['\r', '\n'], " ")
}

fn age_label(created_at_unix: i64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
        .unwrap_or(created_at_unix);
    let secs = now.saturating_sub(created_at_unix).max(0);
    if secs < 60 { format!("{secs}s") }
    else if secs < 3600 { format!("{}m", secs / 60) }
    else if secs < 86_400 { format!("{}h", secs / 3600) }
    else { format!("{}d", secs / 86_400) }
}

fn footer_line(theme: &Theme) -> Line<'static> {
    super::key_hint_footer(
        theme,
        &[("Enter", "include"), ("a", "ack"), ("r", "refresh"), ("Esc", "close")],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyModifiers};
    use ratatui::{Terminal, backend::TestBackend};

    fn row(id: &str, state: Option<&str>) -> TeamInboxSnapshotRow {
        TeamInboxSnapshotRow {
            seq: 1,
            id: id.to_string(),
            channel: "ci".to_string(),
            source: "agent".to_string(),
            created_at_unix: 1,
            priority: "high".to_string(),
            summary: format!("summary {id}"),
            delivery_state: state.map(ToOwned::to_owned),
            retry_count: u32::from(state == Some("failed")),
            task_id: Some("task-1".to_string()),
            status: Some("found".to_string()),
        }
    }

    fn snapshot(rows: Vec<TeamInboxSnapshotRow>) -> TeamInboxSnapshot {
        TeamInboxSnapshot { joined_channels: vec!["ci".to_string()], rows, unread: 2 }
    }

    fn press(code: KeyCode) -> KeyEvent { KeyEvent::new(code, KeyModifiers::NONE) }

    #[test]
    fn refresh_preserves_selection_by_update_id() {
        let mut modal = TeamInboxViewerModal::new(snapshot(vec![row("a", None), row("b", Some("injected"))]));
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));
        modal.refresh(snapshot(vec![row("c", Some("failed")), row("b", Some("acked")), row("a", None)]));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));
        modal.refresh(snapshot(vec![row("c", Some("failed"))]));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));
    }

    #[test]
    fn enter_include_and_ack_actions() {
        let mut modal = TeamInboxViewerModal::new(snapshot(vec![row("a", None)]));
        assert_eq!(modal.handle_key(press(KeyCode::Enter)), Some(TeamInboxViewerAction::Include("[TeamInbox ci/a] summary a".to_string())));
        assert_eq!(modal.handle_key(press(KeyCode::Char('a'))), Some(TeamInboxViewerAction::Ack("a".to_string())));
        modal.refresh(snapshot(vec![row("a", Some("acked"))]));
        assert_eq!(modal.handle_key(press(KeyCode::Char('a'))), None);
    }

    #[test]
    fn enter_include_single_lines_multiline_summaries() {
        let mut multiline = row("a", None);
        multiline.summary = "line one\nline two".to_string();
        let mut modal = TeamInboxViewerModal::new(snapshot(vec![multiline]));
        assert_eq!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(TeamInboxViewerAction::Include(
                "[TeamInbox ci/a] line one line two".to_string()
            )),
            "composer include text must never carry raw newlines"
        );
    }

    #[test]
    fn render_contains_all_state_labels_and_footer() {
        let modal = TeamInboxViewerModal::new(snapshot(vec![
            row("failed", Some("failed")),
            row("stale", Some("stale")),
            row("pending", None),
            row("injected", Some("injected")),
            row("acked", Some("acked")),
        ]));
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal.draw(|frame| modal.draw(frame, frame.area(), &Theme::zo())).expect("draw");
        let rendered = format!("{:?}", terminal.backend().buffer());
        for expected in ["✗ failed", "⚠ stale", "● unread", "○ injected", "✓ acked", "Enter", "include", "a", "ack", "r", "refresh", "Esc", "close"] {
            assert!(rendered.contains(expected), "missing {expected:?} in {rendered}");
        }
    }
}
