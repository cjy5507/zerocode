//! Ctrl+G agents viewer — the structured replacement for the old raw-text
//! agents pager. A flat, session-scoped list of every sub-agent (running AND
//! finished — no live gate) beside a live detail pane, with keyboard/mouse
//! selection that survives refreshes by agent id.
//!
//! Data comes from [`workflow_progress::read_agent_rows_since`]; the row and
//! detail renderers are shared with the Ctrl+O workflow viewer
//! ([`agent_list_line`] / [`agent_detail_body_lines`]) so a fleet reads
//! identically on both surfaces.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph, Wrap};

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::theme::Theme;
use super::super::workflow_progress::AgentRowsSnapshot;
use super::draw_scrollbar;
use super::workflow_viewer::{
    WorkflowAgentRow, agent_detail_body_lines, agent_list_line, short, visible_offset,
};

/// Outcome of a key press the modal handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentsViewerAction {
    /// Close the viewer and return to Normal.
    Close,
    /// Send `message` to the agent with id `target` (the selection at Enter
    /// time). The HOST performs the actual delivery — the modal never touches
    /// the agent registry/disk — and reports back via [`AgentsViewerModal::
    /// set_feedback`].
    Send { target: String, message: String },
}

/// Side-by-side (or stacked, when narrow) list + detail view over the
/// session's agent fleet.
pub struct AgentsViewerModal {
    snapshot: AgentRowsSnapshot,
    /// Index into `snapshot.rows` of the highlighted agent.
    selected: usize,
    /// Top row offset of the list pane (kept so the selection stays visible).
    list_scroll: u16,
    /// Scroll offset of the detail pane (long activity/output tails).
    detail_scroll: u16,
    /// When set, the freshness window is off and the whole session's history
    /// is listed. Flipped by `a`; the host re-reads with this flag.
    show_history: bool,
    /// True while a turn is streaming — the empty state then explains that the
    /// list refreshes live instead of reading as broken.
    turn_active: bool,
    /// Spinner phase for running rows (advanced by the host tick).
    tick: usize,
    /// Compose buffer for the message box (`m` opens it on the selected
    /// agent). `None` = browse mode. While `Some`, printable keys type here
    /// instead of navigating.
    input: Option<String>,
    /// One-line result of the last send (host-reported), shown in the footer:
    /// `(text, is_error)`.
    feedback: Option<(String, bool)>,
}

impl AgentsViewerModal {
    #[must_use]
    pub fn new(snapshot: AgentRowsSnapshot) -> Self {
        Self {
            snapshot,
            selected: 0,
            list_scroll: 0,
            detail_scroll: 0,
            show_history: false,
            turn_active: false,
            tick: 0,
            input: None,
            feedback: None,
        }
    }

    /// True while the message box is open — the host must route ALL printable
    /// keys here (its own shortcuts, e.g. the `a` history toggle, included).
    #[must_use]
    pub fn input_active(&self) -> bool {
        self.input.is_some()
    }

    /// Host-reported result of the last [`AgentsViewerAction::Send`].
    pub fn set_feedback(&mut self, text: String, is_error: bool) {
        self.feedback = Some((text, is_error));
    }

    /// Feed a fresh snapshot. Selection is preserved by **agent id**, not by
    /// index — rows shift as agents finish/sort, and the old pager's
    /// line-offset preservation is exactly what made its content slide under
    /// the reader. The detail scroll survives only when the same agent stays
    /// selected.
    pub fn refresh(&mut self, snapshot: AgentRowsSnapshot) {
        let selected_id = self.selected_row().map(|row| row.id.clone());
        self.snapshot = snapshot;
        let found = selected_id
            .as_deref()
            .and_then(|id| self.snapshot.rows.iter().position(|row| row.id == id));
        if let Some(idx) = found {
            self.selected = idx;
        } else {
            self.selected = self
                .selected
                .min(self.snapshot.rows.len().saturating_sub(1));
            self.detail_scroll = 0;
        }
    }

    pub fn set_turn_active(&mut self, active: bool) {
        self.turn_active = active;
    }

    /// Advance the running-row spinner one frame (host redraw tick).
    pub fn advance_spinner(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    #[must_use]
    pub fn selected_row(&self) -> Option<&WorkflowAgentRow> {
        self.snapshot.rows.get(self.selected)
    }

    /// Pre-select the agent with this id (e.g. a clicked pinned-panel row).
    /// Returns `false` on a miss, leaving the selection unchanged.
    pub fn select_agent_by_id(&mut self, id: &str) -> bool {
        if let Some(idx) = self.snapshot.rows.iter().position(|row| row.id == id) {
            self.selected = idx;
            self.detail_scroll = 0;
            return true;
        }
        false
    }

    /// Flip the history window and report the new state; the host re-reads the
    /// snapshot with it (the modal itself never touches the disk).
    pub fn toggle_history(&mut self) -> bool {
        self.show_history = !self.show_history;
        self.detail_scroll = 0;
        self.show_history
    }

    #[must_use]
    pub const fn show_history(&self) -> bool {
        self.show_history
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

    /// Handle a key. `a` (history) is handled by the App layer because it
    /// needs a disk re-read; everything else is local state.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<AgentsViewerAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(AgentsViewerAction::Close);
        }
        if self.input.is_some() {
            return self.handle_compose_key(key);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return Some(AgentsViewerAction::Close),
            KeyCode::Up | KeyCode::Char('k') => self.select_prev(1),
            KeyCode::Down | KeyCode::Char('j') => self.select_next(1),
            // Message box: talk to the selected agent (steer it mid-run, or
            // resume it with context if it already finished).
            KeyCode::Char('m' | 'i') if self.selected_row().is_some() => {
                self.input = Some(String::new());
                self.feedback = None;
            }
            KeyCode::Home | KeyCode::Char('g') => {
                self.selected = 0;
                self.list_scroll = 0;
                self.detail_scroll = 0;
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.selected = self.snapshot.rows.len().saturating_sub(1);
                self.detail_scroll = 0;
            }
            // The detail pane holds the long content (activity feed, output
            // tail), so paging scrolls it; the list is walked with ↑/↓.
            KeyCode::PageUp => self.detail_scroll = self.detail_scroll.saturating_sub(10),
            KeyCode::PageDown => self.detail_scroll = self.detail_scroll.saturating_add(10),
            _ => {}
        }
        None
    }

    /// Keys while the message box is open. Esc cancels the box (NOT the
    /// modal); Enter sends to the agent selected at this moment.
    fn handle_compose_key(&mut self, key: KeyEvent) -> Option<AgentsViewerAction> {
        let input = self.input.as_mut()?;
        match key.code {
            KeyCode::Esc => {
                self.input = None;
            }
            KeyCode::Enter => {
                let message = input.trim().to_string();
                if message.is_empty() {
                    return None;
                }
                let target = self.selected_row()?.id.clone();
                self.input = None;
                return Some(AgentsViewerAction::Send { target, message });
            }
            KeyCode::Backspace => {
                input.pop();
            }
            // Mirror the main composer's acceptance (`!ctrl`): IME-composed
            // characters (e.g. Hangul) can arrive with modifier bits beyond
            // SHIFT depending on the terminal's keyboard protocol, and the
            // old `empty || SHIFT` guard silently dropped them.
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                input.push(ch);
            }
            _ => {}
        }
        None
    }

    /// Insert pasted (or IME-committed — several terminals deliver a composed
    /// syllable as a bracketed paste) text into the open message box. A no-op
    /// while the box is closed, so stray pastes cannot type into the modal.
    pub fn paste_text(&mut self, text: &str) {
        if let Some(input) = self.input.as_mut() {
            // Single-line box: fold line breaks into spaces instead of
            // dropping the paste or smuggling control characters.
            let cleaned = text.replace(['\r', '\n'], " ");
            input.push_str(&cleaned);
        }
    }

    /// Wheel scroll over the modal: moves the list selection like the arrows.
    pub fn scroll_list(&mut self, up: bool, rows: u16) {
        if up {
            self.select_prev(usize::from(rows));
        } else {
            self.select_next(usize::from(rows));
        }
    }

    /// Route a left-click at absolute `(column, row)` given the same modal
    /// `area` the draw used: a click on a list row selects that agent. The
    /// layout is recomputed with the exact same pure math as [`Self::draw`],
    /// so hit-testing can never drift from the pixels.
    pub fn handle_click(&mut self, column: u16, row: u16, area: Rect, theme: &Theme) {
        let Some(regions) = Self::layout(area, theme) else {
            return;
        };
        let list = regions.list_inner;
        if column < list.x
            || column >= list.x.saturating_add(list.width)
            || row < list.y
            || row >= list.y.saturating_add(list.height)
        {
            return;
        }
        let offset = self.list_offset(list.height);
        let idx = usize::from(row - list.y) + usize::from(offset);
        if idx < self.snapshot.rows.len() {
            self.selected = idx;
            self.detail_scroll = 0;
        }
    }

    /// The list pane's top-row offset for a viewport of `height` rows — shared
    /// by draw and click hit-testing.
    fn list_offset(&self, height: u16) -> u16 {
        let max_scroll = u16::try_from(self.snapshot.rows.len())
            .unwrap_or(u16::MAX)
            .saturating_sub(height);
        let selected = u16::try_from(self.selected).unwrap_or(u16::MAX);
        visible_offset(self.list_scroll.min(max_scroll), selected, height).min(max_scroll)
    }

    /// Pure geometry for one frame: the outer card's inner area split into
    /// header / list / detail / footer. `None` when the area is too small to
    /// show anything.
    fn layout(area: Rect, theme: &Theme) -> Option<AgentsViewerLayout> {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::raw(" Agents "))
            .padding(Padding::symmetric(1, 0))
            .block()
            .inner(area);
        if inner.height < 4 || inner.width == 0 {
            return None;
        }
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);
        // Wide terminals put list and detail side by side; narrow ones stack
        // them (mirrors the workflow viewer's responsive split).
        let (list_area, detail_area) = if body.width >= 110 {
            let [list, detail] =
                Layout::horizontal([Constraint::Length(52), Constraint::Min(24)]).areas(body);
            (list, detail)
        } else {
            let [list, detail] =
                Layout::vertical([Constraint::Percentage(45), Constraint::Min(4)]).areas(body);
            (list, detail)
        };
        let list_inner = CardFrame::new(SurfaceKind::Panel, theme).block().inner(list_area);
        let detail_inner = CardFrame::new(SurfaceKind::Panel, theme)
            .block()
            .inner(detail_area);
        Some(AgentsViewerLayout {
            header,
            list_area,
            list_inner,
            detail_area,
            detail_inner,
            footer,
        })
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(" Agents ", theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let Some(regions) = Self::layout(area, theme) else {
            return;
        };

        frame.render_widget(Paragraph::new(self.header_line(theme)), regions.header);
        self.draw_list(frame, regions.list_area, regions.list_inner, theme);
        self.draw_detail(frame, regions.detail_area, regions.detail_inner, theme);
        frame.render_widget(
            Paragraph::new(self.footer_content(theme, regions.footer.width)),
            regions.footer,
        );
    }

    /// Footer: the compose box while typing, the last send's result after,
    /// and the key hints otherwise.
    fn footer_content(&self, theme: &Theme, width: u16) -> Line<'static> {
        if let Some(input) = self.input.as_ref() {
            let target = self
                .selected_row()
                .map_or_else(|| "agent".to_string(), |row| short(&row.name, 24));
            return Line::from(vec![
                Span::styled(
                    format!("✉ to {target} ❯ "),
                    Style::new().fg(theme.palette.accent),
                ),
                Span::styled(input.clone(), theme.typography.body),
                Span::styled("▌", Style::new().fg(theme.palette.accent)),
                Span::styled("  Enter send · Esc cancel", theme.typography.dim),
            ]);
        }
        if let Some((text, is_error)) = self.feedback.as_ref() {
            let color = if *is_error {
                theme.palette.warn
            } else {
                theme.palette.accent
            };
            return Line::from(Span::styled(text.clone(), Style::new().fg(color)));
        }
        footer_line(theme, width)
    }

    /// Header: running/total tally, scope label, and the honest hidden counts.
    fn header_line(&self, theme: &Theme) -> Line<'static> {
        let dim = theme.typography.dim;
        let running = self
            .snapshot
            .rows
            .iter()
            .filter(|row| !matches!(row.status.as_str(), "completed" | "failed" | "stopped"))
            .count();
        let mut spans = vec![
            Span::styled(
                format!("{running} running"),
                Style::new().fg(theme.palette.accent),
            ),
            Span::styled(
                format!("  ·  {} total  ·  ", self.snapshot.rows.len()),
                dim,
            ),
            Span::styled(
                if self.show_history {
                    "session history"
                } else {
                    "this session"
                }
                .to_string(),
                dim,
            ),
        ];
        if self.snapshot.older_hidden > 0 {
            spans.push(Span::styled(
                format!("  ·  +{} older (a)", self.snapshot.older_hidden),
                Style::new().fg(theme.palette.warn),
            ));
        }
        if self.snapshot.capped > 0 {
            spans.push(Span::styled(
                format!("  ·  +{} beyond read cap", self.snapshot.capped),
                Style::new().fg(theme.palette.warn),
            ));
        }
        Line::from(spans)
    }

    fn draw_list(&self, frame: &mut Frame<'_>, area: Rect, inner: Rect, theme: &Theme) {
        let title = format!(
            " agents · {}/{} ",
            self.selected.saturating_add(1).min(self.snapshot.rows.len()),
            self.snapshot.rows.len()
        );
        CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, theme.typography.dim))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        if self.snapshot.rows.is_empty() {
            frame.render_widget(
                Paragraph::new(self.empty_state_lines(theme)).wrap(Wrap { trim: false }),
                inner,
            );
            return;
        }
        let lines: Vec<Line<'static>> = self
            .snapshot
            .rows
            .iter()
            .enumerate()
            .map(|(idx, agent)| agent_list_line(agent, idx == self.selected, theme, self.tick))
            .collect();
        let offset = self.list_offset(inner.height);
        frame.render_widget(Paragraph::new(lines).scroll((offset, 0)), inner);
        draw_scrollbar(frame, inner, offset, self.snapshot.rows.len(), theme);
    }

    fn empty_state_lines(&self, theme: &Theme) -> Vec<Line<'static>> {
        let dim = theme.typography.dim;
        let mut lines = vec![Line::from(Span::styled(
            "no agents this session yet",
            theme.typography.body,
        ))];
        if self.turn_active {
            lines.push(Line::from(Span::styled(
                "refreshes live while the turn runs — agents may still be spawning",
                dim,
            )));
        }
        if !self.show_history {
            lines.push(Line::from(Span::styled(
                "a — include earlier session history",
                dim,
            )));
        }
        lines
    }

    fn draw_detail(&self, frame: &mut Frame<'_>, area: Rect, inner: Rect, theme: &Theme) {
        let title = self.selected_row().map_or_else(
            || " details ".to_string(),
            |agent| format!(" details · {} ", short(&agent.name, 32)),
        );
        CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, theme.typography.dim))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let Some(agent) = self.selected_row() else {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "select an agent",
                    theme.typography.dim,
                ))),
                inner,
            );
            return;
        };
        let lines = agent_detail_body_lines(agent, theme);
        // Clamp so PageDown past the end cannot scroll everything off-screen.
        // Wrapped rows can exceed the raw line count, so this is a floor — the
        // last content line always stays reachable.
        let max_scroll = u16::try_from(lines.len())
            .unwrap_or(u16::MAX)
            .saturating_sub(inner.height.max(1));
        let scroll = self.detail_scroll.min(max_scroll);
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((scroll, 0)),
            inner,
        );
    }
}

struct AgentsViewerLayout {
    header: Rect,
    list_area: Rect,
    list_inner: Rect,
    detail_area: Rect,
    detail_inner: Rect,
    footer: Rect,
}

fn footer_line(theme: &Theme, width: u16) -> Line<'static> {
    let full = super::key_hint_footer(
        theme,
        &[
            ("↑/↓", "agent"),
            ("PgUp/PgDn", "output"),
            ("m", "message"),
            ("a", "history"),
            ("Esc", "close"),
        ],
    );
    if line_width(&full) <= usize::from(width) {
        return full;
    }
    super::key_hint_footer_with_separator(
        theme,
        &[("↑/↓", "agent"), ("m", "message"), ("Esc", "close")],
        " · ",
    )
}

fn line_width(line: &Line<'_>) -> usize {
    use unicode_width::UnicodeWidthStr;
    line.spans
        .iter()
        .map(|span| span.content.as_ref().width())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::Theme;
    use crossterm::event::{KeyEvent, KeyModifiers};

    fn row(id: &str, status: &str) -> WorkflowAgentRow {
        WorkflowAgentRow {
            id: id.to_string(),
            name: id.to_string(),
            status: status.to_string(),
            ..WorkflowAgentRow::default()
        }
    }

    fn snapshot(rows: Vec<WorkflowAgentRow>) -> AgentRowsSnapshot {
        AgentRowsSnapshot {
            rows,
            ..AgentRowsSnapshot::default()
        }
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// 리프레시는 인덱스가 아니라 agent id 로 선택을 보존한다 — 행 순서가
    /// 바뀌어도(에이전트 종결·정렬 이동) 보던 에이전트를 계속 본다. 이것이
    /// 옛 pager 의 "내용 교체 + 라인 스크롤 보존" 밀림의 근본 해결.
    #[test]
    fn refresh_preserves_selection_by_agent_id() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![
            row("a", "running"),
            row("b", "running"),
        ]));
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));

        // `b` finishes and sorts below a new runner: id survives the shuffle.
        modal.refresh(snapshot(vec![
            row("c", "running"),
            row("a", "running"),
            row("b", "completed"),
        ]));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));

        // The selected agent vanished: clamp, don't panic.
        modal.refresh(snapshot(vec![row("c", "running")]));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));
    }

    #[test]
    fn select_agent_by_id_hits_and_misses() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![
            row("a", "running"),
            row("b", "completed"),
        ]));
        assert!(modal.select_agent_by_id("b"));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));
        assert!(!modal.select_agent_by_id("zzz"));
        assert_eq!(
            modal.selected_row().map(|r| r.id.as_str()),
            Some("b"),
            "a miss leaves the selection unchanged"
        );
    }

    #[test]
    fn esc_q_and_ctrl_c_close() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![row("a", "running")]));
        assert_eq!(
            modal.handle_key(press(KeyCode::Esc)),
            Some(AgentsViewerAction::Close)
        );
        assert_eq!(
            modal.handle_key(press(KeyCode::Char('q'))),
            Some(AgentsViewerAction::Close)
        );
        assert_eq!(
            modal.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Some(AgentsViewerAction::Close)
        );
    }

    #[test]
    fn navigation_clamps_at_both_ends() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![
            row("a", "running"),
            row("b", "running"),
        ]));
        modal.handle_key(press(KeyCode::Up));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("a"));
        modal.handle_key(press(KeyCode::Down));
        modal.handle_key(press(KeyCode::Down));
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));
    }

    /// 클릭 히트테스트는 draw 와 같은 순수 layout 을 재계산한다 — list pane
    /// 안의 행 클릭이 그 에이전트를 선택하고, 밖(디테일/보더)은 no-op.
    #[test]
    fn click_selects_list_row_and_ignores_outside() {
        let theme = Theme::default_dark();
        let mut modal = AgentsViewerModal::new(snapshot(vec![
            row("a", "running"),
            row("b", "running"),
            row("c", "running"),
        ]));
        let area = Rect::new(0, 0, 130, 30);
        let regions = AgentsViewerModal::layout(area, &theme).expect("layout fits");
        let list = regions.list_inner;

        modal.handle_click(list.x + 1, list.y + 2, area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));

        // A click below the last row / outside the list changes nothing.
        modal.handle_click(list.x + 1, list.y + list.height.saturating_sub(1), area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));
        modal.handle_click(area.width.saturating_sub(2), list.y + 1, area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));
    }

    #[test]
    fn history_toggle_flips_and_reports() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![]));
        assert!(!modal.show_history());
        assert!(modal.toggle_history());
        assert!(modal.show_history());
        assert!(!modal.toggle_history());
    }

    /// `m` 은 선택된 에이전트를 향한 메시지 박스를 연다: 입력 중엔 탐색 키가
    /// 문자로 들어가고, Esc 는 박스만 닫고(모달 유지), Enter 는 Enter 시점의
    /// 선택 id 를 target 으로 Send 를 돌려준다. 빈 입력은 no-op.
    #[test]
    fn message_box_types_sends_and_cancels() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![
            row("a", "running"),
            row("b", "completed"),
        ]));
        modal.handle_key(press(KeyCode::Down));
        assert!(!modal.input_active());

        modal.handle_key(press(KeyCode::Char('m')));
        assert!(modal.input_active());

        // Navigation chars type into the box instead of moving the selection.
        for ch in ['g', 'o', ' ', 'o', 'n'] {
            assert_eq!(modal.handle_key(press(KeyCode::Char(ch))), None);
        }
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("b"));

        // Backspace edits; Enter sends to the CURRENT selection's id.
        modal.handle_key(press(KeyCode::Backspace));
        for ch in ['n', ' ', 'd', 'e', 'e', 'p', 'e', 'r'] {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        let action = modal.handle_key(press(KeyCode::Enter));
        assert_eq!(
            action,
            Some(AgentsViewerAction::Send {
                target: "b".to_string(),
                message: "go on deeper".to_string(),
            })
        );
        assert!(!modal.input_active(), "the box closes after a send");

        // Esc cancels the box without closing the modal; empty Enter is a no-op.
        modal.handle_key(press(KeyCode::Char('m')));
        assert_eq!(modal.handle_key(press(KeyCode::Enter)), None);
        assert_eq!(modal.handle_key(press(KeyCode::Esc)), None);
        assert!(!modal.input_active());
        // And a plain Esc in browse mode still closes.
        assert_eq!(
            modal.handle_key(press(KeyCode::Esc)),
            Some(AgentsViewerAction::Close)
        );
    }

    #[test]
    fn q_closes_only_in_browse_mode() {
        let mut modal = AgentsViewerModal::new(snapshot(vec![row("a", "running")]));
        modal.handle_key(press(KeyCode::Char('m')));
        assert_eq!(modal.handle_key(press(KeyCode::Char('q'))), None);
        modal.handle_key(press(KeyCode::Esc));
        assert_eq!(
            modal.handle_key(press(KeyCode::Char('q'))),
            Some(AgentsViewerAction::Close)
        );
    }
}
