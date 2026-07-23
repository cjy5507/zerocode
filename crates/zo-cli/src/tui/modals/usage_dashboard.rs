//! Graphical `/usage` dashboard modal.
//!
//! The modal owns only UI state (active tab and selected row) plus an immutable
//! precomputed snapshot. It performs no file I/O and does not mutate runtime
//! usage counters, so drawing stays deterministic and cheap.

use core_types::{UsageDashboardSnapshot, UsageModelRow, UsagePeriodRow, format_usd};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::super::cards::{CardFrame, SurfaceKind};

use super::{key_hint_footer, selected_style};
use crate::tui::theme::Theme;

const TAB_COUNT: usize = 4;

/// User action emitted by [`UsageDashboardModal::handle_key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageDashboardAction {
    /// Close the modal.
    Close,
}

/// Dashboard tabs available inside the single `/usage` popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageDashboardTab {
    /// Daily usage trend.
    Daily,
    /// Monthly usage trend.
    Monthly,
    /// Model-share breakdown.
    Models,
    /// Estimated savings breakdown.
    Savings,
}

impl UsageDashboardTab {
    const ALL: [Self; TAB_COUNT] = [Self::Daily, Self::Monthly, Self::Models, Self::Savings];

    const fn label(self) -> &'static str {
        match self {
            Self::Daily => "Daily",
            Self::Monthly => "Monthly",
            Self::Models => "Models",
            Self::Savings => "Savings",
        }
    }
}

/// Stateful modal wrapper for the `/usage` dashboard.
#[derive(Debug, Clone)]
pub struct UsageDashboardModal {
    snapshot: UsageDashboardSnapshot,
    tab: UsageDashboardTab,
    selected: usize,
}

impl UsageDashboardModal {
    /// Create a new dashboard modal over a precomputed usage snapshot.
    #[must_use]
    pub const fn new(snapshot: UsageDashboardSnapshot) -> Self {
        Self {
            snapshot,
            tab: UsageDashboardTab::Daily,
            selected: 0,
        }
    }

    /// Active tab, exposed for focused tests.
    #[must_use]
    pub const fn active_tab(&self) -> UsageDashboardTab {
        self.tab
    }

    /// Selected row in the active tab, exposed for focused tests.
    #[must_use]
    pub const fn selected(&self) -> usize {
        self.selected
    }

    /// Handle modal navigation keys.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<UsageDashboardAction> {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(UsageDashboardAction::Close),
            KeyCode::Tab | KeyCode::Right => {
                self.next_tab();
                None
            }
            KeyCode::BackTab | KeyCode::Left => {
                self.prev_tab();
                None
            }
            KeyCode::Char('d' | 'D') => {
                self.set_tab(UsageDashboardTab::Daily);
                None
            }
            KeyCode::Char('m' | 'M') => {
                self.set_tab(UsageDashboardTab::Monthly);
                None
            }
            KeyCode::Char('o' | 'O') => {
                self.set_tab(UsageDashboardTab::Models);
                None
            }
            KeyCode::Char('s' | 'S') => {
                self.set_tab(UsageDashboardTab::Savings);
                None
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(UsageDashboardAction::Close)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_selection(1);
                None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_selection(-1);
                None
            }
            _ => None,
        }
    }

    /// Render the dashboard.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        render_dashboard(self, frame, area, theme);
    }

    fn set_tab(&mut self, tab: UsageDashboardTab) {
        self.tab = tab;
        self.selected = self.selected.min(self.row_count().saturating_sub(1));
    }

    fn next_tab(&mut self) {
        let idx = UsageDashboardTab::ALL
            .iter()
            .position(|tab| *tab == self.tab)
            .unwrap_or(0);
        self.set_tab(UsageDashboardTab::ALL[(idx + 1) % TAB_COUNT]);
    }

    fn prev_tab(&mut self) {
        let idx = UsageDashboardTab::ALL
            .iter()
            .position(|tab| *tab == self.tab)
            .unwrap_or(0);
        self.set_tab(UsageDashboardTab::ALL[(idx + TAB_COUNT - 1) % TAB_COUNT]);
    }

    fn move_selection(&mut self, delta: isize) {
        let rows = self.row_count();
        if rows == 0 {
            self.selected = 0;
            return;
        }
        let current = isize::try_from(self.selected).unwrap_or(0);
        let max = isize::try_from(rows.saturating_sub(1)).unwrap_or(0);
        self.selected = usize::try_from((current + delta).clamp(0, max)).unwrap_or(0);
    }

    fn row_count(&self) -> usize {
        match self.tab {
            UsageDashboardTab::Daily => self.snapshot.daily.len(),
            UsageDashboardTab::Monthly => self.snapshot.monthly.len(),
            UsageDashboardTab::Models => self.snapshot.models.len(),
            UsageDashboardTab::Savings => 4,
        }
    }
}

fn render_dashboard(
    modal: &UsageDashboardModal,
    frame: &mut Frame<'_>,
    area: Rect,
    theme: &Theme,
) {
    let inner = CardFrame::new(SurfaceKind::Modal, theme)
        .title(Line::from(vec![
            Span::styled(" Usage Dashboard ", theme.typography.heading_1),
            Span::styled("/usage ", theme.typography.dim),
        ]))
        .render(frame, area);

    if inner.width < 24 || inner.height < 7 {
        frame.render_widget(
            Paragraph::new("Usage dashboard needs a larger terminal")
                .style(theme.typography.dim)
                .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .split(inner);

    render_kpis(&modal.snapshot, frame, chunks[0], theme);
    render_tabs(modal.tab, frame, chunks[1], theme);
    match modal.tab {
        UsageDashboardTab::Daily => render_period_rows(
            "Daily estimate",
            &modal.snapshot.daily,
            modal.selected,
            frame,
            chunks[2],
            theme,
        ),
        UsageDashboardTab::Monthly => render_period_rows(
            "Monthly estimate",
            &modal.snapshot.monthly,
            modal.selected,
            frame,
            chunks[2],
            theme,
        ),
        UsageDashboardTab::Models => render_model_rows(
            &modal.snapshot.models,
            modal.selected,
            frame,
            chunks[2],
            theme,
        ),
        UsageDashboardTab::Savings => render_savings(&modal.snapshot, frame, chunks[2], theme),
    }
    render_footer(&modal.snapshot.note, frame, chunks[3], theme);
}

fn render_kpis(snapshot: &UsageDashboardSnapshot, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);
    let tokens = compact_tokens(snapshot.total_tokens);
    let saved = format_usd(snapshot.savings.total_savings_usd);
    let top_model = snapshot
        .models
        .first()
        .map_or(snapshot.model.as_str(), |row| row.model.as_str());
    let cards = [
        ("Tokens", tokens, format!("{} turns", snapshot.turns)),
        (
            "Cost",
            format_usd(snapshot.total_cost_usd),
            "estimated".to_string(),
        ),
        ("Saved", saved, "cache + mix".to_string()),
        ("Top model", truncate(top_model, 18), "current".to_string()),
    ];
    for (idx, (label, value, hint)) in cards.into_iter().enumerate() {
        let style = if idx == 2 {
            Style::new()
                .fg(theme.palette.success)
                .add_modifier(Modifier::BOLD)
        } else {
            theme.typography.heading_2
        };
        let lines = vec![
            Line::from(Span::styled(label.to_string(), theme.typography.dim)),
            Line::from(Span::styled(value, style)),
            Line::from(Span::styled(hint, theme.typography.key_hint)),
        ];
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), cols[idx]);
    }
}

fn render_tabs(tab: UsageDashboardTab, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    let mut spans = Vec::with_capacity(TAB_COUNT * 3);
    for (idx, item) in UsageDashboardTab::ALL.iter().enumerate() {
        if idx > 0 {
            spans.push(Span::styled("  ", theme.typography.dim));
        }
        let active = *item == tab;
        let style = if active {
            selected_style(theme)
        } else {
            theme.typography.dim
        };
        let label = if active {
            format!("▰ {} ", item.label())
        } else {
            format!("  {} ", item.label())
        };
        spans.push(Span::styled(label, style));
    }
    let line = Line::from(spans);
    frame.render_widget(Paragraph::new(vec![line, divider_line(area.width, theme)]), area);
}

fn render_period_rows(
    title: &str,
    rows: &[UsagePeriodRow],
    selected: usize,
    frame: &mut Frame<'_>,
    area: Rect,
    theme: &Theme,
) {
    let max_tokens = rows.iter().map(|row| row.tokens).max().unwrap_or(1);
    let visible_rows = usize::from(area.height.saturating_sub(2)).max(1);
    let (start, end) = visible_window(rows.len(), selected, visible_rows);
    let mut lines = Vec::with_capacity(end.saturating_sub(start) + 2);
    let title = if rows.len() > visible_rows {
        format!("{title} · showing {}-{} of {}", start + 1, end, rows.len())
    } else {
        title.to_string()
    };
    lines.push(Line::from(Span::styled(title, theme.typography.bold)));
    lines.push(Line::from(vec![
        Span::styled("Period       ", theme.typography.dim),
        Span::styled("Trend              ", theme.typography.dim),
        Span::styled("Tokens     Cost       Saved      Top model", theme.typography.dim),
    ]));
    for (idx, row) in rows.iter().enumerate().skip(start).take(end.saturating_sub(start)) {
        let is_selected = idx == selected;
        let base = if is_selected {
            selected_style(theme)
        } else {
            theme.typography.body
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{:<12}", truncate(&row.label, 12)), base),
            Span::styled(
                format!("{:<19}", usage_bar(row.tokens, max_tokens, 16)),
                Style::new().fg(theme.palette.accent),
            ),
            Span::styled(format!("{:<10}", compact_tokens(row.tokens)), base),
            Span::styled(format!("{:<11}", format_usd(row.cost_usd)), base),
            Span::styled(
                format!("{:<11}", format_usd(row.saved_usd)),
                Style::new().fg(theme.palette.success),
            ),
            Span::styled(truncate(&row.top_model, 20), theme.typography.dim),
        ]));
    }
    if rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "No usage recorded yet.",
            theme.typography.placeholder,
        )));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
}

fn render_model_rows(
    rows: &[UsageModelRow],
    selected: usize,
    frame: &mut Frame<'_>,
    area: Rect,
    theme: &Theme,
) {
    let visible_rows = usize::from(area.height.saturating_sub(3)).max(1);
    let (start, end) = visible_window(rows.len(), selected, visible_rows);
    let mut lines = Vec::with_capacity(end.saturating_sub(start) + 3);
    let title = if rows.len() > visible_rows {
        format!("Model cost chart · showing {}-{} of {}", start + 1, end, rows.len())
    } else {
        "Model cost chart".to_string()
    };
    lines.push(Line::from(vec![
        Span::styled(title, theme.typography.bold),
        Span::styled("  cost-ranked", theme.typography.dim),
    ]));
    if let Some(top) = rows.first() {
        lines.push(Line::from(vec![
            Span::styled("Top driver ", theme.typography.dim),
            Span::styled(truncate(&top.model, 24), theme.typography.bold),
            Span::styled(
                format!(" · {} · {} tokens", format_usd(top.cost_usd), compact_tokens(top.tokens)),
                theme.typography.dim,
            ),
        ]));
    }
    lines.push(Line::from(Span::styled(
        "#  Model                  Token share          Tokens     Cost       Saved",
        theme.typography.dim,
    )));
    for (idx, row) in rows.iter().enumerate().skip(start).take(end.saturating_sub(start)) {
        let base = if idx == selected {
            selected_style(theme)
        } else {
            theme.typography.body
        };
        let rank_style = if idx == 0 {
            Style::new().fg(theme.palette.success).add_modifier(Modifier::BOLD)
        } else {
            theme.typography.dim
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{:>2} ", idx + 1), rank_style),
            Span::styled(format!("{:<22}", truncate(&row.model, 22)), base),
            Span::styled(
                format!("{:<21}", percent_bar(row.share, 14)),
                Style::new().fg(theme.palette.accent),
            ),
            Span::styled(format!("{:<10}", compact_tokens(row.tokens)), base),
            Span::styled(format!("{:<11}", format_usd(row.cost_usd)), theme.typography.bold),
            Span::styled(format_usd(row.saved_usd), Style::new().fg(theme.palette.success)),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
}

fn render_savings(
    snapshot: &UsageDashboardSnapshot,
    frame: &mut Frame<'_>,
    area: Rect,
    theme: &Theme,
) {
    let savings = &snapshot.savings;
    let max = savings
        .baseline_cost_usd
        .max(savings.actual_cost_usd)
        .max(savings.total_savings_usd)
        .max(0.000_001);
    let lines = vec![
        Line::from(Span::styled("Savings summary", theme.typography.bold)),
        Line::from(Span::styled(
            "Estimated from token usage and model pricing; provider invoices may differ.",
            theme.typography.dim,
        )),
        Line::from(""),
        savings_line(
            "Actual cost",
            savings.actual_cost_usd,
            max,
            theme.palette.accent,
            theme,
        ),
        savings_line(
            "Baseline cost",
            savings.baseline_cost_usd,
            max,
            theme.palette.warn,
            theme,
        ),
        savings_line(
            "Cache savings",
            savings.cache_savings_usd,
            max,
            theme.palette.success,
            theme,
        ),
        savings_line(
            "Model mix",
            savings.model_mix_savings_usd,
            max,
            theme.palette.success,
            theme,
        ),
        savings_line(
            "Total saved",
            savings.total_savings_usd,
            max,
            theme.palette.success,
            theme,
        ),
    ];
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
}

fn savings_line(
    label: &str,
    value: f64,
    max: f64,
    color: ratatui::style::Color,
    theme: &Theme,
) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<15}"), theme.typography.body),
        Span::styled(format!("{:<11}", format_usd(value)), theme.typography.bold),
        Span::styled(ratio_bar(value / max, 22), Style::new().fg(color)),
    ])
}

fn render_footer(note: &str, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    let footer = key_hint_footer(
        theme,
        &[
            ("Tab/←→", "tabs"),
            ("↑↓", "rows"),
            ("d/m/o/s", "views"),
            ("Esc/q", "close"),
        ],
    );
    let compact_note = if note.len() > 72 {
        "Historical estimates · session-level dates/models; ledger pending".to_string()
    } else {
        note.to_string()
    };
    let note_line = if area.width > 88 {
        Line::from(Span::styled(note.to_string(), theme.typography.dim))
    } else {
        Line::from(Span::styled(compact_note, theme.typography.dim))
    };
    let text = vec![note_line, Line::default(), footer];
    frame.render_widget(Paragraph::new(text), area);
}

fn divider_line(width: u16, theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        "─".repeat(usize::from(width).max(1)),
        Style::new().fg(theme.palette.muted),
    ))
}

fn visible_window(row_count: usize, selected: usize, visible_rows: usize) -> (usize, usize) {
    if row_count == 0 || visible_rows == 0 {
        return (0, 0);
    }
    let selected = selected.min(row_count.saturating_sub(1));
    let half = visible_rows / 2;
    let max_start = row_count.saturating_sub(visible_rows);
    let start = selected.saturating_sub(half).min(max_start);
    let end = start.saturating_add(visible_rows).min(row_count);
    (start, end)
}

fn usage_bar(value: u64, max: u64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let width_u64 = u64::try_from(width).unwrap_or(u64::MAX);
    let mut out = String::with_capacity(width);
    for cell in 1..=width {
        let cell_u64 = u64::try_from(cell).unwrap_or(u64::MAX);
        let filled = max > 0
            && u128::from(value) * u128::from(width_u64)
                >= u128::from(max) * u128::from(cell_u64);
        out.push(if filled { '█' } else { '░' });
    }
    out
}

fn ratio_bar(ratio: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let ratio = ratio.clamp(0.0, 1.0);
    let width_u32 = u32::try_from(width).unwrap_or(u32::MAX);
    let mut out = String::with_capacity(width);
    for cell in 1..=width {
        let cell_u32 = u32::try_from(cell).unwrap_or(u32::MAX);
        let threshold = f64::from(cell_u32) / f64::from(width_u32);
        out.push(if ratio >= threshold { '█' } else { '░' });
    }
    out
}

fn percent_bar(share: f64, width: usize) -> String {
    let share = share.clamp(0.0, 1.0);
    format!("{} {:>5.1}%", ratio_bar(share, width), share * 100.0)
}

fn compact_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        compact_decimal(tokens, 1_000_000, "M")
    } else if tokens >= 1_000 {
        compact_decimal(tokens, 1_000, "K")
    } else {
        tokens.to_string()
    }
}

fn compact_decimal(value: u64, divisor: u64, suffix: &str) -> String {
    let value = u128::from(value);
    let divisor = u128::from(divisor);
    let whole = value / divisor;
    let tenth = (value % divisor) * 10 / divisor;
    format!("{whole}.{tenth}{suffix}")
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let mut out = String::new();
    for _ in 0..max_chars {
        let Some(ch) = chars.next() else {
            return out;
        };
        out.push(ch);
    }
    if chars.next().is_some() && max_chars > 1 {
        out.pop();
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::TokenUsage;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn snapshot() -> UsageDashboardSnapshot {
        UsageDashboardSnapshot::from_session(
            "gpt-5.5",
            TokenUsage {
                input_tokens: 10_000,
                output_tokens: 2_000,
                cache_creation_input_tokens: 500,
                cache_read_input_tokens: 20_000,
            },
            2,
        )
    }

    #[test]
    fn usage_bar_preserves_large_u64_ratios() {
        let bar = usage_bar(u64::MAX / 2, u64::MAX, 16);
        let filled = bar.chars().filter(|ch| *ch == '█').count();
        assert!((7..=9).contains(&filled), "bar={bar}");
    }

    #[test]
    fn visible_window_keeps_selected_row_in_view() {
        assert_eq!(visible_window(20, 0, 5), (0, 5));
        assert_eq!(visible_window(20, 10, 5), (8, 13));
        assert_eq!(visible_window(20, 19, 5), (15, 20));
        assert_eq!(visible_window(3, 2, 10), (0, 3));
    }

    #[test]
    fn tab_keys_switch_views_without_allocating_runtime_state() {
        let mut modal = UsageDashboardModal::new(snapshot());
        assert_eq!(modal.active_tab(), UsageDashboardTab::Daily);
        modal.handle_key(KeyEvent::from(KeyCode::Tab));
        assert_eq!(modal.active_tab(), UsageDashboardTab::Monthly);
        modal.handle_key(KeyEvent::from(KeyCode::Char('o')));
        assert_eq!(modal.active_tab(), UsageDashboardTab::Models);
        modal.handle_key(KeyEvent::from(KeyCode::Char('s')));
        assert_eq!(modal.active_tab(), UsageDashboardTab::Savings);
    }

    #[test]
    fn esc_closes_modal() {
        let mut modal = UsageDashboardModal::new(snapshot());
        assert_eq!(
            modal.handle_key(KeyEvent::from(KeyCode::Esc)),
            Some(UsageDashboardAction::Close)
        );
    }

    #[test]
    fn render_contains_graphical_dashboard_sections() {
        let theme = Theme::zo();
        let modal = UsageDashboardModal::new(snapshot());
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| modal.draw(frame, frame.area(), &theme))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        let mut text = String::new();
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                text.push_str(buffer.cell((x, y)).unwrap().symbol());
            }
            text.push('\n');
        }
        assert!(text.contains("Usage Dashboard"));
        assert!(text.contains("Daily"));
        assert!(text.contains("Saved"));
        assert!(text.contains("█"));
    }
}
