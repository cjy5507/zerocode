//! Graphical `/usage` dashboard modal.
//!
//! The modal owns only UI state (active tab and selected row) plus an immutable
//! precomputed snapshot. It performs no file I/O and does not mutate runtime
//! usage counters, so drawing stays deterministic and cheap.

use core_types::{
    UsageDashboardSnapshot, UsageModelRow, UsagePeriodRow, UsageSavingsSummary, format_usd,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph};

use super::super::cards::{CardFrame, SurfaceKind};

use super::{key_hint_footer_reflowing, selected_style};
use crate::tui::charts;
use crate::tui::glyphs;
use crate::tui::theme::Theme;

const TAB_COUNT: usize = 4;
/// Cells in the per-period token trend bar. The trend column is 3 cells wider
/// so the gauge never touches the token figure beside it.
const TREND_CELLS: usize = 16;
/// Cells in the per-model token-share bar. The share column is 21 wide, and the
/// trailing ` NN.N% ` readout claims 8 of them — including the closing space
/// that keeps the percentage from colliding with the token figure beside it.
const SHARE_CELLS: usize = 13;
/// Cells in the savings comparison bar, which owns the rest of its row.
const SAVINGS_CELLS: usize = 22;
/// Rows the chart band claims: four braille rows — 32 vertical levels — plus a
/// caption naming the peak and the range it covers.
const CHART_ROWS: u16 = 5;
/// Smallest inner height that can seat the chart band and still leave the table
/// a usable scroll window. Below it the band is dropped whole rather than
/// squeezed, because a two-row chart is decoration and the table is the data.
const MIN_HEIGHT_WITH_CHART: u16 = 17;
/// Smallest inner width that can seat a composition legend whole. The Savings
/// legend is the widest at roughly fifty cells, and below that the band starts
/// hiding entries — a coloured run whose name is off-screen is a key the reader
/// cannot use, so the band goes instead and the table keeps every figure.
const MIN_WIDTH_WITH_CHART: u16 = 56;

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
            Span::styled("Usage Dashboard", super::modal_title_style(theme)),
            Span::styled(" /usage", theme.typography.dim),
        ]))
        .render(frame, area);

    if inner.width < 24 || inner.height < 7 {
        frame.render_widget(
            Paragraph::new(super::wrap_body_rows(
                &[Line::from("Usage dashboard needs a larger terminal")],
                inner.width,
                true,
            ))
            .style(theme.typography.dim),
            inner,
        );
        return;
    }

    // The band is optional on purpose: it is the first thing to go when the
    // terminal is short, since the table carries the numbers and the chart only
    // carries their shape.
    let chart_rows = if inner.height >= MIN_HEIGHT_WITH_CHART && inner.width >= MIN_WIDTH_WITH_CHART
    {
        CHART_ROWS
    } else {
        0
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(2),
            Constraint::Length(chart_rows),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .split(inner);

    render_kpis(&modal.snapshot, frame, chunks[0], theme);
    render_tabs(modal.tab, frame, chunks[1], theme);
    if chart_rows > 0 {
        render_chart_band(modal, frame, chunks[2], theme);
    }
    match modal.tab {
        UsageDashboardTab::Daily => render_period_rows(
            "Daily estimate",
            &modal.snapshot.daily,
            modal.selected,
            frame,
            chunks[3],
            theme,
        ),
        UsageDashboardTab::Monthly => render_period_rows(
            "Monthly estimate",
            &modal.snapshot.monthly,
            modal.selected,
            frame,
            chunks[3],
            theme,
        ),
        UsageDashboardTab::Models => render_model_rows(
            &modal.snapshot.models,
            modal.selected,
            frame,
            chunks[3],
            theme,
        ),
        UsageDashboardTab::Savings => render_savings(&modal.snapshot, frame, chunks[3], theme),
    }
    render_footer(&modal.snapshot.note, frame, chunks[4], theme);
}

/// The dashboard's one loud element: a band above the table showing the SHAPE
/// of spending, which the per-row gauges cannot — they rank rows against each
/// other, never the run of time across them. Everything else on this screen
/// stays deliberately quiet so this is what the eye lands on.
fn render_chart_band(
    modal: &UsageDashboardModal,
    frame: &mut Frame<'_>,
    area: Rect,
    theme: &Theme,
) {
    let lines = match modal.tab {
        UsageDashboardTab::Daily => {
            period_chart(&modal.snapshot.daily, "Tokens per day", area, theme)
        }
        UsageDashboardTab::Monthly => {
            period_chart(&modal.snapshot.monthly, "Tokens per month", area, theme)
        }
        UsageDashboardTab::Models => model_mix_chart(&modal.snapshot.models, area, theme),
        UsageDashboardTab::Savings => savings_chart(&modal.snapshot.savings, area, theme),
    };
    frame.render_widget(Paragraph::new(super::fit_body_rows(lines, area.width)), area);
}

fn period_chart(
    rows: &[UsagePeriodRow],
    title: &str,
    area: Rect,
    theme: &Theme,
) -> Vec<Line<'static>> {
    // The snapshot orders period rows newest-first, which is what the table
    // below wants — today on top — and the reverse of what a time axis needs.
    // Plotted in that order the week runs right-to-left and every trend reads
    // backwards: a week of rising spend draws as a decline.
    let chronological: Vec<&UsagePeriodRow> = rows.iter().rev().collect();
    let series: Vec<u64> = chronological.iter().map(|row| row.tokens).collect();
    // The same denominator `render_period_rows` gives its trend gauges, so the
    // band and the table underneath cannot disagree about the tall bucket.
    let max = series.iter().copied().max().unwrap_or(0);
    let mut lines = charts::braille_area_chart(
        &series,
        max,
        area.width,
        area.height.saturating_sub(1),
        theme.palette.bright,
        theme,
    );
    lines.push(chart_caption(
        &chronological,
        title,
        max,
        area.width,
        theme,
    ));
    lines
}

/// Peak on the left, covered range on the right. The caption is what keeps the
/// band honest on a terminal whose font has no braille: the numbers still read
/// even when the shape above them does not render.
fn chart_caption(
    rows: &[&UsagePeriodRow],
    title: &str,
    max: u64,
    width: u16,
    theme: &Theme,
) -> Line<'static> {
    let range = match (rows.first(), rows.last()) {
        (Some(first), Some(last)) if rows.len() > 1 => {
            format!("{} → {}", first.label, last.label)
        }
        (Some(only), _) => only.label.clone(),
        _ => String::new(),
    };
    let left = format!("{title} · peak {}", compact_tokens(max));
    // A date cut in half is a different date. When the row cannot seat both
    // halves the range goes whole and the peak stays, since the peak is the
    // number the reader came for.
    if left.chars().count() + range.chars().count() + 1 > usize::from(width) {
        return Line::from(Span::styled(left, theme.typography.dim));
    }
    let pad = usize::from(width)
        .saturating_sub(left.chars().count())
        .saturating_sub(range.chars().count());
    Line::from(vec![
        Span::styled(left, theme.typography.dim),
        Span::styled(" ".repeat(pad), theme.typography.dim),
        Span::styled(range, theme.typography.dim),
    ])
}

fn model_mix_chart(rows: &[UsageModelRow], area: Rect, theme: &Theme) -> Vec<Line<'static>> {
    // Four hues is the ceiling: past that the legend stops being readable and
    // the runs stop being tellable apart. The rest folds into one honest
    // "other" so the bar still sums to the whole.
    let palette = [
        theme.palette.bright,
        theme.palette.violet,
        theme.palette.cyan,
        theme.palette.teal,
    ];
    let mut segments: Vec<(String, u64, Color)> = rows
        .iter()
        .take(palette.len())
        .enumerate()
        .map(|(idx, row)| (truncate(&row.model, 14), row.tokens, palette[idx]))
        .collect();
    let rest: u64 = rows.iter().skip(palette.len()).map(|row| row.tokens).sum();
    if rest > 0 {
        segments.push(("other".to_string(), rest, theme.palette.muted));
    }
    let mut lines = vec![Line::from(Span::styled(
        "Token share by model",
        theme.typography.bold,
    ))];
    lines.extend(charts::stacked_composition_bar(&segments, area.width, theme));
    lines
}

fn savings_chart(savings: &UsageSavingsSummary, area: Rect, theme: &Theme) -> Vec<Line<'static>> {
    let segments = vec![
        (
            "spent".to_string(),
            micro_usd(savings.actual_cost_usd),
            theme.palette.violet,
        ),
        (
            "cache saved".to_string(),
            micro_usd(savings.cache_savings_usd),
            theme.palette.success,
        ),
        (
            "mix saved".to_string(),
            micro_usd(savings.model_mix_savings_usd),
            theme.palette.teal,
        ),
    ];
    let mut lines = vec![Line::from(Span::styled(
        "Baseline cost, split by where it went",
        theme.typography.bold,
    ))];
    lines.extend(charts::stacked_composition_bar(&segments, area.width, theme));
    lines
}

/// USD as whole millionths so the composition bar can weigh segments with the
/// integer arithmetic it uses everywhere else.
///
/// Cents were the obvious unit and the wrong one: a session that has spent
/// four tenths of a cent is real usage, but rounding each figure to whole
/// cents zeroed every segment and the bar disappeared — the screen then read
/// "nothing saved" when the truth was "a little". Millionths keep a sub-cent
/// split proportional. Negative and non-finite input is clamped to zero rather
/// than inverting a run; the upper bound is `f64`'s exactly-representable
/// integer ceiling, past which the unit stops being the unit.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn micro_usd(usd: f64) -> u64 {
    /// Largest integer an `f64` still represents exactly (2^53).
    const EXACT_F64_LIMIT: f64 = 9_007_199_254_740_992.0;
    if !usd.is_finite() || usd <= 0.0 {
        return 0;
    }
    let scaled = (usd * 1_000_000.0).round();
    if scaled >= EXACT_F64_LIMIT {
        return EXACT_F64_LIMIT as u64;
    }
    scaled as u64
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
        // Card roles: tokens neutral-bright, cost violet (the HUD's cost hue),
        // savings success only when something was actually saved. The top model
        // is an identity label rather than a measurement, so it drops the bold
        // that had it competing with the three figures beside it.
        let style = match idx {
            1 => Style::new()
                .fg(theme.palette.violet)
                .add_modifier(Modifier::BOLD),
            2 => savings_value_style(snapshot.savings.total_savings_usd, theme)
                .add_modifier(Modifier::BOLD),
            3 => Style::new().fg(theme.palette.fg),
            _ => Style::new()
                .fg(theme.palette.bright)
                .add_modifier(Modifier::BOLD),
        };
        let lines = vec![
            Line::from(Span::styled(label.to_string(), theme.typography.dim)),
            Line::from(Span::styled(value, style)),
            Line::from(Span::styled(hint, theme.typography.key_hint)),
        ];
        frame.render_widget(
            Paragraph::new(super::wrap_body_rows(&lines, cols[idx].width, true)),
            cols[idx],
        );
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
    frame.render_widget(
        Paragraph::new(super::fit_body_rows(
            vec![line, divider_line(area.width, theme)],
            area.width,
        )),
        area,
    );
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
        let mut spans = vec![Span::styled(
            format!("{:<12}", truncate(&row.label, 12)),
            base,
        )];
        spans.extend(gauge_spans(
            filled_cells(row.tokens, max_tokens, TREND_CELLS),
            TREND_CELLS,
            gauge_fill_color(row.tokens, max_tokens, theme),
            theme,
        ));
        spans.extend([
            Span::styled("   ", base),
            Span::styled(format!("{:<10}", compact_tokens(row.tokens)), base),
            Span::styled(
                format!("{:<11}", format_usd(row.cost_usd)),
                Style::new().fg(theme.palette.violet),
            ),
            Span::styled(
                format!("{:<11}", format_usd(row.saved_usd)),
                savings_value_style(row.saved_usd, theme),
            ),
            Span::styled(truncate(&row.top_model, 20), theme.typography.dim),
        ]);
        lines.push(Line::from(spans));
    }
    if rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "No usage recorded yet.",
            theme.typography.placeholder,
        )));
    }
    frame.render_widget(
        Paragraph::new(super::wrap_body_rows(&lines, area.width, true)),
        area,
    );
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
        // The cost-ranked leader is the row the chart exists to surface, so it
        // takes `bright`; the rest stay body weight. The previous mapping had
        // this inverted and rendered every runner-up brighter than the leader.
        let name_style = if idx == 0 {
            base.fg(theme.palette.bright)
        } else {
            base
        };
        let mut spans = vec![
            Span::styled(format!("{:>2} ", idx + 1), rank_style),
            Span::styled(format!("{:<22}", truncate(&row.model, 22)), name_style),
        ];
        spans.extend(gauge_spans(
            ratio_cells(row.share, SHARE_CELLS),
            SHARE_CELLS,
            theme.metric_color(row.share),
            theme,
        ));
        spans.extend([
            Span::styled(
                format!(" {:>5.1}% ", row.share * 100.0),
                theme.typography.dim,
            ),
            Span::styled(format!("{:<10}", compact_tokens(row.tokens)), base),
            Span::styled(
                format!("{:<11}", format_usd(row.cost_usd)),
                Style::new()
                    .fg(theme.palette.violet)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format_usd(row.saved_usd),
                savings_value_style(row.saved_usd, theme),
            ),
        ]);
        lines.push(Line::from(spans));
    }
    frame.render_widget(
        Paragraph::new(super::wrap_body_rows(&lines, area.width, true)),
        area,
    );
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
        // What you paid takes the app-wide cost hue. The baseline is the
        // hypothetical it is measured against, so it stays a neutral reference
        // rather than `warn` — nothing here is a warning state.
        savings_line(
            "Actual cost",
            savings.actual_cost_usd,
            max,
            theme.palette.violet,
            false,
            theme,
        ),
        savings_line(
            "Baseline cost",
            savings.baseline_cost_usd,
            max,
            theme.palette.muted,
            false,
            theme,
        ),
        savings_line(
            "Cache savings",
            savings.cache_savings_usd,
            max,
            theme.palette.success,
            false,
            theme,
        ),
        savings_line(
            "Model mix",
            savings.model_mix_savings_usd,
            max,
            theme.palette.success,
            false,
            theme,
        ),
        Line::from(""),
        savings_line(
            "Total saved",
            savings.total_savings_usd,
            max,
            theme.palette.success,
            true,
            theme,
        ),
    ];
    frame.render_widget(
        Paragraph::new(super::wrap_body_rows(&lines, area.width, true)),
        area,
    );
}

/// One savings row. `emphasis` marks the summed total, which is the only figure
/// here that earns bold — its own components staying unbold is what makes the
/// row read as a sum instead of a fourth peer.
fn savings_line(
    label: &str,
    value: f64,
    max: f64,
    color: Color,
    emphasis: bool,
    theme: &Theme,
) -> Line<'static> {
    let value_style = if emphasis {
        Style::new().fg(color).add_modifier(Modifier::BOLD)
    } else {
        theme.typography.body
    };
    let mut spans = vec![
        Span::styled(format!("{label:<15}"), theme.typography.body),
        Span::styled(format!("{:<11}", format_usd(value)), value_style),
    ];
    spans.extend(gauge_spans(
        ratio_cells(value / max, SAVINGS_CELLS),
        SAVINGS_CELLS,
        color,
        theme,
    ));
    Line::from(spans)
}

fn render_footer(note: &str, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    let footer = key_hint_footer_reflowing(
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
    frame.render_widget(Paragraph::new(super::fit_body_rows(text, area.width)), area);
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

/// Gauge fill color on the shared success → warn → error ramp
/// ([`Theme::metric_color`]): the same thresholds the HUD context gauge uses,
/// so a "hot" bar means the same thing everywhere in the app.
fn gauge_fill_color(value: u64, max: u64, theme: &Theme) -> Color {
    theme.metric_color(cell_ratio(filled_cells(value, max, TREND_CELLS), TREND_CELLS))
}

/// Filled-cell count for a `value/max` bar, in integer arithmetic so a
/// `u64::MAX` denominator cannot lose precision the way an `f64` cast would.
fn filled_cells(value: u64, max: u64, width: usize) -> usize {
    if width == 0 || max == 0 {
        return 0;
    }
    let width_u128 = u128::from(u64::try_from(width).unwrap_or(u64::MAX));
    let filled = u128::from(value) * width_u128 / u128::from(max);
    usize::try_from(filled).unwrap_or(width).min(width)
}

/// Filled-cell count for an already-normalized `0.0..=1.0` share.
fn ratio_cells(ratio: f64, width: usize) -> usize {
    if width == 0 {
        return 0;
    }
    let ratio = ratio.clamp(0.0, 1.0);
    let width_u32 = u32::try_from(width).unwrap_or(u32::MAX);
    (1..=width)
        .filter(|cell| {
            let cell_u32 = u32::try_from(*cell).unwrap_or(u32::MAX);
            ratio >= f64::from(cell_u32) / f64::from(width_u32)
        })
        .count()
}

/// Ratio a filled-cell count represents, for feeding the metric ramp.
fn cell_ratio(filled: usize, width: usize) -> f64 {
    if width == 0 {
        return 0.0;
    }
    let filled = u32::try_from(filled).unwrap_or(u32::MAX);
    let width = u32::try_from(width).unwrap_or(u32::MAX);
    f64::from(filled) / f64::from(width)
}

/// A gauge as two spans: the filled run carries the reading, the track recedes
/// to `faint`.
///
/// A single-span bar paints both runs one color, which is what made these
/// dashboards read as a wash — a spent cell and an empty one were the same hue
/// and only the glyph shape distinguished them. The glyphs are the EAW-Neutral
/// [`glyphs::GAUGE_FILL`]/[`glyphs::GAUGE_EMPTY`] pair the sidebar and HUD
/// gauges already use, not the block `█`: that one is East-Asian Ambiguous, so
/// under a `ko_KR` wide-ambiguous tmux every filled cell painted two columns and
/// shoved the fixed-width figures to its right out of alignment.
fn gauge_spans(filled: usize, width: usize, fill: Color, theme: &Theme) -> Vec<Span<'static>> {
    let color = !theme.no_color;
    let filled = filled.min(width);
    vec![
        Span::styled(
            glyphs::pick(color, glyphs::GAUGE_FILL, glyphs::GAUGE_FILL_NC).repeat(filled),
            Style::new().fg(fill),
        ),
        Span::styled(
            glyphs::pick(color, glyphs::GAUGE_EMPTY, glyphs::GAUGE_EMPTY_NC)
                .repeat(width.saturating_sub(filled)),
            Style::new().fg(theme.palette.faint),
        ),
    ]
}

/// Savings are money recovered, so zero is not an achievement: a green `$0.00`
/// on every row teaches the eye to skip the column that matters. Only a
/// positive figure earns `success`.
fn savings_value_style(value: f64, theme: &Theme) -> Style {
    if value > 0.0 {
        Style::new().fg(theme.palette.success)
    } else {
        Style::new().fg(theme.palette.faint)
    }
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
    use unicode_width::UnicodeWidthStr;

    fn is_braille(ch: char) -> bool {
        ('\u{2800}'..='\u{28ff}').contains(&ch)
    }

    fn painted_rows(modal: &UsageDashboardModal, theme: &Theme, w: u16, h: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal
            .draw(|frame| modal.draw(frame, frame.area(), theme))
            .unwrap();
        let buffer = terminal.backend().buffer();
        (buffer.area.y..buffer.area.y + buffer.area.height)
            .map(|y| {
                (buffer.area.x..buffer.area.x + buffer.area.width)
                    .map(|x| buffer.cell((x, y)).unwrap().symbol())
                    .collect::<String>()
            })
            .collect()
    }

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
    fn filled_cells_preserves_large_u64_ratios() {
        // Half of `u64::MAX` over sixteen cells is seven full cells and change.
        // An `f64` cast would round the operands and hand back eight.
        assert_eq!(filled_cells(u64::MAX / 2, u64::MAX, 16), 7);
        assert_eq!(filled_cells(u64::MAX, u64::MAX, 16), 16);
        assert_eq!(filled_cells(0, u64::MAX, 16), 0);
        assert_eq!(filled_cells(5, 0, 16), 0);
    }

    #[test]
    fn gauge_paints_its_track_apart_from_its_fill() {
        let theme = Theme::zo();
        let spans = gauge_spans(6, 16, theme.palette.success, &theme);
        assert_eq!(spans.len(), 2, "a gauge is a fill run plus a track run");
        assert_eq!(spans[0].content.chars().count(), 6);
        assert_eq!(spans[1].content.chars().count(), 10);
        assert_eq!(spans[0].style.fg, Some(theme.palette.success));
        assert_eq!(spans[1].style.fg, Some(theme.palette.faint));
        assert_ne!(
            spans[0].style.fg, spans[1].style.fg,
            "a track painted in the fill color is what made the bar unreadable"
        );
    }

    #[test]
    fn gauge_glyphs_stay_one_column_and_degrade_without_color() {
        use unicode_width::UnicodeWidthStr;

        let theme = Theme::zo();
        let spans = gauge_spans(2, 4, theme.palette.success, &theme);
        let bar: String = spans.iter().map(|span| span.content.as_ref()).collect();
        assert_eq!(bar, "\u{25ac}\u{25ac}\u{2591}\u{2591}");
        // The block `█` is East-Asian Ambiguous and doubles under a `ko_KR`
        // wide-ambiguous tmux, which is what pushed the columns to its right
        // out of alignment. Both gauge glyphs must stay Neutral.
        assert_eq!(bar.width_cjk(), 4, "gauge must hold its budget: {bar}");

        let mut mono = theme.clone();
        mono.no_color = true;
        let plain: String = gauge_spans(2, 4, mono.palette.success, &mono)
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(plain, "##..");
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

    /// A week of daily buckets — the shape the band exists to show. The
    /// single-session `snapshot()` has one bucket and deliberately plots no
    /// trend, so it cannot exercise the chart.
    fn week_snapshot() -> UsageDashboardSnapshot {
        let mut snap = snapshot();
        // Newest first, the order `UsageDashboardSnapshot` actually produces.
        // Listing it oldest-first read naturally and was a lie about the input,
        // and it hid a band that plotted the whole week backwards.
        snap.daily = [
            ("2026-08-03", 9_000_u64),
            ("2026-08-02", 18_000),
            ("2026-08-01", 3_000),
            ("2026-07-31", 26_000),
            ("2026-07-30", 12_000),
            ("2026-07-29", 31_000),
            ("2026-07-28", 4_000),
        ]
        .into_iter()
        .map(|(label, tokens)| UsagePeriodRow {
            label: label.to_string(),
            tokens,
            cost_usd: 0.02,
            saved_usd: 0.01,
            top_model: "gpt-5.5".to_string(),
        })
        .collect();
        snap
    }

    /// Paint the modal and read the cells back, so the band is judged as the
    /// user sees it. Braille is the one thing here the row model cannot check
    /// for us: a span-level assertion passes happily on a chart that never
    /// reached the pane — and it passed on a solid block that only a look
    /// revealed.
    #[test]
    fn a_tall_pane_paints_the_chart_band_above_the_table() {
        let theme = Theme::zo();
        let rows = painted_rows(&UsageDashboardModal::new(week_snapshot()), &theme, 100, 24);

        let ink = rows
            .iter()
            .position(|row| row.chars().any(is_braille))
            .expect("the chart band paints braille ink");
        let table = rows
            .iter()
            .position(|row| row.contains("Period"))
            .expect("the table header still paints");
        assert!(
            ink < table,
            "the band reads above the table it summarizes:\n{}",
            rows.join("\n")
        );
        assert!(
            rows.iter().any(|row| row.contains("peak")),
            "the caption keeps the band readable without braille:\n{}",
            rows.join("\n")
        );
        assert!(
            rows.iter()
                .any(|row| row.contains("2026-07-28 → 2026-08-03")),
            "the painted axis runs oldest → newest:\n{}",
            rows.join("\n")
        );
        // Scoped to the band's own rows: the card border is box-drawing, which
        // is EAW-Ambiguous by design here (see `glyphs::ANVIL_LINE`) and which
        // target terminals force narrow. The data ink is what must never widen.
        for row in rows.iter().filter(|row| row.chars().any(is_braille)) {
            assert_eq!(
                UnicodeWidthStr::width_cjk(row.as_str()),
                100,
                "a wide-ambiguous locale must not widen the chart band: {row:?}"
            );
        }
    }

    fn flat(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect()
    }

    fn period_rows_newest_first(rows: &[(&str, u64)]) -> Vec<UsagePeriodRow> {
        rows.iter()
            .map(|(label, tokens)| UsagePeriodRow {
                label: (*label).to_string(),
                tokens: *tokens,
                cost_usd: 0.01,
                saved_usd: 0.0,
                top_model: "gpt-5.5".to_string(),
            })
            .collect()
    }

    /// `UsageDashboardSnapshot` sorts period rows newest-first, which is right
    /// for the table (today on top) and wrong for a time axis. Plotted as
    /// given, the week runs right-to-left and every trend reads inverted — a
    /// rise looks like a fall.
    #[test]
    fn the_band_plots_time_left_to_right_though_the_table_is_newest_first() {
        let theme = Theme::zo();
        // Newest first, exactly the order the snapshot hands over.
        let rows = period_rows_newest_first(&[
            ("2026-08-04", 1_000),
            ("2026-08-03", 1_000),
            ("2026-08-02", 1_000),
            ("2026-08-01", 50_000),
        ]);
        let lines = period_chart(&rows, "Tokens per day", Rect::new(0, 0, 60, 5), &theme);

        let caption = flat(&lines[lines.len() - 1..]);
        assert!(
            caption.contains("2026-08-01 → 2026-08-04"),
            "the axis caption runs oldest → newest: {caption:?}"
        );

        let top: Vec<char> = flat(&lines[..1]).chars().collect();
        let half = top.len() / 2;
        let left = top[..half].iter().any(|ch| *ch != '\u{2800}');
        let right = top[half..].iter().any(|ch| *ch != '\u{2800}');
        assert!(
            left && !right,
            "the older, taller bucket belongs on the left: {top:?}"
        );
    }

    /// Rounding each figure to whole cents erased money that exists: a
    /// sub-cent split summed to zero and the bar vanished entirely, which
    /// reads as "no savings" rather than "a small amount".
    #[test]
    fn a_sub_cent_savings_split_still_draws_its_bar() {
        let theme = Theme::zo();
        let savings = UsageSavingsSummary {
            actual_cost_usd: 0.004,
            baseline_cost_usd: 0.006,
            cache_savings_usd: 0.001,
            model_mix_savings_usd: 0.001,
            total_savings_usd: 0.002,
        };

        let text = flat(&savings_chart(&savings, Rect::new(0, 0, 60, 5), &theme));
        assert!(
            text.contains(glyphs::GAUGE_FILL),
            "sub-cent money is still money: {text:?}"
        );
    }

    /// A pane too narrow to seat a legend drops the band whole. Cutting it
    /// instead leaves a colour run whose name is off-screen — a key the reader
    /// cannot use, which is worse than no chart.
    #[test]
    fn a_narrow_pane_drops_the_band_rather_than_cutting_its_legend() {
        let theme = Theme::zo();
        let rows = painted_rows(&UsageDashboardModal::new(week_snapshot()), &theme, 40, 24);
        let text = rows.join("\n");

        assert!(
            !text.chars().any(is_braille),
            "no half-legible band in a narrow pane:\n{text}"
        );
        assert!(text.contains("Period"), "the table survives:\n{text}");
    }

    /// A short terminal drops the band whole. Squeezing it would spend the
    /// table's scroll window on a chart too short to read.
    #[test]
    fn a_short_pane_drops_the_band_and_keeps_the_table() {
        let theme = Theme::zo();
        let rows = painted_rows(&UsageDashboardModal::new(snapshot()), &theme, 100, 14);
        let text = rows.join("\n");

        assert!(
            !text.chars().any(is_braille),
            "no band on a short pane:\n{text}"
        );
        assert!(text.contains("Period"), "the table survives:\n{text}");
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
        assert!(
            text.contains(glyphs::GAUGE_FILL),
            "trend gauge missing:\n{text}"
        );
        assert!(
            !text.contains('\u{2588}'),
            "the dashboard must not paint the EAW-Ambiguous block glyph:\n{text}"
        );
    }
}
