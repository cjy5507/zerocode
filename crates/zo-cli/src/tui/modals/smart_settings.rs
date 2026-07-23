//! Large dashboard modal for `/smart` Smart Model Router settings.
//!
//! The widget owns only interactive UI state: selected target, active pane,
//! edit cursors, staged routing choices, and rendering. It does not read or
//! write settings files. The session layer supplies a [`SmartSettingsView`] and
//! persists the [`SmartSettingsCommit`] returned on Enter.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

use super::super::cards::{CardFrame, SurfaceKind};

use super::{ModalResult, ModalSelection, key_hint_footer, selected_style};
use crate::tui::theme::Theme;

const MODE_COUNT: usize = 3;
const FAMILY_FIELD_COUNT: usize = 4;
/// Most provider bars rendered in the dashboard's provider-mix chart.
const MIX_CHART_MAX_BARS: usize = 4;

/// Freshness policy for a Smart Router family selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartSettingsFreshness {
    /// Prefer the newest matching model, including previews if the inventory
    /// exposes them.
    Latest,
    /// Prefer the newest stable matching model.
    LatestStable,
}

impl SmartSettingsFreshness {
    fn label(self) -> &'static str {
        match self {
            Self::Latest => "latest",
            Self::LatestStable => "latestStable",
        }
    }

    fn cycle(self) -> Self {
        match self {
            Self::Latest => Self::LatestStable,
            Self::LatestStable => Self::Latest,
        }
    }
}

/// Staged routing update for a role or subagent target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmartSettingsUpdate {
    /// Let the Smart Router choose from the usable model pool.
    Auto,
    /// Pin this target to one concrete model id.
    ExactPin {
        /// Model id to pin.
        model: String,
    },
    /// Track a provider/family/class selector instead of one fixed model.
    FamilyLock {
        /// Provider id, e.g. `openai`.
        provider: String,
        /// Family label, e.g. `gpt`.
        family: String,
        /// Class label, e.g. `coding` or `balanced`.
        class: String,
        /// Freshness policy.
        freshness: SmartSettingsFreshness,
    },
}

impl SmartSettingsUpdate {
    fn mode(&self) -> RoutingMode {
        match self {
            Self::Auto => RoutingMode::Auto,
            Self::ExactPin { .. } => RoutingMode::Pin,
            Self::FamilyLock { .. } => RoutingMode::Family,
        }
    }

    fn short_label(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::ExactPin { .. } => "Pin",
            Self::FamilyLock { .. } => "Family",
        }
    }
}

/// One usable model row displayed by the Smart settings modal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartSettingsModel {
    /// Model id.
    pub id: String,
    /// Provider id.
    pub provider: String,
    /// Family label.
    pub family: String,
    /// Class label.
    pub class: String,
}

/// Kind of target being configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartSettingsTargetKind {
    /// Route role fallback target.
    Role,
    /// Subagent profile target.
    Subagent,
}

/// One role/subagent row in the target list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartSettingsTarget {
    /// Stable key used by the settings file.
    pub key: String,
    /// Human-readable label rendered in the target list.
    pub label: String,
    /// Target kind.
    pub kind: SmartSettingsTargetKind,
    /// Saved/current update at modal open time.
    pub update: SmartSettingsUpdate,
}

/// Recommendation preview supplied by the session layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartSettingsRecommendation {
    /// Target kind this recommendation belongs to.
    pub kind: SmartSettingsTargetKind,
    /// Target key.
    pub key: String,
    /// Selected model id.
    pub selected_model: String,
    /// Confidence label: `High`, `Medium`, or `Low`.
    pub confidence: String,
    /// Short reason for the recommendation.
    pub reason: String,
    /// Audit/guardrail lines supplied by the session layer.
    pub audit: Vec<String>,
}

/// Observed routing outcome for a target, aggregated across every model that ran
/// for it, read from the durable route-outcome log. Surfaced inline on the
/// resolution line so the user sees what actually ran, not only the
/// recommendation. Counts are over DECISIVE runs (completed + failed); user
/// cancels are excluded so the `ok` ratio matches the feedback score's neutrality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartSettingsObservedRoute {
    /// Target kind this outcome belongs to.
    pub kind: SmartSettingsTargetKind,
    /// Target key.
    pub key: String,
    /// Completed (successful) decisive runs.
    pub completed: usize,
    /// Decisive runs (completed + failed; excludes user cancels).
    pub decisive: usize,
    /// The model the runs were on, only when a single model ran for this target
    /// (otherwise `None` — naming one of several would mislead).
    pub model: Option<String>,
}

/// Immutable dashboard input used to initialize the modal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartSettingsView {
    /// Whether Smart Router is currently enabled.
    pub enabled: bool,
    /// Whether verifier/reviewer recommendations may prefer a different provider family.
    pub allow_cross_provider_diversity: bool,
    /// Whether bounded aggregate feedback may adjust auto scores.
    pub feedback_informed_auto: bool,
    /// Read-only display label for the auto-classifier mode (`off` /
    /// `deterministic` / `assisted`), surfaced in the preview's guardrails.
    pub auto_classifier: String,
    /// Current main model id.
    pub main_model: String,
    /// Settings file path shown in the footer.
    pub settings_path: String,
    /// Usable model inventory.
    pub models: Vec<SmartSettingsModel>,
    /// Notes about configured providers excluded from the usable inventory.
    pub model_notes: Vec<String>,
    /// Role fallback targets.
    pub roles: Vec<SmartSettingsTarget>,
    /// Subagent targets.
    pub subagents: Vec<SmartSettingsTarget>,
    /// Auto recommendation previews with cross-provider diversity OFF.
    pub recommendations: Vec<SmartSettingsRecommendation>,
    /// Auto recommendation previews with cross-provider diversity ON. The modal
    /// picks between this and `recommendations` by the live toggle so flipping
    /// `d` previews the effect immediately, instead of waiting for a save.
    pub recommendations_with_diversity: Vec<SmartSettingsRecommendation>,
    /// Per-turn output-token usage for the current session, oldest first.
    /// Rendered as a usage sparkline; empty when the session has no recorded
    /// turns yet (the dashboard simply omits that chart).
    pub turn_output_tokens: Vec<u32>,
    /// Observed routing outcomes (from the durable route-outcome log), surfaced
    /// on each target's resolution line. Empty when nothing has been recorded.
    pub observed_routes: Vec<SmartSettingsObservedRoute>,
}

/// Confirmed Smart settings payload returned from the modal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartSettingsCommit {
    /// Final enabled state.
    pub enabled: bool,
    /// Final cross-provider diversity setting.
    pub allow_cross_provider_diversity: bool,
    /// Final feedback-informed auto setting.
    pub feedback_informed_auto: bool,
    /// Staged role updates.
    pub roles: Vec<(String, SmartSettingsUpdate)>,
    /// Staged subagent updates.
    pub subagents: Vec<(String, SmartSettingsUpdate)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetTab {
    Roles,
    Subagents,
}

impl TargetTab {
    fn label(self) -> &'static str {
        match self {
            Self::Roles => "Roles",
            Self::Subagents => "Subagents",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Roles => Self::Subagents,
            Self::Subagents => Self::Roles,
        }
    }

    fn prev(self) -> Self {
        self.next()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivePane {
    /// The role/subagent list.
    Targets,
    /// The merged editor + live preview for the selected target.
    Detail,
}

impl ActivePane {
    fn next(self) -> Self {
        match self {
            Self::Targets => Self::Detail,
            Self::Detail => Self::Targets,
        }
    }

    fn prev(self) -> Self {
        // Two panes: prev == next.
        self.next()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingMode {
    Auto,
    Pin,
    Family,
}

impl RoutingMode {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto recommended",
            Self::Pin => "Pin exact model",
            Self::Family => "Track provider/family/class",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Auto => 0,
            Self::Pin => 1,
            Self::Family => 2,
        }
    }

    fn from_index(index: usize) -> Self {
        match index % MODE_COUNT {
            0 => Self::Auto,
            1 => Self::Pin,
            _ => Self::Family,
        }
    }

    fn next(self) -> Self {
        Self::from_index(self.index() + 1)
    }

    fn prev(self) -> Self {
        Self::from_index(self.index() + MODE_COUNT - 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TargetState {
    key: String,
    label: String,
    kind: SmartSettingsTargetKind,
    original: SmartSettingsUpdate,
    current: SmartSettingsUpdate,
}

impl TargetState {
    fn from_target(target: SmartSettingsTarget) -> Self {
        Self {
            key: target.key,
            label: target.label,
            kind: target.kind,
            original: target.update.clone(),
            current: target.update,
        }
    }

    fn changed(&self) -> bool {
        self.original != self.current
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SmartPolicyToggles {
    allow_cross_provider_diversity: bool,
    feedback_informed_auto: bool,
}

impl SmartPolicyToggles {
    fn from_view(view: &SmartSettingsView) -> Self {
        Self {
            allow_cross_provider_diversity: view.allow_cross_provider_diversity,
            feedback_informed_auto: view.feedback_informed_auto,
        }
    }

    fn changed_count(self, original: Self) -> usize {
        usize::from(self.allow_cross_provider_diversity != original.allow_cross_provider_diversity)
            + usize::from(self.feedback_informed_auto != original.feedback_informed_auto)
    }
}

/// Interactive `/smart` dashboard modal.
#[derive(Debug, Clone)]
pub struct SmartSettingsModal {
    enabled: bool,
    original_enabled: bool,
    policy: SmartPolicyToggles,
    original_policy: SmartPolicyToggles,
    auto_classifier: String,
    main_model: String,
    settings_path: String,
    models: Vec<SmartSettingsModel>,
    model_notes: Vec<String>,
    recommendations: Vec<SmartSettingsRecommendation>,
    recommendations_with_diversity: Vec<SmartSettingsRecommendation>,
    roles: Vec<TargetState>,
    subagents: Vec<TargetState>,
    tab: TargetTab,
    pane: ActivePane,
    role_cursor: usize,
    subagent_cursor: usize,
    editor_cursor: usize,
    model_cursor: usize,
    model_filter: String,
    filtering_models: bool,
    turn_output_tokens: Vec<u32>,
    observed_routes: Vec<SmartSettingsObservedRoute>,
}

impl SmartSettingsModal {
    /// Build a modal from the current Smart settings view.
    #[must_use]
    pub fn new(view: SmartSettingsView) -> Self {
        Self {
            enabled: view.enabled,
            original_enabled: view.enabled,
            policy: SmartPolicyToggles::from_view(&view),
            original_policy: SmartPolicyToggles::from_view(&view),
            auto_classifier: view.auto_classifier,
            main_model: view.main_model,
            settings_path: view.settings_path,
            models: view.models,
            model_notes: view.model_notes,
            recommendations: view.recommendations,
            recommendations_with_diversity: view.recommendations_with_diversity,
            roles: view.roles.into_iter().map(TargetState::from_target).collect(),
            subagents: view.subagents.into_iter().map(TargetState::from_target).collect(),
            tab: TargetTab::Roles,
            pane: ActivePane::Targets,
            role_cursor: 0,
            subagent_cursor: 0,
            editor_cursor: 0,
            model_cursor: 0,
            model_filter: String::new(),
            filtering_models: false,
            turn_output_tokens: view.turn_output_tokens,
            observed_routes: view.observed_routes,
        }
    }

    /// Currently active target tab.
    #[must_use]
    pub fn active_tab_label(&self) -> &'static str {
        self.tab.label()
    }

    /// Count staged changes, including the master enable toggle.
    #[must_use]
    pub fn pending_change_count(&self) -> usize {
        let global_delta = usize::from(self.enabled != self.original_enabled)
            + self.policy.changed_count(self.original_policy);
        global_delta
            + self.roles.iter().filter(|target| target.changed()).count()
            + self.subagents.iter().filter(|target| target.changed()).count()
    }

    /// Handle a single key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if self.filtering_models {
            return self.handle_filter_key(key);
        }
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(ModalResult::Cancelled);
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Enter => Some(ModalResult::Selected(ModalSelection::SmartSettings(
                self.commit(),
            ))),
            KeyCode::Tab => {
                self.pane = if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.pane.prev()
                } else {
                    self.pane.next()
                };
                None
            }
            KeyCode::BackTab => {
                self.pane = self.pane.prev();
                None
            }
            KeyCode::Char(' ') => {
                self.enabled = !self.enabled;
                None
            }
            KeyCode::Char('d' | 'D') => {
                self.policy.allow_cross_provider_diversity = !self.policy.allow_cross_provider_diversity;
                None
            }
            KeyCode::Char('b' | 'B') => {
                self.policy.feedback_informed_auto = !self.policy.feedback_informed_auto;
                None
            }
            KeyCode::Char('A') => {
                self.apply_recommended_setup();
                None
            }
            KeyCode::Char('a') => {
                self.set_current_mode(RoutingMode::Auto);
                None
            }
            KeyCode::Char('p' | 'P') => {
                self.set_current_mode(RoutingMode::Pin);
                None
            }
            KeyCode::Char('f' | 'F') => {
                self.set_current_mode(RoutingMode::Family);
                None
            }
            KeyCode::Char('r' | 'R') => {
                self.reset_current_target();
                None
            }
            KeyCode::Char('/') => {
                if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin) {
                    self.filtering_models = true;
                    self.pane = ActivePane::Detail;
                }
                None
            }
            KeyCode::Left => {
                self.move_left();
                None
            }
            KeyCode::Right => {
                self.move_right();
                None
            }
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
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
            _ => None,
        }
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::from(vec![
                Span::styled(" Smart Model Router ", theme.typography.heading_1),
                Span::styled("/smart ", theme.typography.dim),
            ]))
            .render(frame, area);

        if inner.width < 78 || inner.height < 18 {
            frame.render_widget(
                Paragraph::new("Smart settings dashboard needs a larger terminal")
                    .style(theme.typography.dim)
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(inner);
        self.render_header(frame, rows[0], theme);
        self.render_body(frame, rows[1], theme);
        self.render_footer(frame, rows[2], theme);
    }

    fn handle_filter_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        match key.code {
            KeyCode::Esc => {
                self.filtering_models = false;
                None
            }
            KeyCode::Enter => {
                self.apply_selected_model();
                self.filtering_models = false;
                None
            }
            KeyCode::Backspace => {
                self.model_filter.pop();
                self.model_cursor = self.model_cursor.min(self.filtered_model_count().saturating_sub(1));
                None
            }
            KeyCode::Up => {
                self.model_cursor = self.model_cursor.saturating_sub(1);
                None
            }
            KeyCode::Down => {
                let count = self.filtered_model_count();
                if count > 0 {
                    self.model_cursor = (self.model_cursor + 1).min(count - 1);
                }
                None
            }
            KeyCode::Char(ch) => {
                self.model_filter.push(ch);
                self.model_cursor = self.model_cursor.min(self.filtered_model_count().saturating_sub(1));
                None
            }
            _ => None,
        }
    }

    /// Status + global-settings bar. Replaces the old five-card header: the
    /// global toggles (the things users actually change) are surfaced here with
    /// their keys instead of being buried inside the editor pane.
    fn render_header(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let smart_style = if self.enabled {
            Style::new().fg(theme.palette.success).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(theme.palette.warn).add_modifier(Modifier::BOLD)
        };
        let pending = self.pending_change_count();
        let mut status = vec![
            Span::styled(if self.enabled { "● Smart ON" } else { "○ Smart OFF" }, smart_style),
            Span::styled("   Main: ", theme.typography.dim),
            Span::styled(truncate(&self.main_model, 28), theme.typography.bold),
            Span::styled(format!("   {} models", self.models.len()), theme.typography.dim),
        ];
        if pending > 0 {
            status.push(Span::styled(
                format!("   ● {pending} unsaved"),
                Style::new().fg(theme.palette.warn).add_modifier(Modifier::BOLD),
            ));
        }

        let mut settings = vec![Span::styled("Global  ", theme.typography.dim)];
        settings.extend(toggle_spans("Space", "Smart", self.enabled, theme));
        settings.push(Span::raw("   "));
        settings.extend(toggle_spans("d", "Cross-provider", self.policy.allow_cross_provider_diversity, theme));
        settings.push(Span::raw("   "));
        settings.extend(toggle_spans("b", "Feedback", self.policy.feedback_informed_auto, theme));
        settings.push(Span::styled(format!("   Classifier: {}", self.auto_classifier), theme.typography.dim));

        let mut lines = vec![Line::from(status), Line::from(""), Line::from(settings)];
        if !self.model_notes.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("{} provider(s) hidden from the usable pool", self.model_notes.len()),
                theme.typography.dim,
            )));
        }
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
    }

    fn render_body(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        // Two panes: the target list and a merged editor + live preview. The old
        // third "Preview" pane was redundant with the editor (both described the
        // selected target); folding it in removes a column and a focus stop.
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(area);
        self.render_targets(frame, cols[0], theme);
        self.render_detail(frame, cols[1], theme);
    }

    fn render_targets(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let active = self.pane == ActivePane::Targets;
        let block = pane_block("Targets", active, theme);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Reserve a bottom strip for the overview charts (provider mix + usage
        // sparkline), but only when the pane is tall enough to keep the target
        // list usable. These summarize the whole config/session, so they stay
        // visible alongside the list rather than hiding behind a toggle.
        let overview = self.overview_lines(inner.width, theme);
        let overview_rows = u16::try_from(overview.len()).unwrap_or(u16::MAX);
        let (list_area, chart_area) = if overview.is_empty() || inner.height < overview_rows + 6 {
            (inner, None)
        } else {
            let parts = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(5), Constraint::Length(overview_rows)])
                .split(inner);
            (parts[0], Some(parts[1]))
        };

        let mut lines = Vec::new();
        lines.push(render_tabs(self.tab, theme));
        lines.push(Line::from(""));
        let visible = usize::from(list_area.height.saturating_sub(3));
        let targets = self.current_targets();
        let cursor = self.current_cursor();
        let (start, end) = visible_window(targets.len(), cursor, visible.max(1));
        for (idx, target) in targets.iter().enumerate().skip(start).take(end.saturating_sub(start)) {
            let selected = idx == cursor;
            let marker = if selected { ">" } else { " " };
            let changed = if target.changed() { "*" } else { " " };
            let fit = self
                .recommendation_for(target)
                .map_or("---".to_string(), |rec| confidence_meter(&rec.confidence).to_string());
            let style = if selected {
                selected_style(theme)
            } else if target.changed() {
                Style::new().fg(theme.palette.warn)
            } else {
                theme.typography.body
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{marker}{changed} "), style),
                Span::styled(format!("{:<16}", truncate(&target.label, 16)), style),
                Span::styled(format!(" {:<6}", target.current.short_label()), badge_style(&target.current, theme)),
                Span::styled(format!(" {fit}"), theme.typography.dim),
            ]));
        }
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), list_area);

        if let Some(chart_area) = chart_area {
            frame.render_widget(Paragraph::new(overview), chart_area);
        }
    }

    /// Overview charts for the Targets pane bottom strip: a horizontal
    /// provider-mix bar chart (which provider each role/subagent resolves to)
    /// and a per-turn output-token usage sparkline for the session. Both are
    /// computed from current state so they update live as settings are staged.
    fn overview_lines(&self, width: u16, theme: &Theme) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();

        let mix = self.provider_mix();
        let total: usize = mix.iter().map(|(_, count)| count).sum();
        if total > 0 {
            let bar_max = usize::from(width).saturating_sub(20).clamp(4, 28);
            let main_provider = self.main_provider();
            lines.push(Line::from(vec![
                Span::styled("Provider mix ", theme.typography.bold),
                Span::styled(format!("· {total} targets"), theme.typography.dim),
            ]));
            for (provider, count) in mix.iter().take(MIX_CHART_MAX_BARS) {
                // Integer geometry only — avoids float casts (clippy precision lints).
                let filled = (count * bar_max / total).clamp(usize::from(*count > 0), bar_max);
                let pct = count * 100 / total;
                let is_main = main_provider.as_deref() == Some(provider.as_str());
                let bar_style = if is_main {
                    Style::new().fg(theme.palette.accent).add_modifier(Modifier::BOLD)
                } else {
                    Style::new().fg(theme.palette.accent)
                };
                let label_style = if is_main { theme.typography.bold } else { theme.typography.body };
                lines.push(Line::from(vec![
                    Span::styled(format!("{:<8}", truncate(provider, 8)), label_style),
                    Span::styled("█".repeat(filled), bar_style),
                    Span::styled("░".repeat(bar_max - filled), theme.typography.dim),
                    Span::styled(format!(" {count} ({pct}%)"), theme.typography.dim),
                ]));
            }
        }

        if !self.turn_output_tokens.is_empty() {
            if !lines.is_empty() {
                lines.push(Line::from(""));
            }
            let turns = self.turn_output_tokens.len();
            let last = self.turn_output_tokens.last().copied().unwrap_or(0);
            let spark_width = usize::from(width).saturating_sub(2).clamp(8, 40);
            let spark = usage_sparkline(&self.turn_output_tokens, spark_width, theme);
            lines.push(Line::from(vec![
                Span::styled("Output tokens ", theme.typography.bold),
                Span::styled(format!("· {turns} turns"), theme.typography.dim),
            ]));
            lines.push(Line::from(vec![
                Span::styled(spark, Style::new().fg(theme.palette.accent)),
                Span::styled(format!("  last {}", fmt_compact(last)), theme.typography.dim),
            ]));
        }

        lines
    }

    fn render_detail(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let active = self.pane == ActivePane::Detail;
        let block = pane_block("Detail", active, theme);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let Some(target) = self.current_target() else {
            frame.render_widget(Paragraph::new("No target"), inner);
            return;
        };
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Target ", theme.typography.dim),
                Span::styled(target.label.clone(), theme.typography.bold),
                Span::styled(format!("  {}", target_kind_label(target.kind)), theme.typography.dim),
            ]),
            Line::from(""),
            Line::from(Span::styled("Routing mode", theme.typography.bold)),
        ];
        for (idx, mode) in [RoutingMode::Auto, RoutingMode::Pin, RoutingMode::Family]
            .into_iter()
            .enumerate()
        {
            let selected_mode = target.current.mode() == mode;
            let cursor = self.editor_cursor == 0 && selected_mode && self.pane == ActivePane::Detail;
            let symbol = if selected_mode { "◉" } else { "○" };
            let style = if cursor {
                selected_style(theme)
            } else if selected_mode {
                theme.typography.bold
            } else {
                theme.typography.dim
            };
            let key = match idx { 0 => "a", 1 => "p", _ => "f" };
            lines.push(Line::from(vec![
                Span::styled(format!("  {symbol} "), style),
                Span::styled(format!("{:<28}", mode.label()), style),
                Span::styled(format!(" {key}"), theme.typography.key_hint),
            ]));
        }
        lines.push(Line::from(""));
        match &target.current {
            SmartSettingsUpdate::Auto => self.render_auto_editor(&mut lines, theme),
            SmartSettingsUpdate::ExactPin { model } => self.render_pin_editor(&mut lines, model, inner.height, theme),
            SmartSettingsUpdate::FamilyLock { provider, family, class, freshness } => {
                self.render_family_editor(&mut lines, provider, family, class, *freshness, theme);
            }
        }
        self.render_resolution(&mut lines, target, theme);
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
    }

    fn render_auto_editor(&self, lines: &mut Vec<Line<'static>>, theme: &Theme) {
        lines.push(Line::from(Span::styled("Auto picks from the usable model pool.", theme.typography.dim)));
        lines.push(Line::from(Span::styled("p = pin a model · f = track a family.", theme.typography.dim)));
        self.render_model_notes(lines, theme);
    }

    /// The live "what does this resolve to" preview, folded into the detail pane
    /// (previously a separate third column). Shows the resolved model, confidence,
    /// and the reason so the effect of the chosen mode is visible inline.
    fn render_resolution(&self, lines: &mut Vec<Line<'static>>, target: &TargetState, theme: &Theme) {
        let preview = self.preview_for(target);
        lines.push(Line::from(""));
        if !self.enabled {
            // Routing is unchanged while Smart is OFF; the preview is hypothetical.
            lines.push(Line::from(Span::styled(
                "Smart is OFF — preview only (turns use the main model)",
                Style::new().fg(theme.palette.warn),
            )));
        }
        lines.push(Line::from(vec![
            Span::styled("→ resolves to ", theme.typography.dim),
            Span::styled(preview.model, Style::new().fg(theme.palette.accent).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  confidence ", theme.typography.dim),
            Span::styled(preview.confidence, theme.typography.body),
            Span::styled(format!(" · {}", preview.source), theme.typography.dim),
        ]));
        lines.push(Line::from(Span::styled(format!("  {}", preview.reason), theme.typography.dim)));
        // What actually ran for this target (from the durable outcome log), so the
        // recommendation and its evidence are visible together. Only shown when
        // there are decisive (non-cancelled) runs.
        if let Some(observed) = self.observed_for(target).filter(|observed| observed.decisive > 0) {
            let mut spans = vec![
                Span::styled("  ⤷ observed ", theme.typography.dim),
                Span::styled(
                    format!("{}/{} ok", observed.completed, observed.decisive),
                    theme.typography.body,
                ),
            ];
            if let Some(model) = &observed.model {
                spans.push(Span::styled(format!(" on {}", truncate(model, 28)), theme.typography.dim));
            }
            lines.push(Line::from(spans));
        }
    }

    fn observed_for(&self, target: &TargetState) -> Option<&SmartSettingsObservedRoute> {
        self.observed_routes
            .iter()
            .find(|observed| observed.kind == target.kind && observed.key == target.key)
    }

    fn render_pin_editor(
        &self,
        lines: &mut Vec<Line<'static>>,
        model: &str,
        height: u16,
        theme: &Theme,
    ) {
        let matches = self.filtered_models();
        let total = matches.len();
        lines.push(Line::from(vec![
            Span::styled("Pinned: ", theme.typography.dim),
            Span::styled(truncate(model, 32), theme.typography.bold),
        ]));
        let filter = if self.filtering_models {
            format!("{}▌", self.model_filter)
        } else if self.model_filter.is_empty() {
            "(press / to filter)".to_string()
        } else {
            self.model_filter.clone()
        };
        lines.push(Line::from(vec![
            Span::styled("Filter / ", theme.typography.key_hint),
            Span::styled(filter, Style::new().fg(theme.palette.accent).add_modifier(Modifier::BOLD)),
            Span::styled(format!("   {total} match"), theme.typography.dim),
        ]));
        if total == 0 {
            lines.push(Line::from(Span::styled("  No models match — Esc clears the filter", theme.typography.dim)));
            self.render_model_notes(lines, theme);
            return;
        }
        // Labeled columns so the id/provider/family columns are scannable. The
        // header also carries the live "row N/total" position so the user always
        // knows where the cursor sits in a list that scrolls past the viewport.
        let position = self.model_cursor.min(total.saturating_sub(1)).saturating_add(1);
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<22} {:<9} family/class", "model", "provider"),
                theme.typography.key_hint,
            ),
            Span::styled(format!("   {position}/{total}"), theme.typography.dim),
        ]));
        // Slide a fixed-height window so the selected row is always visible: the
        // old `take(max_rows)` rendered from index 0, so moving the cursor below
        // the fold scrolled the highlight (and the silently-changing pin) off
        // screen. `visible_window` centers the cursor and clamps at both ends.
        let max_rows = usize::from(height.saturating_sub(12)).clamp(3, 10);
        let (start, end) = visible_window(total, self.model_cursor, max_rows);
        if start > 0 {
            lines.push(Line::from(Span::styled(
                format!("  ↑ {start} more above"),
                theme.typography.dim,
            )));
        }
        for (idx, candidate) in matches.iter().enumerate().skip(start).take(end.saturating_sub(start)) {
            let selected = idx == self.model_cursor;
            let is_current = candidate.id == *model;
            let marker = if selected { "> " } else if is_current { "● " } else { "  " };
            let style = if selected {
                selected_style(theme)
            } else if is_current {
                theme.typography.bold
            } else {
                theme.typography.body
            };
            lines.push(Line::from(vec![
                Span::styled(marker, style),
                Span::styled(format!("{:<22}", truncate(&candidate.id, 22)), style),
                Span::styled(format!(" {:<9}", truncate(&candidate.provider, 9)), theme.typography.dim),
                Span::styled(format!(" {}/{}", candidate.family, candidate.class), theme.typography.dim),
            ]));
        }
        if end < total {
            lines.push(Line::from(Span::styled(
                format!("  ↓ {} more — ↑↓ browse · / filter", total - end),
                theme.typography.dim,
            )));
        }
        self.render_model_notes(lines, theme);
    }

    fn render_model_notes(&self, lines: &mut Vec<Line<'static>>, theme: &Theme) {
        if self.model_notes.is_empty() {
            return;
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Hidden providers", theme.typography.bold)));
        for note in self.model_notes.iter().take(3) {
            lines.push(Line::from(Span::styled(format!("  {note}"), theme.typography.dim)));
        }
        if self.model_notes.len() > 3 {
            lines.push(Line::from(Span::styled(
                format!("  … {} more", self.model_notes.len() - 3),
                theme.typography.dim,
            )));
        }
    }

    fn render_family_editor(
        &self,
        lines: &mut Vec<Line<'static>>,
        provider: &str,
        family: &str,
        class: &str,
        freshness: SmartSettingsFreshness,
        theme: &Theme,
    ) {
        lines.push(Line::from(Span::styled("Family selector", theme.typography.bold)));
        for (idx, (label, value)) in [
            ("Provider", provider.to_string()),
            ("Family", family.to_string()),
            ("Class", class.to_string()),
            ("Freshness", freshness.label().to_string()),
        ]
        .into_iter()
        .enumerate()
        {
            let cursor = self.editor_cursor == idx + 1 && self.pane == ActivePane::Detail;
            let style = if cursor { selected_style(theme) } else { theme.typography.body };
            lines.push(Line::from(vec![
                Span::styled(if cursor { "> " } else { "  " }, style),
                Span::styled(format!("{label:<10}"), theme.typography.dim),
                Span::styled(format!("[ {:<18} ]", truncate(&value, 18)), style),
                Span::styled(" ←→", theme.typography.key_hint),
            ]));
        }
    }


    fn render_footer(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        // Only the keys relevant to the focused pane, so the footer is not a wall
        // of twelve shortcuts. Global toggles (Space/d/b) live in the header bar.
        let context: &[(&str, &str)] = match self.pane {
            ActivePane::Targets => &[
                ("↑↓", "pick"),
                ("←→", "Roles/Subagents"),
                ("Tab", "edit"),
                ("a", "auto"),
                ("A", "auto all"),
            ],
            ActivePane::Detail => &[
                ("↑↓←→", "adjust"),
                ("a/p/f", "mode"),
                ("/", "filter models"),
                ("r", "revert"),
                ("Tab", "back to list"),
            ],
        };
        let mut hints: Vec<(&str, &str)> = context.to_vec();
        hints.push(("Enter", "save"));
        hints.push(("Esc", "cancel"));
        let lines = vec![
            Line::from(vec![
                Span::styled("Settings ", theme.typography.dim),
                Span::styled(
                    truncate(&self.settings_path, usize::from(area.width.saturating_sub(10))),
                    theme.typography.key_hint,
                ),
            ]),
            Line::default(),
            key_hint_footer(theme, &hints),
        ];
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), area);
    }

    fn move_left(&mut self) {
        match self.pane {
            ActivePane::Targets => self.switch_tab(self.tab.prev()),
            ActivePane::Detail => self.editor_left(),
        }
    }

    fn move_right(&mut self) {
        match self.pane {
            ActivePane::Targets => self.switch_tab(self.tab.next()),
            ActivePane::Detail => self.editor_right(),
        }
    }

    fn move_up(&mut self) {
        match self.pane {
            ActivePane::Targets => self.move_target(-1),
            ActivePane::Detail => self.editor_up(),
        }
    }

    fn move_down(&mut self) {
        match self.pane {
            ActivePane::Targets => self.move_target(1),
            ActivePane::Detail => self.editor_down(),
        }
    }

    fn move_home(&mut self) {
        match self.pane {
            ActivePane::Targets => self.set_current_cursor(0),
            ActivePane::Detail => self.editor_home(),
        }
    }

    fn move_end(&mut self) {
        match self.pane {
            ActivePane::Targets => self.set_current_cursor(self.current_targets().len().saturating_sub(1)),
            ActivePane::Detail => self.editor_end(),
        }
    }

    fn editor_left(&mut self) {
        if self.editor_cursor == 0 {
            let mode = self.current_target().map_or(RoutingMode::Auto, |target| target.current.mode());
            self.set_current_mode(mode.prev());
            return;
        }
        if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin) {
            self.model_cursor = self.model_cursor.saturating_sub(1);
            self.apply_selected_model();
        } else {
            self.cycle_family_field(false);
        }
    }

    fn editor_right(&mut self) {
        if self.editor_cursor == 0 {
            let mode = self.current_target().map_or(RoutingMode::Auto, |target| target.current.mode());
            self.set_current_mode(mode.next());
            return;
        }
        if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin) {
            let count = self.filtered_model_count();
            if count > 0 {
                self.model_cursor = (self.model_cursor + 1).min(count - 1);
            }
            self.apply_selected_model();
        } else {
            self.cycle_family_field(true);
        }
    }

    fn editor_up(&mut self) {
        if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin)
            && self.editor_cursor == 1
        {
            if self.model_cursor == 0 {
                self.editor_cursor = 0;
            } else {
                self.model_cursor -= 1;
                self.apply_selected_model();
            }
            return;
        }
        self.editor_cursor = self.editor_cursor.saturating_sub(1);
    }

    fn editor_down(&mut self) {
        if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin)
            && self.editor_cursor == 1
        {
            let count = self.filtered_model_count();
            if count > 0 {
                self.model_cursor = (self.model_cursor + 1).min(count - 1);
                self.apply_selected_model();
            }
            return;
        }
        let max = self.editor_cursor_max();
        self.editor_cursor = (self.editor_cursor + 1).min(max);
    }

    fn editor_home(&mut self) {
        if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin)
            && self.editor_cursor == 1
            && self.filtered_model_count() > 0
        {
            self.model_cursor = 0;
            self.apply_selected_model();
            return;
        }
        self.editor_cursor = 0;
    }

    fn editor_end(&mut self) {
        if self.current_target().is_some_and(|target| target.current.mode() == RoutingMode::Pin)
            && self.editor_cursor == 1
        {
            let count = self.filtered_model_count();
            if count > 0 {
                self.model_cursor = count - 1;
                self.apply_selected_model();
            }
            return;
        }
        self.editor_cursor = self.editor_cursor_max();
    }

    fn editor_cursor_max(&self) -> usize {
        match self.current_target().map(|target| target.current.mode()) {
            Some(RoutingMode::Family) => FAMILY_FIELD_COUNT,
            Some(RoutingMode::Pin) => 1,
            _ => 0,
        }
    }

    fn set_current_mode(&mut self, mode: RoutingMode) {
        let default_model = self.default_model_id();
        let default_family = self.default_family_values();
        if let Some(target) = self.current_target_mut() {
            target.current = match mode {
                RoutingMode::Auto => SmartSettingsUpdate::Auto,
                RoutingMode::Pin => SmartSettingsUpdate::ExactPin { model: default_model },
                RoutingMode::Family => SmartSettingsUpdate::FamilyLock {
                    provider: default_family.0,
                    family: default_family.1,
                    class: default_family.2,
                    freshness: SmartSettingsFreshness::LatestStable,
                },
            };
        }
        self.editor_cursor = self.editor_cursor.min(self.editor_cursor_max());
    }

    fn cycle_family_field(&mut self, forward: bool) {
        let providers = self.unique_values(|model| model.provider.as_str());
        let families = self.unique_values(|model| model.family.as_str());
        let classes = self.unique_values(|model| model.class.as_str());
        let cursor = self.editor_cursor;
        let Some(target) = self.current_target_mut() else { return; };
        let SmartSettingsUpdate::FamilyLock { provider, family, class, freshness } = &mut target.current else {
            return;
        };
        match cursor {
            1 => cycle_string(provider, &providers, forward),
            2 => cycle_string(family, &families, forward),
            3 => cycle_string(class, &classes, forward),
            4 => *freshness = freshness.cycle(),
            _ => {}
        }
    }

    fn apply_selected_model(&mut self) {
        let Some(model) = self.filtered_models().get(self.model_cursor).cloned() else {
            return;
        };
        if let Some(target) = self.current_target_mut() {
            target.current = SmartSettingsUpdate::ExactPin { model: model.id };
        }
    }

    fn reset_current_target(&mut self) {
        if let Some(target) = self.current_target_mut() {
            target.current = target.original.clone();
        }
    }

    fn apply_recommended_setup(&mut self) {
        self.enabled = true;
        for target in &mut self.roles {
            target.current = SmartSettingsUpdate::Auto;
        }
        for target in &mut self.subagents {
            target.current = SmartSettingsUpdate::Auto;
        }
    }

    fn commit(&self) -> SmartSettingsCommit {
        SmartSettingsCommit {
            enabled: self.enabled,
            allow_cross_provider_diversity: self.policy.allow_cross_provider_diversity,
            feedback_informed_auto: self.policy.feedback_informed_auto,
            roles: self
                .roles
                .iter()
                .map(|target| (target.key.clone(), target.current.clone()))
                .collect(),
            subagents: self
                .subagents
                .iter()
                .map(|target| (target.key.clone(), target.current.clone()))
                .collect(),
        }
    }

    fn current_targets(&self) -> &[TargetState] {
        match self.tab {
            TargetTab::Roles => &self.roles,
            TargetTab::Subagents => &self.subagents,
        }
    }

    fn current_targets_mut(&mut self) -> &mut [TargetState] {
        match self.tab {
            TargetTab::Roles => &mut self.roles,
            TargetTab::Subagents => &mut self.subagents,
        }
    }

    fn current_cursor(&self) -> usize {
        match self.tab {
            TargetTab::Roles => self.role_cursor,
            TargetTab::Subagents => self.subagent_cursor,
        }
    }

    fn set_current_cursor(&mut self, cursor: usize) {
        match self.tab {
            TargetTab::Roles => self.role_cursor = cursor.min(self.roles.len().saturating_sub(1)),
            TargetTab::Subagents => self.subagent_cursor = cursor.min(self.subagents.len().saturating_sub(1)),
        }
    }

    fn move_target(&mut self, delta: isize) {
        let len = self.current_targets().len();
        if len == 0 {
            self.set_current_cursor(0);
            return;
        }
        let current = self.current_cursor();
        let next = if delta.is_negative() {
            current.saturating_sub(delta.unsigned_abs())
        } else {
            current.saturating_add(delta.unsigned_abs()).min(len.saturating_sub(1))
        };
        self.set_current_cursor(next);
    }

    fn current_target(&self) -> Option<&TargetState> {
        self.current_targets().get(self.current_cursor())
    }

    fn current_target_mut(&mut self) -> Option<&mut TargetState> {
        let cursor = self.current_cursor();
        self.current_targets_mut().get_mut(cursor)
    }

    fn switch_tab(&mut self, tab: TargetTab) {
        self.tab = tab;
        self.editor_cursor = self.editor_cursor.min(self.editor_cursor_max());
    }

    fn filtered_models(&self) -> Vec<SmartSettingsModel> {
        if self.model_filter.trim().is_empty() {
            return self.models.clone();
        }
        let needle = self.model_filter.to_ascii_lowercase();
        self.models
            .iter()
            .filter(|model| {
                let haystack = format!("{} {} {} {}", model.id, model.provider, model.family, model.class)
                    .to_ascii_lowercase();
                haystack.contains(&needle)
            })
            .cloned()
            .collect()
    }

    fn filtered_model_count(&self) -> usize {
        self.filtered_models().len()
    }

    fn unique_values(&self, value: impl Fn(&SmartSettingsModel) -> &str) -> Vec<String> {
        let mut values = self
            .models
            .iter()
            .map(value)
            .filter(|item| !item.trim().is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        values.sort();
        values.dedup();
        values
    }

    fn default_model_id(&self) -> String {
        self.models
            .first()
            .map_or_else(|| self.main_model.clone(), |model| model.id.clone())
    }

    fn default_family_values(&self) -> (String, String, String) {
        self.models.first().map_or_else(
            || ("unknown".to_string(), "custom".to_string(), "balanced".to_string()),
            |model| (model.provider.clone(), model.family.clone(), model.class.clone()),
        )
    }

    /// Provider that `target` currently resolves to, used by the mix chart.
    /// Family locks name their provider directly; everything else is looked up
    /// from the resolved model id, falling back to `other` when unknown.
    fn resolved_provider(&self, target: &TargetState) -> String {
        if let SmartSettingsUpdate::FamilyLock { provider, .. } = &target.current {
            return provider.clone();
        }
        let model_id = self.preview_for(target).model;
        self.models
            .iter()
            .find(|model| model.id == model_id)
            .map_or_else(|| "other".to_string(), |model| model.provider.clone())
    }

    /// Provider of the current main model, if it is in the usable pool.
    fn main_provider(&self) -> Option<String> {
        self.models
            .iter()
            .find(|model| model.id == self.main_model)
            .map(|model| model.provider.clone())
    }

    /// Provider distribution across every role and subagent, descending by count
    /// then name (deterministic). One entry per distinct resolved provider.
    fn provider_mix(&self) -> Vec<(String, usize)> {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for target in self.roles.iter().chain(self.subagents.iter()) {
            let provider = self.resolved_provider(target);
            if let Some(entry) = counts.iter_mut().find(|(name, _)| *name == provider) {
                entry.1 += 1;
            } else {
                counts.push((provider, 1));
            }
        }
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        counts
    }

    fn recommendation_for(&self, target: &TargetState) -> Option<&SmartSettingsRecommendation> {
        // Pick the set matching the *currently staged* diversity toggle so the
        // preview reflects `d` immediately, before the user saves.
        let set = if self.policy.allow_cross_provider_diversity {
            &self.recommendations_with_diversity
        } else {
            &self.recommendations
        };
        set.iter().find(|rec| rec.kind == target.kind && rec.key == target.key)
    }

    fn preview_for(&self, target: &TargetState) -> PreviewDetails {
        match &target.current {
            SmartSettingsUpdate::Auto => self.recommendation_for(target).map_or_else(
                || PreviewDetails {
                    model: self.main_model.clone(),
                    confidence: "Low".to_string(),
                    source: "main fallback".to_string(),
                    reason: "no auto recommendation is available for this target".to_string(),
                    audit: vec!["auto recommendation unavailable".to_string()],
                },
                |rec| PreviewDetails {
                    model: rec.selected_model.clone(),
                    confidence: rec.confidence.clone(),
                    source: "auto recommendation".to_string(),
                    reason: rec.reason.clone(),
                    audit: rec.audit.clone(),
                },
            ),
            SmartSettingsUpdate::ExactPin { model } => PreviewDetails {
                model: model.clone(),
                confidence: if self.models.iter().any(|entry| entry.id == *model) { "High" } else { "Low" }.to_string(),
                source: "exact pin".to_string(),
                reason: if self.models.iter().any(|entry| entry.id == *model) {
                    "pinned model is in the usable pool".to_string()
                } else {
                    "pinned model is not currently in the usable pool; runtime will fallback".to_string()
                },
                audit: Vec::new(),
            },
            SmartSettingsUpdate::FamilyLock { provider, family, class, freshness } => {
                let selected = self.models.iter().find(|model| {
                    model.provider == *provider && model.family == *family && model.class == *class
                });
                PreviewDetails {
                    model: selected.map_or_else(
                        || format!("{provider}/{family}/{class}"),
                        |model| model.id.clone(),
                    ),
                    confidence: if selected.is_some() { "Medium" } else { "Low" }.to_string(),
                    source: "family selector".to_string(),
                    reason: format!("tracks {provider}/{family}/{class}/{}", freshness.label()),
                    audit: Vec::new(),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreviewDetails {
    model: String,
    confidence: String,
    source: String,
    reason: String,
    audit: Vec<String>,
}

/// Block-glyph sparkline of `series` (most recent `width` samples), max-relative
/// so the tallest bar is always full height. Mirrors the sidebar's agent
/// sparkline: degrades to a flat `#` run under `NO_COLOR`, where the 8-step ramp
/// would be indistinguishable.
fn usage_sparkline(series: &[u32], width: usize, theme: &Theme) -> String {
    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if series.is_empty() || width == 0 {
        return String::new();
    }
    let recent = if series.len() > width {
        &series[series.len() - width..]
    } else {
        series
    };
    if theme.no_color {
        return "#".repeat(recent.len());
    }
    let max = recent.iter().copied().max().unwrap_or(1).max(1);
    recent
        .iter()
        .map(|value| {
            let scaled = (u64::from(*value) * (GLYPHS.len() as u64 - 1)) / u64::from(max);
            let idx = usize::try_from(scaled).unwrap_or(0).min(GLYPHS.len() - 1);
            GLYPHS[idx]
        })
        .collect()
}

/// Compact token count: `1234` → `1.2k`, `2_500_000` → `2.5M`. Integer math
/// only (no float casts) so it stays clippy-clean under the precision lints.
fn fmt_compact(value: u32) -> String {
    if value >= 1_000_000 {
        format!("{}.{}M", value / 1_000_000, (value % 1_000_000) / 100_000)
    } else if value >= 1_000 {
        format!("{}.{}k", value / 1_000, (value % 1_000) / 100)
    } else {
        value.to_string()
    }
}

/// `[key] Label: on/off` spans for the global settings bar, with the value
/// colored by state so toggles are scannable at a glance.
fn toggle_spans(key: &'static str, label: &'static str, on: bool, theme: &Theme) -> Vec<Span<'static>> {
    let value_style = if on {
        Style::new().fg(theme.palette.success).add_modifier(Modifier::BOLD)
    } else {
        theme.typography.dim
    };
    vec![
        Span::styled(format!("[{key}] "), theme.typography.key_hint),
        Span::styled(format!("{label}: "), theme.typography.body),
        Span::styled(if on { "on" } else { "off" }, value_style),
    ]
}

fn pane_block<'a>(title: &'static str, active: bool, theme: &'a Theme) -> Block<'a> {
    let style = if active {
        Style::new().fg(theme.palette.accent).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(theme.palette.dim)
    };
    CardFrame::new(SurfaceKind::Panel, theme)
        .border_style(style)
        .title(format!(" {title} "))
        .block()
}

fn render_tabs(tab: TargetTab, theme: &Theme) -> Line<'static> {
    let mut spans = Vec::new();
    for item in [TargetTab::Roles, TargetTab::Subagents] {
        let active = item == tab;
        let style = if active { selected_style(theme) } else { theme.typography.dim };
        spans.push(Span::styled(
            if active { format!("[{}] ", item.label()) } else { format!(" {}  ", item.label()) },
            style,
        ));
    }
    Line::from(spans)
}

fn badge_style(update: &SmartSettingsUpdate, theme: &Theme) -> Style {
    match update {
        SmartSettingsUpdate::Auto => Style::new().fg(theme.palette.success),
        SmartSettingsUpdate::ExactPin { .. } => Style::new().fg(theme.palette.violet).add_modifier(Modifier::BOLD),
        SmartSettingsUpdate::FamilyLock { .. } => Style::new().fg(theme.palette.cyan).add_modifier(Modifier::BOLD),
    }
}

fn confidence_meter(confidence: &str) -> &'static str {
    match confidence {
        "High" => "███",
        "Medium" => "██-",
        _ => "█--",
    }
}

fn target_kind_label(kind: SmartSettingsTargetKind) -> &'static str {
    match kind {
        SmartSettingsTargetKind::Role => "role fallback",
        SmartSettingsTargetKind::Subagent => "subagent",
    }
}

fn visible_window(len: usize, selected: usize, visible_rows: usize) -> (usize, usize) {
    if len == 0 || visible_rows == 0 {
        return (0, 0);
    }
    let half = visible_rows / 2;
    let start = selected.saturating_sub(half).min(len.saturating_sub(visible_rows));
    let end = (start + visible_rows).min(len);
    (start, end)
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let mut out = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        out.push('…');
    }
    out
}

fn cycle_string(value: &mut String, options: &[String], forward: bool) {
    if options.is_empty() {
        return;
    }
    let current = options.iter().position(|candidate| candidate == value).unwrap_or(0);
    let next = if forward {
        (current + 1) % options.len()
    } else {
        (current + options.len() - 1) % options.len()
    };
    value.clone_from(&options[next]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEventState, KeyModifiers};

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent { code, modifiers: KeyModifiers::empty(), kind: KeyEventKind::Press, state: KeyEventState::empty() }
    }

    fn sample_modal() -> SmartSettingsModal {
        SmartSettingsModal::new(SmartSettingsView {
            enabled: false,
            allow_cross_provider_diversity: false,
            feedback_informed_auto: false,
            auto_classifier: "deterministic".to_string(),
            main_model: "main-model".to_string(),
            settings_path: "/tmp/settings.json".to_string(),
            models: vec![
                SmartSettingsModel { id: "fast-model".to_string(), provider: "openai".to_string(), family: "gpt".to_string(), class: "fast".to_string() },
                SmartSettingsModel { id: "code-model".to_string(), provider: "openai".to_string(), family: "gpt".to_string(), class: "coding".to_string() },
            ],
            model_notes: Vec::new(),
            roles: vec![SmartSettingsTarget { key: "coding".to_string(), label: "Coding".to_string(), kind: SmartSettingsTargetKind::Role, update: SmartSettingsUpdate::Auto }],
            subagents: vec![SmartSettingsTarget { key: "Verification".to_string(), label: "Verification".to_string(), kind: SmartSettingsTargetKind::Subagent, update: SmartSettingsUpdate::Auto }],
            recommendations: vec![SmartSettingsRecommendation { kind: SmartSettingsTargetKind::Subagent, key: "Verification".to_string(), selected_model: "code-model".to_string(), confidence: "High".to_string(), reason: "verification fit".to_string(), audit: vec!["cross-provider diversity: disabled by default".to_string()] }],
            recommendations_with_diversity: vec![SmartSettingsRecommendation { kind: SmartSettingsTargetKind::Subagent, key: "Verification".to_string(), selected_model: "fast-model".to_string(), confidence: "High".to_string(), reason: "verification fit; cross-provider diversity explicitly allowed".to_string(), audit: vec!["cross-provider diversity: allowed".to_string()] }],
            observed_routes: vec![SmartSettingsObservedRoute { kind: SmartSettingsTargetKind::Subagent, key: "Verification".to_string(), completed: 8, decisive: 9, model: Some("code-model".to_string()) }],
            turn_output_tokens: vec![120, 340, 90, 510, 220],
        })
    }

    #[test]
    fn enter_returns_staged_commit() {
        let mut modal = sample_modal();
        modal.handle_key(press(KeyCode::Char(' ')));
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        let Some(ModalResult::Selected(ModalSelection::SmartSettings(commit))) = modal.handle_key(press(KeyCode::Enter)) else {
            panic!("expected smart settings commit");
        };
        assert!(commit.enabled);
        assert_eq!(commit.roles.len(), 1);
        assert!(matches!(commit.roles[0].1, SmartSettingsUpdate::ExactPin { .. }));
    }

    #[test]
    fn recommended_shortcut_resets_overrides_and_enables() {
        let mut modal = sample_modal();
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        assert_eq!(modal.pending_change_count(), 1);
        modal.handle_key(press(KeyCode::Char('A')));
        assert!(modal.enabled);
        assert!(modal.roles.iter().all(|target| matches!(target.current, SmartSettingsUpdate::Auto)));
    }

    #[test]
    fn slash_filters_pin_models_without_saving() {
        let mut modal = sample_modal();
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        modal.handle_key(press(KeyCode::Char('/')));
        modal.handle_key(press(KeyCode::Char('c')));
        assert!(modal.filtering_models);
        modal.handle_key(press(KeyCode::Enter));
        let Some(target) = modal.current_target() else { panic!("target"); };
        assert!(matches!(&target.current, SmartSettingsUpdate::ExactPin { model } if model == "code-model"));
    }

    #[test]
    fn lowercase_a_sets_only_current_target_to_auto() {
        let mut modal = sample_modal();
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        assert_eq!(modal.pending_change_count(), 1);

        modal.handle_key(press(KeyCode::Char('a')));

        assert!(!modal.enabled, "lowercase a must not enable global recommended setup");
        let Some(target) = modal.current_target() else { panic!("target"); };
        assert!(matches!(target.current, SmartSettingsUpdate::Auto));
    }

    #[test]
    fn pin_model_list_moves_with_arrow_keys() {
        let mut modal = sample_modal();
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        modal.handle_key(press(KeyCode::Down));
        modal.handle_key(press(KeyCode::Down));

        let Some(target) = modal.current_target() else { panic!("target"); };
        assert!(matches!(&target.current, SmartSettingsUpdate::ExactPin { model } if model == "code-model"));

        modal.handle_key(press(KeyCode::Up));
        let Some(target) = modal.current_target() else { panic!("target"); };
        assert!(matches!(&target.current, SmartSettingsUpdate::ExactPin { model } if model == "fast-model"));
    }

    #[test]
    fn pin_model_list_scrolls_to_keep_selected_row_visible() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // A pool larger than any viewport window: the old `take(max_rows)` rendered
        // from index 0, so a cursor past the fold scrolled the highlight off screen
        // (the reported "목록이 다 표시되지 않는" bug). The window must now slide.
        let models: Vec<SmartSettingsModel> = (0..20)
            .map(|i| SmartSettingsModel {
                id: format!("model-{i:02}"),
                provider: "openai".to_string(),
                family: "gpt".to_string(),
                class: "coding".to_string(),
            })
            .collect();
        let mut modal = SmartSettingsModal::new(SmartSettingsView {
            enabled: true,
            allow_cross_provider_diversity: false,
            feedback_informed_auto: false,
            auto_classifier: "deterministic".to_string(),
            main_model: "main-model".to_string(),
            settings_path: "/tmp/settings.json".to_string(),
            models,
            model_notes: Vec::new(),
            roles: vec![SmartSettingsTarget { key: "coding".to_string(), label: "Coding".to_string(), kind: SmartSettingsTargetKind::Role, update: SmartSettingsUpdate::Auto }],
            subagents: vec![SmartSettingsTarget { key: "Verification".to_string(), label: "Verification".to_string(), kind: SmartSettingsTargetKind::Subagent, update: SmartSettingsUpdate::Auto }],
            recommendations: Vec::new(),
            recommendations_with_diversity: Vec::new(),
            observed_routes: Vec::new(),
            turn_output_tokens: Vec::new(),
        });

        // Tab into the Detail pane, enter pin mode on the role, then drive the
        // cursor deep past the fold. The first Down moves focus from the mode row
        // into the model list; each subsequent Down advances the model cursor.
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        for _ in 0..19 {
            modal.handle_key(press(KeyCode::Down));
        }
        let Some(target) = modal.current_target() else { panic!("target"); };
        let SmartSettingsUpdate::ExactPin { model } = &target.current else {
            panic!("expected pin");
        };
        let selected = model.clone();
        assert_eq!(selected, "model-18");

        let mut term = Terminal::new(TestBackend::new(120, 40)).expect("backend");
        let theme = Theme::zo();
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 40), &theme))
            .expect("draw");
        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }

        // The deep-selected row is on screen (the window followed the cursor)...
        assert!(dump.contains(&selected), "selected row scrolled off screen:\n{dump}");
        // ...the position counter reflects where we are...
        assert!(dump.contains("19/20"), "position counter missing:\n{dump}");
        // ...and the top of the list is hidden behind an overflow indicator.
        assert!(dump.contains("more above"), "above-overflow indicator missing:\n{dump}");
        assert!(!dump.contains("model-00"), "top row should be scrolled out:\n{dump}");
    }


    #[test]
    fn global_policy_toggles_are_staged_in_commit() {
        let mut modal = sample_modal();
        modal.handle_key(press(KeyCode::Char('d')));
        modal.handle_key(press(KeyCode::Char('b')));
        assert_eq!(modal.pending_change_count(), 2);

        let Some(ModalResult::Selected(ModalSelection::SmartSettings(commit))) = modal.handle_key(press(KeyCode::Enter)) else {
            panic!("expected smart settings commit");
        };
        assert!(commit.allow_cross_provider_diversity);
        assert!(commit.feedback_informed_auto);
    }

    #[test]
    fn auto_preview_carries_recommendation_audit() {
        let modal = sample_modal();
        let target = modal.subagents.first().expect("subagent target");
        let preview = modal.preview_for(target);
        assert_eq!(preview.model, "code-model");
        assert!(preview.audit.iter().any(|line| line.contains("disabled by default")));
    }

    #[test]
    fn resolution_surfaces_observed_outcome() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let theme = Theme::zo();
        let mut modal = sample_modal();
        // Select the Verification subagent, which has an observed outcome (8/9).
        modal.handle_key(press(KeyCode::Right));

        let backend = TestBackend::new(120, 40);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 40), &theme))
            .expect("draw");
        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        assert!(dump.contains("observed"), "observed line missing:\n{dump}");
        assert!(dump.contains("8/9 ok"), "observed counts missing:\n{dump}");
    }

    #[test]
    fn diversity_toggle_flips_recommendation_preview_live() {
        let mut modal = sample_modal();
        // Move from the Roles tab to Subagents and select the Verification target.
        modal.handle_key(press(KeyCode::Right));
        let target = modal.current_target().expect("subagent target").clone();

        let before = modal.preview_for(&target);
        assert_eq!(before.model, "code-model", "diversity off → off recommendation");

        modal.handle_key(press(KeyCode::Char('d')));
        assert!(modal.policy.allow_cross_provider_diversity);
        let after = modal.preview_for(&target);
        assert_eq!(after.model, "fast-model", "toggling d previews the diversity recommendation live");
    }

    #[test]
    fn dashboard_renders_provider_mix_chart() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let theme = Theme::zo();
        let backend = TestBackend::new(120, 40);
        let mut term = Terminal::new(backend).expect("backend");
        let modal = sample_modal();
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 40), &theme))
            .expect("draw smart modal");

        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        assert!(dump.contains("Provider mix"), "chart title missing:\n{dump}");
        assert!(dump.contains('█'), "chart bar missing:\n{dump}");
        // Two targets resolve (one openai recommendation, one main fallback).
        assert!(dump.contains("2 targets"), "chart total missing:\n{dump}");
    }

    #[test]
    fn dashboard_renders_usage_sparkline() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let theme = Theme::zo();
        let backend = TestBackend::new(120, 40);
        let mut term = Terminal::new(backend).expect("backend");
        let modal = sample_modal();
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 40), &theme))
            .expect("draw smart modal");

        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        assert!(dump.contains("Output tokens"), "usage title missing:\n{dump}");
        assert!(dump.contains("5 turns"), "turn count missing:\n{dump}");
        // The sparkline ramp includes the tallest glyph for the peak sample.
        assert!(dump.contains('█'), "sparkline glyph missing:\n{dump}");
    }

    #[test]
    fn usage_sparkline_scales_and_degrades() {
        let theme = Theme::zo();
        let spark = usage_sparkline(&[0, 50, 100], 8, &theme);
        // Max-relative: smallest sample is the floor glyph, peak is full block.
        assert!(spark.starts_with('▁'), "min glyph: {spark}");
        assert!(spark.ends_with('█'), "max glyph: {spark}");

        let mut mono = theme.clone();
        mono.no_color = true;
        assert_eq!(usage_sparkline(&[1, 2, 3], 8, &mono), "###");
        assert_eq!(usage_sparkline(&[], 8, &theme), "");
    }

    #[test]
    fn fmt_compact_thresholds() {
        assert_eq!(fmt_compact(512), "512");
        assert_eq!(fmt_compact(1_234), "1.2k");
        assert_eq!(fmt_compact(2_500_000), "2.5M");
    }

    #[test]
    fn pin_picker_surfaces_count_columns_and_saved_marker() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let theme = Theme::zo();
        let mut modal = sample_modal();
        // Enter pin mode (pins the first model), then browse the filter list down
        // one row WITHOUT committing — the cursor now sits off the saved pin.
        modal.handle_key(press(KeyCode::Tab));
        modal.handle_key(press(KeyCode::Char('p')));
        modal.handle_key(press(KeyCode::Char('/')));
        modal.handle_key(press(KeyCode::Down));

        let backend = TestBackend::new(120, 40);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 40), &theme))
            .expect("draw pin picker");

        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        assert!(dump.contains("2 match"), "match count missing:\n{dump}");
        assert!(dump.contains("provider"), "column header missing:\n{dump}");
        // The saved pin (fast-model) is marked distinctly while the cursor is elsewhere.
        assert!(dump.contains('●'), "saved-pin marker missing:\n{dump}");
    }

    #[test]
    fn header_bar_surfaces_global_settings() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let theme = Theme::zo();
        let backend = TestBackend::new(120, 40);
        let mut term = Terminal::new(backend).expect("backend");
        let modal = sample_modal();
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 40), &theme))
            .expect("draw smart modal");

        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        // The global settings bar surfaces the toggles and classifier up top.
        assert!(dump.contains("Classifier: deterministic"), "{dump}");
        assert!(dump.contains("Cross-provider"), "{dump}");
        assert!(dump.contains("Feedback"), "{dump}");
    }

}
