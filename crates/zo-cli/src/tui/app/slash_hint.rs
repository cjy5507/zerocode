//! Slash-command hint popup that surfaces above the input row.
//!
//! [`draw_slash_hint`] renders the live suggestion list (sourced from
//! `commands::slash_help`) when the user's buffer begins with `/`.
//! [`slash_completion_for`] is the matching Tab/Space auto-expander
//! used by the input handler.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use unicode_width::UnicodeWidthStr;

use crate::tui::cards::{CardFrame, SurfaceKind};
use crate::tui::fuzzy;
use crate::tui::input::InputWidget;
use crate::tui::keybindings::command_hint;
use crate::tui::theme::Theme;

use super::App;
use super::AppMode;

const SLASH_HINT_LABEL_WIDTH: usize = 14;
pub(super) const SLASH_HINT_LIMIT: usize = 10;
const PROMPT_COMMAND_RISK: &str = "Medium: effect depends on the prompt body and available tools";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SlashHintSuggestion {
    pub(super) command: String,
    pub(super) summary: String,
    pub(super) risk: Option<&'static str>,
    /// Char positions within `command` (the `/name` label) that the query
    /// matched, used to highlight the match. Empty when the match came from the
    /// summary/aliases or when there is no active query.
    pub(super) matched_indices: Vec<usize>,
    /// `true` when the command takes a *required* argument (`<arg>`). Picking
    /// such a command from the hint fills the input and waits for the argument
    /// instead of running immediately.
    pub(super) requires_arg: bool,
}

/// A command needs an inline argument before it can run only when its hint
/// names a *required* placeholder (`<arg>`). Optional `[arg]` hints — including
/// the bare pickers like `/model [model]` — run immediately and open their own
/// modal or act on a default, matching the Claude Code `/` experience.
fn requires_argument(hint: Option<&str>) -> bool {
    hint.is_some_and(|h| h.trim_start().starts_with('<'))
}

/// Render a slash-command hint popup just above the input region when
/// the user's current buffer starts with `/`. Shows up to 10 matching
/// commands with their summaries, sourced from `commands::slash_help`.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_slash_hint(
    frame: &mut ratatui::Frame<'_>,
    input_area: Rect,
    inline_bounds: Option<Rect>,
    input: &InputWidget,
    prompt_commands: &[commands::PromptCommandDef],
    recent: &[String],
    theme: &Theme,
    mode: AppMode,
    selected: Option<usize>,
) {
    if !matches!(mode, AppMode::Normal) {
        return;
    }
    let buffer = input.text();
    let first_line = buffer.lines().next().unwrap_or("");
    let trimmed = first_line.trim_start();
    if !trimmed.starts_with('/') {
        return;
    }

    let suggestions = slash_hint_suggestions(trimmed, prompt_commands, recent, SLASH_HINT_LIMIT);
    if suggestions.is_empty() {
        return;
    }

    // Match the input-box width so the popup fully covers the input
    // row beneath it and lines up visually with the prompt below.
    // On very wide terminals we still cap at 90 columns to keep
    // summaries readable.
    let Some(mut popup_area) = slash_hint_popup_area(input_area, suggestions.len()) else {
        return;
    };
    if let Some(bounds) = inline_bounds {
        let Some(bounded) = fit_popup_above_input(popup_area, input_area, bounds) else {
            return;
        };
        popup_area = bounded;
    }
    let line_width = usize::from(popup_area.width.saturating_sub(2));

    // Dynamic label width: check the maximum length of suggested commands
    // and clamp it between 14 and 24 to keep things clean.
    let max_len = suggestions
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.command.as_str()))
        .max()
        .unwrap_or(SLASH_HINT_LABEL_WIDTH);
    let label_width = max_len.clamp(14, 24);

    // Resolve each suggestion's summary for a two-column line.
    let mut lines: Vec<Line<'_>> = Vec::with_capacity(suggestions.len());
    for (i, suggestion) in suggestions.iter().enumerate() {
        let is_selected = selected == Some(i);
        // Global accelerator for this command (e.g. `/agents` → ^A), shown
        // right-aligned and dim like the palette's key-hint column.
        let name = suggestion.command.trim_start_matches('/');
        let accel = command_hint(name, !theme.no_color);
        lines.push(render_slash_hint_line(
            &suggestion.command,
            &suggestion.matched_indices,
            &suggestion.summary,
            suggestion.risk,
            accel.as_deref(),
            line_width,
            theme,
            is_selected,
            label_width,
        ));
    }

    frame.render_widget(Clear, popup_area);
    let block = CardFrame::new(SurfaceKind::Popup, theme)
        .title(Span::styled(
            " /commands ",
            Style::default()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(
            " tab accept \u{00b7} enter run \u{00b7} esc close ",
            Style::default().fg(theme.palette.dim),
        ))
        .block();
    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

pub(super) fn slash_hint_suggestions(
    buffer: &str,
    prompt_commands: &[commands::PromptCommandDef],
    recent: &[String],
    limit: usize,
) -> Vec<SlashHintSuggestion> {
    if limit == 0 {
        return Vec::new();
    }
    let first_line = buffer.lines().next().unwrap_or("");
    let trimmed = first_line.trim_start();
    if !trimmed.starts_with('/') {
        return Vec::new();
    }

    let public_specs = commands::public_slash_command_specs();
    let query = trimmed.trim().trim_start_matches('/').to_ascii_lowercase();

    // Empty query (`/` alone): catalogue order, but float the user's
    // recently-run commands to the top so the hint inherits the old palette's
    // "Suggested" pins. `recent` is most-recent-first; the stable sort leaves
    // every other command in its original catalogue position.
    if query.is_empty() {
        let mut specs = public_specs.clone();
        specs.sort_by_key(|spec| {
            recent
                .iter()
                .position(|cmd| cmd.trim_start_matches('/') == spec.name)
                .unwrap_or(usize::MAX)
        });
        return specs
            .into_iter()
            .take(limit)
            .map(|spec| builtin_suggestion(spec, Vec::new()))
            .collect();
    }

    // Non-empty query: rank every command by the *same* subsequence rule the
    // palette uses, searching name + summary + aliases — so `/git` surfaces
    // `/diff`, `/commit`, `/pr` by their descriptions, not just by name.
    let mut scored: Vec<ScoredHint> = Vec::new();
    for spec in &public_specs {
        if let Some((rank, name_indices)) = rank_builtin(spec, &query) {
            scored.push(ScoredHint {
                rank,
                tiebreak: spec.name.len(),
                suggestion: builtin_suggestion(spec, name_indices),
            });
        }
    }
    for command in prompt_commands {
        // A project prompt command sharing a built-in's name is shown once.
        if public_specs
            .iter()
            .any(|spec| spec.name.eq_ignore_ascii_case(&command.name))
        {
            continue;
        }
        if let Some((rank, name_indices)) = rank_prompt(command, &query) {
            scored.push(ScoredHint {
                rank,
                tiebreak: command.name.len(),
                suggestion: prompt_suggestion(command, name_indices),
            });
        }
    }

    // Lower rank first (name-prefix < name-fuzzy < body), then shorter names, so
    // the top row is the most direct match.
    scored.sort_by(|a, b| a.rank.cmp(&b.rank).then(a.tiebreak.cmp(&b.tiebreak)));
    scored
        .into_iter()
        .take(limit)
        .map(|scored| scored.suggestion)
        .collect()
}

pub(super) fn slash_hint_popup_area(input_area: Rect, suggestion_count: usize) -> Option<Rect> {
    if input_area.y == 0 || input_area.width == 0 || suggestion_count == 0 {
        return None;
    }
    let popup_width = popup_width_for(input_area.width);
    // +2 for top/bottom border rows. The footer hint is embedded in the
    // bottom border.
    let popup_rows = u16::try_from(suggestion_count)
        .unwrap_or(u16::try_from(SLASH_HINT_LIMIT).unwrap_or(10))
        .saturating_add(2);
    let popup_y = input_area.y.saturating_sub(popup_rows);
    Some(Rect::new(input_area.x, popup_y, popup_width, popup_rows))
}

/// Reposition an above-input popup so every cell lands inside `bounds`.
///
/// Inline viewport frame buffers use terminal-absolute coordinates and can
/// start below row zero. A popup whose desired height exceeds the rows between
/// that origin and the input must shrink upward from the input, rather than
/// saturating toward terminal row zero (which is outside the frame buffer).
pub(super) fn fit_popup_above_input(
    popup_area: Rect,
    input_area: Rect,
    bounds: Rect,
) -> Option<Rect> {
    let anchor_y = input_area.y.min(bounds.bottom()).max(bounds.y);
    let height = popup_area
        .height
        .min(anchor_y.saturating_sub(bounds.y));
    if height == 0 {
        return None;
    }
    let repositioned = Rect::new(
        popup_area.x,
        anchor_y.saturating_sub(height),
        popup_area.width,
        height,
    );
    let bounded = repositioned.intersection(bounds);
    (bounded.width > 0 && bounded.height > 0).then_some(bounded)
}

/// One ranked slash-hint candidate before truncation to `limit`.
struct ScoredHint {
    /// 0 = query is a prefix of the name, 1 = query is a fuzzy subsequence of
    /// the name, 2 = query only matched the summary/aliases.
    rank: u8,
    /// Shorter names win ties so the most specific command floats up.
    tiebreak: usize,
    suggestion: SlashHintSuggestion,
}

/// Rank a built-in command against `query` (already lowercased): name prefix,
/// then name subsequence, then summary/alias subsequence. Returns the rank and
/// the matched char positions within the name (empty for a body-only match).
fn rank_builtin(spec: &commands::SlashCommandSpec, query: &str) -> Option<(u8, Vec<usize>)> {
    let name_lower = spec.name.to_ascii_lowercase();
    if name_lower.starts_with(query) {
        return Some((
            0,
            fuzzy::subsequence_indices(&name_lower, query).unwrap_or_default(),
        ));
    }
    if let Some(indices) = fuzzy::subsequence_indices(&name_lower, query) {
        return Some((1, indices));
    }
    let mut body = spec.summary.to_ascii_lowercase();
    for alias in spec.aliases {
        body.push(' ');
        body.push_str(&alias.to_ascii_lowercase());
    }
    fuzzy::is_subsequence(&body, query).then_some((2, Vec::new()))
}

/// Rank a project prompt command against `query`: name prefix, then name
/// subsequence, then summary subsequence.
fn rank_prompt(command: &commands::PromptCommandDef, query: &str) -> Option<(u8, Vec<usize>)> {
    let name_lower = command.name.to_ascii_lowercase();
    if name_lower.starts_with(query) {
        return Some((
            0,
            fuzzy::subsequence_indices(&name_lower, query).unwrap_or_default(),
        ));
    }
    if let Some(indices) = fuzzy::subsequence_indices(&name_lower, query) {
        return Some((1, indices));
    }
    let summary = command.summary().to_ascii_lowercase();
    fuzzy::is_subsequence(&summary, query).then_some((2, Vec::new()))
}

fn builtin_suggestion(
    spec: &commands::SlashCommandSpec,
    matched_indices: Vec<usize>,
) -> SlashHintSuggestion {
    SlashHintSuggestion {
        command: format!("/{}", spec.name),
        summary: spec.summary.to_string(),
        risk: commands::slash_command_metadata(spec.name).map(|metadata| metadata.risk),
        matched_indices,
        requires_arg: requires_argument(spec.argument_hint),
    }
}

fn prompt_suggestion(
    command: &commands::PromptCommandDef,
    matched_indices: Vec<usize>,
) -> SlashHintSuggestion {
    SlashHintSuggestion {
        command: format!("/{}", command.name),
        summary: command.summary(),
        risk: Some(PROMPT_COMMAND_RISK),
        matched_indices,
        requires_arg: requires_argument(command.argument_hint.as_deref()),
    }
}

fn popup_width_for(input_width: u16) -> u16 {
    if input_width < 32 {
        input_width
    } else {
        input_width.min(90)
    }
}

#[allow(clippy::too_many_arguments)]
fn render_slash_hint_line<'a>(
    suggestion: &str,
    matched: &[usize],
    summary: &'a str,
    risk: Option<&str>,
    accel: Option<&str>,
    width: usize,
    theme: &Theme,
    is_selected: bool,
    label_width: usize,
) -> Line<'a> {
    let indicator = if is_selected { "\u{25b8} " } else { "  " };
    let indicator_w = UnicodeWidthStr::width(indicator);
    let risk_badge = risk.and_then(risk_badge);
    let risk_width = risk_badge.map_or(0, |badge| UnicodeWidthStr::width(badge.text));
    // Reserve a trailing column for the accelerator (plus its leading space) so
    // the right-aligned key hint never collides with the summary.
    let accel_width = accel.map_or(0, |a| UnicodeWidthStr::width(a) + 1);
    let summary_width = width
        .saturating_sub(indicator_w)
        .saturating_sub(label_width)
        .saturating_sub(1)
        .saturating_sub(risk_width)
        .saturating_sub(accel_width);
    let label = pad_or_truncate_cells(suggestion, label_width);
    let summary = truncate_cells(summary, summary_width);
    let summary_w = UnicodeWidthStr::width(summary.as_str());

    let (indicator_style, label_style, summary_style) = if is_selected {
        let selected_bg = Style::default().bg(theme.palette.faint);
        (
            selected_bg.fg(theme.palette.dim),
            selected_bg
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
            selected_bg.fg(theme.palette.fg),
        )
    } else {
        (
            Style::default().fg(theme.palette.dim),
            Style::default()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(theme.palette.dim),
        )
    };
    // Matched name chars are brightened + underlined so the reason a command
    // surfaced reads at a glance, even on the selected row.
    let match_style = {
        let base = if is_selected {
            Style::default().bg(theme.palette.faint)
        } else {
            Style::default()
        };
        base.fg(theme.palette.bright)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    };

    let mut spans = Vec::with_capacity(label.chars().count() + 5);
    spans.push(Span::styled(indicator, indicator_style));
    // Per-char label: the leading `/` is char 0, so a name match index `n`
    // lands at label char `n + 1`.
    for (ci, ch) in label.chars().enumerate() {
        let hit = ci
            .checked_sub(1)
            .is_some_and(|name_idx| matched.contains(&name_idx));
        let style = if hit { match_style } else { label_style };
        spans.push(Span::styled(ch.to_string(), style));
    }
    spans.push(Span::styled(" ", summary_style));
    spans.push(Span::styled(summary, summary_style));
    if let Some(badge) = risk_badge {
        spans.push(Span::styled(badge.text, badge.style(theme, is_selected)));
    }
    if let Some(accel) = accel {
        // Pad to the right edge so every accelerator lines up in a column.
        let used = indicator_w + label_width + 1 + summary_w + risk_width;
        let pad = width.saturating_sub(used).saturating_sub(accel_width);
        if pad > 0 {
            spans.push(Span::raw(" ".repeat(pad)));
        }
        spans.push(Span::styled(
            format!(" {accel}"),
            Style::default().fg(theme.palette.dim),
        ));
    }
    Line::from(spans)
}

#[derive(Clone, Copy)]
struct SlashHintRiskBadge {
    text: &'static str,
    severity: SlashHintRiskSeverity,
}

#[derive(Clone, Copy)]
enum SlashHintRiskSeverity {
    Medium,
    High,
}

impl SlashHintRiskBadge {
    fn style(self, theme: &Theme, is_selected: bool) -> Style {
        let color = match self.severity {
            SlashHintRiskSeverity::Medium => theme.palette.warn,
            SlashHintRiskSeverity::High => theme.palette.error,
        };
        let style = Style::default().fg(color).add_modifier(Modifier::BOLD);
        if is_selected {
            style.bg(theme.palette.faint)
        } else {
            style
        }
    }
}

fn risk_badge(risk: &str) -> Option<SlashHintRiskBadge> {
    let label = risk.split_once(':').map_or(risk, |(label, _)| label).trim();
    if label.eq_ignore_ascii_case("medium") {
        Some(SlashHintRiskBadge {
            text: " med",
            severity: SlashHintRiskSeverity::Medium,
        })
    } else if label.eq_ignore_ascii_case("high") {
        Some(SlashHintRiskBadge {
            text: " high",
            severity: SlashHintRiskSeverity::High,
        })
    } else {
        None
    }
}

fn pad_or_truncate_cells(text: &str, width: usize) -> String {
    let mut rendered = truncate_cells(text, width);
    let pad = width.saturating_sub(UnicodeWidthStr::width(rendered.as_str()));
    if pad > 0 {
        rendered.push_str(&" ".repeat(pad));
    }
    rendered
}

fn truncate_cells(text: &str, max_cells: usize) -> String {
    if max_cells == 0 {
        return String::new();
    }
    if UnicodeWidthStr::width(text) <= max_cells {
        return text.to_string();
    }
    let body_cells = max_cells.saturating_sub(1);
    let mut rendered = String::new();
    let mut used = 0;
    for ch in text.chars() {
        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + ch_width > body_cells {
            break;
        }
        rendered.push(ch);
        used += ch_width;
    }
    rendered.push('\u{2026}');
    rendered
}

/// Try to expand the current `/partial` token to a full slash command.
///
/// Returns the completed buffer (command + trailing space) when the
/// buffer is a single-line `/token` where `token` is a strict prefix
/// of a known slash command. Built-in commands keep priority; project
/// prompt commands fill the same inline surface when the built-in top
/// suggestion does not extend the user's token. Returns `None`
/// otherwise so the caller can fall through to normal input handling.
pub(super) fn slash_completion_for(
    buffer: &str,
    prompt_commands: &[commands::PromptCommandDef],
) -> Option<String> {
    // Single line only: multi-line prompts must not be rewritten.
    if buffer.contains('\n') {
        return None;
    }
    let trimmed = buffer.trim_end_matches(' ');
    // Must start with `/` and have at least one name character.
    let rest = trimmed.strip_prefix('/')?;
    if rest.is_empty() {
        return None;
    }
    // Only trigger while the user is still typing the token itself —
    // never once arguments have been entered.
    if rest.chars().any(|c| c == ' ' || c == '\t') {
        return None;
    }
    // Only auto-expand when the top built-in match extends the user's
    // token — we never want Space to silently change characters they
    // already typed. Case-insensitive prefix match mirrors the suggester.
    let token_lc = rest.to_ascii_lowercase();
    let builtin_suggestions = commands::suggest_slash_commands(trimmed, 4);
    if let Some(top) = builtin_suggestions.first() {
        let top_name = top.strip_prefix('/')?;
        let top_lc = top_name.to_ascii_lowercase();
        if top_lc != token_lc && top_lc.starts_with(&token_lc) {
            return Some(format!("/{top_name} "));
        }
    }

    let mut prompt_matches = prompt_commands.iter().filter(|command| {
        let name = command.name.to_ascii_lowercase();
        name != token_lc && name.starts_with(&token_lc)
    });
    let command = prompt_matches.next()?;
    if prompt_matches.next().is_some() {
        return None;
    }
    Some(format!("/{} ", command.name))
}

/// App 상태에 얹힌 슬래시-힌트 팝업 조회·선택 메서드.
impl App {
    /// Whether the slash-command hint popup is visible.
    pub(super) fn slash_hint_active(&self) -> bool {
        if !self.input_enabled
            || !matches!(self.mode, AppMode::Normal)
            || self.input.has_collapsed_paste()
        {
            return false;
        }
        let buffer = self.input.text();
        if buffer.contains('\n') || self.hints.slash_hidden_for.as_deref() == Some(buffer.as_str()) {
            return false;
        }
        let first_line = buffer.lines().next().unwrap_or("");
        first_line.trim_start().starts_with('/')
    }

    /// The `/command` at the hint cursor plus whether it needs a required
    /// inline argument (`<arg>`). The caller runs argument-less commands
    /// immediately and fills the input for the rest.
    pub(super) fn slash_hint_selected(&self) -> Option<(String, bool)> {
        let cursor = self.hints.slash_cursor?;
        self.slash_hint_candidate_at(cursor)
    }

    /// The highlighted slash hint, or the first visible candidate when no
    /// row is highlighted. Used by Tab completion.
    pub(super) fn slash_hint_candidate(&self) -> Option<(String, bool)> {
        self.slash_hint_candidate_at(self.hints.slash_cursor.unwrap_or(0))
    }

    pub(super) fn slash_hint_candidate_at(&self, cursor: usize) -> Option<(String, bool)> {
        if self.input.has_collapsed_paste() {
            return None;
        }
        let recent = self.command_history.top_recent(SLASH_HINT_LIMIT);
        slash_hint_suggestions(
            self.input.text().as_str(),
            &self.prompt_commands,
            &recent,
            SLASH_HINT_LIMIT,
        )
        .get(cursor)
        .map(|suggestion| (suggestion.command.clone(), suggestion.requires_arg))
    }

    pub(super) fn slash_hint_suggestion_count(&self) -> usize {
        if !self.slash_hint_active() {
            return 0;
        }
        let recent = self.command_history.top_recent(SLASH_HINT_LIMIT);
        slash_hint_suggestions(
            self.input.text().as_str(),
            &self.prompt_commands,
            &recent,
            SLASH_HINT_LIMIT,
        )
        .len()
    }

    pub(super) fn slash_hint_popup_rect(&self) -> Option<Rect> {
        let regions = self.regions?;
        let popup = slash_hint_popup_area(regions.input, self.slash_hint_suggestion_count())?;
        if !self.terminal_mode.is_inline() {
            return Some(popup);
        }
        let bounds = Rect::new(
            regions.input.x,
            regions.transcript.y,
            regions.input.width,
            regions.hud.bottom().saturating_sub(regions.transcript.y),
        );
        fit_popup_above_input(popup, regions.input, bounds)
    }

    pub(super) fn select_prev_slash_hint(&mut self) {
        let count = self.slash_hint_suggestion_count();
        if count == 0 {
            self.hints.slash_cursor = None;
            return;
        }
        let current = self.hints.slash_cursor.unwrap_or(0);
        self.hints.slash_cursor = Some(current.saturating_sub(1));
    }

    pub(super) fn select_next_slash_hint(&mut self) {
        let count = self.slash_hint_suggestion_count();
        if count == 0 {
            self.hints.slash_cursor = None;
            return;
        }
        let current = self.hints.slash_cursor.unwrap_or(0);
        self.hints.slash_cursor = Some(current.saturating_add(1).min(count - 1));
    }

    pub(super) fn hide_slash_hint_for_current_input(&mut self) {
        self.hints.slash_cursor = None;
        self.hints.slash_hidden_for = Some(self.input.text());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn prompt_command(name: &str) -> commands::PromptCommandDef {
        commands::PromptCommandDef {
            name: name.to_string(),
            description: Some(format!("Run {name} prompt")),
            argument_hint: Some("<scope>".to_string()),
            model: None,
            effort: None,
            body: "Do the thing for $ARGUMENTS".to_string(),
            allowed_tools: Vec::new(),
            path: PathBuf::from(format!(".zo/commands/{name}.md")),
        }
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn slash_hint_line_projects_medium_and_high_risk() {
        let theme = Theme::no_color();

        let medium = render_slash_hint_line(
            "/model",
            &[],
            "Switch model",
            Some("Medium: changes subsequent model/provider behavior"),
            None,
            72,
            &theme,
            false,
            SLASH_HINT_LABEL_WIDTH,
        );
        let high = render_slash_hint_line(
            "/commit",
            &[],
            "Create commit",
            Some("High: mutates git index and repository history"),
            None,
            72,
            &theme,
            false,
            SLASH_HINT_LABEL_WIDTH,
        );

        assert!(line_text(&medium).contains(" med"));
        assert!(line_text(&high).contains(" high"));
    }

    #[test]
    fn slash_hint_line_omits_low_risk_badge() {
        let theme = Theme::no_color();
        let line = render_slash_hint_line(
            "/help",
            &[],
            "Show help",
            Some("Low: no workspace or session mutation"),
            None,
            72,
            &theme,
            false,
            SLASH_HINT_LABEL_WIDTH,
        );

        assert!(!line_text(&line).contains(" low"));
    }

    #[test]
    fn slash_hint_line_stays_within_cell_width() {
        let theme = Theme::no_color();
        let line = render_slash_hint_line(
            "/한글명령어가길다",
            &[],
            "설명이 길어도 입력창 위 힌트 폭을 넘기지 않아야 한다",
            Some("High: 테스트"),
            None,
            28,
            &theme,
            false,
            SLASH_HINT_LABEL_WIDTH,
        );
        let rendered = line_text(&line);

        assert!(
            UnicodeWidthStr::width(rendered.as_str()) <= 28,
            "rendered: {rendered}"
        );
        assert!(rendered.contains('\u{2026}'), "rendered: {rendered}");
    }

    #[test]
    fn slash_hint_line_highlights_matched_name_chars() {
        let theme = Theme::no_color();
        // Matched name indices [0, 1] land on label chars 1, 2 — the `c`, `o`
        // of `/commit`. They carry the underline modifier; the unmatched name
        // chars do not.
        let line = render_slash_hint_line(
            "/commit",
            &[0, 1],
            "Create a commit",
            None,
            None,
            72,
            &theme,
            false,
            SLASH_HINT_LABEL_WIDTH,
        );
        let underlined: Vec<&str> = line
            .spans
            .iter()
            .filter(|span| span.style.add_modifier.contains(Modifier::UNDERLINED))
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(
            underlined,
            vec!["c", "o"],
            "only the matched name chars are underlined"
        );
    }

    #[test]
    fn slash_hint_line_appends_global_accelerator_at_right_edge() {
        let theme = Theme::no_color();
        let line = render_slash_hint_line(
            "/agents",
            &[],
            "List configured agents",
            None,
            Some("^A"),
            72,
            &theme,
            false,
            SLASH_HINT_LABEL_WIDTH,
        );
        let text: String = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(
            text.trim_end().ends_with("^A"),
            "the accelerator sits at the right edge: {text:?}"
        );
    }

    #[test]
    fn popup_width_matches_input_or_wide_cap() {
        assert_eq!(popup_width_for(20), 20);
        assert_eq!(popup_width_for(60), 60);
        assert_eq!(popup_width_for(140), 90);
    }

    #[test]
    fn slash_hint_suggestions_include_matching_prompt_commands() {
        let commands = vec![prompt_command("review-local")];

        let suggestions = slash_hint_suggestions("/review-local", &commands, &[], 10);
        let suggestion = suggestions
            .iter()
            .find(|suggestion| suggestion.command == "/review-local")
            .expect("prompt command suggestion");

        assert_eq!(suggestion.summary, "Run review-local prompt");
        assert_eq!(suggestion.risk, Some(PROMPT_COMMAND_RISK));
    }

    #[test]
    fn slash_hint_suggestions_keep_builtin_defaults_for_empty_query() {
        let commands = vec![prompt_command("review-local")];

        let suggestions = slash_hint_suggestions("/", &commands, &[], 10);

        assert!(
            suggestions
                .iter()
                .any(|suggestion| suggestion.command == "/help"),
            "empty slash query should keep the built-in command list: {suggestions:?}"
        );
    }

    #[test]
    fn slash_hint_matches_commands_by_summary_not_just_name() {
        // No built-in is *named* "git", but /diff and /commit mention git in
        // their summary. The old name-only filter found nothing; the fuzzy +
        // summary search surfaces them — the core search-usability fix.
        let suggestions = slash_hint_suggestions("/git", &[], &[], 24);
        let names: Vec<&str> = suggestions.iter().map(|s| s.command.as_str()).collect();

        assert!(
            names.contains(&"/diff"),
            "/git surfaces /diff by its 'git diff' summary: {names:?}"
        );
        assert!(
            names.contains(&"/commit"),
            "/git surfaces /commit by its 'git commit' summary: {names:?}"
        );
        // Neither is *named* git — both matched purely on their summary text.
        assert!(
            !names.contains(&"/git"),
            "there is no /git command; matches came from descriptions: {names:?}"
        );
    }

    #[test]
    fn slash_hint_fuzzy_matches_name_subsequence_and_ranks_prefix_first() {
        // `cmt` is a gapped subsequence of `commit` (c-m-t).
        let suggestions = slash_hint_suggestions("/cmt", &[], &[], 24);
        assert!(
            suggestions.iter().any(|s| s.command == "/commit"),
            "fuzzy name match finds /commit for 'cmt': {suggestions:?}"
        );

        // A prefix query ranks an exact-prefix command into the first row.
        let suggestions = slash_hint_suggestions("/co", &[], &[], 24);
        let first = suggestions.first().map(|s| s.command.as_str());
        assert!(
            matches!(first, Some(name) if name.starts_with("/co")),
            "prefix query ranks a /co* command first: {first:?}"
        );
    }

    #[test]
    fn slash_hint_records_name_match_indices_for_highlighting() {
        // The `/co` query matches the first two chars of the name; those char
        // positions are recorded (offset within the name, not the label) so the
        // renderer can highlight them.
        let suggestions = slash_hint_suggestions("/co", &[], &[], 24);
        let commit = suggestions
            .iter()
            .find(|s| s.command == "/commit")
            .expect("/commit present for /co");
        assert_eq!(
            commit.matched_indices,
            vec![0, 1],
            "co matches commit's first two name chars"
        );
    }

    #[test]
    fn slash_completion_expands_prompt_command_prefix() {
        let commands = vec![prompt_command("review-local")];

        assert_eq!(
            slash_completion_for("/review-l", &commands).as_deref(),
            Some("/review-local ")
        );
    }

    #[test]
    fn slash_completion_keeps_builtin_priority_for_shared_prefix() {
        let commands = vec![prompt_command("review-local")];

        assert_eq!(
            slash_completion_for("/rev", &commands).as_deref(),
            Some("/review ")
        );
    }

    #[test]
    fn slash_completion_does_not_rewrite_prompt_command_arguments() {
        let commands = vec![prompt_command("review-local")];

        assert_eq!(slash_completion_for("/review-l src", &commands), None);
    }

    #[test]
    fn slash_completion_does_not_pick_between_ambiguous_prompt_commands() {
        let commands = vec![
            prompt_command("review-local"),
            prompt_command("review-remote"),
        ];

        assert_eq!(slash_completion_for("/review-", &commands), None);
        assert_eq!(
            slash_completion_for("/review-l", &commands).as_deref(),
            Some("/review-local ")
        );
    }
}
