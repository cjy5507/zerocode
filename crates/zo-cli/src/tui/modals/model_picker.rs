//! `/model` picker modal (Phase 3, Lane L6).
//!
//! The current product surface is Claude-first. The modal still keeps
//! grouping metadata internally so adapter-backed catalogs can exist,
//! but a single-group registry renders as a plain model list without a
//! provider banner.
//!
//! Code-rules: R1 (neutral vocabulary — the modal never names a
//! specific provider in its logic; the caller decides what goes in
//! the registry), R2 (no ANSI), R9 (`&Theme` drives every style
//! decision).

use api::AuthRoute;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Margin, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use runtime::message_stream::ActiveModel;
use runtime::model_catalog::{CatalogProvider, CatalogRow, ModelCatalog};
use unicode_width::UnicodeWidthStr;

use super::super::theme::Theme;
use super::{ModalResult, ModalSelection, blank_marker, cursor_marker, key_hint_footer};
use crate::tui::fuzzy;

/// Cursor rows a PageUp/PageDown jumps through. Matches the page stride used by
/// the other selection-list modals (see `tool_toggle::page_down`).
const PAGE_STRIDE: usize = 8;

/// One entry in the model picker registry.
#[derive(Debug, Clone)]
pub struct ModelPickerEntry {
    /// Provider id (e.g. `"anthropic"`, `"codex"`). Used for the
    /// group header.
    pub provider: String,
    /// The underlying model that will be returned if selected.
    pub model: ActiveModel,
}

/// Model picker modal.
#[derive(Debug, Clone)]
pub struct ModelPickerModal {
    entries: Vec<ModelPickerEntry>,
    /// Live type-ahead query. Filters the visible list by a fuzzy
    /// subsequence match against `display_name` and provider label.
    query: String,
    /// Indices into `entries` that survive the current `query`, in
    /// registry order. The visible list iterates this view.
    filtered: Vec<usize>,
    /// Cursor position *within `filtered`* (not a raw `entries` index).
    /// Group headers are computed on the fly so they cannot be "selected".
    cursor: usize,
    scroll_offset: std::cell::Cell<usize>,
    manager: Option<ModelManager>,
    manager_error: Option<String>,
}

impl ModelPickerModal {
    /// Construct a modal from a registry of entries.
    ///
    /// The order of `entries` determines the render order inside each
    /// group. Entries sharing the same `provider` must already be
    /// contiguous — the modal does not re-sort the list.
    #[must_use]
    pub fn new(entries: Vec<ModelPickerEntry>) -> Self {
        let filtered: Vec<usize> = (0..entries.len()).collect();
        Self {
            entries,
            query: String::new(),
            filtered,
            cursor: 0,
            scroll_offset: std::cell::Cell::new(0),
            manager: None,
            manager_error: None,
        }
    }

    /// Number of entries currently visible (after type-ahead filtering).
    #[must_use]
    pub fn len(&self) -> usize {
        self.filtered.len()
    }

    /// The live type-ahead query string.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Re-derive `filtered` from `entries` against the current `query` and
    /// clamp the cursor back into range. An empty query shows everything.
    fn refilter(&mut self) {
        let needle = self.query.to_lowercase();
        self.filtered = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                if needle.is_empty() {
                    return true;
                }
                let name = entry.model.display_name.to_lowercase();
                if fuzzy::is_subsequence(&name, &needle) {
                    return true;
                }
                let provider = entry.provider.to_lowercase();
                if fuzzy::is_subsequence(&provider, &needle) {
                    return true;
                }
                // A model promoted into the SUGGESTED group has its group label
                // rewritten to "suggested" but keeps its real provider on
                // `model.provider`; match that so provider filtering (e.g.
                // typing "claude") still finds the promoted row.
                let model_provider = entry.model.provider.to_lowercase();
                fuzzy::is_subsequence(&model_provider, &needle)
            })
            .map(|(i, _)| i)
            .collect();
        if self.cursor >= self.filtered.len() {
            self.cursor = self.filtered.len().saturating_sub(1);
        }
    }

    /// The `entries` index the cursor currently points at, if any visible.
    fn selected_entry_index(&self) -> Option<usize> {
        self.filtered.get(self.cursor).copied()
    }

    /// Total rendered row count including provider group headers (mirrors
    /// [`Self::render_lines`]) so the host can size the modal to fit. Group
    /// headers only appear when more than one *visible* provider group is
    /// present. Counts the filtered view so a narrowing query shrinks the modal.
    #[must_use]
    pub fn visual_rows(&self) -> usize {
        if let Some(manager) = &self.manager {
            return manager.visual_rows();
        }
        let error_rows = usize::from(self.manager_error.is_some());
        let Some(&first_idx) = self.filtered.first() else {
            return error_rows;
        };
        let first = self.entries[first_idx].provider.as_str();
        let multi_group = self
            .filtered
            .iter()
            .any(|&i| self.entries[i].provider != first);
        if !multi_group {
            return self.filtered.len() + error_rows;
        }
        let mut groups = 0usize;
        let mut last: Option<&str> = None;
        for &i in &self.filtered {
            let provider = self.entries[i].provider.as_str();
            if Some(provider) != last {
                groups += 1;
                last = Some(provider);
            }
        }
        self.filtered.len() + groups + error_rows
    }

    /// `true` if no entries are currently visible (registry empty or the
    /// type-ahead query matched nothing).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filtered.is_empty()
    }

    /// Current cursor index *within the filtered view*.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Move cursor to the next visible entry. Wraps to the first
    /// entry if we fall off the end.
    pub fn move_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + 1) % self.filtered.len();
    }

    /// Move cursor to the previous visible entry. Wraps to the last entry.
    pub fn move_up(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        if self.cursor == 0 {
            self.cursor = self.filtered.len() - 1;
        } else {
            self.cursor -= 1;
        }
    }

    /// Move the cursor down by a page, clamping at the last visible entry.
    pub fn page_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + PAGE_STRIDE).min(self.filtered.len() - 1);
    }

    /// Move the cursor up by a page, clamping at the first visible entry.
    pub fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(PAGE_STRIDE);
    }

    /// Jump the cursor to the first visible entry.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Jump the cursor to the last visible entry.
    pub fn move_end(&mut self) {
        self.cursor = self.filtered.len().saturating_sub(1);
    }

    /// Jump to the first entry of the next *visible* provider group. Wraps to
    /// the first group when called on the last group. No-op on an empty view.
    pub fn next_group(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        let current_provider = self.entries[self.filtered[self.cursor]].provider.as_str();
        let n = self.filtered.len();
        for offset in 1..=n {
            let pos = (self.cursor + offset) % n;
            if self.entries[self.filtered[pos]].provider != current_provider {
                self.cursor = self.first_pos_of_group(pos);
                return;
            }
        }
    }

    /// Jump to the first entry of the previous *visible* provider group. Wraps.
    pub fn prev_group(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        let current_provider = self.entries[self.filtered[self.cursor]].provider.as_str();
        let n = self.filtered.len();
        for offset in 1..=n {
            let pos = (self.cursor + n - offset) % n;
            if self.entries[self.filtered[pos]].provider != current_provider {
                self.cursor = self.first_pos_of_group(pos);
                return;
            }
        }
    }

    /// First *filtered position* of the provider group that visible position
    /// `pos` belongs to.
    fn first_pos_of_group(&self, pos: usize) -> usize {
        let provider = self.entries[self.filtered[pos]].provider.as_str();
        let mut first = pos;
        while first > 0 && self.entries[self.filtered[first - 1]].provider == provider {
            first -= 1;
        }
        first
    }

    /// Insert terminal paste or IME-committed text into the focused field.
    pub fn paste_text(&mut self, text: &str) {
        if let Some(manager) = self.manager.as_mut() {
            manager.paste_text(text);
            return;
        }
        let printable = text
            .chars()
            .filter(|ch| !ch.is_control())
            .collect::<String>();
        if !printable.is_empty() {
            self.manager_error = None;
            self.query.push_str(&printable);
            self.refilter();
        }
    }

    /// Hardware cursor for the active model add/edit text field.
    #[must_use]
    pub fn cursor_position(&self, area: Rect) -> Option<Position> {
        self.manager.as_ref()?.cursor_position(area)
    }

    /// Handle a single key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if let Some(manager) = self.manager.as_mut() {
            let close = manager.handle_key(key);
            let changed = manager.take_changed();
            let sync = changed.then(|| (manager.catalog.clone(), manager.connected.clone()));
            if let Some((catalog, connected)) = sync.as_ref() {
                self.sync_catalog_entries(catalog, connected);
            }
            if close {
                self.manager = None;
            }
            return None;
        }
        if matches!(key.code, KeyCode::F(2)) && key.modifiers.is_empty() {
            match ModelManager::from_entries(&self.entries) {
                Ok(manager) => {
                    self.manager = Some(manager);
                    self.manager_error = None;
                }
                Err(error) => {
                    self.manager_error = Some(format!("Could not load model settings: {error}"));
                }
            }
            return None;
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
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
            KeyCode::Left => {
                self.prev_group();
                None
            }
            KeyCode::Right => {
                self.next_group();
                None
            }
            KeyCode::Backspace => {
                self.manager_error = None;
                self.query.pop();
                self.refilter();
                None
            }
            KeyCode::Char(ch) => {
                self.manager_error = None;
                self.query.push(ch);
                self.refilter();
                None
            }
            KeyCode::Enter => {
                let Some(idx) = self.selected_entry_index() else {
                    return Some(ModalResult::Cancelled);
                };
                let mut model = self.entries[idx].model.clone();
                if let Some(provider) = CatalogProvider::from_key(model.provider) {
                    match ModelCatalog::load() {
                        Ok(catalog) => {
                            model.alias = catalog.selection_token(provider, &model.alias);
                        }
                        Err(error) => {
                            self.manager_error =
                                Some(format!("Could not load model settings: {error}"));
                            return None;
                        }
                    }
                }
                Some(ModalResult::Selected(ModalSelection::Model(model)))
            }
            _ => None,
        }
    }

    fn sync_catalog_entries(
        &mut self,
        catalog: &ModelCatalog,
        connected: &[CatalogProvider],
    ) {
        let old = self.entries.clone();
        let active = old
            .iter()
            .find(|entry| entry.provider == "suggested")
            .cloned();
        let mut entries = Vec::new();
        for row in catalog.rows(connected, false) {
            let suggested = active.as_ref().is_some_and(|entry| {
                entry.model.provider == row.provider.key()
                    && same_picker_model(&entry.model.alias, &row.id)
            });
            entries.push(ModelPickerEntry {
                provider: if suggested { "suggested" } else { row.provider.key() }.to_string(),
                model: ActiveModel {
                    provider: row.provider.key(),
                    alias: row.id.clone(),
                    display_name: if row.builtin {
                        row.display_name.clone()
                    } else {
                        format!("{} · UNVERIFIED", row.display_name)
                    },
                    context_limit: u32::try_from(api::context_window_for_model(&row.id))
                        .unwrap_or(u32::MAX),
                },
            });
        }
        for entry in old {
            if CatalogProvider::from_key(entry.model.provider).is_none() {
                entries.push(entry);
            }
        }
        if let Some(active) = active {
            let present = entries.iter().any(|entry| {
                entry.model.provider == active.model.provider
                    && same_picker_model(&entry.model.alias, &active.model.alias)
            });
            if !present {
                entries.push(active);
            }
        }
        entries.sort_by_key(|entry| {
            (
                entry.provider != "suggested",
                provider_sort_key(entry.model.provider),
            )
        });
        self.entries = entries;
        self.refilter();
    }

    /// Move the cursor down by `count` visible rows, clamping at the end. Used
    /// by the host's mouse-wheel routing (which owns the app-level dispatch).
    pub fn scroll_down(&mut self, count: usize) {
        if self.filtered.is_empty() {
            return;
        }
        self.cursor = (self.cursor + count).min(self.filtered.len() - 1);
    }

    /// Move the cursor up by `count` visible rows, clamping at the top. Used by
    /// the host's mouse-wheel routing.
    pub fn scroll_up(&mut self, count: usize) {
        self.cursor = self.cursor.saturating_sub(count);
    }

    /// Build the rendered line set used by both [`Self::draw`] and
    /// tests. Group headers are shown only when multiple *visible* groups
    /// exist. The list iterates the type-ahead filtered view.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let mut lines: Vec<Line<'a>> = Vec::with_capacity(self.filtered.len() * 2 + 1);

        // Live type-ahead query line: only shown once the user starts typing so
        // an untouched picker keeps its original look.
        if !self.query.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  Filter: {}_", self.query),
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            )));
        }

        let first_provider = self
            .filtered
            .first()
            .map(|&i| self.entries[i].provider.as_str());
        let show_group_headers = first_provider.is_some_and(|first| {
            self.filtered
                .iter()
                .any(|&i| self.entries[i].provider != first)
        });

        let max_name_len = self
            .filtered
            .iter()
            .map(|&i| self.entries[i].model.display_name.chars().count())
            .max()
            .unwrap_or(20);

        let mut last_provider: Option<&str> = None;
        for (pos, &idx) in self.filtered.iter().enumerate() {
            let entry = &self.entries[idx];
            let provider = entry.provider.as_str();
            if show_group_headers && Some(provider) != last_provider {
                let header_text = if theme.no_color {
                    if provider == "suggested" {
                        "* SUGGESTED ------------------------------------------------".to_string()
                    } else {
                        format!("-- {provider} ------------------------------------------------")
                    }
                } else {
                    let line_char = "─";
                    if provider == "suggested" {
                        format!("── ★ SUGGESTED {}", line_char.repeat(40))
                    } else {
                        format!("── {provider} {}", line_char.repeat(40))
                    }
                };
                let header_color = if theme.no_color {
                    theme.palette.dim
                } else {
                    match provider {
                        "suggested" | "claude" | "anthropic" => theme.palette.accent,
                        "openai" => theme.palette.violet,
                        "google" => theme.palette.cyan,
                        _ => theme.palette.dim,
                    }
                };
                lines.push(Line::from(Span::styled(
                    header_text,
                    Style::new().fg(header_color).add_modifier(Modifier::BOLD),
                )));
                last_provider = Some(provider);
            }

            let selected = pos == self.cursor;
            let marker = if selected {
                cursor_marker(!theme.no_color)
            } else {
                blank_marker()
            };

            let mut spans = Vec::new();

            // Marker
            let marker_style = if selected {
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(theme.palette.dim)
            };
            spans.push(Span::styled(marker.to_string(), marker_style));

            // Model name
            let name_style = if selected {
                Style::new()
                    .fg(theme.palette.bright)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(theme.palette.fg)
            };
            spans.push(Span::styled(entry.model.display_name.clone(), name_style));

            // Padding after model name
            let name_chars = entry.model.display_name.chars().count();
            let pad1 = (max_name_len + 4).saturating_sub(name_chars);
            spans.push(Span::raw(" ".repeat(pad1)));

            // Provider label
            let provider_label = clean_provider_label(provider);
            let provider_color = if theme.no_color {
                if selected {
                    theme.palette.bright
                } else {
                    theme.palette.fg
                }
            } else {
                match provider {
                    "claude" | "suggested" => theme.palette.accent,
                    "openai" => theme.palette.violet,
                    "google" => theme.palette.cyan,
                    _ => theme.palette.dim,
                }
            };
            let provider_style = if selected {
                Style::new().fg(provider_color).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(provider_color)
            };
            spans.push(Span::styled(
                format!("{provider_label:<15}"),
                provider_style,
            ));

            // Context window limit
            let limit_label = format_limit(entry.model.context_limit);
            let limit_style = if selected {
                Style::new()
                    .fg(theme.palette.bright)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(theme.palette.dim)
            };
            spans.push(Span::styled(limit_label, limit_style));

            lines.push(Line::from(spans));
        }
        if self.filtered.is_empty() && !self.query.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No models match",
                Style::new().fg(theme.palette.dim),
            )));
        }
        if let Some(error) = &self.manager_error {
            lines.push(Line::from(Span::styled(
                error.clone(),
                Style::new().fg(theme.palette.error),
            )));
        }
        lines.push(Line::from(""));
        let mut hints: Vec<(&str, &str)> = vec![("↑↓", "move")];
        if show_group_headers {
            hints.push(("←→", "group"));
        }
        hints.push(("type", "filter"));
        hints.push(("Enter", "confirm"));
        hints.push(("F2", "Manage models"));
        hints.push(("Esc", "cancel"));
        lines.push(key_hint_footer(theme, &hints));
        lines
    }

    /// Draw the modal into `area` using `theme`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if let Some(manager) = &self.manager {
            manager.draw(frame, area, theme);
            return;
        }
        let inner = super::modal_frame(frame, area, "/model", theme).inner(Margin {
            vertical: 1,
            horizontal: 1,
        });

        if inner.height == 0 || inner.width == 0 {
            return;
        }

        let mut lines = self.render_lines(theme);
        let footer_lines = if lines.len() >= 2 {
            lines.split_off(lines.len() - 2)
        } else {
            Vec::new()
        };

        // Height reserved for the scrollable list
        let footer_height = u16::try_from(footer_lines.len()).unwrap_or(0);
        let list_height = inner.height.saturating_sub(footer_height);

        // Find the line index of the selected cursor within the rendered
        // `lines`. A non-empty query renders a leading "Filter:" row, so the
        // list starts one row lower — mirror that offset here.
        let mut cursor_line_idx = 0;
        let mut current_line_count = usize::from(!self.query.is_empty());
        let first_provider = self
            .filtered
            .first()
            .map(|&i| self.entries[i].provider.as_str());
        let show_group_headers = first_provider.is_some_and(|first| {
            self.filtered
                .iter()
                .any(|&i| self.entries[i].provider != first)
        });
        let mut last_provider: Option<&str> = None;
        for (pos, &idx) in self.filtered.iter().enumerate() {
            let provider = self.entries[idx].provider.as_str();
            if show_group_headers && Some(provider) != last_provider {
                current_line_count += 1;
                last_provider = Some(provider);
            }
            if pos == self.cursor {
                cursor_line_idx = current_line_count;
            }
            current_line_count += 1;
        }

        // Calculate scroll offset to keep cursor_line_idx visible
        let mut scroll_offset = self.scroll_offset.get();
        let visible_height = usize::from(list_height);
        if visible_height > 0 {
            if cursor_line_idx < scroll_offset {
                scroll_offset = cursor_line_idx;
            } else if cursor_line_idx >= scroll_offset + visible_height {
                scroll_offset = cursor_line_idx - visible_height + 1;
            }
            // Clamp scroll_offset in case the height changed or it's out of bounds
            let max_scroll = lines.len().saturating_sub(visible_height);
            scroll_offset = scroll_offset.min(max_scroll);
        } else {
            scroll_offset = 0;
        }
        self.scroll_offset.set(scroll_offset);

        // Render the visible slice of list lines
        let list_slice = lines
            .into_iter()
            .skip(scroll_offset)
            .take(visible_height)
            .collect::<Vec<_>>();

        let list_rect = Rect::new(inner.x, inner.y, inner.width, list_height);
        let list_para = Paragraph::new(list_slice)
            .style(theme.typography.body)
            .wrap(Wrap { trim: false });
        frame.render_widget(list_para, list_rect);

        // Render the pinned footer
        if footer_height > 0 {
            let footer_rect = Rect::new(
                inner.x,
                inner.y + list_height,
                inner.width,
                footer_height.min(inner.height - list_height),
            );
            let footer_para = Paragraph::new(footer_lines)
                .style(theme.typography.body)
                .wrap(Wrap { trim: false });
            frame.render_widget(footer_para, footer_rect);
        }
    }
}


#[derive(Debug, Clone)]
struct ModelManager {
    catalog: ModelCatalog,
    connected: Vec<CatalogProvider>,
    rows: Vec<CatalogRow>,
    cursor: usize,
    form: Option<ModelForm>,
    confirm: bool,
    changed: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelTextField {
    value: String,
    cursor: usize,
}

impl ModelTextField {
    fn new(value: String) -> Self {
        let cursor = value.len();
        Self { value, cursor }
    }

    fn text(&self) -> &str {
        &self.value
    }

    fn before_cursor(&self) -> &str {
        &self.value[..self.cursor]
    }

    fn insert_char(&mut self, ch: char) {
        if ch.is_control() {
            return;
        }
        self.value.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    fn insert_text(&mut self, text: &str) {
        let printable = text
            .chars()
            .filter(|ch| !ch.is_control())
            .collect::<String>();
        if printable.is_empty() {
            return;
        }
        self.value.insert_str(self.cursor, &printable);
        self.cursor += printable.len();
    }

    fn backspace(&mut self) {
        let Some((previous, _)) = self.value[..self.cursor].char_indices().next_back() else {
            return;
        };
        self.value.drain(previous..self.cursor);
        self.cursor = previous;
    }

    fn delete(&mut self) {
        let Some(ch) = self.value[self.cursor..].chars().next() else {
            return;
        };
        self.value.drain(self.cursor..self.cursor + ch.len_utf8());
    }

    fn move_left(&mut self) {
        if let Some((previous, _)) = self.value[..self.cursor].char_indices().next_back() {
            self.cursor = previous;
        }
    }

    fn move_right(&mut self) {
        if let Some(ch) = self.value[self.cursor..].chars().next() {
            self.cursor += ch.len_utf8();
        }
    }

    fn move_home(&mut self) {
        self.cursor = 0;
    }

    fn move_end(&mut self) {
        self.cursor = self.value.len();
    }
}

#[derive(Debug, Clone)]
struct ModelForm {
    original: Option<CatalogRow>,
    provider: usize,
    providers: Vec<CatalogProvider>,
    id: ModelTextField,
    display_name: ModelTextField,
    auth_route: AuthRoute,
    focus: usize,
    error: Option<String>,
}

impl ModelForm {
    fn active_text_field(&self) -> Option<&ModelTextField> {
        match self.focus {
            1 => Some(&self.id),
            2 => Some(&self.display_name),
            _ => None,
        }
    }

    fn active_text_field_mut(&mut self) -> Option<&mut ModelTextField> {
        match self.focus {
            1 => Some(&mut self.id),
            2 => Some(&mut self.display_name),
            _ => None,
        }
    }

    fn paste_text(&mut self, text: &str) {
        if let Some(field) = self.active_text_field_mut() {
            field.insert_text(text);
            self.error = None;
        }
    }
}

fn auth_route_label(provider: CatalogProvider, route: AuthRoute) -> &'static str {
    match (provider, route) {
        (_, AuthRoute::Auto) => "Auto",
        (CatalogProvider::Google, AuthRoute::OAuth) => "OAuth · Code Assist",
        (CatalogProvider::Google, AuthRoute::ApiKey) => "API key · Gemini API",
        (CatalogProvider::Openai, AuthRoute::OAuth) => "OAuth · ChatGPT",
        (CatalogProvider::Openai, AuthRoute::ApiKey) => "API key · OpenAI API",
        (CatalogProvider::Anthropic, AuthRoute::OAuth) => "OAuth · Claude",
        (CatalogProvider::Anthropic, AuthRoute::ApiKey) => "API key · Anthropic API",
    }
}

fn previous_auth_route(route: AuthRoute) -> AuthRoute {
    match route {
        AuthRoute::Auto | AuthRoute::OAuth => AuthRoute::Auto,
        AuthRoute::ApiKey => AuthRoute::OAuth,
    }
}

fn next_auth_route(route: AuthRoute) -> AuthRoute {
    match route {
        AuthRoute::Auto => AuthRoute::OAuth,
        AuthRoute::OAuth | AuthRoute::ApiKey => AuthRoute::ApiKey,
    }
}

impl ModelManager {
    fn from_entries(entries: &[ModelPickerEntry]) -> std::io::Result<Self> {
        let catalog = ModelCatalog::load()?;
        let oauth_connected = oauth_connected_catalog_providers();
        let mut connected = connected_catalog_providers(entries);
        for provider in &oauth_connected {
            if !connected.contains(provider) {
                connected.push(*provider);
            }
        }
        let rows = catalog.rows(&connected, true);
        Ok(Self {
            catalog,
            connected,
            rows,
            cursor: 0,
            form: None,
            confirm: false,
            changed: false,
            error: None,
        })
    }

    fn visual_rows(&self) -> usize {
        if let Some(form) = &self.form {
            return 8 + usize::from(form.error.is_some());
        }
        self.rows.len().clamp(1, 7) * 2 + 2 + usize::from(self.error.is_some())
    }

    fn selected(&self) -> Option<&CatalogRow> {
        self.rows.get(self.cursor)
    }

    fn refresh(&mut self) {
        self.rows = self.catalog.rows(&self.connected, true);
        self.cursor = self.cursor.min(self.rows.len().saturating_sub(1));
    }

    fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if self.confirm {
            match key.code {
                KeyCode::Esc | KeyCode::Char('n' | 'N') => self.confirm = false,
                KeyCode::Enter | KeyCode::Char('y' | 'Y') => {
                    if let Some(row) = self.selected().cloned() {
                        match self.catalog.delete_or_hide(&row) {
                            Ok(()) => { self.changed = true; self.confirm = false; self.refresh(); }
                            Err(error) => { self.error = Some(error); self.confirm = false; }
                        }
                    }
                }
                _ => {}
            }
            return false;
        }
        if self.form.is_some() {
            return self.handle_form_key(key);
        }
        match key.code {
            KeyCode::Esc => return true,
            KeyCode::Up | KeyCode::Char('k') => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.cursor = (self.cursor + 1).min(self.rows.len().saturating_sub(1));
            }
            KeyCode::Char('a' | 'A') => {
                let providers = self.connected.clone();
                if providers.is_empty() {
                    self.error = Some("Connect a provider before adding a model".to_string());
                } else {
                    self.error = None;
                    self.form = Some(ModelForm {
                        original: None,
                        provider: 0,
                        providers,
                        id: ModelTextField::new(String::new()),
                        display_name: ModelTextField::new(String::new()),
                        auth_route: AuthRoute::OAuth,
                        focus: 0,
                        error: None,
                    });
                }
            }
            KeyCode::Enter | KeyCode::Char('e' | 'E') => {
                if let Some(row) = self.selected().filter(|row| !row.hidden).cloned() {
                    let mut providers = self.connected.clone();
                    if !providers.contains(&row.provider) {
                        providers.push(row.provider);
                    }
                    let provider = providers
                        .iter()
                        .position(|candidate| *candidate == row.provider)
                        .unwrap_or(0);
                    self.error = None;
                    self.form = Some(ModelForm {
                        original: Some(row.clone()),
                        provider,
                        providers,
                        id: ModelTextField::new(row.id),
                        display_name: ModelTextField::new(row.display_name),
                        auth_route: row.auth_route,
                        focus: 0,
                        error: None,
                    });
                }
            }
            KeyCode::Char('d' | 'D') => {
                if self.selected().is_some_and(|row| !row.hidden) { self.confirm = true; }
            }
            KeyCode::Char('r' | 'R') => {
                if let Some(row) = self.selected().filter(|row| row.hidden).cloned() {
                    match self.catalog.restore(&row) {
                        Ok(()) => { self.changed = true; self.refresh(); }
                        Err(error) => self.error = Some(error),
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn handle_form_key(&mut self, key: KeyEvent) -> bool {
        let form = self.form.as_mut().expect("form checked above");
        match key.code {
            KeyCode::Esc => self.form = None,
            KeyCode::Tab | KeyCode::Down => form.focus = (form.focus + 1) % 4,
            KeyCode::BackTab | KeyCode::Up => form.focus = (form.focus + 3) % 4,
            KeyCode::Left if form.focus == 0 => {
                form.provider = form.provider.saturating_sub(1);
            }
            KeyCode::Right if form.focus == 0 => {
                form.provider =
                    (form.provider + 1).min(form.providers.len().saturating_sub(1));
            }
            KeyCode::Left if form.focus == 3 => {
                form.auth_route = previous_auth_route(form.auth_route);
            }
            KeyCode::Right if form.focus == 3 => {
                form.auth_route = next_auth_route(form.auth_route);
            }
            KeyCode::Left => {
                if let Some(field) = form.active_text_field_mut() {
                    field.move_left();
                }
            }
            KeyCode::Right => {
                if let Some(field) = form.active_text_field_mut() {
                    field.move_right();
                }
            }
            KeyCode::Home => {
                if let Some(field) = form.active_text_field_mut() {
                    field.move_home();
                }
            }
            KeyCode::End => {
                if let Some(field) = form.active_text_field_mut() {
                    field.move_end();
                }
            }
            KeyCode::Backspace => {
                if let Some(field) = form.active_text_field_mut() {
                    field.backspace();
                    form.error = None;
                }
            }
            KeyCode::Delete => {
                if let Some(field) = form.active_text_field_mut() {
                    field.delete();
                    form.error = None;
                }
            }
            KeyCode::Enter if form.focus < 3 => form.focus += 1,
            KeyCode::Enter => {
                let provider = form.providers[form.provider];
                let result = if let Some(original) = form.original.as_ref() {
                    self.catalog.edit_with_auth_route(
                        original,
                        provider,
                        form.id.text(),
                        form.display_name.text(),
                        form.auth_route,
                    )
                } else {
                    self.catalog.add_with_auth_route(
                        provider,
                        form.id.text(),
                        form.display_name.text(),
                        form.auth_route,
                    )
                };
                match result {
                    Ok(()) => {
                        self.form = None;
                        self.changed = true;
                        self.refresh();
                    }
                    Err(error) => form.error = Some(error),
                }
            }
            KeyCode::Char(ch)
                if !key.modifiers.intersects(
                    KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                ) =>
            {
                if let Some(field) = form.active_text_field_mut() {
                    field.insert_char(ch);
                    form.error = None;
                }
            }
            _ => {}
        }
        false
    }

    fn paste_text(&mut self, text: &str) {
        if let Some(form) = self.form.as_mut() {
            form.paste_text(text);
        }
    }

    fn cursor_position(&self, area: Rect) -> Option<Position> {
        let form = self.form.as_ref()?;
        let field = form.active_text_field()?;
        let row = u16::try_from(form.focus).ok()?;
        let inner = area.inner(Margin {
            vertical: 2,
            horizontal: 2,
        });
        if inner.width == 0 || row >= inner.height {
            return None;
        }
        let text_width = u16::try_from(UnicodeWidthStr::width(field.before_cursor()))
            .unwrap_or(u16::MAX);
        let x = inner
            .x
            .saturating_add(15)
            .saturating_add(text_width)
            .min(inner.right().saturating_sub(1));
        Some(Position::new(x, inner.y.saturating_add(row)))
    }

    fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let title = if self.form.is_some() { "Manage models · Add/Edit" } else { "Manage models" };
        let inner = super::modal_frame(frame, area, title, theme).inner(Margin { vertical: 1, horizontal: 1 });
        if inner.width == 0 || inner.height == 0 { return; }
        let mut lines = Vec::new();
        if let Some(form) = &self.form {
            let provider = form
                .providers
                .get(form.provider)
                .map_or("", |provider| provider.label());
            lines.push(field_line(theme, form.focus == 0, "Provider", provider));
            lines.push(text_field_line(theme, form.focus == 1, "Model ID", &form.id));
            lines.push(text_field_line(
                theme,
                form.focus == 2,
                "Display name",
                &form.display_name,
            ));
            lines.push(field_line(
                theme,
                form.focus == 3,
                "Auth route",
                auth_route_label(form.providers[form.provider], form.auth_route),
            ));
            lines.push(Line::from(""));
            if let Some(error) = &form.error {
                lines.push(Line::from(Span::styled(
                    error.clone(),
                    Style::new().fg(theme.palette.error),
                )));
            }
            lines.push(Line::from(""));
            lines.push(key_hint_footer(
                theme,
                &[
                    ("↑↓/Tab", "field"),
                    ("←→", "choice/cursor"),
                    ("Home/End", "text"),
                    ("Enter", "next/save"),
                    ("Esc", "cancel"),
                ],
            ));
        } else {
            if let Some(error) = &self.error { lines.push(Line::from(Span::styled(error.clone(), Style::new().fg(theme.palette.error)))); }
            let visible = usize::from(inner.height.saturating_sub(3)) / 2;
            let start = self.cursor.saturating_sub(visible.saturating_sub(1));
            for (index, row) in self.rows.iter().enumerate().skip(start).take(visible) {
                let marker = if index == self.cursor { cursor_marker(!theme.no_color) } else { blank_marker() };
                let state = if row.hidden {
                    "hidden"
                } else if row.builtin {
                    "built-in"
                } else {
                    "UNVERIFIED"
                };
                let style = if index == self.cursor {
                    Style::new()
                        .fg(theme.palette.bright)
                        .add_modifier(Modifier::BOLD)
                } else if row.hidden {
                    Style::new().fg(theme.palette.dim)
                } else if row.builtin {
                    Style::new().fg(theme.palette.fg)
                } else {
                    Style::new().fg(theme.palette.warn)
                };
                lines.push(Line::from(Span::styled(
                    format!(
                        "{marker}{} · {}  [{}]",
                        row.display_name,
                        row.provider.label(),
                        state
                    ),
                    style,
                )));
                lines.push(Line::from(Span::styled(
                    format!(
                        "  Auth route: {}",
                        auth_route_label(row.provider, row.auth_route)
                    ),
                    style,
                )));
            }
            lines.push(Line::from(""));
            if self.confirm {
                lines.push(Line::from(Span::styled("Delete this user model or hide this built-in? Enter/y confirms; Esc/n cancels", Style::new().fg(theme.palette.warn))));
            } else {
                lines.push(key_hint_footer(theme, &[("↑↓/j/k", "move"), ("a", "add"), ("Enter/e", "edit"), ("d", "delete/hide"), ("r", "restore"), ("Esc", "back")]));
            }
        }
        frame.render_widget(Paragraph::new(lines).style(theme.typography.body).wrap(Wrap { trim: false }), inner);
    }
}

fn field_line<'a>(theme: &Theme, focused: bool, label: &'a str, value: &'a str) -> Line<'a> {
    let marker = if focused { ">" } else { " " };
    Line::from(vec![
        Span::styled(
            format!("{marker} {label:<13}"),
            Style::new().fg(if focused {
                theme.palette.accent
            } else {
                theme.palette.dim
            }),
        ),
        Span::styled(value, Style::new().fg(theme.palette.fg)),
    ])
}

fn text_field_line(
    theme: &Theme,
    focused: bool,
    label: &str,
    field: &ModelTextField,
) -> Line<'static> {
    let marker = if focused { ">" } else { " " };
    let label_style = Style::new().fg(if focused {
        theme.palette.accent
    } else {
        theme.palette.dim
    });
    let text_style = Style::new().fg(theme.palette.fg);
    if !focused {
        return Line::from(vec![
            Span::styled(format!("{marker} {label:<13}"), label_style),
            Span::styled(field.text().to_string(), text_style),
        ]);
    }

    let (before, after) = field.value.split_at(field.cursor);
    Line::from(vec![
        Span::styled(format!("{marker} {label:<13}"), label_style),
        Span::styled(before.to_string(), text_style),
        Span::styled("▏", Style::new().fg(theme.palette.accent)),
        Span::styled(after.to_string(), text_style),
    ])
}

fn same_picker_model(left: &str, right: &str) -> bool {
    api::resolve_model_alias(&api::wire_model_id(left))
        .eq_ignore_ascii_case(&api::resolve_model_alias(&api::wire_model_id(right)))
}

fn connected_catalog_providers(entries: &[ModelPickerEntry]) -> Vec<CatalogProvider> {
    let mut providers = Vec::new();
    for entry in entries {
        if let Some(provider) = CatalogProvider::from_key(entry.model.provider) {
            if !providers.contains(&provider) { providers.push(provider); }
        }
    }
    providers
}

fn oauth_connected_catalog_providers() -> Vec<CatalogProvider> {
    let mut providers = Vec::new();
    let anthropic = api::oauth_store::load_oauth_credentials().ok().flatten().is_some()
        || api::read_claude_code_keychain_session().is_some();
    if anthropic { providers.push(CatalogProvider::Anthropic); }
    if api::oauth_store::load_openai_oauth().ok().flatten().is_some() {
        providers.push(CatalogProvider::Openai);
    }
    if api::google_code_assist_oauth_present() || api::google_gemini_oauth_available() {
        providers.push(CatalogProvider::Google);
    }
    providers
}

fn provider_sort_key(provider: &str) -> u8 {
    match provider {
        "claude" | "anthropic" => 0,
        "openai" => 1,
        "google" => 2,
        _ => 3,
    }
}

fn clean_provider_label(provider: &str) -> &'static str {
    match provider {
        "claude" | "anthropic" => "Anthropic",
        "openai" => "OpenAI",
        "google" => "Google",
        "xai" => "xAI",
        "ollama" => "Ollama",
        _ => "Suggested",
    }
}

fn format_limit(limit: u32) -> String {
    if limit == u32::MAX || limit == 0 {
        "unknown limit".to_string()
    } else if limit >= 1_000_000 {
        let mb = f64::from(limit) / 1_000_000.0;
        format!("{mb:.1}M tokens")
    } else {
        let kb = limit / 1_000;
        format!("{kb}k tokens")
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

    fn entry(provider: &str, alias: &str) -> ModelPickerEntry {
        ModelPickerEntry {
            provider: provider.to_string(),
            model: ActiveModel {
                provider: "anthropic",
                alias: alias.to_string(),
                display_name: format!("{provider}:{alias}"),
                context_limit: 200_000,
            },
        }
    }

    fn registry() -> Vec<ModelPickerEntry> {
        vec![
            entry("anthropic", "opus"),
            entry("anthropic", "sonnet"),
            entry("anthropic", "haiku"),
            entry("codex", "gpt-5"),
            entry("codex", "gpt-5-mini"),
        ]
    }

    #[test]
    fn type_ahead_filters_and_confirms_match() {
        let mut modal = ModelPickerModal::new(registry());
        assert_eq!(modal.len(), 5);

        // Typing "son" should fuzzy-narrow to the single "anthropic:sonnet"
        // entry, and Enter must resolve through the filtered view (not the raw
        // registry index that "son" no longer corresponds to).
        for ch in "son".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert_eq!(modal.query(), "son");
        assert_eq!(modal.len(), 1);

        match modal.handle_key(press(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::Model(m))) => {
                assert_eq!(m.alias, "sonnet");
            }
            other => panic!("expected sonnet selection, got {other:?}"),
        }
    }

    #[test]
    fn backspace_widens_the_filter() {
        let mut modal = ModelPickerModal::new(registry());
        for ch in "gpt".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert_eq!(modal.len(), 2);
        modal.handle_key(press(KeyCode::Backspace));
        modal.handle_key(press(KeyCode::Backspace));
        modal.handle_key(press(KeyCode::Backspace));
        assert_eq!(modal.query(), "");
        assert_eq!(modal.len(), 5);
    }

    #[test]
    fn page_down_advances_by_a_page() {
        // 20 entries, PAGE_STRIDE = 8: one PageDown jumps the cursor by a page,
        // a second clamps at the final row.
        let entries: Vec<ModelPickerEntry> = (0..20)
            .map(|i| entry("anthropic", &format!("m{i}")))
            .collect();
        let mut modal = ModelPickerModal::new(entries);
        assert_eq!(modal.cursor(), 0);

        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), PAGE_STRIDE);

        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), PAGE_STRIDE * 2);

        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), 19, "PageDown clamps at the last entry");

        modal.handle_key(press(KeyCode::PageUp));
        assert_eq!(modal.cursor(), 19 - PAGE_STRIDE);
    }

    #[test]
    fn home_and_end_jump_to_bounds() {
        let mut modal = ModelPickerModal::new(registry());
        modal.handle_key(press(KeyCode::End));
        assert_eq!(modal.cursor(), 4);
        modal.handle_key(press(KeyCode::Home));
        assert_eq!(modal.cursor(), 0);
    }

    #[test]
    fn provider_filter_finds_promoted_suggested_row() {
        // A model promoted into SUGGESTED has its group label rewritten to
        // "suggested" but keeps its real provider on `model.provider`. Filtering
        // by that provider (e.g. "claude") must still surface the promoted row —
        // otherwise deduping it out of its provider group hides it from search.
        let promoted = ModelPickerEntry {
            provider: "suggested".to_string(),
            model: ActiveModel {
                provider: "claude",
                alias: "fable".to_string(),
                display_name: "Fable 5".to_string(),
                context_limit: 200_000,
            },
        };
        let other = ModelPickerEntry {
            provider: "openai".to_string(),
            model: ActiveModel {
                provider: "openai",
                alias: "gpt-5".to_string(),
                display_name: "GPT-5".to_string(),
                context_limit: 200_000,
            },
        };
        let mut modal = ModelPickerModal::new(vec![promoted, other]);
        for ch in "claude".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert_eq!(
            modal.len(),
            1,
            "promoted suggested row is still found by its real provider"
        );
    }

    fn google_entry(alias: &str, display_name: &str) -> ModelPickerEntry {
        ModelPickerEntry {
            provider: "google".to_string(),
            model: ActiveModel {
                provider: "google",
                alias: alias.to_string(),
                display_name: display_name.to_string(),
                context_limit: 1_000_000,
            },
        }
    }

    fn test_manager(name: &str) -> (ModelManager, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "zo-model-manager-{name}-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        let catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let connected = vec![CatalogProvider::Google];
        let rows = catalog.rows(&connected, true);
        (
            ModelManager {
                catalog,
                connected,
                rows,
                cursor: 0,
                form: None,
                confirm: false,
                changed: false,
                error: None,
            },
            path,
        )
    }

    fn type_text(manager: &mut ModelManager, text: &str) {
        for ch in text.chars() {
            manager.handle_key(press(KeyCode::Char(ch)));
        }
    }

    #[test]
    fn manager_event_flow_add_edit_delete_hide_restore_and_cancel() {
        let (mut manager, path) = test_manager("events");

        manager.form = Some(ModelForm { original: None, provider: 0, providers: vec![CatalogProvider::Google], id: ModelTextField::new(String::new()), display_name: ModelTextField::new(String::new()), auth_route: AuthRoute::OAuth, focus: 0, error: None });
        manager.handle_key(press(KeyCode::Enter));
        type_text(&mut manager, "gemini-4.0-flash");
        manager.handle_key(press(KeyCode::Enter));
        type_text(&mut manager, "Gemini 4.0 Flash");
        manager.handle_key(press(KeyCode::Enter));
        manager.handle_key(press(KeyCode::Enter));
        let added = manager
            .rows
            .iter()
            .find(|row| !row.builtin && row.id == "gemini-4.0-flash")
            .unwrap();
        assert_eq!(added.auth_route, AuthRoute::OAuth);

        manager.cursor = manager.rows.iter().position(|row| row.id == "gemini-4.0-flash").unwrap();
        manager.handle_key(press(KeyCode::Char('e')));
        let form = manager.form.as_mut().unwrap();
        form.display_name = ModelTextField::new("Future Flash".to_string());
        form.auth_route = AuthRoute::ApiKey;
        form.focus = 3;
        manager.handle_key(press(KeyCode::Enter));
        let edited = manager
            .rows
            .iter()
            .find(|row| row.display_name == "Future Flash")
            .unwrap();
        assert_eq!(edited.auth_route, AuthRoute::ApiKey);

        manager.cursor = manager.rows.iter().position(|row| row.id == "gemini-4.0-flash").unwrap();
        manager.handle_key(press(KeyCode::Char('d')));
        manager.handle_key(press(KeyCode::Esc));
        assert!(manager.rows.iter().any(|row| row.id == "gemini-4.0-flash"), "Esc cancels confirmation");
        manager.handle_key(press(KeyCode::Char('d')));
        manager.handle_key(press(KeyCode::Enter));
        assert!(!manager.rows.iter().any(|row| row.id == "gemini-4.0-flash"));

        manager.cursor = manager.rows.iter().position(|row| row.id == "gemini-3.5-flash").unwrap();
        manager.handle_key(press(KeyCode::Char('d')));
        manager.handle_key(press(KeyCode::Enter));
        let hidden = manager.rows.iter().position(|row| row.id == "gemini-3.5-flash").unwrap();
        assert!(manager.rows[hidden].hidden);
        manager.cursor = hidden;
        manager.handle_key(press(KeyCode::Char('r')));
        assert!(!manager.rows.iter().find(|row| row.id == "gemini-3.5-flash").unwrap().hidden);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn manager_invalid_form_error_is_inline_and_escape_cancels_form() {
        let (mut manager, path) = test_manager("invalid");
        manager.form = Some(ModelForm { original: None, provider: 0, providers: vec![CatalogProvider::Google], id: ModelTextField::new(String::new()), display_name: ModelTextField::new(String::new()), auth_route: AuthRoute::OAuth, focus: 0, error: None });
        manager.handle_key(press(KeyCode::Enter));
        manager.handle_key(press(KeyCode::Enter));
        manager.handle_key(press(KeyCode::Enter));
        manager.handle_key(press(KeyCode::Enter));
        assert_eq!(manager.form.as_ref().and_then(|form| form.error.as_deref()), Some("Model ID cannot be empty"));
        manager.handle_key(press(KeyCode::Esc));
        assert!(manager.form.is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn manager_duplicate_alias_error_is_inline() {
        let (mut manager, path) = test_manager("duplicate");
        manager.form = Some(ModelForm {
            original: None,
            provider: 0,
            providers: vec![CatalogProvider::Google],
            id: ModelTextField::new("gemini-flash".to_string()),
            display_name: ModelTextField::new("Duplicate Flash".to_string()),
            auth_route: AuthRoute::OAuth,
            focus: 3,
            error: None,
        });
        manager.handle_key(press(KeyCode::Enter));
        assert_eq!(
            manager.form.as_ref().and_then(|form| form.error.as_deref()),
            Some("A model with this provider and ID already exists")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn manager_write_refreshes_picker_immediately_and_escape_is_focus_trapped() {
        let (manager, path) = test_manager("refresh");
        let mut modal = ModelPickerModal::new(vec![google_entry("gemini-3.5-flash", "Gemini 3.5 Flash")]);
        modal.manager = Some(manager);

        modal.manager.as_mut().unwrap().form = Some(ModelForm { original: None, provider: 0, providers: vec![CatalogProvider::Google], id: ModelTextField::new(String::new()), display_name: ModelTextField::new(String::new()), auth_route: AuthRoute::OAuth, focus: 0, error: None });
        modal.handle_key(press(KeyCode::Enter));
        for ch in "gemini-4.0-flash".chars() { modal.handle_key(press(KeyCode::Char(ch))); }
        modal.handle_key(press(KeyCode::Enter));
        for ch in "Gemini 4.0 Flash".chars() { modal.handle_key(press(KeyCode::Char(ch))); }
        modal.handle_key(press(KeyCode::Enter));
        modal.handle_key(press(KeyCode::Enter));
        assert!(modal.entries.iter().any(|entry| entry.model.alias == "gemini-4.0-flash"));
        assert!(modal.manager.is_some());
        assert!(modal.handle_key(press(KeyCode::Esc)).is_none());
        assert!(modal.manager.is_none(), "Esc closes only the nested manager, not the picker");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn manager_renders_in_short_narrow_terminal_without_panicking() {
        use ratatui::{Terminal, backend::TestBackend};

        let (manager, path) = test_manager("render");
        let mut modal = ModelPickerModal::new(vec![google_entry("gemini-3.5-flash", "Gemini 3.5 Flash")]);
        modal.manager = Some(manager);
        let backend = TestBackend::new(32, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let theme = Theme::default_dark();
        terminal.draw(|frame| modal.draw(frame, frame.area(), &theme)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(rendered.contains("Manage models"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn uppercase_m_remains_filter_input_and_footer_advertises_f2_manager() {
        let mut modal = ModelPickerModal::new(registry());
        modal.handle_key(press(KeyCode::Char('M')));
        assert_eq!(modal.query(), "M");
        assert!(modal.manager.is_none());
        let rendered = modal
            .render_lines(&Theme::default_dark())
            .into_iter()
            .map(|line| line.to_string())
            .collect::<String>();
        assert!(rendered.contains("F2"));
        assert!(rendered.contains("Manage models"));
    }

    #[test]
    fn hiding_active_model_keeps_it_pinned_until_the_session_switches() {
        let path = std::env::temp_dir().join(format!(
            "zo-model-picker-active-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let active = CatalogRow {
            provider: CatalogProvider::Google,
            id: "gemini-3.5-flash".to_string(),
            display_name: "Gemini 3.5 Flash".to_string(),
            auth_route: AuthRoute::Auto,
            builtin: true,
            hidden: false,
        };
        catalog.delete_or_hide(&active).unwrap();
        let mut entry = google_entry("gemini-3.5-flash", "Gemini 3.5 Flash");
        entry.provider = "suggested".to_string();
        let mut modal = ModelPickerModal::new(vec![entry]);

        modal.sync_catalog_entries(&catalog, &[CatalogProvider::Google]);

        assert!(modal.entries.iter().any(|entry| {
            entry.provider == "suggested" && entry.model.alias == "gemini-3.5-flash"
        }));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn edit_form_offers_all_connected_auth_providers() {
        let (mut manager, path) = test_manager("provider-gate");
        manager.connected = vec![
            CatalogProvider::Anthropic,
            CatalogProvider::Openai,
            CatalogProvider::Google,
        ];
        manager.refresh();
        manager.cursor = manager
            .rows
            .iter()
            .position(|row| row.provider == CatalogProvider::Anthropic)
            .unwrap();

        manager.handle_key(press(KeyCode::Char('e')));

        let providers = &manager.form.as_ref().unwrap().providers;
        assert!(providers.contains(&CatalogProvider::Google));
        assert!(providers.contains(&CatalogProvider::Anthropic));
        assert!(providers.contains(&CatalogProvider::Openai));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn auth_route_field_selects_google_labels_without_claiming_hardware_cursor() {
        assert_eq!(
            auth_route_label(CatalogProvider::Google, AuthRoute::OAuth),
            "OAuth · Code Assist"
        );
        assert_eq!(
            auth_route_label(CatalogProvider::Google, AuthRoute::ApiKey),
            "API key · Gemini API"
        );
        let (mut manager, path) = test_manager("auth-field");
        manager.form = Some(ModelForm {
            original: None,
            provider: 0,
            providers: vec![CatalogProvider::Google],
            id: ModelTextField::new("gemini-3.6-flash".to_string()),
            display_name: ModelTextField::new("Gemini 3.6 Flash".to_string()),
            auth_route: AuthRoute::OAuth,
            focus: 3,
            error: None,
        });
        manager.handle_key(press(KeyCode::Right));
        assert_eq!(manager.form.as_ref().unwrap().auth_route, AuthRoute::ApiKey);
        assert_eq!(manager.cursor_position(Rect::new(0, 0, 60, 20)), None);
        manager.handle_key(press(KeyCode::Left));
        assert_eq!(manager.form.as_ref().unwrap().auth_route, AuthRoute::OAuth);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn text_field_edits_at_unicode_character_boundaries() {
        let mut field = ModelTextField::new("ab한글".to_string());
        field.move_left();
        field.insert_char('X');
        assert_eq!(field.text(), "ab한X글");
        field.backspace();
        assert_eq!(field.text(), "ab한글");
        field.move_left();
        field.delete();
        assert_eq!(field.text(), "ab글");
        field.move_home();
        field.insert_text("Gemini ");
        assert_eq!(field.text(), "Gemini ab글");
        field.move_end();
        assert_eq!(field.cursor, field.text().len());
    }

    #[test]
    fn model_form_accepts_shifted_uppercase_and_pastes_at_cursor() {
        let (manager, path) = test_manager("editor");
        let mut modal = ModelPickerModal::new(vec![google_entry(
            "gemini-3.5-flash",
            "Gemini 3.5 Flash",
        )]);
        modal.manager = Some(manager);
        let mut id = ModelTextField::new("gemini-flash".to_string());
        id.move_home();
        for _ in 0.."gemini-".chars().count() {
            id.move_right();
        }
        modal.manager.as_mut().unwrap().form = Some(ModelForm {
            original: None,
            provider: 0,
            providers: vec![CatalogProvider::Google],
            id,
            display_name: ModelTextField::new(String::new()),
            auth_route: AuthRoute::OAuth,
            focus: 1,
            error: None,
        });

        modal.paste_text("3.6-");
        modal.handle_key(KeyEvent {
            code: KeyCode::Char('X'),
            modifiers: KeyModifiers::SHIFT,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        });

        let form = modal.manager.as_ref().unwrap().form.as_ref().unwrap();
        assert_eq!(form.id.text(), "gemini-3.6-Xflash");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn model_form_reports_the_hardware_cursor_at_the_insertion_point() {
        let (manager, path) = test_manager("cursor");
        let mut modal = ModelPickerModal::new(vec![google_entry(
            "gemini-3.5-flash",
            "Gemini 3.5 Flash",
        )]);
        modal.manager = Some(manager);
        let mut id = ModelTextField::new("abc".to_string());
        id.move_home();
        id.move_right();
        modal.manager.as_mut().unwrap().form = Some(ModelForm {
            original: None,
            provider: 0,
            providers: vec![CatalogProvider::Google],
            id,
            display_name: ModelTextField::new(String::new()),
            auth_route: AuthRoute::OAuth,
            focus: 1,
            error: None,
        });

        assert_eq!(
            modal.cursor_position(Rect::new(10, 5, 60, 20)),
            Some(Position::new(28, 8))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn promoted_builtin_refresh_is_not_marked_unverified() {
        let path = std::env::temp_dir().join(format!(
            "zo-model-picker-promoted-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::write(
            &path,
            r#"{"modelCatalog":{"models":[{"provider":"google","id":"gemini-3.6-flash","displayName":"Gemini 3.6 Flash","authRoute":"oauth"}],"hidden":[{"provider":"google","id":"gemini-3.5-flash"}]}}"#,
        )
        .unwrap();
        let catalog = ModelCatalog::load_from(path.clone()).unwrap();
        let mut modal = ModelPickerModal::new(vec![google_entry(
            "gemini-3.5-flash",
            "Gemini 3.5 Flash",
        )]);

        modal.sync_catalog_entries(&catalog, &[CatalogProvider::Google]);

        let promoted = modal
            .entries
            .iter()
            .find(|entry| entry.model.alias == "gemini-3.6-flash")
            .unwrap();
        assert_eq!(promoted.model.display_name, "Gemini 3.6 Flash");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn custom_model_refresh_is_visibly_marked_unverified() {
        let path = std::env::temp_dir().join(format!(
            "zo-model-picker-unverified-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut catalog = ModelCatalog::load_from(path.clone()).unwrap();
        catalog
            .add(
                CatalogProvider::Google,
                "gemini-4.0-flash",
                "Gemini 4.0 Flash",
            )
            .unwrap();
        let mut modal = ModelPickerModal::new(vec![google_entry(
            "gemini-3.5-flash",
            "Gemini 3.5 Flash",
        )]);

        modal.sync_catalog_entries(&catalog, &[CatalogProvider::Google]);

        let custom = modal
            .entries
            .iter()
            .find(|entry| entry.model.alias == "gemini-4.0-flash")
            .unwrap();
        assert!(custom.model.display_name.contains("UNVERIFIED"));
        let _ = std::fs::remove_file(path);
    }
}
