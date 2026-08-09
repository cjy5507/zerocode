//! Ctrl+G agents viewer — the structured replacement for the old raw-text
//! agents pager. A flat, session-scoped list of every sub-agent (running AND
//! finished — no live gate) beside a live detail pane, with keyboard/mouse
//! selection that survives refreshes by agent id.
//!
//! Data comes from [`workflow_progress::read_agent_rows_since`]; the row and
//! detail renderers are shared with the Ctrl+O workflow viewer
//! ([`agent_list_line`], [`agent_meta_column`] / [`agent_output_column`] and
//! the [`DetailBand`] split) so a fleet reads identically on both surfaces.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph};

use super::super::theme::Theme;
use super::super::workflow_progress::AgentRowsSnapshot;
use super::draw_scrollbar;
use super::workflow_viewer::{
    COLUMN_RAIL_WIDTH, DetailBand, DetailColumn, DetailFolds, DetailSection, PANE_GUTTER,
    WorkflowAgentRow, agent_list_line, agent_meta_column, agent_output_column,
    detail_label_style, detail_value_style, draw_column_rail, draw_detail_rail,
    draw_output_column, fleet_width, hit_row, section_rule, short, split_detail_band_for,
    visible_offset,
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
    /// Scroll offset of the stacked (narrow-terminal) detail card, counted from
    /// its first row.
    detail_scroll: u16,
    /// Rows the reader has walked *back from the live tail* in the output
    /// column. `0` follows the stream — the newest line sits on the band's last
    /// row, which is what makes a running agent read as live. `PgUp` walks
    /// back, `PgDn` returns to the tail.
    output_scroll_back: u16,
    /// Which detail sections are folded away (clicked headings). Modal state,
    /// so a fold survives walking the fleet.
    folds: DetailFolds,
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
            output_scroll_back: 0,
            folds: DetailFolds::default(),
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
            self.reset_detail_scroll();
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
            self.reset_detail_scroll();
            return true;
        }
        false
    }

    /// Flip the history window and report the new state; the host re-reads the
    /// snapshot with it (the modal itself never touches the disk).
    pub fn toggle_history(&mut self) -> bool {
        self.show_history = !self.show_history;
        self.reset_detail_scroll();
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

    /// Back to the top of the stacked card and to the live tail of the output
    /// column — a fresh selection is read from its start, and its stream from
    /// its newest line.
    fn reset_detail_scroll(&mut self) {
        self.detail_scroll = 0;
        self.output_scroll_back = 0;
    }

    fn select_prev(&mut self, step: usize) {
        self.selected = self.selected.saturating_sub(step);
        self.reset_detail_scroll();
    }

    fn select_next(&mut self, step: usize) {
        let max = self.snapshot.rows.len().saturating_sub(1);
        self.selected = self.selected.saturating_add(step).min(max);
        self.reset_detail_scroll();
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
            // resume it with context if it already finished). Enter carries it
            // too — this is the viewer's only mutation, and bare `m`/`i` never
            // arrive while a Korean IME is composing.
            KeyCode::Enter | KeyCode::Char('m' | 'i') if self.selected_row().is_some() => {
                self.input = Some(String::new());
                self.feedback = None;
            }
            KeyCode::Home | KeyCode::Char('g') => {
                self.selected = 0;
                self.list_scroll = 0;
                self.reset_detail_scroll();
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.selected = self.snapshot.rows.len().saturating_sub(1);
                self.reset_detail_scroll();
            }
            // The detail pane holds the long content (activity feed, output
            // tail), so paging scrolls it; the list is walked with ↑/↓.
            // Both axes move together because only one of them is on screen:
            // the stacked card scrolls from its top, the output column back
            // from its live tail.
            KeyCode::PageUp => {
                self.detail_scroll = self.detail_scroll.saturating_sub(10);
                self.output_scroll_back = self.output_scroll_back.saturating_add(10);
            }
            KeyCode::PageDown => {
                self.detail_scroll = self.detail_scroll.saturating_add(10);
                self.output_scroll_back = self.output_scroll_back.saturating_sub(10);
            }
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
    /// `area` the draw used:
    ///
    /// * a list row selects that agent
    /// * a detail **section heading** (`▾ LIVE OUTPUT ───`) folds or unfolds it
    ///
    /// The layout is recomputed with the exact same pure math as
    /// [`Self::draw`] — down to each section's wrapped row span — so
    /// hit-testing can never drift from the pixels.
    pub fn handle_click(&mut self, column: u16, row: u16, area: Rect, theme: &Theme) {
        let Some(regions) = self.layout(area, theme) else {
            return;
        };
        if let Some(hit) = hit_row(regions.list_area, column, row) {
            let offset = self.list_offset(regions.list_area.height);
            let idx = usize::from(offset.saturating_add(hit));
            if idx < self.snapshot.rows.len() {
                self.selected = idx;
                self.reset_detail_scroll();
            }
            return;
        }
        self.toggle_detail_section_at(&regions, column, row, theme);
    }

    /// Fold/unfold the detail section whose heading the click landed on.
    fn toggle_detail_section_at(
        &mut self,
        regions: &AgentsViewerLayout,
        column: u16,
        row: u16,
        theme: &Theme,
    ) -> bool {
        let Some(agent) = self.selected_row() else {
            return false;
        };
        let section = match regions.detail_band {
            // The output column pins its heading to row 0 and scrolls the tail
            // under it, so that row — and only that row — is the fold target.
            Some(band) if column >= band.output.x => hit_row(band.output, column, row)
                .filter(|hit| *hit == 0)
                .and_then(|hit| {
                    self.output_column(agent, band.output.width, theme)
                        .section_at_row(usize::from(hit), band.output.width)
                }),
            Some(band) => hit_row(band.meta, column, row).and_then(|hit| {
                self.meta_column(agent, band.meta.width, theme)
                    .section_at_row(usize::from(hit), band.meta.width)
            }),
            None => hit_row(regions.detail_area, column, row).and_then(|hit| {
                let width = regions.detail_area.width;
                self.stacked_column(agent, width, theme)
                    .section_at_row(usize::from(hit) + usize::from(self.detail_scroll), width)
            }),
        };
        let Some(section) = section else {
            return false;
        };
        self.folds.toggle(section);
        true
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

    /// Rows the detail pane always keeps, so it can never collapse into a
    /// caption no matter how long the fleet list grows.
    const DETAIL_MIN_ROWS: u16 = 4;

    /// Rows of air between two sections of the modal.
    const SECTION_AIR_ROWS: u16 = 1;

    /// Width at which the fleet moves beside the inspector instead of above it.
    ///
    /// Higher than the workflow viewer's own split floor on purpose. There the
    /// tree *replaces* two panes and has to be a column at any width it fits;
    /// here the list stacks perfectly well, so taking thirty-odd columns from
    /// the inspector only pays once the inspector can still split into
    /// meta │ output with them gone.
    const COLUMN_LAYOUT_MIN_WIDTH: u16 = super::workflow_viewer::FLEET_MIN_WIDTH
        + COLUMN_RAIL_WIDTH
        + super::workflow_viewer::DETAIL_SPLIT_MIN_WIDTH
        + 2 * PANE_GUTTER;

    /// Rows the list pane will actually paint — the same count
    /// [`Self::draw_list`] renders, so measure and paint cannot drift.
    fn list_content_rows(&self, theme: &Theme) -> usize {
        if self.snapshot.rows.is_empty() {
            self.empty_state_lines(theme).len()
        } else {
            self.snapshot.rows.len()
        }
    }

    /// Pure geometry for one frame: header / list / detail band / footer, plus
    /// the band's two columns when the terminal is wide enough to split them.
    /// `None` when the area is too small to show anything.
    fn layout(&self, area: Rect, theme: &Theme) -> Option<AgentsViewerLayout> {
        // Rules top and bottom, no side borders and no nested panels: the
        // frame the old version drew cost twelve columns of content on a
        // 60-column terminal and clipped agent names for it.
        if area.height < 6 || area.width == 0 {
            return None;
        }
        let [top_rule, header, body, footer, bottom_rule] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(2),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(area);

        // Fleet beside the inspector, both full height — the Ctrl+O shape
        // without the phase headers, so an agent reads the same on either
        // surface. Stacked, the two panes had a two-row fleet spanning the
        // whole width and a short card under it, leaving well over half a tall
        // terminal empty: the *screen* had no structure, whatever the card did.
        let (list_area, list_rail, band) = if body.width >= Self::COLUMN_LAYOUT_MIN_WIDTH {
            let [list, rail, band] = Layout::horizontal([
                Constraint::Length(fleet_width(body.width)),
                Constraint::Length(COLUMN_RAIL_WIDTH),
                Constraint::Min(1),
            ])
            .spacing(PANE_GUTTER)
            .areas(body);
            (list, Some(rail), band)
        } else {
            // Narrow: the list is sized to its own content, not to "whatever is
            // left over" — `Min(1)` had a one-agent fleet reserve the entire
            // body and paint ~20 blank rows above the detail pane. One row of
            // air separates them, or the card's `─ name ───` rule reads as the
            // list's last row rather than the head of a new section.
            let list_rows = u16::try_from(self.list_content_rows(theme))
                .unwrap_or(u16::MAX)
                .min(
                    body.height
                        .saturating_sub(Self::DETAIL_MIN_ROWS + Self::SECTION_AIR_ROWS),
                )
                .max(1);
            let [list, _air, band] = Layout::vertical([
                Constraint::Length(list_rows),
                Constraint::Length(Self::SECTION_AIR_ROWS),
                Constraint::Min(1),
            ])
            .areas(body);
            (list, None, band)
        };
        // The band takes the rows it was given, whole. Hugging it to its content
        // left a tall terminal with forty blank rows under a *clipped* output
        // tail — the pane with more to show was the one being starved.
        let detail_area = band;
        let detail_band = self
            .selected_row()
            .and_then(|agent| {
                split_detail_band_for(
                    detail_area,
                    agent,
                    None,
                    self.folds.folded(DetailSection::Output),
                )
            });

        Some(AgentsViewerLayout {
            top_rule,
            header,
            list_area,
            list_rail,
            detail_area,
            detail_band,
            footer,
            bottom_rule,
        })
    }

    /// Left column: the selected agent's name on a rule, then the shared meta
    /// card (status → task → model → metrics → plumbing → ACTIVITY).
    fn meta_column(&self, agent: &WorkflowAgentRow, width: u16, theme: &Theme) -> DetailColumn {
        let mut column = DetailColumn::default();
        column.push(section_rule(&short(&agent.name, 40), width, theme));
        column.append(agent_meta_column(agent, width, theme, self.folds));
        column
    }

    /// Right column: the agent's streamed output, which is what fills the band.
    fn output_column(&self, agent: &WorkflowAgentRow, width: u16, theme: &Theme) -> DetailColumn {
        agent_output_column(
            agent,
            width,
            theme,
            self.folds.folded(DetailSection::Output),
            None,
        )
    }

    /// The narrow-terminal fallback: one stacked card, exactly as before the
    /// band could split.
    fn stacked_column(&self, agent: &WorkflowAgentRow, width: u16, theme: &Theme) -> DetailColumn {
        let mut column = self.meta_column(agent, width, theme);
        column.push(Line::from(""));
        column.append(self.output_column(agent, width, theme));
        column
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let Some(regions) = self.layout(area, theme) else {
            return;
        };
        // Blank the transcript cells this modal sits on *and* lay down the
        // dialog's glass fill: this viewer draws its own rules rather than a
        // `CardFrame` block, so nothing else paints its interior, and a bare
        // `Clear` would strip the fill the frosted backdrop put down.
        super::dialog_surface(frame, area, theme);

        // Pi dialog chrome: a titled top rule and a plain bottom rule in the
        // resting `border` hue, no sides — the same delimiters `CardFrame`
        // draws for every other modal.
        let rule_style = Style::new().fg(theme.palette.border);
        frame.render_widget(
            Paragraph::new(super::workflow_viewer::section_rule(
                "Agents",
                area.width,
                theme,
            )),
            regions.top_rule,
        );
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "\u{2500}".repeat(area.width as usize),
                rule_style,
            ))),
            regions.bottom_rule,
        );

        // Fitted: the header is assembled from `format!` (tally, scope label,
        // hidden counts) with no width bound of its own, and renders without
        // `wrap` — so on a narrow pane its tail was cut mid-glyph.
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                vec![self.header_line(theme)],
                regions.header.width,
            )),
            regions.header,
        );
        self.draw_list(frame, regions.list_area, theme);
        if let Some(rail) = regions.list_rail {
            draw_column_rail(frame, rail, theme);
        }
        self.draw_detail(frame, &regions, theme);
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                vec![self.footer_content(theme, regions.footer.width)],
                regions.footer.width,
            )),
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
            let prompt = format!("✉ to {target} ❯ ");
            // The compose row's prefix grows with the agent name *and* whatever
            // has been typed, so the hints get only what is left over. Budgeting
            // them against the full width would let a long message push
            // `Esc cancel` off the edge mid-word; here they simply drop out, and
            // the text being typed is what keeps the space.
            let spent = crate::tui::text_metrics::display_width(&prompt)
                + crate::tui::text_metrics::display_width(input)
                + 3;
            let hint_budget = width.saturating_sub(
                u16::try_from(spent).unwrap_or(u16::MAX),
            );
            return Line::from(vec![
                Span::styled(prompt, Style::new().fg(theme.palette.accent)),
                Span::styled(input.clone(), detail_value_style(theme)),
                Span::styled("▌", Style::new().fg(theme.palette.accent)),
                Span::styled("  ", detail_label_style(theme)),
            ]
            .into_iter()
            .chain(
                super::key_hint_footer_fitted(
                    theme,
                    &[("Enter", "send"), ("Esc", "cancel")],
                    hint_budget,
                )
                .spans,
            )
            .collect::<Vec<_>>());
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
        // `muted`, not `typography.dim`: the latter stacks `Modifier::DIM` on
        // top of the dim hue and sank the whole tally into the dialog surface.
        let dim = detail_label_style(theme);
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

    fn draw_list(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        if self.snapshot.rows.is_empty() {
            frame.render_widget(
                Paragraph::new(super::wrap_body_rows(
                    &self.empty_state_lines(theme),
                    area.width,
                    false,
                )),
                area,
            );
            return;
        }
        // The selected row is washed across its full width rather than framed —
        // the same cue Pi's selectors use, and the reason no panel border is
        // needed to show where the selection is.
        let selection_bg = theme.selection_bg();
        let lines: Vec<Line<'static>> = self
            .snapshot
            .rows
            .iter()
            .enumerate()
            .map(|(idx, agent)| {
                let selected = idx == self.selected;
                let line = agent_list_line(agent, selected, theme, self.tick, area.width);
                match (selected, selection_bg) {
                    (true, Some(bg)) => line.style(Style::new().bg(bg)),
                    _ => line,
                }
            })
            .collect();
        let offset = self.list_offset(area.height);
        // Fitted: agent names and their status/model suffixes are caller data of
        // unbounded length, and the list renders without `wrap`, so a long name
        // used to lose its tail silently rather than visibly.
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(lines, area.width)).scroll((offset, 0)),
            area,
        );
        draw_scrollbar(frame, area, offset, self.snapshot.rows.len(), theme);
    }

    fn empty_state_lines(&self, theme: &Theme) -> Vec<Line<'static>> {
        let dim = detail_label_style(theme);
        let mut lines = vec![Line::from(Span::styled(
            "no agents this session yet",
            detail_value_style(theme),
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

    /// Detail for the selected agent, rendered *under* the selection rather
    /// than beside it — so it gets the full width instead of half.
    ///
    /// Wide enough, the band splits into `meta │ output`: the meta rows are
    /// forty cells of label + value and stop there, so a 200-column terminal
    /// spent every row's remaining 160 columns on nothing while the streamed
    /// output trickled down the left edge with a band of empty surface beneath
    /// it. The output column now takes the band's whole height and fills it
    /// with the tail.
    fn draw_detail(&self, frame: &mut Frame<'_>, regions: &AgentsViewerLayout, theme: &Theme) {
        let area = regions.detail_area;
        if area.height == 0 || area.width == 0 {
            return;
        }
        let Some(agent) = self.selected_row() else {
            // Lead with the same rule the selected state uses: the rule is what
            // marks the list/detail split, so dropping it when nothing is
            // selected left the two panes running together.
            frame.render_widget(
                Paragraph::new(super::fit_body_rows(
                    vec![
                        super::workflow_viewer::section_heading_rule(
                            "detail", area.width, theme,
                        ),
                        Line::from(Span::styled(
                            "select an agent",
                            detail_label_style(theme),
                        )),
                    ],
                    area.width,
                )),
                area,
            );
            return;
        };
        let Some(band) = regions.detail_band else {
            // Narrow fallback: one stacked card, scrolled from its top. Clamp so
            // PageDown past the end cannot scroll everything off-screen.
            let column = self.stacked_column(agent, area.width, theme);
            // Wrapped up front, so the scroll clamp counts the rows that are
            // actually painted instead of asking ratatui's wrapper for a second
            // opinion (which is off by one where a wide glyph meets the edge).
            let rows = super::wrap_body_rows(&column.into_lines(), area.width, false);
            let max_scroll = u16::try_from(rows.len())
                .unwrap_or(u16::MAX)
                .saturating_sub(area.height.max(1));
            let scroll = self.detail_scroll.min(max_scroll);
            frame.render_widget(Paragraph::new(rows).scroll((scroll, 0)), area);
            return;
        };
        let meta = self.meta_column(agent, band.meta.width, theme);
        let output = self.output_column(agent, band.output.width, theme);
        // The rail separates two columns, so it runs as far as the taller of
        // them and no further. Sized to the output column alone it stopped
        // mid-card whenever the meta side was longer, and the ACTIVITY block
        // below the cut read as having fallen out of the layout.
        let content_rows = output
            .row_count(band.output.width)
            .max(meta.row_count(band.meta.width))
            .min(usize::from(band.rail.height));
        frame.render_widget(
            Paragraph::new(super::wrap_body_rows(
                &meta.into_lines(),
                band.meta.width,
                false,
            )),
            band.meta,
        );
        draw_detail_rail(
            frame,
            band.rail,
            theme,
            u16::try_from(content_rows).unwrap_or(u16::MAX),
        );
        draw_output_column(frame, band.output, output, self.output_scroll_back);
    }
}

struct AgentsViewerLayout {
    top_rule: Rect,
    header: Rect,
    list_area: Rect,
    /// The `│` between the fleet column and the inspector; `None` when the
    /// modal is too narrow and stacks them instead.
    list_rail: Option<Rect>,
    /// The whole detail band — the body's full remaining height.
    detail_area: Rect,
    /// The band's two columns, or `None` on a terminal too narrow to split.
    detail_band: Option<DetailBand>,
    footer: Rect,
    bottom_rule: Rect,
}

fn footer_line(theme: &Theme, width: u16) -> Line<'static> {
    // `click` is advertised because it now does something on every surface of
    // this modal: a row selects, a section heading folds.
    let full = super::key_hint_footer_reflowing(
        theme,
        &[
            ("↑/↓", "agent"),
            ("PgUp/PgDn", "output"),
            ("click", "select/fold"),
            ("Enter", "message"),
            ("Tab", "history"),
            ("Esc", "close"),
        ],
    );
    if line_width(&full) <= usize::from(width) {
        return full;
    }
    // The reduced set is a deliberate editorial choice — these three are the
    // hints worth keeping when there is no room for six — but it is not
    // automatically narrow enough: at ~33 cells it overflows a genuinely cramped
    // pane too. Fitting it drops from the end instead of letting the renderer cut
    // `Esc clo`, and always keeps the first hint.
    super::key_hint_footer_fitted(
        theme,
        &[("↑/↓", "agent"), ("m", "message"), ("Esc", "close")],
        width,
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
    use super::super::workflow_viewer::split_detail_band;
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

    /// A streaming agent with a long rolling output tail — the case the split
    /// band exists for.
    fn streaming_row(id: &str, tail_lines: usize) -> WorkflowAgentRow {
        WorkflowAgentRow {
            description: "Implement and test the PayNow contract".to_string(),
            model: "claude-opus-5".to_string(),
            tool_calls: Some(2),
            elapsed_secs: 6,
            recent_tools: vec!["bash · cd svc && ls".to_string()],
            output_tail: Some(
                (0..tail_lines)
                    .map(|idx| format!("streamed prose row {idx}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            ..row(id, "running")
        }
    }

    fn render(modal: &AgentsViewerModal, w: u16, h: u16) -> ratatui::buffer::Buffer {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let mut term = Terminal::new(TestBackend::new(w, h)).expect("backend");
        term.draw(|frame| modal.draw(frame, Rect::new(0, 0, w, h), &Theme::pi()))
            .expect("draw");
        term.backend().buffer().clone()
    }

    fn row_text(buf: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect()
    }

    /// The whole frame as text, one row per line.
    fn dump_of(buf: &ratatui::buffer::Buffer) -> String {
        let mut out = String::new();
        for y in 0..buf.area.height {
            out.push_str(&row_text(buf, y));
            out.push('\n');
        }
        out
    }

    /// `(y, x)` of the first cell of `needle` in the rendered buffer — click
    /// coordinates come from the pixels, never from a hardcoded guess.
    fn find(buf: &ratatui::buffer::Buffer, needle: &str) -> (u16, u16) {
        for y in 0..buf.area.height {
            let row = row_text(buf, y);
            if let Some(byte_idx) = row.find(needle) {
                let col = row[..byte_idx].chars().count();
                return (y, u16::try_from(col).unwrap_or(0));
            }
        }
        panic!("{needle:?} not found in:\n{buf:?}");
    }

    /// **공간 배분 회귀 핀** — 넓은 폭에서 detail 은 2단으로 갈라지고, 우측
    /// LIVE OUTPUT 컬럼이 밴드의 마지막 행까지 내용을 채운다(빈 행 밴드 금지).
    #[test]
    fn a_wide_detail_splits_into_meta_and_output_columns() {
        let theme = Theme::pi();
        let modal = AgentsViewerModal::new(snapshot(vec![
            streaming_row("opus-paynow", 40),
            row("classifier", "completed"),
        ]));
        let area = Rect::new(0, 0, 200, 40);
        let regions = modal.layout(area, &theme).expect("layout fits");
        let band = regions.detail_band.expect("200 columns splits the band");

        // Two real columns with a one-cell gutter of air on each side of the
        // rail — Pi separates panes with space, never with a frame.
        assert!(band.meta.width >= 40 && band.output.width >= 40);
        assert_eq!(band.rail.x, band.meta.x + band.meta.width + 1);
        assert_eq!(band.output.x, band.rail.x + 2);

        let buf = render(&modal, 200, 40);
        // The output tail is 40 rows long and the band is shorter, so every row
        // of the column carries output — the ~34-row void under the old
        // single-stack detail is gone.
        for y in band.output.y..band.output.y + band.output.height {
            let text = row_text(&buf, y);
            let column: String = text
                .chars()
                .skip(usize::from(band.output.x))
                .collect::<String>()
                .trim()
                .to_string();
            assert!(
                !column.is_empty(),
                "row {y} of the output column is blank — the band must be filled by the tail"
            );
        }
        // ...and the newest line is the one on the last row: the column follows
        // its own tail.
        let last = row_text(&buf, band.output.y + band.output.height - 1);
        assert!(
            last.contains("streamed prose row 39"),
            "the output column must follow the live tail: {last}"
        );

        // The meta column keeps the reading order the pane is for.
        let (status_y, status_x) = find(&buf, "status   running");
        assert!(status_x < band.rail.x, "meta rows live left of the rail");
        let (task_y, _) = find(&buf, "task     Implement");
        assert!(task_y > status_y);
        assert_eq!(
            buf[(band.rail.x, status_y)].symbol(),
            "\u{2502}",
            "a `│` rail separates the columns"
        );
        assert_eq!(
            buf[(band.rail.x, status_y)].style().fg,
            Some(theme.palette.border),
            "the rail takes the resting `border` hue, like every other rule"
        );
        assert_eq!(
            buf[(band.rail.x, status_y)].style().bg,
            theme.surface2(),
            "the rail rides the dialog's own glass — it must not punch a hole \
             through the surface the modal painted"
        );
    }

    /// **결함 A 회귀 핀** — 라이브 `outputTail` 은 모델이 개행을 내보내기
    /// 전까지 **개행 없는 한 덩어리**(매니페스트 캡 2000자)다. 예전 렌더러는
    /// 각 줄을 200자에 잘라서, 55행짜리 출력 컬럼이 **1~2행만** 쓰고 나머지를
    /// 텅 비웠다. 캡은 문자 수가 아니라 행 수 기준이어야 하고, tail 은 컬럼
    /// 폭에 랩되어야 한다.
    #[test]
    fn a_newline_free_tail_fills_the_output_column() {
        let theme = Theme::pi();
        let mut blob = String::new();
        for idx in 0..250 {
            use std::fmt::Write as _;
            let _ = write!(blob, "token{idx:03} ");
        }
        assert!(blob.len() >= 2000 && !blob.contains('\n'));
        let mut agent = row("opus-paynow", "running");
        agent.output_tail = Some(blob);
        let modal = AgentsViewerModal::new(snapshot(vec![agent]));

        let area = Rect::new(0, 0, 200, 40);
        let band = modal
            .layout(area, &theme)
            .expect("layout")
            .detail_band
            .expect("200 columns splits the band");
        let buf = render(&modal, 200, 40);
        let filled = (band.output.y..band.output.y + band.output.height)
            .filter(|y| {
                !row_text(&buf, *y)
                    .chars()
                    .skip(usize::from(band.output.x))
                    .collect::<String>()
                    .trim()
                    .is_empty()
            })
            .count();
        assert!(
            filled >= 10,
            "an unbroken 2000-char tail must wrap into the column, not stop at \
             one row (only {filled} of {} rows carried output)",
            band.output.height
        );
        // ...and it still follows the tail: the newest token lands last.
        let dump = dump_of(&buf);
        assert!(dump.contains("token249"), "the newest token is visible: {dump}");
    }

    /// **결함 B 회귀 핀** — ACTIVITY 항목은 컬럼 폭과 무관한 고정 길이로
    /// 잘려서 2~3행으로 줄바꿈됐고, 이어지는 행이 0열에서 시작해 목록이
    /// 뭉개졌다. 항목당 정확히 1행, 컬럼 폭 이내, 그리고 경로가 살아남도록
    /// **중간 절단**.
    #[test]
    fn a_long_activity_entry_stays_on_one_row_of_its_column() {
        let theme = Theme::pi();
        let long = "bash · echo \"### PayNow\" >> /Users/joe/2026/forge-code/crates/\
                    zo-cli/src/tui/modals/tests/PayNowContractSpec.tsx";
        let mut agent = row("opus-paynow", "running");
        agent.recent_tools = vec![long.to_string()];
        let modal = AgentsViewerModal::new(snapshot(vec![agent.clone()]));
        let area = Rect::new(0, 0, 200, 40);
        let band = modal
            .layout(area, &theme)
            .expect("layout")
            .detail_band
            .expect("splits");

        let column = modal.meta_column(&agent, band.meta.width, &theme);
        let width = band.meta.width;
        let lines = {
            let rows = column.row_count(width);
            let lines = column.into_lines();
            assert_eq!(
                rows,
                lines.len(),
                "every meta row must occupy exactly one terminal row at width {width}"
            );
            lines
        };
        let entry = lines
            .iter()
            .find(|line| {
                line.spans
                    .first()
                    .is_some_and(|span| span.content.starts_with("  \u{00b7} "))
            })
            .expect("the activity entry");
        let text: String = entry
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(
            unicode_width::UnicodeWidthStr::width(text.as_str()) <= usize::from(width),
            "the entry must not overflow its {width}-cell column: {text:?}"
        );
        assert!(text.contains("bash"), "the head survives: {text:?}");
        assert!(
            text.contains("PayNowContractSpec.tsx"),
            "shortening the path's shared prefix keeps the file that names the \
             call: {text:?}"
        );
        assert!(
            !text.contains("/Users/joe"),
            "the 40-cell prefix nobody reads is what gets dropped: {text:?}"
        );

        // An argument that is still too long once its paths are shortened is
        // middle-cut, so the verb at the head and the tail both survive.
        let mut wide = row("opus-paynow", "running");
        wide.recent_tools = vec![format!("bash · {}", "abcdefghij".repeat(12))];
        let column = modal.meta_column(&wide, width, &theme);
        let text: String = column
            .into_lines()
            .iter()
            .find(|line| {
                line.spans
                    .first()
                    .is_some_and(|span| span.content.starts_with("  \u{00b7} "))
            })
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .expect("the activity entry");
        assert!(text.contains('…'), "an unshortenable brief is cut: {text:?}");
        assert!(
            unicode_width::UnicodeWidthStr::width(text.as_str()) <= usize::from(width),
            "and the cut keeps it inside the column: {text:?}"
        );
    }

    /// **결함 C 회귀 핀** — 메타/출력 비율은 고정이 아니라 밴드 폭의 46%,
    /// 하한 40 · 상한 64 칸이다. 예전 "하한 + slack 2/5" 식은 넓은 폭에서
    /// 천장에 눌러붙고 좁은 폭에서 라벨 컬럼을 굶겼다.
    #[test]
    fn the_meta_column_is_a_share_of_the_band_not_a_fixed_floor() {
        for (width, expected) in [(83u16, 40u16), (100, 46), (140, 64), (200, 64), (300, 64)] {
            let band = split_detail_band(Rect::new(0, 0, width, 10))
                .unwrap_or_else(|| panic!("width {width} must split"));
            assert_eq!(band.meta.width, expected, "meta width at band width {width}");
            assert!(
                band.output.width >= 40,
                "the output column keeps its floor at width {width}"
            );
        }
        assert!(
            split_detail_band(Rect::new(0, 0, 82, 10)).is_none(),
            "below the split floor the band still stacks"
        );
    }

    /// **결함 E 회귀 핀** — 컬럼 사이 세로 레일은 밴드가 **실제로 그린 행**
    /// 까지만 내려간다. 짧은 출력(접힌 스트림/미시작 에이전트) 옆에서 레일이
    /// 화면 바닥까지 매달리던 것이 어색함의 원인이었다.
    #[test]
    fn the_column_rail_stops_at_the_last_painted_row() {
        let theme = Theme::pi();
        // A tall meta card beside a *silent* stream: the case where the rail
        // used to run a dozen rows past the only two rows on its right.
        let mut agent = row("opus-paynow", "running");
        agent.description = "Implement and test the PayNow contract".to_string();
        agent.model = "claude-opus-5".to_string();
        agent.subagent_type = Some("analysis".to_string());
        agent.last_event = Some("lane.started".to_string());
        agent.output_file = Some("/tmp/opus-paynow.md".to_string());
        agent.recent_tools = vec!["bash · cd svc && ls".to_string()];
        let modal = AgentsViewerModal::new(snapshot(vec![agent.clone()]));
        let area = Rect::new(0, 0, 200, 40);
        let regions = modal.layout(area, &theme).expect("layout");
        let band = regions.detail_band.expect("splits");
        let buf = render(&modal, 200, 40);

        // The rail spans the *taller* column — it separates two of them — and
        // stops there: drawn to the band's full height it hangs below the card
        // with nothing on either side of it.
        let content_rows = modal
            .output_column(&agent, band.output.width, &theme)
            .row_count(band.output.width)
            .max(
                modal
                    .meta_column(&agent, band.meta.width, &theme)
                    .row_count(band.meta.width),
            );
        assert!(
            content_rows < regions.detail_area.height.into(),
            "precondition: the card is shorter than its band ({content_rows} rows)"
        );
        let last_content = band.output.y + u16::try_from(content_rows).unwrap_or(1) - 1;
        let last_rail = (band.rail.y..area.height)
            .rev()
            .find(|y| buf[(band.rail.x, *y)].symbol() == "\u{2502}")
            .expect("the rail is drawn beside the columns it separates");
        assert!(
            last_rail <= last_content,
            "the rail hangs {} rows below the content it separates",
            last_rail - last_content
        );
    }

    /// 좁은 폭에서는 기존 1단 스택으로 폴백한다 — 좁은 터미널은 두 번째
    /// 컬럼을 잃을 뿐, 카드를 잃어서는 안 된다.
    #[test]
    fn a_narrow_detail_falls_back_to_the_single_column_stack() {
        let theme = Theme::pi();
        let modal = AgentsViewerModal::new(snapshot(vec![streaming_row("opus-paynow", 6)]));
        for width in [60, 78, 82] {
            let area = Rect::new(0, 0, width, 30);
            let regions = modal.layout(area, &theme).expect("layout fits");
            assert!(
                regions.detail_band.is_none(),
                "width {width} is below the split floor and must stack"
            );
            let buf = render(&modal, width, 30);
            let dump = dump_of(&buf);
            assert!(dump.contains("status   running"), "{dump}");
            assert!(dump.contains("LIVE OUTPUT"), "{dump}");
            assert!(
                dump.contains("streamed prose row 5"),
                "the stacked card still shows the tail: {dump}"
            );
            assert!(
                !dump.contains('\u{2502}'),
                "no column rail is drawn in the stacked fallback: {dump}"
            );
        }
    }

    /// 섹션 헤딩 rule 은 **자기 컬럼 폭까지만** 그린다 — 전폭 rule 이 값과
    /// 어긋나던 것이 어색함의 원인이었다.
    #[test]
    fn section_rules_stop_at_their_own_column() {
        let theme = Theme::pi();
        let modal = AgentsViewerModal::new(snapshot(vec![streaming_row("opus-paynow", 12)]));
        let area = Rect::new(0, 0, 200, 40);
        let band = modal
            .layout(area, &theme)
            .expect("layout")
            .detail_band
            .expect("splits");
        let buf = render(&modal, 200, 40);

        // The agent-name rule heads the meta column and stops at its width.
        let (title_y, title_x) = find(&buf, "opus-paynow");
        assert!(title_x < band.rail.x);
        let title_row = row_text(&buf, title_y);
        let after_meta: String = title_row
            .chars()
            .skip(usize::from(band.meta.x + band.meta.width))
            .take(usize::from(band.rail.x - (band.meta.x + band.meta.width)) + 1)
            .collect();
        assert!(
            after_meta.trim().is_empty(),
            "the meta rule must stop inside its own column, got {after_meta:?}"
        );

        // The output heading is an inlay of the *right* column.
        let (out_y, out_x) = find(&buf, "LIVE OUTPUT");
        assert_eq!(out_y, band.output.y, "the output heading tops its column");
        assert!(out_x > band.rail.x);
        // Scoped to the band's own columns: the fleet column's separator is a
        // different rail and legitimately runs the full height.
        let inside_band: String = row_text(&buf, out_y)
            .chars()
            .skip(usize::from(band.meta.x))
            .take(usize::from(band.output.x - band.meta.x))
            .collect();
        assert!(
            !inside_band.contains('\u{2502}'),
            "the band's own rail starts below the headings, so they read as two \
             inlays: {inside_band:?}"
        );
    }

    /// 섹션 헤딩 클릭 = 접기/펼치기. 좌표는 렌더 버퍼에서 찾아온다(하드코딩
    /// 금지) — 그리기와 히트테스트가 어긋나면 이 테스트가 먼저 깨진다.
    #[test]
    fn clicking_a_section_heading_folds_and_unfolds_it() {
        let theme = Theme::pi();
        let mut modal = AgentsViewerModal::new(snapshot(vec![streaming_row("opus-paynow", 8)]));
        let area = Rect::new(0, 0, 200, 40);

        let buf = render(&modal, 200, 40);
        let (y, x) = find(&buf, "LIVE OUTPUT");
        assert!(row_text(&buf, y).contains('\u{25be}'), "open sections show ▾");

        modal.handle_click(x, y, area, &theme);
        let folded = render(&modal, 200, 40);
        let (fy, _) = find(&folded, "LIVE OUTPUT");
        let heading = row_text(&folded, fy);
        assert!(
            heading.contains('\u{25b8}') && heading.contains("8 lines"),
            "a folded section states what it is holding back: {heading}"
        );
        let dump = dump_of(&folded);
        assert!(
            !dump.contains("streamed prose row 0"),
            "folded output must actually be gone: {dump}"
        );

        // ...and the same click brings it back.
        modal.handle_click(x, fy, area, &theme);
        let reopened = render(&modal, 200, 40);
        let dump = dump_of(&reopened);
        assert!(dump.contains("streamed prose row 0"), "{dump}");

        // The ACTIVITY heading in the *other* column folds independently.
        let (ay, ax) = find(&reopened, "ACTIVITY");
        modal.handle_click(ax, ay, area, &theme);
        let dump = dump_of(&render(&modal, 200, 40));
        assert!(!dump.contains("cd svc && ls"), "activity folded: {dump}");
        assert!(
            dump.contains("streamed prose row 0"),
            "folding one section must not disturb the other: {dump}"
        );
    }

    /// 출력 컬럼은 기본적으로 최신(tail)을 보여주고 `PgUp` 이 과거로 되돌린다.
    #[test]
    fn the_output_column_follows_the_tail_and_pgup_walks_back() {
        let theme = Theme::pi();
        let mut modal = AgentsViewerModal::new(snapshot(vec![streaming_row("opus-paynow", 60)]));
        let area = Rect::new(0, 0, 200, 30);
        let band = modal
            .layout(area, &theme)
            .expect("layout")
            .detail_band
            .expect("splits");
        let last_row = band.output.y + band.output.height - 1;

        let buf = render(&modal, 200, 30);
        assert!(
            row_text(&buf, last_row).contains("streamed prose row 59"),
            "the newest line sits on the band's last row"
        );

        modal.handle_key(press(KeyCode::PageUp));
        let paged = render(&modal, 200, 30);
        assert!(
            !row_text(&paged, last_row).contains("streamed prose row 59"),
            "PgUp walks back from the tail"
        );
        modal.handle_key(press(KeyCode::PageDown));
        let back = render(&modal, 200, 30);
        assert!(
            row_text(&back, last_row).contains("streamed prose row 59"),
            "PgDn returns to the live tail"
        );
    }

    /// 리프레시는 인덱스가 아니라 agent id 로 선택을 보존한다 — 행 순서가
    /// 바뀌어도(에이전트 종결·정렬 이동) 보던 에이전트를 계속 본다. 이것이
    /// The fleet list speaks the Pi vocabulary: a `→` cursor over the selection
    /// wash, the agent's own hue on its (bold) name, and the status glyph set
    /// `✓ success / ✗ error / ⊘ warn / spinner cyan`.
    #[test]
    fn agent_rows_use_the_pi_selection_and_status_vocabulary() {
        let theme = Theme::pi();
        let rows = [
            row("alpha", "running"),
            row("bravo", "completed"),
            row("charlie", "failed"),
            row("delta", "stopped"),
        ];
        let lines: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(idx, agent)| {
                crate::tui::modals::workflow_viewer::agent_list_line(agent, idx == 0, &theme, 0, 80)
            })
            .collect();

        // Row 0 is selected: Pi arrow in the accent, washed.
        assert_eq!(lines[0].spans[0].content.as_ref(), "\u{2192} ");
        assert_eq!(lines[0].spans[0].style.fg, Some(theme.palette.accent));
        assert_eq!(lines[0].spans[0].style.bg, theme.selection_bg());
        assert_eq!(lines[1].spans[0].content.as_ref(), "  ");

        // Status glyph + hue per state.
        let status = |idx: usize| {
            (
                lines[idx].spans[1].content.trim().to_string(),
                lines[idx].spans[1].style.fg,
            )
        };
        assert_eq!(status(1), ("\u{2713}".into(), Some(theme.palette.success)));
        assert_eq!(status(2), ("\u{2717}".into(), Some(theme.palette.error)));
        assert_eq!(status(3), ("\u{2298}".into(), Some(theme.palette.warn)));
        assert_eq!(
            status(0).1,
            Some(theme.palette.cyan),
            "a running agent spins in cyan, never in the selection accent"
        );

        // The name carries the agent's identity hue, bold.
        assert_eq!(
            lines[1].spans[2].style.fg,
            Some(theme.agent_color("bravo"))
        );
        assert!(
            lines[1].spans[2]
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::BOLD)
        );
    }

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
        let regions = modal.layout(area, &theme).expect("layout fits");
        let list = regions.list_area;

        modal.handle_click(list.x + 1, list.y + 2, area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));

        // Outside the fleet column is a no-op: below its last row, and — now
        // that the fleet is a column rather than a full-width band — anywhere
        // in the inspector beside it.
        modal.handle_click(list.x + 1, list.y + list.height, area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));
        modal.handle_click(area.width.saturating_sub(2), regions.detail_area.y, area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("c"));

        // ...and the far right of the fleet column itself still selects its row.
        modal.handle_click(list.x + list.width - 1, list.y, area, &theme);
        assert_eq!(modal.selected_row().map(|r| r.id.as_str()), Some("a"));
    }

    /// 대비 회귀 핀 — the shared detail ladder must survive on this surface
    /// too: `muted` labels (never the DIM-modified `key_hint`), `fg` values,
    /// and an uppercase section inlay on a `border` rule.
    #[test]
    fn detail_rows_keep_the_readable_contrast_ladder() {
        use ratatui::style::Modifier;
        let theme = Theme::pi();
        let mut agent = row("explorer", "running");
        agent.description = "Audit the modal contrast ladder".to_string();
        agent.recent_tools = vec!["read_file · crates/zo-cli/src/tui/theme.rs".to_string()];
        let mut column = agent_meta_column(&agent, 80, &theme, DetailFolds::default());
        column.append(agent_output_column(&agent, 80, &theme, false, None));
        let lines = column.into_lines();

        let task = lines
            .iter()
            .find(|line| line.spans[0].content.starts_with("task"))
            .expect("task row");
        assert_eq!(task.spans[0].style.fg, Some(theme.palette.muted));
        assert!(!task.spans[0].style.add_modifier.contains(Modifier::DIM));
        assert_ne!(task.spans[0].style.fg, Some(theme.palette.faint));
        assert_eq!(task.spans[1].style.fg, Some(theme.palette.fg));

        let heading = lines
            .iter()
            .find(|line| line.spans.iter().any(|s| s.content == "ACTIVITY"))
            .expect("uppercase section inlay");
        assert_eq!(heading.spans[0].style.fg, Some(theme.palette.border));
        assert_eq!(heading.spans[1].style.fg, Some(theme.palette.accent));
        assert!(heading.spans[1].style.add_modifier.contains(Modifier::BOLD));
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


    /// **골격 회귀 핀** — Ctrl+G 는 Ctrl+O 와 같은 2단이다. 예전엔 폭 전체를 쓰는
    /// 2행짜리 목록 위에 짧은 카드가 붙어, 61행 터미널의 절반 이상이 빈 채로
    /// 남았다 — 카드가 무엇을 하든 **화면 자체에 구조가 없었다**.
    #[test]
    fn a_wide_agents_viewer_puts_the_fleet_beside_the_inspector() {
        let theme = Theme::pi();
        let modal = AgentsViewerModal::new(snapshot(vec![
            streaming_row("opus-paynow", 4),
            row("classifier", "completed"),
        ]));
        let area = Rect::new(0, 0, 200, 40);
        let regions = modal.layout(area, &theme).expect("layout fits");

        let rail = regions.list_rail.expect("a wide modal separates the columns");
        assert!(
            regions.list_area.x + regions.list_area.width <= rail.x
                && rail.x < regions.detail_area.x,
            "fleet │ inspector, left to right"
        );
        assert_eq!(
            regions.list_area.height, regions.detail_area.height,
            "both columns take the body's full height"
        );
        assert!(
            regions.list_area.height > 10,
            "the fleet column is no longer hugged to its two rows"
        );

        // Narrow keeps the stack it always had.
        let stacked = modal.layout(Rect::new(0, 0, 70, 30), &theme).expect("layout");
        assert!(stacked.list_rail.is_none(), "no rail when stacked");
        assert!(
            stacked.list_area.y < stacked.detail_area.y,
            "the fleet sits above the card"
        );
    }
}
