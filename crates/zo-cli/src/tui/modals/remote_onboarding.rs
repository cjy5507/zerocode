//! `/remote` onboarding modal.
//!
//! This modal owns only display and selection state. Its action rows return
//! explicit `/remote …` command strings so the session loop remains the sole
//! owner of Remote lifecycle and pairing behavior.

use std::cell::Cell;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, draw_scrollbar, key_hint_footer,
    selected_style,
};

const PAGE_STRIDE: usize = 8;

/// A local-only pairing request safe to render in the Remote onboarding modal.
///
/// It deliberately carries no offer secret, cookie, or device credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemotePendingPair {
    /// Pairing device name supplied by the local tailnet client.
    pub device_name: String,
    /// Short comparison code the user must verify before approving.
    pub comparison_code: String,
}

/// Read-only Remote status rendered by [`RemoteOnboardingModal`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteOnboardingView {
    /// Whether the local Remote gateway is currently running.
    pub running: bool,
    /// Tailnet origin while running; never includes a pairing secret.
    pub url: Option<String>,
    /// Number of authenticated devices.
    pub device_count: usize,
    /// Number of waiting pairing requests.
    pub pending_count: usize,
    /// Current controller device name, if assigned.
    pub controller: Option<String>,
    /// Human-readable current turn phase (`idle` or `running`).
    pub turn_state: String,
    /// Waiting pairing requests in deterministic display order.
    pub pending_pairs: Vec<RemotePendingPair>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RemoteActionRow {
    label: String,
    command: String,
}

/// Keyboard-accessible Zo Remote onboarding and status modal.
#[derive(Debug, Clone)]
pub struct RemoteOnboardingModal {
    view: RemoteOnboardingView,
    actions: Vec<RemoteActionRow>,
    cursor: usize,
    scroll: Cell<usize>,
    viewport_rows: Cell<usize>,
}

impl RemoteOnboardingModal {
    /// Build the modal from a safe, local status snapshot.
    #[must_use]
    pub fn new(view: RemoteOnboardingView) -> Self {
        let actions = action_rows(&view);
        Self {
            view,
            actions,
            cursor: 0,
            scroll: Cell::new(0),
            viewport_rows: Cell::new(0),
        }
    }

    /// Current selected action index.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Number of available lifecycle or pairing actions.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Move to the next action, wrapping at the end.
    pub fn move_down(&mut self) {
        if self.actions.is_empty() {
            return;
        }
        self.cursor = (self.cursor + 1) % self.actions.len();
        self.ensure_selection_visible();
    }

    /// Move to the previous action, wrapping at the beginning.
    pub fn move_up(&mut self) {
        if self.actions.is_empty() {
            return;
        }
        self.cursor = if self.cursor == 0 {
            self.actions.len() - 1
        } else {
            self.cursor - 1
        };
        self.ensure_selection_visible();
    }

    fn page_down(&mut self) {
        if self.actions.is_empty() {
            return;
        }
        self.cursor = (self.cursor + PAGE_STRIDE).min(self.actions.len() - 1);
        self.ensure_selection_visible();
    }

    fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(PAGE_STRIDE);
        self.ensure_selection_visible();
    }

    const fn action_start_line() -> usize {
        // Heading, 3 onboarding steps, status heading and four status lines,
        // security note, spacer, and the action heading.
        12
    }

    fn ensure_selection_visible(&self) {
        let viewport = self.viewport_rows.get();
        if viewport == 0 {
            return;
        }
        let selected_line = Self::action_start_line() + self.cursor;
        let mut scroll = self.scroll.get();
        if selected_line < scroll {
            scroll = selected_line;
        } else if selected_line >= scroll.saturating_add(viewport) {
            scroll = selected_line.saturating_add(1).saturating_sub(viewport);
        }
        self.scroll.set(scroll);
    }

    /// Handle one keyboard event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_up();
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
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
                self.cursor = 0;
                self.ensure_selection_visible();
                None
            }
            KeyCode::End => {
                self.cursor = self.actions.len().saturating_sub(1);
                self.ensure_selection_visible();
                None
            }
            KeyCode::Enter => self.actions.get(self.cursor).map(|action| {
                ModalResult::Selected(ModalSelection::RemoteCommand(action.command.clone()))
            }),
            _ => None,
        }
    }

    /// Draw the modal into `area` using the current theme.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled("Zo Remote", theme.typography.body))
            .render(frame, area);
        let content = Rect::new(
            inner.x,
            inner.y,
            inner.width.saturating_sub(1),
            inner.height,
        );
        self.viewport_rows.set(usize::from(content.height));
        self.ensure_selection_visible();
        let lines = self.render_lines(theme);
        let scroll = self.scroll.get().min(lines.len().saturating_sub(1));
        let visible = lines
            .into_iter()
            .skip(scroll)
            .take(usize::from(content.height))
            .collect::<Vec<_>>();
        frame.render_widget(Paragraph::new(visible), content);
        draw_scrollbar(
            frame,
            inner,
            u16::try_from(scroll).unwrap_or(u16::MAX),
            self.content_rows(),
            theme,
        );
    }

    /// Number of rendered content rows including the footer.
    #[must_use]
    pub fn content_rows(&self) -> usize {
        Self::action_start_line() + self.actions.len() + 2
    }

    fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let status = if self.view.running { "RUNNING" } else { "STOPPED" };
        let status_style = if self.view.running {
            Style::new()
                .fg(theme.palette.cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            theme.typography.dim
        };
        let controller = self.view.controller.as_deref().unwrap_or("none");
        let url = self.view.url.as_deref().unwrap_or("not running");
        let mut lines = vec![
            Line::from(vec![
                Span::styled("REMOTE ACCESS  ", theme.typography.body.add_modifier(Modifier::BOLD)),
                Span::styled(status, status_style),
            ]),
            Line::from(Span::styled(
                "1  Secure tailnet link — Zo listens locally through Tailscale.",
                theme.typography.body,
            )),
            Line::from(Span::styled(
                "2  Pair your phone — scan the QR code and compare the code shown.",
                theme.typography.body,
            )),
            Line::from(Span::styled(
                "3  Approve, then control this session from the paired device.",
                theme.typography.body,
            )),
            Line::from(""),
            Line::from(Span::styled(
                "SESSION STATUS",
                theme.typography.body.add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(format!("URL          {url}"), theme.typography.body)),
            Line::from(Span::styled(
                format!(
                    "Devices      {} connected · {} pending",
                    self.view.device_count, self.view.pending_count
                ),
                theme.typography.body,
            )),
            Line::from(Span::styled(
                format!("Controller   {controller} · Turn: {}", self.view.turn_state),
                theme.typography.body,
            )),
            Line::from(Span::styled(
                "Security     Tailnet only. Keep Funnel off; every pairing needs local approval.",
                theme.typography.dim,
            )),
            Line::from(""),
            Line::from(Span::styled(
                "ACTIONS",
                theme.typography.body.add_modifier(Modifier::BOLD),
            )),
        ];
        lines.extend(self.actions.iter().enumerate().map(|(index, action)| {
            let selected = index == self.cursor;
            let marker = if selected {
                cursor_marker(!theme.no_color)
            } else {
                blank_marker()
            };
            let style = if selected {
                selected_style(theme)
            } else {
                theme.typography.body
            };
            Line::from(Span::styled(format!("{marker}{}", action.label), style))
        }));
        lines.push(Line::from(""));
        lines.push(key_hint_footer(
            theme,
            &[("↑↓/j k", "move"), ("Enter", "run"), ("Esc", "cancel")],
        ));
        lines
    }
}

fn action_rows(view: &RemoteOnboardingView) -> Vec<RemoteActionRow> {
    if !view.running {
        return vec![RemoteActionRow {
            label: "Start Zo Remote".to_string(),
            command: "/remote start".to_string(),
        }];
    }

    let mut actions = Vec::with_capacity(view.pending_pairs.len().saturating_mul(2) + 4);
    for pair in &view.pending_pairs {
        actions.push(RemoteActionRow {
            label: format!("Approve {} · {}", pair.device_name, pair.comparison_code),
            command: format!("/remote approve {}", pair.comparison_code),
        });
        actions.push(RemoteActionRow {
            label: format!("Deny {} · {}", pair.device_name, pair.comparison_code),
            command: format!("/remote deny {}", pair.comparison_code),
        });
    }
    actions.extend([
        RemoteActionRow {
            label: "Show QR code".to_string(),
            command: "/remote qr".to_string(),
        },
        RemoteActionRow {
            label: "Refresh status".to_string(),
            command: "/remote status".to_string(),
        },
        RemoteActionRow {
            label: "Rotate credentials".to_string(),
            command: "/remote rotate".to_string(),
        },
        RemoteActionRow {
            label: "Stop Zo Remote".to_string(),
            command: "/remote stop".to_string(),
        },
    ]);
    actions
}
