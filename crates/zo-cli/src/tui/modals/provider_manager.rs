//! `/providers` — one screen to register, edit, and delete model providers.
//!
//! The surface is a two-level tree: each registered provider is a parent row
//! whose children are the model ids it serves. Everything the user can do to a
//! provider is reachable from the row itself (F2 edit, Del delete, F5
//! rediscover) and everything they can do to a single model is reachable from
//! the model row (Del delete), so there is no second menu to learn and no mode
//! to remember. Every mutation answers to a key an IME cannot swallow — the
//! `e`/`d`/`r` letters remain as aliases.
//!
//! Deleting is the only destructive action, so it routes through an inline
//! confirmation that spells out what disappears and — when the credential is
//! not shared with another provider — offers to forget the stored API key in
//! the same keystroke.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

use super::super::glyphs;
use super::super::theme::Theme;
use super::{ModalResult, ModalSelection, key_hint_footer_fitted, modal_frame, selected_style};

/// Whether a provider can actually authenticate right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKeyState {
    /// No credential needed (a local server such as Ollama or LM Studio).
    Keyless,
    /// A key is saved in Zo's credential store.
    Stored,
    /// A key is present in this process's environment.
    FromEnv,
    /// The provider declares `auth_env` but no key can be found — its models
    /// are registered but every request would fail.
    Missing,
}

impl ProviderKeyState {
    fn glyph(self, color: bool) -> &'static str {
        match self {
            Self::Stored | Self::FromEnv => glyphs::pick(color, glyphs::CHECK, glyphs::CHECK_NC),
            Self::Keyless => glyphs::pick(color, "\u{25cb}", "o"),
            Self::Missing => glyphs::pick(color, glyphs::CROSS, glyphs::CROSS_NC),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Stored => "key saved",
            Self::FromEnv => "key from env",
            Self::Keyless => "keyless",
            Self::Missing => "key missing",
        }
    }

    const fn color(self, theme: &Theme) -> ratatui::style::Color {
        match self {
            Self::Stored | Self::FromEnv => theme.palette.success,
            Self::Keyless => theme.palette.dim,
            Self::Missing => theme.palette.error,
        }
    }
}

/// Where a provider entry came from, which decides whether it is editable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOrigin {
    /// The user-global `settings.json` — the only rows Zo may rewrite.
    GlobalSettings,
    /// Injected through the `ZO_CUSTOM_PROVIDERS` environment override, which
    /// belongs to whoever launched the process. Shown for completeness, never
    /// edited or deleted from here.
    EnvOverride,
}

/// One registered provider, projected for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderManagerRow {
    pub name: String,
    pub base_url: String,
    pub auth_env: Option<String>,
    pub models: Vec<String>,
    pub key_state: ProviderKeyState,
    pub origin: ProviderOrigin,
    /// `true` when this provider's `auth_env` also authenticates another
    /// provider, so deleting the key here would break that one too.
    pub key_shared: bool,
}

impl ProviderManagerRow {
    const fn editable(&self) -> bool {
        matches!(self.origin, ProviderOrigin::GlobalSettings)
    }
}

/// A subscription / OAuth account, listed above the registered endpoints.
///
/// These are a different kind of connection — a browser login or a shell
/// credential rather than a settings entry — but from the user's side they
/// answer the same question ("what can I actually use right now?"), so the
/// manager shows both and `/connect`, `/login`, and `/logout` all land here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderAccountRow {
    /// Canonical id re-submitted as `/login <id>` when connecting.
    pub id: String,
    /// Display name, e.g. `Claude`.
    pub label: String,
    /// How this account is (or would be) authenticated, shown dim.
    pub detail: String,
    pub connected: bool,
    /// `false` when the credential lives somewhere zo cannot clear (an env var
    /// exported by the shell, `gcloud` ADC), so `d` explains instead of lying.
    pub disconnectable: bool,
}

/// A mutation the manager asks the host to perform. The host owns every write;
/// the modal only decides *what* the user asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderManagerAction {
    /// Open the picker of everything that can be connected or registered.
    Add,
    /// Start the OAuth / setup flow for an account row.
    ConnectAccount { id: String },
    /// Forget this account's saved credentials.
    DisconnectAccount { id: String },
    /// Open the wizard pre-filled with this provider, so saving edits it.
    Edit { name: String },
    /// Re-probe the endpoint's `/models` and union the result in.
    Rediscover { name: String },
    /// Drop the whole entry; `delete_key` also forgets its stored API key.
    DeleteProvider { name: String, delete_key: bool },
    /// Drop one model id, leaving the provider and its key registered.
    DeleteModel { name: String, model: String },
}

/// A tree row as the user sees it: either a provider header or one of its
/// models. Rebuilt from `providers` + `expanded` on every mutation so the
/// cursor arithmetic only ever deals with what is actually on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeRow {
    Account { account: usize },
    Provider { provider: usize },
    Model { provider: usize, model: usize },
    AddNew,
}

/// The pending destructive confirmation, if any.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Confirm {
    Provider {
        name: String,
        model_count: usize,
        auth_env: Option<String>,
        /// `None` when there is no key to offer, so no checkbox is drawn.
        delete_key: Option<bool>,
        key_shared: bool,
    },
    Model {
        name: String,
        model: String,
    },
    Account {
        id: String,
        label: String,
        detail: String,
    },
}

/// `/providers` manager modal.
#[derive(Debug, Clone)]
pub struct ProviderManagerModal {
    accounts: Vec<ProviderAccountRow>,
    providers: Vec<ProviderManagerRow>,
    expanded: Vec<bool>,
    rows: Vec<TreeRow>,
    cursor: usize,
    scroll: usize,
    /// The global settings file every mutation writes to, shown in the header
    /// so the user can see the registration is machine-wide, not per-project.
    settings_path: String,
    confirm: Option<Confirm>,
}

impl ProviderManagerModal {
    /// Build the manager over `accounts` + `providers`, expanding the first
    /// registered provider so the tree's shape is obvious without pressing
    /// anything.
    #[must_use]
    pub fn new(
        accounts: Vec<ProviderAccountRow>,
        providers: Vec<ProviderManagerRow>,
        settings_path: impl Into<String>,
    ) -> Self {
        let mut expanded = vec![false; providers.len()];
        if let Some(first) = expanded.first_mut() {
            *first = true;
        }
        let mut modal = Self {
            accounts,
            providers,
            expanded,
            rows: Vec::new(),
            cursor: 0,
            scroll: 0,
            settings_path: settings_path.into(),
            confirm: None,
        };
        modal.rebuild_rows();
        modal
    }

    /// Swap in a refreshed listing after the host applied a mutation, keeping
    /// the expansion state and landing the cursor near where it was.
    pub fn refresh(
        &mut self,
        accounts: Vec<ProviderAccountRow>,
        providers: Vec<ProviderManagerRow>,
    ) {
        self.accounts = accounts;
        let focused = self.selected_provider_name();
        let previously_expanded: Vec<(String, bool)> = self
            .providers
            .iter()
            .zip(self.expanded.iter())
            .map(|(provider, expanded)| (provider.name.clone(), *expanded))
            .collect();
        self.expanded = providers
            .iter()
            .map(|provider| {
                previously_expanded
                    .iter()
                    .find(|(name, _)| *name == provider.name)
                    .is_some_and(|(_, expanded)| *expanded)
            })
            .collect();
        self.providers = providers;
        self.confirm = None;
        self.rebuild_rows();
        if let Some(name) = focused {
            if let Some(index) = self.rows.iter().position(|row| {
                matches!(row, TreeRow::Provider { provider }
                    if self.providers[*provider].name == name)
            }) {
                self.cursor = index;
            }
        }
        self.clamp_cursor();
    }

    /// Provider list currently rendered, for tests.
    #[cfg(test)]
    #[must_use]
    pub fn providers(&self) -> &[ProviderManagerRow] {
        &self.providers
    }

    /// Zero-based index of the highlighted tree row, for tests.
    #[cfg(test)]
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    fn selected_provider_name(&self) -> Option<String> {
        match self.rows.get(self.cursor)? {
            TreeRow::Provider { provider } | TreeRow::Model { provider, .. } => {
                Some(self.providers[*provider].name.clone())
            }
            TreeRow::Account { .. } | TreeRow::AddNew => None,
        }
    }

    fn rebuild_rows(&mut self) {
        let mut rows = Vec::new();
        for account in 0..self.accounts.len() {
            rows.push(TreeRow::Account { account });
        }
        for (index, provider) in self.providers.iter().enumerate() {
            rows.push(TreeRow::Provider { provider: index });
            if self.expanded.get(index).copied().unwrap_or_default() {
                for model in 0..provider.models.len() {
                    rows.push(TreeRow::Model {
                        provider: index,
                        model,
                    });
                }
            }
        }
        rows.push(TreeRow::AddNew);
        self.rows = rows;
        self.clamp_cursor();
    }

    fn clamp_cursor(&mut self) {
        if self.cursor >= self.rows.len() {
            self.cursor = self.rows.len().saturating_sub(1);
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if self.confirm.is_some() {
            return self.handle_confirm_key(key);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(ModalResult::Cancelled),
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_cursor(-1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_cursor(1);
                None
            }
            KeyCode::PageUp => {
                self.move_cursor(-8);
                None
            }
            KeyCode::PageDown => {
                self.move_cursor(8);
                None
            }
            KeyCode::Home => {
                self.cursor = 0;
                None
            }
            KeyCode::End => {
                self.cursor = self.rows.len().saturating_sub(1);
                None
            }
            KeyCode::Right => {
                self.set_expanded(true);
                None
            }
            KeyCode::Left => {
                self.set_expanded(false);
                None
            }
            // Enter and Space both act on the row: a provider folds open, the
            // trailing row starts the add wizard. One key does the obvious
            // thing for whatever is highlighted.
            KeyCode::Enter | KeyCode::Char(' ') => self.activate(),
            KeyCode::Char('a') => Some(ModalResult::Selected(ModalSelection::ProviderManage(
                ProviderManagerAction::Add,
            ))),
            // F2/F5 carry edit and re-probe because Enter is already taken by
            // fold-open and a bare `e`/`r` never arrives while a Korean IME is
            // composing — the letters stay as aliases.
            KeyCode::F(2) | KeyCode::Char('e') => self.edit_current(),
            KeyCode::F(5) | KeyCode::Char('r') => self.rediscover_current(),
            KeyCode::Char('d') | KeyCode::Delete => {
                self.begin_delete();
                None
            }
            _ => None,
        }
    }

    fn handle_confirm_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        match key.code {
            KeyCode::Esc | KeyCode::Char('n') => {
                self.confirm = None;
                None
            }
            // Arrows toggle the "also wipe the stored key" checkbox alongside
            // Space, so the choice is reachable with a composing IME instead of
            // the user being stuck with whatever the default was.
            KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right | KeyCode::Tab => {
                if let Some(Confirm::Provider {
                    delete_key: Some(delete_key),
                    ..
                }) = &mut self.confirm
                {
                    *delete_key = !*delete_key;
                }
                None
            }
            KeyCode::Enter | KeyCode::Char('y') => {
                let confirm = self.confirm.take()?;
                Some(ModalResult::Selected(ModalSelection::ProviderManage(
                    match confirm {
                        Confirm::Provider {
                            name, delete_key, ..
                        } => ProviderManagerAction::DeleteProvider {
                            name,
                            delete_key: delete_key.unwrap_or_default(),
                        },
                        Confirm::Model { name, model } => {
                            ProviderManagerAction::DeleteModel { name, model }
                        }
                        Confirm::Account { id, .. } => {
                            ProviderManagerAction::DisconnectAccount { id }
                        }
                    },
                )))
            }
            _ => None,
        }
    }

    fn move_cursor(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let last = self.rows.len() - 1;
        let next = isize::try_from(self.cursor).unwrap_or(0).saturating_add(delta);
        self.cursor = usize::try_from(next).unwrap_or(0).min(last);
    }

    fn set_expanded(&mut self, expanded: bool) {
        let Some(row) = self.rows.get(self.cursor).copied() else {
            return;
        };
        match row {
            TreeRow::Provider { provider } => {
                if let Some(slot) = self.expanded.get_mut(provider) {
                    *slot = expanded;
                }
                self.rebuild_rows();
                // Keep the highlight on the provider the user just folded.
                if let Some(index) = self
                    .rows
                    .iter()
                    .position(|row| matches!(row, TreeRow::Provider { provider: p } if *p == provider))
                {
                    self.cursor = index;
                }
            }
            // Collapsing from a model row folds its parent and lands there, so
            // Left always feels like "go up a level".
            TreeRow::Model { provider, .. } if !expanded => {
                if let Some(slot) = self.expanded.get_mut(provider) {
                    *slot = false;
                }
                self.rebuild_rows();
                if let Some(index) = self
                    .rows
                    .iter()
                    .position(|row| matches!(row, TreeRow::Provider { provider: p } if *p == provider))
                {
                    self.cursor = index;
                }
            }
            TreeRow::Account { .. } | TreeRow::Model { .. } | TreeRow::AddNew => {}
        }
    }

    fn activate(&mut self) -> Option<ModalResult> {
        match self.rows.get(self.cursor).copied()? {
            TreeRow::AddNew => Some(ModalResult::Selected(ModalSelection::ProviderManage(
                ProviderManagerAction::Add,
            ))),
            // Enter on an account starts (or re-runs) its login — re-connecting
            // an already-connected account is how a stale token is refreshed.
            TreeRow::Account { account } => {
                let id = self.accounts.get(account)?.id.clone();
                Some(ModalResult::Selected(ModalSelection::ProviderManage(
                    ProviderManagerAction::ConnectAccount { id },
                )))
            }
            TreeRow::Provider { provider } => {
                let expanded = self.expanded.get(provider).copied().unwrap_or_default();
                self.set_expanded(!expanded);
                None
            }
            TreeRow::Model { .. } => None,
        }
    }

    fn current_provider(&self) -> Option<&ProviderManagerRow> {
        match self.rows.get(self.cursor)? {
            TreeRow::Provider { provider } | TreeRow::Model { provider, .. } => {
                self.providers.get(*provider)
            }
            TreeRow::Account { .. } | TreeRow::AddNew => None,
        }
    }

    fn edit_current(&mut self) -> Option<ModalResult> {
        let provider = self.current_provider()?;
        if !provider.editable() {
            return None;
        }
        Some(ModalResult::Selected(ModalSelection::ProviderManage(
            ProviderManagerAction::Edit {
                name: provider.name.clone(),
            },
        )))
    }

    fn rediscover_current(&mut self) -> Option<ModalResult> {
        let provider = self.current_provider()?;
        if !provider.editable() {
            return None;
        }
        Some(ModalResult::Selected(ModalSelection::ProviderManage(
            ProviderManagerAction::Rediscover {
                name: provider.name.clone(),
            },
        )))
    }

    fn begin_delete(&mut self) {
        let Some(row) = self.rows.get(self.cursor).copied() else {
            return;
        };
        match row {
            TreeRow::AddNew => {}
            TreeRow::Account { account } => {
                let Some(entry) = self.accounts.get(account) else {
                    return;
                };
                if !entry.connected {
                    return;
                }
                self.confirm = Some(Confirm::Account {
                    id: entry.id.clone(),
                    label: entry.label.clone(),
                    // A credential zo does not own cannot be cleared from here,
                    // so the confirmation says where it actually lives.
                    detail: if entry.disconnectable {
                        entry.detail.clone()
                    } else {
                        format!("{} — zo cannot clear this from here", entry.detail)
                    },
                });
            }
            TreeRow::Provider { provider } => {
                let Some(entry) = self.providers.get(provider) else {
                    return;
                };
                if !entry.editable() {
                    return;
                }
                // Offer the key checkbox only when there is a stored key to
                // forget and no sibling provider still depends on it — a shared
                // credential is not this entry's to delete, so the choice is
                // withheld rather than shown unchecked.
                let delete_key = entry
                    .auth_env
                    .as_ref()
                    .filter(|_| {
                        !entry.key_shared
                            && matches!(
                                entry.key_state,
                                ProviderKeyState::Stored | ProviderKeyState::Missing
                            )
                    })
                    .map(|_| true);
                self.confirm = Some(Confirm::Provider {
                    name: entry.name.clone(),
                    model_count: entry.models.len(),
                    auth_env: entry.auth_env.clone(),
                    delete_key,
                    key_shared: entry.key_shared,
                });
            }
            TreeRow::Model { provider, model } => {
                let Some(entry) = self.providers.get(provider) else {
                    return;
                };
                if !entry.editable() {
                    return;
                }
                let Some(model) = entry.models.get(model) else {
                    return;
                };
                self.confirm = Some(Confirm::Model {
                    name: entry.name.clone(),
                    model: model.clone(),
                });
            }
        }
    }

    fn ensure_visible(&mut self, viewport: usize) {
        if viewport == 0 {
            return;
        }
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if self.cursor >= self.scroll + viewport {
            self.scroll = self.cursor + 1 - viewport;
        }
    }

    /// Scroll one wheel notch without moving the highlight.
    pub fn scroll_by(&mut self, up: bool, rows: usize) {
        if up {
            self.scroll = self.scroll.saturating_sub(rows);
        } else {
            self.scroll = (self.scroll + rows).min(self.rows.len().saturating_sub(1));
        }
    }

    /// Build the render lines. Split from [`Self::draw`] so a `TestBackend`
    /// dump test can assert the exact text without a terminal.
    #[must_use]
    pub fn render_lines(&self, theme: &Theme, width: usize, height: usize) -> Vec<Line<'static>> {
        if let Some(confirm) = &self.confirm {
            return Self::confirm_lines(theme, confirm, width);
        }
        let color = !theme.no_color;
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Providers", theme.typography.heading_1),
                Span::styled(
                    format!("   {} registered", self.providers.len()),
                    theme.typography.dim,
                ),
            ]),
            Line::from(Span::styled(
                format!("global · {}", self.settings_path),
                theme.typography.dim,
            )),
            Line::from(""),
        ];

        if self.providers.is_empty() {
            lines.push(Line::from(Span::styled(
                "Nothing registered yet — press a to add a provider.",
                theme.typography.dim,
            )));
            lines.push(Line::from(""));
        }

        // Reserve the header/footer rows so the list scrolls inside the modal
        // instead of overflowing it.
        let chrome = lines.len() + 3;
        let viewport = height.saturating_sub(chrome).max(1);
        let mut scrolled = self.clone();
        scrolled.ensure_visible(viewport);
        let start = scrolled.scroll.min(self.rows.len().saturating_sub(1));

        // Each provider (and the trailing add row) opens with a blank spacer so
        // the groups read as blocks rather than one dense list. The spacers are
        // rendering-only — they are not selectable — so the budget is counted in
        // emitted lines, not in tree rows, or a long registry would overflow the
        // modal by exactly the number of separators.
        let mut emitted = 0usize;
        let mut index = start;
        let mut section: Option<&'static str> = None;
        while index < self.rows.len() && emitted < viewport {
            let row = self.rows[index];
            // Section captions and blank separators are rendering-only; the
            // budget is counted in emitted lines rather than tree rows so a long
            // registry cannot overflow the modal by exactly the chrome it adds.
            let heading = match row {
                TreeRow::Account { .. } => Some("Accounts"),
                TreeRow::Provider { .. } => Some("Registered providers"),
                TreeRow::Model { .. } | TreeRow::AddNew => None,
            };
            let mut wrote_heading = false;
            if let Some(heading) = heading {
                if section != Some(heading) {
                    let needs = if section.is_some() { 2 } else { 1 };
                    if emitted + needs >= viewport {
                        break;
                    }
                    if section.is_some() {
                        lines.push(Line::from(""));
                        emitted += 1;
                    }
                    lines.push(Line::from(Span::styled(
                        heading.to_string(),
                        theme.typography.heading_3,
                    )));
                    emitted += 1;
                    section = Some(heading);
                    wrote_heading = true;
                }
            }
            // A blank line opens each provider block (but not the first one
            // under its caption) and the trailing add row, so the tree reads as
            // groups rather than one dense list.
            let opens_group = !wrote_heading
                && index > start
                && matches!(row, TreeRow::Provider { .. } | TreeRow::AddNew);
            if opens_group {
                if emitted + 1 >= viewport {
                    break;
                }
                lines.push(Line::from(""));
                emitted += 1;
            }
            lines.push(self.tree_line(theme, color, row, index == self.cursor, width));
            emitted += 1;
            index += 1;
        }

        lines.push(Line::from(""));
        lines.push(self.footer(theme, width));
        lines
    }

    fn tree_line(
        &self,
        theme: &Theme,
        color: bool,
        row: TreeRow,
        focused: bool,
        width: usize,
    ) -> Line<'static> {
        let marker = if focused {
            super::cursor_marker(color)
        } else {
            super::blank_marker()
        };
        let marker_style = if focused {
            selected_style(theme)
        } else {
            theme.typography.dim
        };
        match row {
            TreeRow::Account { account } => {
                self.account_line(theme, color, account, focused, width, marker, marker_style)
            }
            TreeRow::Provider { provider } => {
                self.provider_line(theme, color, provider, focused, width, marker, marker_style)
            }
            TreeRow::Model { provider, model } => {
                let name = self.providers[provider].models[model].clone();
                let style = if focused {
                    selected_style(theme)
                } else {
                    theme.typography.body
                };
                Line::from(vec![
                    Span::styled(marker.to_string(), marker_style),
                    Span::styled("    ".to_string(), theme.typography.dim),
                    Span::styled(name, style),
                ])
            }
            TreeRow::AddNew => {
                let style = if focused {
                    selected_style(theme)
                } else {
                    theme.typography.dim
                };
                Line::from(vec![
                    Span::styled(marker.to_string(), marker_style),
                    Span::styled("+ ".to_string(), style),
                    Span::styled("Add a provider…".to_string(), style),
                ])
            }
        }
    }

    /// One OAuth / subscription account row.
    #[allow(clippy::too_many_arguments)] // shared row chrome, computed once by the caller
    fn account_line(
        &self,
        theme: &Theme,
        color: bool,
        account: usize,
        focused: bool,
        width: usize,
        marker: &str,
        marker_style: Style,
    ) -> Line<'static> {
        let entry = &self.accounts[account];
                let (glyph, key_color) = if entry.connected {
                    (
                        glyphs::pick(color, glyphs::CHECK, glyphs::CHECK_NC),
                        theme.palette.success,
                    )
                } else {
                    (glyphs::pick(color, "\u{25cb}", "o"), theme.palette.dim)
                };
                let name_style = if focused {
                    selected_style(theme)
                } else {
                    theme.typography.body.add_modifier(Modifier::BOLD)
                };
                let detail = if entry.connected {
                    format!("connected · {}", entry.detail)
                } else {
                    format!("not connected · {}", entry.detail)
                };
                let mut spans = vec![
                    Span::styled(marker.to_string(), marker_style),
                    Span::styled("  ".to_string(), theme.typography.dim),
                    Span::styled(glyph.to_string(), Style::new().fg(key_color)),
                    Span::styled(" ".to_string(), theme.typography.body),
                    Span::styled(entry.label.clone(), name_style),
                ];
                spans.push(Span::styled(
                    pad_to_column(&spans, width, &detail),
                    theme.typography.dim,
                ));
                Line::from(spans)
    }

    /// One registered OpenAI-compatible provider row.
    #[allow(clippy::too_many_arguments)] // shared row chrome, computed once by the caller
    fn provider_line(
        &self,
        theme: &Theme,
        color: bool,
        provider: usize,
        focused: bool,
        width: usize,
        marker: &str,
        marker_style: Style,
    ) -> Line<'static> {
        let entry = &self.providers[provider];
                let expanded = self.expanded.get(provider).copied().unwrap_or_default();
                let chevron = if expanded {
                    glyphs::pick(color, glyphs::CHEVRON_DOWN, glyphs::CHEVRON_DOWN_NC)
                } else {
                    glyphs::pick(color, glyphs::CHEVRON_RIGHT, glyphs::CHEVRON_RIGHT_NC)
                };
                let name_style = if focused {
                    selected_style(theme)
                } else {
                    theme.typography.body.add_modifier(Modifier::BOLD)
                };
                let detail = format!(
                    "{} · {} model{} · {}",
                    entry.key_state.label(),
                    entry.models.len(),
                    if entry.models.len() == 1 { "" } else { "s" },
                    short_endpoint(&entry.base_url),
                );
                let mut spans = vec![
                    Span::styled(marker.to_string(), marker_style),
                    Span::styled(format!("{chevron} "), theme.typography.dim),
                    Span::styled(
                        entry.key_state.glyph(color).to_string(),
                        Style::new().fg(entry.key_state.color(theme)),
                    ),
                    Span::styled(" ", theme.typography.body),
                    Span::styled(entry.name.clone(), name_style),
                ];
                if !entry.editable() {
                    spans.push(Span::styled(
                        "  (env, read-only)".to_string(),
                        theme.typography.dim,
                    ));
                }
                spans.push(Span::styled(
                    pad_to_column(&spans, width, &detail),
                    theme.typography.dim,
                ));
                Line::from(spans)
    }

    /// Footer hints track the highlighted row, so the keys shown are always the
    /// keys that will do something.
    /// `width` is the modal's content width; the hint row is fitted to it
    /// because `draw` renders without `Wrap` and would otherwise have it cut
    /// mid-word at the rect edge.
    fn footer(&self, theme: &Theme, width: usize) -> Line<'static> {
        let hints: &[(&str, &str)] = match self.rows.get(self.cursor) {
            Some(TreeRow::Account { account })
                if self.accounts.get(*account).is_some_and(|row| row.connected) =>
            {
                &[
                    ("↑↓", "move"),
                    ("Enter", "reconnect"),
                    ("Del", "disconnect"),
                    ("Esc", "close"),
                ]
            }
            Some(TreeRow::Account { .. }) => &[
                ("↑↓", "move"),
                ("Enter", "connect"),
                ("Esc", "close"),
            ],
            Some(TreeRow::Provider { provider })
                if self
                    .providers
                    .get(*provider)
                    .is_some_and(ProviderManagerRow::editable) =>
            {
                &[
                    ("↑↓", "move"),
                    ("Enter", "fold"),
                    ("F2", "edit"),
                    ("F5", "rediscover"),
                    ("Del", "delete"),
                    ("Esc", "close"),
                ]
            }
            Some(TreeRow::Model { .. }) => &[
                ("↑↓", "move"),
                ("←", "collapse"),
                ("Del", "remove model"),
                ("Esc", "close"),
            ],
            _ => &[
                ("↑↓", "move"),
                ("Enter", "add"),
                ("Esc", "close"),
            ],
        };
        key_hint_footer_fitted(theme, hints, u16::try_from(width).unwrap_or(u16::MAX))
    }

    fn confirm_lines(theme: &Theme, confirm: &Confirm, width: usize) -> Vec<Line<'static>> {
        match confirm {
            Confirm::Provider { .. } => Self::provider_confirm_lines(theme, confirm, width),
            Confirm::Account { .. } | Confirm::Model { .. } => {
                Self::simple_confirm_lines(theme, confirm, width)
            }
        }
    }

    /// The provider confirmation is the only one with a toggle, so it owns the
    /// checkbox and the shared-credential explanation.
    fn provider_confirm_lines(
        theme: &Theme,
        confirm: &Confirm,
        width: usize,
    ) -> Vec<Line<'static>> {
        let color = !theme.no_color;
        let mut lines = Vec::new();
        match confirm {
            Confirm::Provider {
                name,
                model_count,
                auth_env,
                delete_key,
                key_shared,
            } => {
                lines.push(Line::from(Span::styled(
                    format!("Delete provider “{name}”?"),
                    theme.typography.heading_1,
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!(
                        "· {model_count} model{} disappear from /model",
                        if *model_count == 1 { "" } else { "s" }
                    ),
                    theme.typography.dim,
                )));
                match (auth_env, delete_key) {
                    (Some(env), Some(checked)) => {
                        let box_glyph = if *checked {
                            glyphs::pick(color, "[x]", "[x]")
                        } else {
                            glyphs::pick(color, "[ ]", "[ ]")
                        };
                        lines.push(Line::from(""));
                        lines.push(Line::from(vec![
                            Span::styled(
                                format!("{box_glyph} "),
                                if *checked {
                                    selected_style(theme)
                                } else {
                                    theme.typography.dim
                                },
                            ),
                            Span::styled(
                                "Also delete the stored API key".to_string(),
                                theme.typography.body,
                            ),
                        ]));
                        lines.push(Line::from(Span::styled(
                            format!("    {env}"),
                            theme.typography.dim,
                        )));
                    }
                    (Some(env), None) if *key_shared => {
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled(
                            format!("· {env} stays — another provider still uses it"),
                            theme.typography.dim,
                        )));
                    }
                    _ => {}
                }
                lines.push(Line::from(""));
                lines.push(key_hint_footer_fitted(
                    theme,
                    if delete_key.is_some() {
                        &[
                            ("Enter", "delete"),
                            ("Space", "toggle key"),
                            ("Esc", "cancel"),
                        ]
                    } else {
                        &[("Enter", "delete"), ("Esc", "cancel")]
                    },
                    u16::try_from(width).unwrap_or(u16::MAX),
                ));
            }
            Confirm::Account { .. } | Confirm::Model { .. } => {}
        }
        lines
    }

    /// Account and single-model confirmations: a heading, what changes, and the
    /// two keys that resolve it.
    fn simple_confirm_lines(
        theme: &Theme,
        confirm: &Confirm,
        width: usize,
    ) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        match confirm {
            Confirm::Provider { .. } => {}
            Confirm::Account { label, detail, .. } => {
                lines.push(Line::from(Span::styled(
                    format!("Disconnect {label}?"),
                    theme.typography.heading_1,
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!("· {detail}"),
                    theme.typography.dim,
                )));
                lines.push(Line::from(Span::styled(
                    "· reconnect any time with Enter on this row".to_string(),
                    theme.typography.dim,
                )));
                lines.push(Line::from(""));
                lines.push(key_hint_footer_fitted(
                    theme,
                    &[("Enter", "disconnect"), ("Esc", "cancel")],
                    u16::try_from(width).unwrap_or(u16::MAX),
                ));
            }
            Confirm::Model { name, model } => {
                lines.push(Line::from(Span::styled(
                    format!("Remove model “{model}”?"),
                    theme.typography.heading_1,
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!("· {name} and its API key stay registered"),
                    theme.typography.dim,
                )));
                lines.push(Line::from(Span::styled(
                    "· re-add it any time with r (rediscover)".to_string(),
                    theme.typography.dim,
                )));
                lines.push(Line::from(""));
                lines.push(key_hint_footer_fitted(
                    theme,
                    &[("Enter", "remove"), ("Esc", "cancel")],
                    u16::try_from(width).unwrap_or(u16::MAX),
                ));
            }
        }
        lines
    }

    /// Content-sized overlay: tall enough for the tree, clamped to the screen.
    #[must_use]
    pub fn size(&self, area: Rect) -> (u16, u16) {
        let width = area.width.clamp(48, 86).min(area.width.saturating_sub(4).max(40));
        let content = if self.confirm.is_some() {
            10
        } else {
            // rows + one spacer per provider group + the add row's spacer.
            // one spacer per provider block + the add row's, plus two captions.
            let spacers = u16::try_from(self.providers.len())
                .unwrap_or(u16::MAX)
                .saturating_add(3);
            u16::try_from(self.rows.len())
                .unwrap_or(u16::MAX)
                .saturating_add(spacers)
                .saturating_add(7)
        };
        let height = content
            .clamp(9, 26)
            .min(area.height.saturating_sub(2).max(9));
        (width, height)
    }

    /// Draw the modal.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = modal_frame(frame, area, " Providers ", theme);
        let lines = self.render_lines(
            theme,
            usize::from(inner.width),
            usize::from(inner.height),
        );
        // Fitted: this body renders without `wrap`, so an over-wide row would be
        // cut mid-glyph by `LineTruncator` instead of losing its tail visibly.
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(lines, inner.width))
                .style(theme.typography.body),
            inner,
        );
    }
}

/// Trim an endpoint to `host[:port]` so the detail column stays readable.
fn short_endpoint(base_url: &str) -> String {
    let without_scheme = base_url
        .split_once("://")
        .map_or(base_url, |(_, rest)| rest);
    without_scheme
        .split('/')
        .next()
        .unwrap_or(without_scheme)
        .to_string()
}

/// Right-align `detail` within `width`, collapsing to two spaces when the row
/// is already wide — never truncating the name to make room.
fn pad_to_column(spans: &[Span<'static>], width: usize, detail: &str) -> String {
    let used: usize = spans.iter().map(|span| span.content.width()).sum();
    let detail_width = detail.width();
    let gap = width
        .saturating_sub(used)
        .saturating_sub(detail_width)
        .max(2);
    format!("{}{detail}", " ".repeat(gap))
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderAccountRow, ProviderKeyState, ProviderManagerAction, ProviderManagerModal,
        ProviderManagerRow, ProviderOrigin,
    };
    use crate::tui::modals::{ModalResult, ModalSelection};
    use crate::tui::theme::Theme;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn row(name: &str, models: &[&str], key_state: ProviderKeyState) -> ProviderManagerRow {
        ProviderManagerRow {
            name: name.to_string(),
            base_url: format!("https://{name}.example/v1"),
            auth_env: matches!(
                key_state,
                ProviderKeyState::Stored | ProviderKeyState::Missing
            )
            .then(|| format!("ZO_{}_API_KEY", name.to_uppercase())),
            models: models.iter().map(|model| (*model).to_string()).collect(),
            key_state,
            origin: ProviderOrigin::GlobalSettings,
            key_shared: false,
        }
    }

    fn account(id: &str, label: &str, connected: bool) -> ProviderAccountRow {
        ProviderAccountRow {
            id: id.to_string(),
            label: label.to_string(),
            detail: "saved OAuth".to_string(),
            connected,
            disconnectable: connected,
        }
    }

    fn accounts() -> Vec<ProviderAccountRow> {
        vec![
            account("claude", "Claude", true),
            account("openai", "ChatGPT", false),
        ]
    }

    fn modal() -> ProviderManagerModal {
        ProviderManagerModal::new(
            accounts(),
            vec![
                row("deepseek", &["deepseek-chat", "deepseek-reasoner"], ProviderKeyState::Stored),
                row("ollama", &["llama3.1"], ProviderKeyState::Keyless),
            ],
            "/home/u/.zo/settings.json",
        )
    }

    /// Cursor index of the first registered-provider row, so the tests stay
    /// readable as the accounts section grows.
    const FIRST_PROVIDER: usize = 2;

    fn on_first_provider() -> ProviderManagerModal {
        let mut modal = modal();
        for _ in 0..FIRST_PROVIDER {
            modal.handle_key(press(KeyCode::Down));
        }
        modal
    }

    fn text_of(modal: &ProviderManagerModal) -> String {
        modal
            .render_lines(&Theme::default_dark(), 70, 30)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn action(result: Option<ModalResult>) -> ProviderManagerAction {
        match result {
            Some(ModalResult::Selected(ModalSelection::ProviderManage(action))) => action,
            other => panic!("expected a manager action, got {other:?}"),
        }
    }

    #[test]
    fn the_first_provider_starts_expanded_so_the_tree_shape_is_visible() {
        let modal = modal();
        let lines = modal.render_lines(&Theme::default_dark(), 70, 24);
        let text: Vec<String> = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(
            text.iter().any(|line| line.contains("deepseek-chat")),
            "expanded provider shows its models: {text:?}"
        );
        assert!(
            text.iter().any(|line| line.contains("Add a provider")),
            "registration is reachable from the same screen: {text:?}"
        );
        assert!(
            text.iter().any(|line| line.contains("global · /home/u/.zo/settings.json")),
            "the global scope is stated on screen: {text:?}"
        );
    }

    #[test]
    fn collapsing_hides_model_rows_and_keeps_the_highlight_on_the_provider() {
        let mut modal = on_first_provider();
        assert!(modal.handle_key(press(KeyCode::Left)).is_none());
        assert_eq!(modal.cursor(), FIRST_PROVIDER);
        let lines = modal.render_lines(&Theme::default_dark(), 70, 24);
        let text = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("deepseek-chat"), "collapsed: {text}");
        assert!(text.contains("ollama"), "siblings stay visible: {text}");
    }

    #[test]
    fn deleting_a_provider_asks_first_and_defaults_to_forgetting_its_key() {
        let mut modal = on_first_provider();
        assert!(
            modal.handle_key(press(KeyCode::Char('d'))).is_none(),
            "delete must not fire without a confirmation"
        );
        let text = modal
            .render_lines(&Theme::default_dark(), 70, 24)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Delete provider"), "{text}");
        assert!(text.contains("2 models disappear"), "{text}");
        assert!(text.contains("[x] Also delete the stored API key"), "{text}");

        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::DeleteProvider {
                name: "deepseek".to_string(),
                delete_key: true,
            }
        );
    }

    #[test]
    fn the_key_checkbox_toggles_off_so_a_key_can_be_kept_for_re_registration() {
        let mut modal = on_first_provider();
        modal.handle_key(press(KeyCode::Char('d')));
        modal.handle_key(press(KeyCode::Char(' ')));
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::DeleteProvider {
                name: "deepseek".to_string(),
                delete_key: false,
            }
        );
    }

    #[test]
    fn a_shared_key_is_never_offered_for_deletion() {
        let mut shared = row("gateway-a", &["m"], ProviderKeyState::Stored);
        shared.key_shared = true;
        let mut modal =
            ProviderManagerModal::new(Vec::new(), vec![shared], "/home/u/.zo/settings.json");
        modal.handle_key(press(KeyCode::Char('d')));
        let text = modal
            .render_lines(&Theme::default_dark(), 70, 24)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains("Also delete the stored API key"),
            "a key another provider still needs must not be offered: {text}"
        );
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::DeleteProvider {
                name: "gateway-a".to_string(),
                delete_key: false,
            }
        );
    }

    #[test]
    fn a_model_row_deletes_only_that_model() {
        let mut modal = on_first_provider();
        modal.handle_key(press(KeyCode::Down));
        modal.handle_key(press(KeyCode::Char('d')));
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::DeleteModel {
                name: "deepseek".to_string(),
                model: "deepseek-chat".to_string(),
            }
        );
    }

    #[test]
    fn escape_cancels_a_pending_delete_instead_of_closing_the_manager() {
        let mut modal = on_first_provider();
        modal.handle_key(press(KeyCode::Char('d')));
        assert!(modal.handle_key(press(KeyCode::Esc)).is_none());
        assert!(
            matches!(modal.handle_key(press(KeyCode::Esc)), Some(ModalResult::Cancelled)),
            "a second Esc closes the manager"
        );
    }

    #[test]
    fn env_provided_providers_are_read_only() {
        let mut env_row = row("env-only", &["env-model"], ProviderKeyState::Keyless);
        env_row.origin = ProviderOrigin::EnvOverride;
        let mut modal =
            ProviderManagerModal::new(Vec::new(), vec![env_row], "/home/u/.zo/settings.json");
        assert!(modal.handle_key(press(KeyCode::Char('d'))).is_none());
        assert!(modal.handle_key(press(KeyCode::Char('e'))).is_none());
        let text = modal
            .render_lines(&Theme::default_dark(), 70, 24)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("(env, read-only)"), "{text}");
        assert!(!text.contains("Delete provider"), "no confirmation opened: {text}");
    }

    #[test]
    fn edit_and_rediscover_name_the_highlighted_provider() {
        let mut modal = on_first_provider();
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Char('e')))),
            ProviderManagerAction::Edit {
                name: "deepseek".to_string()
            }
        );
        // A model row still edits/rediscovers its parent provider.
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Char('r')))),
            ProviderManagerAction::Rediscover {
                name: "deepseek".to_string()
            }
        );
    }

    #[test]
    fn the_trailing_row_starts_registration() {
        let mut modal = modal();
        modal.handle_key(press(KeyCode::End));
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::Add
        );
    }

    #[test]
    fn refresh_keeps_expansion_and_focus_after_a_mutation() {
        let mut modal = on_first_provider();
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.providers().len(), 2);

        modal.refresh(accounts(), vec![
            row("deepseek", &["deepseek-reasoner"], ProviderKeyState::Stored),
            row("ollama", &["llama3.1"], ProviderKeyState::Keyless),
        ]);
        let text = modal
            .render_lines(&Theme::default_dark(), 70, 24)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("deepseek-reasoner"),
            "the refreshed list stays expanded: {text}"
        );
        assert!(!text.contains("deepseek-chat"), "removed model is gone: {text}");
    }

    /// Accounts and registered endpoints answer the same question, so both are
    /// on one screen under captions that say which is which.
    #[test]
    fn accounts_and_registered_providers_share_one_screen() {
        let text = text_of(&modal());
        assert!(text.contains("Accounts"), "{text}");
        assert!(text.contains("Claude"), "{text}");
        assert!(text.contains("connected · saved OAuth"), "{text}");
        assert!(text.contains("not connected"), "ChatGPT row: {text}");
        assert!(text.contains("Registered providers"), "{text}");
        assert!(text.contains("deepseek"), "{text}");
    }

    /// Enter on an account starts its login — including on a connected one,
    /// which is how an expired token gets refreshed.
    #[test]
    fn enter_on_an_account_connects_it() {
        let mut claude_row = modal();
        assert_eq!(
            action(claude_row.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::ConnectAccount {
                id: "claude".to_string()
            }
        );
        let mut modal = modal();
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::ConnectAccount {
                id: "openai".to_string()
            }
        );
    }

    /// Disconnect is per-account and confirmed — the old `/logout` cleared
    /// every saved credential at once with no prompt.
    #[test]
    fn disconnecting_an_account_asks_first_and_names_only_that_account() {
        let mut modal = modal();
        assert!(modal.handle_key(press(KeyCode::Char('d'))).is_none());
        let text = text_of(&modal);
        assert!(text.contains("Disconnect Claude?"), "{text}");
        assert_eq!(
            action(modal.handle_key(press(KeyCode::Enter))),
            ProviderManagerAction::DisconnectAccount {
                id: "claude".to_string()
            }
        );
    }

    /// Nothing is connected, so there is nothing to disconnect: `d` must not
    /// open a confirmation that would resolve to a no-op.
    #[test]
    fn a_disconnected_account_offers_no_disconnect() {
        let mut modal = modal();
        modal.handle_key(press(KeyCode::Down));
        assert!(modal.handle_key(press(KeyCode::Char('d'))).is_none());
        assert!(
            !text_of(&modal).contains("Disconnect"),
            "no confirmation should open"
        );
    }

    /// An account is not a settings entry, so the provider-only actions must
    /// not fire on it and silently target the wrong row.
    #[test]
    fn account_rows_ignore_edit_and_rediscover() {
        let mut modal = modal();
        assert!(modal.handle_key(press(KeyCode::Char('e'))).is_none());
        assert!(modal.handle_key(press(KeyCode::Char('r'))).is_none());
    }

    #[test]
    fn an_empty_registry_explains_how_to_start() {
        let modal = ProviderManagerModal::new(Vec::new(), Vec::new(), "/home/u/.zo/settings.json");
        let text = modal
            .render_lines(&Theme::default_dark(), 70, 24)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Nothing registered yet"), "{text}");
        assert!(text.contains("press a to add"), "{text}");
    }
}
