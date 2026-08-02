//! Guided custom OpenAI-compatible provider onboarding modal.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph};

use super::super::theme::Theme;
use super::{ModalResult, ModalSelection, key_hint_footer_reflowing, selected_style};

/// Draft emitted by the custom provider wizard.
#[derive(Clone, PartialEq, Eq)]
pub struct CustomProviderDraft {
    /// Human-readable provider name stored in `settings.json`.
    pub name: String,
    /// OpenAI-compatible base URL, normally ending in `/v1`.
    pub base_url: String,
    /// Credential env/credential-store key. `None` means keyless.
    pub auth_env: Option<String>,
    /// Secret API key pasted in this modal. Never rendered.
    pub api_key: Option<String>,
    /// Model ids to save. Empty means the backend should try `/models`.
    pub models: Vec<String>,
    /// Optional context-window override in tokens for every saved model.
    pub context_window: Option<u64>,
    /// Optional max-output-token override for every saved model.
    pub max_output_tokens: Option<u64>,
    /// Whether to send `stream_options.include_usage` on streaming requests.
    pub include_usage: bool,
    /// Whether this endpoint's models accept OpenAI's `reasoning_effort`.
    /// Off means `/effort` (and the Smart dynamic band) cannot reach them.
    pub supports_reasoning_effort: bool,
    /// `true` when this draft edits a provider that is already registered, so
    /// its fields are authoritative: a model the user deleted from the list
    /// really disappears instead of being unioned back in, and clearing the
    /// auth env really makes the provider keyless. A fresh registration leaves
    /// this `false` and merges, which is what makes repeated `/connect` runs
    /// against one provider name accumulate rather than reset.
    pub edit_existing: bool,
}

/// Field values used to pre-fill the wizard when editing a registered provider.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CustomProviderPrefill {
    pub name: String,
    pub base_url: String,
    pub auth_env: Option<String>,
    pub models: Vec<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub include_usage: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    /// Whether a credential for `auth_env` already resolves — from the process
    /// environment or from the credential store — so the form can keep it
    /// instead of demanding the secret be pasted again. Deliberately not
    /// "saved": an exported variable is never written to the store, and
    /// counting only the store made an env-authenticated provider uneditable
    /// without its key on hand.
    pub has_existing_key: bool,
}


impl std::fmt::Debug for CustomProviderDraft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomProviderDraft")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("auth_env", &self.auth_env)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("models", &self.models)
            .field("context_window", &self.context_window)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("include_usage", &self.include_usage)
            .field("supports_reasoning_effort", &self.supports_reasoning_effort)
            .field("edit_existing", &self.edit_existing)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthMode {
    NoKey,
    PasteKey,
    ExistingEnv,
}

impl AuthMode {
    fn label(self) -> &'static str {
        match self {
            Self::NoKey => "No key",
            Self::PasteKey => "Paste API key",
            Self::ExistingEnv => "Existing env/credential",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::NoKey => Self::PasteKey,
            Self::PasteKey => Self::ExistingEnv,
            Self::ExistingEnv => Self::NoKey,
        }
    }

    fn prev(self) -> Self {
        match self {
            Self::NoKey => Self::ExistingEnv,
            Self::PasteKey => Self::NoKey,
            Self::ExistingEnv => Self::PasteKey,
        }
    }

    fn needs_env(self) -> bool {
        matches!(self, Self::PasteKey | Self::ExistingEnv)
    }

    fn needs_key_input(self) -> bool {
        matches!(self, Self::PasteKey)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Name,
    BaseUrl,
    AuthMode,
    AuthEnv,
    ApiKey,
    Models,
    ContextWindow,
    MaxOutputTokens,
    IncludeUsage,
    ReasoningEffort,
}

const FIELDS: &[Field] = &[
    Field::Name,
    Field::BaseUrl,
    Field::AuthMode,
    Field::AuthEnv,
    Field::ApiKey,
    Field::Models,
    Field::ContextWindow,
    Field::MaxOutputTokens,
    Field::IncludeUsage,
    Field::ReasoningEffort,
];

/// Step-by-step form for adding a custom OpenAI-compatible provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomProviderWizardModal {
    name: String,
    base_url: String,
    auth_mode: AuthMode,
    auth_env: String,
    api_key: String,
    models_text: String,
    context_window_text: String,
    max_output_tokens_text: String,
    include_usage: bool,
    supports_reasoning_effort: bool,
    focus: usize,
    error: Option<String>,
    auth_env_touched: bool,
    /// Whether this form registers a new provider or edits a stored one.
    mode: WizardMode,
}

/// What the wizard is for. Editing changes three things at once — the title,
/// whether a blank API-key field means "keep the existing key", and whether the
/// emitted draft replaces or merges — so they travel together rather than as
/// separate booleans that can disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WizardMode {
    Create,
    Edit {
        /// `true` when a credential for the provider's `auth_env` already
        /// resolves, from the environment or from the credential store.
        existing_key: bool,
    },
}

impl WizardMode {
    const fn is_edit(self) -> bool {
        matches!(self, Self::Edit { .. })
    }

    const fn has_existing_key(self) -> bool {
        matches!(self, Self::Edit { existing_key: true })
    }
}

impl CustomProviderWizardModal {
    /// Create an empty custom provider wizard.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: String::new(),
            base_url: String::new(),
            auth_mode: AuthMode::PasteKey,
            // Derived from the (still empty) name and re-derived on every
            // keystroke, so each provider gets its own credential slot. A shared
            // literal here would put two unrelated endpoints on one key.
            auth_env: default_auth_env_for(""),
            api_key: String::new(),
            models_text: String::new(),
            context_window_text: String::new(),
            max_output_tokens_text: String::new(),
            include_usage: false,
            supports_reasoning_effort: false,
            focus: 0,
            error: None,
            auth_env_touched: false,
            mode: WizardMode::Create,
        }
    }

    /// Open the wizard on an already-registered provider. Its fields become
    /// authoritative on save, which is what lets an edit remove a model or drop
    /// the credential binding — a fresh registration only ever merges.
    #[must_use]
    pub fn editing(prefill: &CustomProviderPrefill) -> Self {
        let auth_mode = match (&prefill.auth_env, prefill.has_existing_key) {
            (None, _) => AuthMode::NoKey,
            (Some(_), true) => AuthMode::ExistingEnv,
            (Some(_), false) => AuthMode::PasteKey,
        };
        Self {
            name: prefill.name.clone(),
            base_url: prefill.base_url.clone(),
            auth_mode,
            auth_env: prefill
                .auth_env
                .clone()
                .unwrap_or_else(|| default_auth_env_for(&prefill.name)),
            api_key: String::new(),
            models_text: prefill.models.join("\n"),
            context_window_text: prefill
                .context_window
                .map(|value| value.to_string())
                .unwrap_or_default(),
            max_output_tokens_text: prefill
                .max_output_tokens
                .map(|value| value.to_string())
                .unwrap_or_default(),
            include_usage: prefill.include_usage.unwrap_or_default(),
            supports_reasoning_effort: prefill.supports_reasoning_effort.unwrap_or_default(),
            focus: 0,
            error: None,
            // The stored env name is the user's existing choice; never silently
            // re-derive it from the name while they are editing.
            auth_env_touched: prefill.auth_env.is_some(),
            mode: WizardMode::Edit {
                existing_key: prefill.has_existing_key,
            },
        }
    }

    /// Current provider name, for tests.
    #[cfg(test)]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Current auth env, for tests.
    #[cfg(test)]
    #[must_use]
    pub fn auth_env(&self) -> &str {
        &self.auth_env
    }

    /// Current models parsed from the text area, for tests.
    #[cfg(test)]
    #[must_use]
    pub fn models(&self) -> Vec<String> {
        parse_model_ids(&self.models_text)
    }

    /// Paste text into the currently focused editable field. API keys are trimmed
    /// like the preset key modal; model lists preserve internal newlines.
    pub fn paste_text(&mut self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        match self.current_field() {
            Field::Name | Field::BaseUrl => self.replace_or_insert(trimmed),
            Field::AuthEnv => {
                self.auth_env_touched = true;
                self.replace_or_insert(trimmed);
            }
            Field::ApiKey => self.api_key.push_str(trimmed),
            Field::Models => {
                if !self.models_text.is_empty() && !self.models_text.ends_with('\n') {
                    self.models_text.push('\n');
                }
                self.models_text.push_str(trimmed);
            }
            Field::ContextWindow => self.context_window_text.push_str(trimmed),
            Field::MaxOutputTokens => self.max_output_tokens_text.push_str(trimmed),
            Field::AuthMode | Field::IncludeUsage | Field::ReasoningEffort => {}
        }
        self.sync_default_auth_env();
    }

    /// Handle one key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        self.error = None;
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Enter => {
                if self.focus + 1 == FIELDS.len() {
                    self.submit()
                } else {
                    self.next_field();
                    None
                }
            }
            KeyCode::Tab => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.prev_field();
                } else {
                    self.next_field();
                }
                None
            }
            KeyCode::BackTab | KeyCode::Up => {
                self.prev_field();
                None
            }
            KeyCode::Down => {
                self.next_field();
                None
            }
            KeyCode::Left => {
                if matches!(self.current_field(), Field::AuthMode) {
                    self.auth_mode = self.auth_mode.prev();
                }
                None
            }
            KeyCode::Right => {
                if matches!(self.current_field(), Field::AuthMode) {
                    self.auth_mode = self.auth_mode.next();
                    self.sync_default_auth_env();
                }
                None
            }
            KeyCode::Char('s' | 'S') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.submit()
            }
            KeyCode::Char('u' | 'U') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.clear_current_field();
                None
            }
            KeyCode::Char(' ') if matches!(self.current_field(), Field::IncludeUsage) => {
                self.include_usage = !self.include_usage;
                None
            }
            KeyCode::Char(' ') if matches!(self.current_field(), Field::ReasoningEffort) => {
                self.supports_reasoning_effort = !self.supports_reasoning_effort;
                None
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_char(ch);
                None
            }
            KeyCode::Backspace => {
                self.backspace();
                None
            }
            _ => None,
        }
    }

    fn current_field(&self) -> Field {
        FIELDS[self.focus]
    }

    fn next_field(&mut self) {
        self.focus = (self.focus + 1).min(FIELDS.len().saturating_sub(1));
    }

    fn prev_field(&mut self) {
        self.focus = self.focus.saturating_sub(1);
    }

    fn replace_or_insert(&mut self, text: &str) {
        match self.current_field() {
            Field::Name => self.name.push_str(text),
            Field::BaseUrl => self.base_url.push_str(text),
            Field::AuthEnv => self.auth_env.push_str(text),
            Field::ApiKey => self.api_key.push_str(text),
            Field::Models => self.models_text.push_str(text),
            Field::ContextWindow => self.context_window_text.push_str(text),
            Field::MaxOutputTokens => self.max_output_tokens_text.push_str(text),
            Field::AuthMode | Field::IncludeUsage | Field::ReasoningEffort => {}
        }
    }

    fn clear_current_field(&mut self) {
        match self.current_field() {
            Field::Name => self.name.clear(),
            Field::BaseUrl => self.base_url.clear(),
            Field::AuthEnv => {
                self.auth_env.clear();
                self.auth_env_touched = true;
            }
            Field::ApiKey => self.api_key.clear(),
            Field::Models => self.models_text.clear(),
            Field::ContextWindow => self.context_window_text.clear(),
            Field::MaxOutputTokens => self.max_output_tokens_text.clear(),
            Field::AuthMode => self.auth_mode = AuthMode::NoKey,
            Field::IncludeUsage => self.include_usage = false,
            Field::ReasoningEffort => self.supports_reasoning_effort = false,
        }
    }

    fn insert_char(&mut self, ch: char) {
        match self.current_field() {
            Field::Name => {
                self.name.push(ch);
                self.sync_default_auth_env();
            }
            Field::BaseUrl => self.base_url.push(ch),
            Field::AuthMode => match ch {
                '1' => self.auth_mode = AuthMode::NoKey,
                '2' => self.auth_mode = AuthMode::PasteKey,
                '3' => self.auth_mode = AuthMode::ExistingEnv,
                ' ' => self.auth_mode = self.auth_mode.next(),
                _ => {}
            },
            Field::AuthEnv => {
                self.auth_env_touched = true;
                self.auth_env.push(ch);
            }
            Field::ApiKey => self.api_key.push(ch),
            Field::Models => self.models_text.push(ch),
            Field::ContextWindow => self.context_window_text.push(ch),
            Field::MaxOutputTokens => self.max_output_tokens_text.push(ch),
            Field::IncludeUsage => {
                if matches!(ch, ' ' | 'y' | 'Y') {
                    self.include_usage = !self.include_usage;
                } else if matches!(ch, 'n' | 'N') {
                    self.include_usage = false;
                }
            }
            Field::ReasoningEffort => {
                if matches!(ch, ' ' | 'y' | 'Y') {
                    self.supports_reasoning_effort = !self.supports_reasoning_effort;
                } else if matches!(ch, 'n' | 'N') {
                    self.supports_reasoning_effort = false;
                }
            }
        }
    }

    fn backspace(&mut self) {
        match self.current_field() {
            Field::Name => {
                self.name.pop();
                self.sync_default_auth_env();
            }
            Field::BaseUrl => {
                self.base_url.pop();
            }
            Field::AuthEnv => {
                self.auth_env_touched = true;
                self.auth_env.pop();
            }
            Field::ApiKey => {
                self.api_key.pop();
            }
            Field::Models => {
                self.models_text.pop();
            }
            Field::ContextWindow => {
                self.context_window_text.pop();
            }
            Field::MaxOutputTokens => {
                self.max_output_tokens_text.pop();
            }
            Field::AuthMode | Field::IncludeUsage | Field::ReasoningEffort => {}
        }
    }

    fn sync_default_auth_env(&mut self) {
        if self.auth_env_touched || !self.auth_mode.needs_env() {
            return;
        }
        self.auth_env = default_auth_env_for(&self.name);
    }

    fn submit(&mut self) -> Option<ModalResult> {
        match self.build_draft() {
            Ok(draft) => Some(ModalResult::Selected(ModalSelection::CustomProvider(draft))),
            Err(error) => {
                self.error = Some(error);
                None
            }
        }
    }

    fn build_draft(&self) -> Result<CustomProviderDraft, String> {
        let name = self.name.trim();
        if name.is_empty() {
            return Err("Provider name is required".to_string());
        }
        let base_url = self.base_url.trim();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err("Base URL must start with http:// or https://".to_string());
        }
        let auth_env = if self.auth_mode.needs_env() {
            let env = self.auth_env.trim();
            if env.is_empty() {
                return Err("Auth env is required for keyed providers".to_string());
            }
            if !valid_auth_env_name(env) {
                return Err("Auth env must match [A-Za-z_][A-Za-z0-9_]*".to_string());
            }
            Some(env.to_string())
        } else {
            None
        };
        let api_key = if self.auth_mode.needs_key_input() {
            let key = self.api_key.trim();
            if key.is_empty() {
                // Editing a provider whose key already resolves must not force
                // the secret to be pasted again just to change, say, a model id.
                if self.mode.has_existing_key() {
                    None
                } else {
                    return Err("Paste an API key or choose a different auth mode".to_string());
                }
            } else {
                Some(key.to_string())
            }
        } else {
            None
        };
        let context_window =
            parse_optional_token_count(&self.context_window_text, "Context window")?;
        let max_output_tokens =
            parse_optional_token_count(&self.max_output_tokens_text, "Max output tokens")?;
        Ok(CustomProviderDraft {
            name: name.to_string(),
            base_url: base_url.to_string(),
            auth_env,
            api_key,
            models: parse_model_ids(&self.models_text),
            context_window,
            max_output_tokens,
            include_usage: self.include_usage,
            supports_reasoning_effort: self.supports_reasoning_effort,
            edit_existing: self.mode.is_edit(),
        })
    }

    fn masked_api_key(&self) -> String {
        if self.api_key.is_empty() {
            if self.mode.has_existing_key() {
                return "<existing key kept — type to replace>".to_string();
            }
            "<paste API key>".to_string()
        } else {
            "•".repeat(self.api_key.chars().count().min(64))
        }
    }

    fn field_line<'a>(
        &'a self,
        theme: &Theme,
        field: Field,
        label: &'static str,
        value: String,
    ) -> Line<'a> {
        let focused = self.current_field() == field;
        // Pi SettingsList row: `→ label   value` — cursor + label accent while
        // focused, the value column muted otherwise.
        let marker = if focused {
            super::cursor_marker(!theme.no_color)
        } else {
            super::blank_marker()
        };
        let label_style = if focused {
            selected_style(theme)
        } else {
            theme.typography.dim
        };
        let value_style = if focused {
            selected_style(theme)
        } else {
            theme.typography.body
        };
        Line::from(vec![
            Span::styled(marker.to_string(), label_style),
            Span::styled(format!("{label:<12}"), label_style),
            Span::styled(value, value_style),
        ])
    }

    /// One on/off endpoint-capability row. Both capabilities render the same
    /// way — a state word plus what turning it on actually sends — so they share
    /// a builder rather than inlining two near-identical blocks into
    /// [`Self::render_lines`] (which the 100-line lint rejects anyway).
    fn capability_line<'a>(&'a self, theme: &Theme, field: Field) -> Line<'a> {
        let (label, value) = match field {
            Field::ReasoningEffort => (
                "Effort",
                if self.supports_reasoning_effort {
                    "on — send reasoning_effort (/effort reaches this provider)"
                } else {
                    "off — no effort field; /effort has no effect here"
                },
            ),
            _ => (
                "Usage opt",
                if self.include_usage {
                    "on — send stream_options.include_usage"
                } else {
                    "off — safest for compatible servers"
                },
            ),
        };
        self.field_line(theme, field, label, value.to_string())
    }

    /// Build render lines for drawing and tests. Secrets are masked.
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let name_value = if self.name.is_empty() {
            "<provider-name>".to_string()
        } else {
            self.name.clone()
        };
        let base_url_value = if self.base_url.is_empty() {
            "<https://host.example/v1>".to_string()
        } else {
            self.base_url.clone()
        };
        let auth_env_value = if self.auth_env.is_empty() {
            "<ENV_OR_CREDENTIAL_NAME>".to_string()
        } else {
            self.auth_env.clone()
        };
        let models_value = if self.models_text.trim().is_empty() {
            "<auto-discover from /models; or paste ids>".to_string()
        } else {
            parse_model_ids(&self.models_text).join(", ")
        };
        let context_window_value = if self.context_window_text.trim().is_empty() {
            "<auto/unknown; optional tokens>".to_string()
        } else {
            self.context_window_text.clone()
        };
        let max_output_value = if self.max_output_tokens_text.trim().is_empty() {
            "<auto/default; optional tokens>".to_string()
        } else {
            self.max_output_tokens_text.clone()
        };
        let mut lines = vec![
            Line::from(vec![Span::styled(
                if self.mode.is_edit() {
                    format!("Edit provider · {}", self.name)
                } else {
                    "Add custom OpenAI-compatible provider".to_string()
                },
                theme.typography.heading_1,
            )]),
            Line::from(Span::styled(
                if self.mode.is_edit() {
                    "These fields replace what is stored — a model you delete here is removed."
                        .to_string()
                } else {
                    "Fill in order. Leave models empty to discover from /models after save."
                        .to_string()
                },
                theme.typography.dim,
            )),
            Line::from(""),
            self.field_line(theme, Field::Name, "Name", name_value),
            self.field_line(theme, Field::BaseUrl, "Base URL", base_url_value),
            self.field_line(
                theme,
                Field::AuthMode,
                "Auth",
                format!("{}  (←/→ or 1/2/3)", self.auth_mode.label()),
            ),
            self.field_line(theme, Field::AuthEnv, "Auth env", auth_env_value),
            self.field_line(theme, Field::ApiKey, "API key", self.masked_api_key()),
            self.field_line(theme, Field::Models, "Models", models_value),
            self.field_line(theme, Field::ContextWindow, "Context", context_window_value),
            self.field_line(theme, Field::MaxOutputTokens, "Max output", max_output_value),
            self.capability_line(theme, Field::IncludeUsage),
            self.capability_line(theme, Field::ReasoningEffort),
        ];
        if let Some(error) = &self.error {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("Error: {error}"),
                Style::new().fg(theme.palette.error),
            )));
        }
        lines.push(Line::from(""));
        lines.push(key_hint_footer_reflowing(
            theme,
            &[
                ("Enter", "next/save"),
                ("Tab", "next"),
                ("Shift+Tab", "back"),
                ("Ctrl+S", "save"),
                ("Ctrl+V", "paste"),
                ("Esc", "cancel"),
            ],
        ));
        lines
    }

    /// Draw the modal.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let title = if self.mode.is_edit() {
            " Edit provider "
        } else {
            " Connect custom provider "
        };
        let inner = super::modal_frame(frame, area, title, theme);
        let paragraph =
            Paragraph::new(super::wrap_body_rows(&self.render_lines(theme), inner.width, false))
                .style(theme.typography.body);
        frame.render_widget(paragraph, inner);
    }
}

impl Default for CustomProviderWizardModal {
    fn default() -> Self {
        Self::new()
    }
}

fn default_auth_env_for(name: &str) -> String {
    let slug = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if slug.is_empty() {
        "ZO_CUSTOM_API_KEY".to_string()
    } else {
        format!("ZO_{slug}_API_KEY")
    }
}

fn valid_auth_env_name(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn parse_model_ids(raw: &str) -> Vec<String> {
    raw.split([',', '\n', '\r', '\t'])
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_optional_token_count(raw: &str, label: &str) -> Result<Option<u64>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let normalized = trimmed.replace([',', '_'], "");
    let value = normalized
        .parse::<u64>()
        .map_err(|_| format!("{label} must be a positive integer token count"))?;
    if value == 0 {
        return Err(format!("{label} must be greater than 0"));
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventState;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn ctrl(ch: char) -> KeyEvent {
        KeyEvent {
            code: KeyCode::Char(ch),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn parse_model_ids_accepts_commas_and_newlines() {
        assert_eq!(
            parse_model_ids("a, b\nc\t d\r\n"),
            vec!["a", "b", "c", "d"]
        );
    }

    #[test]
    fn api_key_is_masked_in_rendered_lines() {
        let mut modal = CustomProviderWizardModal::new();
        modal.focus = 4;
        modal.paste_text("dummy-secret");
        let rendered = modal
            .render_lines(&Theme::zo())
            .into_iter()
            .map(|line| {
                line.spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!rendered.contains("dummy-secret"));
        assert!(rendered.contains("••••"));
    }

    #[test]
    fn default_auth_env_tracks_provider_name_until_touched() {
        let mut modal = CustomProviderWizardModal::new();
        modal.focus = 0;
        for ch in "nvidia-nim".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert_eq!(modal.auth_env(), "ZO_NVIDIA_NIM_API_KEY");
        modal.focus = 3;
        modal.handle_key(ctrl('u'));
        modal.paste_text("NVIDIA_API_KEY");
        modal.focus = 0;
        modal.handle_key(press(KeyCode::Char('x')));
        assert_eq!(modal.auth_env(), "NVIDIA_API_KEY");
    }

    #[test]
    fn submit_rejects_invalid_auth_env_name() {
        let mut modal = CustomProviderWizardModal::new();
        modal.name = "bad-env".to_string();
        modal.base_url = "https://example.com/v1".to_string();
        modal.auth_env = "FOO=bar".to_string();
        modal.api_key = "sk-secret".to_string();
        modal.focus = FIELDS.len() - 1;

        assert!(modal.handle_key(press(KeyCode::Enter)).is_none());
        assert!(
            modal
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Auth env must match")),
            "invalid env should be rejected before emitting a draft: {:?}",
            modal.error
        );
    }

    #[test]
    fn submit_parses_context_and_max_output_overrides() {
        let mut modal = CustomProviderWizardModal::new();
        modal.name = "xai-custom".to_string();
        modal.base_url = "https://api.x.ai/v1".to_string();
        modal.auth_env = "XAI_API_KEY".to_string();
        modal.api_key = "dummy-secret".to_string();
        modal.models_text = "grok-4.5".to_string();
        modal.context_window_text = "256,000".to_string();
        modal.max_output_tokens_text = "32_000".to_string();
        modal.focus = FIELDS.len() - 1;
        match modal.handle_key(press(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::CustomProvider(draft))) => {
                assert_eq!(draft.context_window, Some(256_000));
                assert_eq!(draft.max_output_tokens, Some(32_000));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn submit_rejects_invalid_context_override() {
        let mut modal = CustomProviderWizardModal::new();
        modal.name = "xai-custom".to_string();
        modal.base_url = "https://api.x.ai/v1".to_string();
        modal.auth_env = "XAI_API_KEY".to_string();
        modal.api_key = "dummy-secret".to_string();
        modal.models_text = "grok-4.5".to_string();
        modal.context_window_text = "many".to_string();
        modal.focus = FIELDS.len() - 1;

        assert!(modal.handle_key(press(KeyCode::Enter)).is_none());
        assert!(
            modal
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Context window must be")),
            "invalid context override should be rejected: {:?}",
            modal.error
        );
    }

    #[test]
    fn submit_returns_custom_provider_draft() {
        let mut modal = CustomProviderWizardModal::new();
        modal.name = "nvidia-nim".to_string();
        modal.base_url = "https://integrate.api.nvidia.com/v1".to_string();
        modal.auth_env = "NVIDIA_API_KEY".to_string();
        modal.api_key = "dummy-secret".to_string();
        modal.models_text = "meta/llama-3.1-8b-instruct, z-ai/glm-5.2".to_string();
        modal.focus = FIELDS.len() - 1;
        match modal.handle_key(press(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::CustomProvider(draft))) => {
                assert_eq!(draft.name, "nvidia-nim");
                assert_eq!(draft.auth_env.as_deref(), Some("NVIDIA_API_KEY"));
                assert_eq!(draft.api_key.as_deref(), Some("dummy-secret"));
                assert_eq!(
                    draft.models,
                    vec!["meta/llama-3.1-8b-instruct", "z-ai/glm-5.2"]
                );
                assert_eq!(draft.context_window, None);
                assert_eq!(draft.max_output_tokens, None);
                assert!(!draft.include_usage);
                assert!(!draft.supports_reasoning_effort);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn reasoning_effort_field_toggles_and_reaches_the_draft() {
        // The capability had no UI at all: it could only be turned on by
        // hand-editing `settings.json`, so a gateway-served reasoning model
        // silently ignored `/effort` with nothing in the interface to explain
        // why. Space (and y/n) toggles it, and it must survive submit.
        let mut modal = CustomProviderWizardModal::new();
        modal.name = "agent".to_string();
        modal.base_url = "https://agentrouter.org/v1".to_string();
        modal.auth_env = "ZO_AGENT_API_KEY".to_string();
        modal.api_key = "dummy-secret".to_string();
        modal.models_text = "kimi-k3".to_string();

        let effort_index = FIELDS
            .iter()
            .position(|field| *field == Field::ReasoningEffort)
            .expect("the effort field is on the form");
        modal.focus = effort_index;
        assert!(!modal.supports_reasoning_effort);
        assert!(modal.handle_key(press(KeyCode::Char(' '))).is_none());
        assert!(modal.supports_reasoning_effort, "space must toggle it on");
        assert!(modal.handle_key(press(KeyCode::Char('n'))).is_none());
        assert!(!modal.supports_reasoning_effort, "n must turn it off");
        assert!(modal.handle_key(press(KeyCode::Char('y'))).is_none());
        assert!(modal.supports_reasoning_effort, "y must turn it back on");

        modal.focus = FIELDS.len() - 1;
        match modal.handle_key(press(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::CustomProvider(draft))) => {
                assert!(draft.supports_reasoning_effort);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn editing_prefill_restores_the_reasoning_effort_toggle() {
        // Round-trip: a provider stored with the capability on must reopen with
        // the toggle on, or the next edit would silently turn it back off.
        let modal = CustomProviderWizardModal::editing(&CustomProviderPrefill {
            name: "agent".to_string(),
            base_url: "https://agentrouter.org/v1".to_string(),
            auth_env: Some("ZO_AGENT_API_KEY".to_string()),
            models: vec!["kimi-k3".to_string()],
            supports_reasoning_effort: Some(true),
            ..CustomProviderPrefill::default()
        });
        assert!(modal.supports_reasoning_effort);
    }
}
