//! API-key entry modal for `/connect` OpenAI-compatible cloud adapters.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::super::theme::Theme;
use super::{ModalResult, ModalSelection, key_hint_footer};

/// Metadata shown in the API-key setup modal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiKeyConnectInfo {
    /// Canonical provider id submitted to `/connect`, e.g. `deepseek`.
    pub provider: String,
    /// Human-facing provider label.
    pub label: String,
    /// Env var name used by the adapter, e.g. `DEEPSEEK_API_KEY`.
    pub auth_env: String,
    /// Curated model ids the preset will register.
    pub models: Vec<String>,
}

/// Secret text-entry modal for cloud OpenAI-compatible providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiKeyModal {
    info: ApiKeyConnectInfo,
    input: String,
}

impl ApiKeyModal {
    /// Create a new empty API-key input modal.
    #[must_use]
    pub fn new(info: ApiKeyConnectInfo) -> Self {
        Self {
            info,
            input: String::new(),
        }
    }

    /// Provider id this modal will connect on submit.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.info.provider
    }

    /// Current unmasked input; intended for tests only.
    #[cfg(test)]
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Paste text into the secret input. Bracketed paste and Ctrl+V clipboard
    /// payloads both route here; trim only the outer newlines/spaces users often
    /// get when copying keys from dashboards, but do not render the secret.
    pub fn paste_text(&mut self, text: &str) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            self.input.push_str(trimmed);
        }
    }

    /// Handle one key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Enter => {
                let trimmed = self.input.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(ModalResult::Selected(ModalSelection::ApiKey {
                        provider: self.info.provider.clone(),
                        api_key: trimmed.to_string(),
                    }))
                }
            }
            KeyCode::Backspace => {
                self.input.pop();
                None
            }
            KeyCode::Char(ch) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if matches!(ch, 'u' | 'U') {
                    self.input.clear();
                }
                None
            }
            KeyCode::Char(ch) => {
                self.input.push(ch);
                None
            }
            _ => None,
        }
    }

    fn masked_input(&self) -> String {
        if self.input.is_empty() {
            "<paste API key>".to_string()
        } else {
            "•".repeat(self.input.chars().count().min(64))
        }
    }

    /// Build render lines for draw and tests. The actual key is never rendered.
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Provider  ".to_string(), theme.typography.dim),
                Span::styled(self.info.label.clone(), theme.typography.heading_1),
            ]),
            Line::from(vec![
                Span::styled("Stores    ".to_string(), theme.typography.dim),
                Span::styled(
                    format!("provider in settings.json; key in credentials.json ({})", self.info.auth_env),
                    theme.typography.body,
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled("Models", theme.typography.dim)),
        ];
        for model in &self.info.models {
            lines.push(Line::from(format!("  • {model}")));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("API key  ".to_string(), theme.typography.dim),
            Span::styled(self.masked_input(), theme.typography.body),
        ]));
        lines.push(Line::from(""));
        lines.push(key_hint_footer(
            theme,
            &[
                ("Enter", "save + connect"),
                ("Ctrl+V", "paste"),
                ("Esc", "cancel"),
                ("Ctrl+U", "clear"),
            ],
        ));
        lines
    }

    /// Draw the modal into `area` using `theme`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = super::modal_frame(
            frame,
            area,
            format!(" Connect {} ", self.info.label),
            theme,
        );
        let paragraph = Paragraph::new(self.render_lines(theme))
            .style(theme.typography.body)
            .wrap(Wrap { trim: false });
        frame.render_widget(paragraph, inner);
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

    #[test]
    fn paste_text_trims_dashboard_newline_and_stays_masked() {
        let mut modal = ApiKeyModal::new(ApiKeyConnectInfo {
            provider: "deepseek".to_string(),
            label: "DeepSeek".to_string(),
            auth_env: "DEEPSEEK_API_KEY".to_string(),
            models: vec!["deepseek-chat".to_string()],
        });

        modal.paste_text("  sk-pasted\n");
        assert_eq!(modal.input(), "sk-pasted");
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
        assert!(!rendered.contains("sk-pasted"));
        assert!(rendered.contains("••••"));
        assert!(rendered.contains("Ctrl+V"));
        assert!(rendered.contains("paste"));
    }

    #[test]
    fn enter_returns_provider_and_key_without_rendering_secret() {
        let mut modal = ApiKeyModal::new(ApiKeyConnectInfo {
            provider: "deepseek".to_string(),
            label: "DeepSeek".to_string(),
            auth_env: "DEEPSEEK_API_KEY".to_string(),
            models: vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()],
        });
        for ch in "sk-secret".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
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
        assert!(!rendered.contains("sk-secret"));
        assert!(rendered.contains("DEEPSEEK_API_KEY"));

        match modal.handle_key(press(KeyCode::Enter)) {
            Some(ModalResult::Selected(ModalSelection::ApiKey { provider, api_key })) => {
                assert_eq!(provider, "deepseek");
                assert_eq!(api_key, "sk-secret");
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }
}
