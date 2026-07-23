//! `@`-mention inline autocomplete: file completion under the cursor.
//!
//! Sister module to [`super::slash_hint`]. Where `slash_hint` completes
//! `/commands` at the start of the line, this completes `@file` mentions
//! at the cursor — triggered as-you-type and ranked by frecency. opencode
//! `autocomplete.tsx` parity (the `@` arm): the popup lists workspace files
//! matching the partial token, most-frequently/recently mentioned first.
//!
//! Trigger and apply are pure string functions (unit-tested here); the host
//! [`super::App`] owns the file cache, frecency log, selection cursor, and
//! key routing — mirroring the slash-hint wiring.

use std::collections::HashMap;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};

use crate::tui::cards::{CardFrame, SurfaceKind};
use crate::tui::theme::Theme;

use std::sync::Arc;

use super::App;
use super::AppMode;
use super::slash_hint::fit_popup_above_input;
use super::{collect_workspace_files, new_scan_cancel_token};

/// Locate the active `@`-mention token on a single-line buffer.
///
/// Returns `(at_byte, query)` when the buffer ends in an `@token`: the last
/// `@` that (a) starts a word — preceded by whitespace or the buffer start,
/// so `user@host` never triggers — and (b) has no whitespace between it and
/// the end of the line (the cursor is parked at the token's end, mirroring
/// `slash_completion_for`). `query` is the text after the `@`.
pub(super) fn mention_trigger(line: &str) -> Option<(usize, &str)> {
    let at = line.rfind('@')?;
    // Word-start guard: the char before `@` must be whitespace (or none),
    // so an email address or `a@b` inside a word does not open the popup.
    if at > 0 {
        let before = line[..at].chars().next_back();
        if before.is_some_and(|c| !c.is_whitespace()) {
            return None;
        }
    }
    let query = &line[at + 1..];
    // A space after the token means the mention is already complete.
    if query.chars().any(char::is_whitespace) {
        return None;
    }
    Some((at, query))
}

/// Rank workspace `files` for the partial `query`, frecency-weighted.
///
/// opencode `autocomplete.tsx` scoring parity: a base relevance (basename
/// prefix > substring > subsequence) multiplied by `1 + frecency`, so
/// frequently-mentioned files float up. An empty query lists the
/// highest-frecency files (the "just typed `@`" case). Ties break on path
/// depth then lexically, matching opencode's secondary sort.
pub(super) fn rank_mentions(
    query: &str,
    files: &[String],
    frecency: &HashMap<String, f64>,
    limit: usize,
) -> Vec<String> {
    let q = query.to_ascii_lowercase();
    let mut scored: Vec<(f64, usize, &String)> = files
        .iter()
        .filter_map(|file| {
            let base = match_score(&q, file)?;
            let frec = frecency.get(file).copied().unwrap_or(0.0);
            let depth = file.bytes().filter(|&b| b == b'/').count();
            Some((base * (1.0 + frec), depth, file))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(b.2))
    });
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, f)| f.clone())
        .collect()
}

/// Relevance of `file` to the already-lowercased `query`; `None` if it does
/// not match. Basename prefix beats a path substring beats a subsequence;
/// an empty query matches everything (relevance comes from frecency alone).
fn match_score(query: &str, file: &str) -> Option<f64> {
    if query.is_empty() {
        return Some(1.0);
    }
    let lower = file.to_ascii_lowercase();
    let base = lower.rsplit('/').next().unwrap_or(lower.as_str());
    if base.starts_with(query) {
        Some(4.0)
    } else if lower.contains(query) {
        Some(2.0)
    } else if is_subsequence(query, &lower) {
        Some(1.0)
    } else {
        None
    }
}

/// `true` when every char of `needle` appears in `haystack` in order.
fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut hay = haystack.chars();
    needle.chars().all(|nc| hay.any(|hc| hc == nc))
}

/// Replace the active `@token` on `line` with `@path ` (trailing space so
/// the user can keep typing after the inserted mention).
pub(super) fn apply_mention(line: &str, at_byte: usize, path: &str) -> String {
    format!("{}@{} ", &line[..at_byte], path)
}

/// Render the `@`-mention popup above the input row. `suggestions` is the
/// already-ranked path list; `selected` highlights the active row.
pub(super) fn draw_mention_hint(
    frame: &mut ratatui::Frame<'_>,
    input_area: Rect,
    inline_bounds: Option<Rect>,
    suggestions: &[String],
    theme: &Theme,
    selected: Option<usize>,
) {
    if suggestions.is_empty() || input_area.y == 0 {
        return;
    }
    let mut lines: Vec<Line<'_>> = Vec::with_capacity(suggestions.len());
    for (i, path) in suggestions.iter().enumerate() {
        let is_selected = selected == Some(i);
        let indicator = if is_selected { "\u{25b8} " } else { "  " };
        if is_selected {
            let sel_bg = Style::default().bg(theme.palette.faint);
            let sel_label = sel_bg.fg(theme.palette.accent).add_modifier(Modifier::BOLD);
            lines.push(Line::from(vec![
                Span::styled(indicator, sel_label),
                Span::styled(format!("@{path}"), sel_label),
            ]));
        } else {
            let name_style = Style::default().fg(theme.palette.accent);
            let dim = Style::default().fg(theme.palette.dim);
            lines.push(Line::from(vec![
                Span::styled(indicator, dim),
                Span::styled("@", dim),
                Span::styled(path.clone(), name_style),
            ]));
        }
    }

    // +2 for the top/bottom border rows.
    let popup_rows = u16::try_from(lines.len()).unwrap_or(10).saturating_add(2);
    let popup_y = input_area.y.saturating_sub(popup_rows);
    let popup_width = if input_area.width < 32 {
        input_area.width
    } else {
        input_area.width.min(90)
    };
    let mut popup_area = Rect::new(input_area.x, popup_y, popup_width, popup_rows);
    if let Some(bounds) = inline_bounds {
        let Some(bounded) = fit_popup_above_input(popup_area, input_area, bounds) else {
            return;
        };
        popup_area = bounded;
    }

    frame.render_widget(Clear, popup_area);
    let block = CardFrame::new(SurfaceKind::Popup, theme)
        .title(Span::styled(
            " @mention ",
            Style::default()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(
            " tab insert \u{00b7} enter insert \u{00b7} esc close ",
            Style::default().fg(theme.palette.dim),
        ))
        .block();
    frame.render_widget(Paragraph::new(lines).block(block), popup_area);
}

/// App 상태에 얹힌 at-멘션(파일 픽커) 힌트 조회·선택 메서드.
impl App {
    pub(super) fn hide_mention_hint_for_current_input(&mut self) {
        self.hints.mention_cursor = None;
        self.hints.mention_hidden_for = Some(self.input.text());
    }

    /// Whether the `@`-mention hint popup is visible: a single-line buffer
    /// whose text ends in an active `@token` (see [`mention_trigger`]).
    pub(super) fn mention_hint_active(&self) -> bool {
        if !self.input_enabled
            || !matches!(self.mode, AppMode::Normal)
            || self.input.has_collapsed_paste()
        {
            return false;
        }
        let buffer = self.input.text();
        if buffer.contains('\n') || self.hints.mention_hidden_for.as_deref() == Some(buffer.as_str())
        {
            return false;
        }
        mention_trigger(&buffer).is_some()
    }

    /// `true` when an `@` typed now begins an *inline* mention (the cursor
    /// sits past the start of a single-line buffer) rather than opening the
    /// full picker modal (the bare-`@`-at-line-start case).
    pub(super) fn mention_opens_inline(&self) -> bool {
        if self.input.has_collapsed_paste() {
            return true;
        }
        if self.input.text().contains('\n') {
            return false;
        }
        self.input.cursor().1 > 0
    }

    /// Collect the workspace file list once and cache it for `@`-mention
    /// completion. Runs the scan on a `spawn_blocking` worker (landed by
    /// [`Self::poll_workspace_scan`] from the run loop) so typing `@` never
    /// blocks the UI thread on a large repo walk; without a tokio runtime
    /// (unit tests) it falls back to a synchronous fill.
    pub(super) fn ensure_workspace_files(&mut self) {
        if self.input.has_collapsed_paste()
            || !self.workspace_files.is_empty()
            || self.scans.workspace_task.is_some()
        {
            return;
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let cancel = new_scan_cancel_token();
            let worker_cancel = Arc::clone(&cancel);
            self.scans.workspace_cancel = Some(cancel);
            self.scans.workspace_task =
                Some(handle.spawn_blocking(move || collect_workspace_files(&worker_cancel)));
        } else {
            let cancel = new_scan_cancel_token();
            self.workspace_files = collect_workspace_files(&cancel);
        }
    }

    /// Land a finished background workspace scan into `workspace_files`.
    /// Returns `true` when new entries arrived (caller should redraw so the
    /// open mention hint swaps from empty to the real suggestions). Called
    /// each tick from the run loop; a no-op when no scan is pending.
    pub fn poll_workspace_scan(&mut self) -> bool {
        let Some(task) = &mut self.scans.workspace_task else {
            return false;
        };
        if !task.is_finished() {
            return false;
        }
        let task = self
            .scans
            .workspace_task
            .take()
            .expect("checked Some above");
        self.scans.workspace_cancel.take();
        match futures_util::future::FutureExt::now_or_never(task) {
            Some(Ok(items)) if !items.is_empty() => {
                self.workspace_files = items;
                true
            }
            // Join error or empty workspace: leave the list empty; the next
            // `@` retriggers the scan (cheap, and rare).
            _ => false,
        }
    }

    /// Ranked `@`-mention suggestions for the active token (frecency-
    /// weighted), or empty when no mention token is active.
    pub(super) fn mention_hint_suggestions(&self) -> Vec<String> {
        if self.input.has_collapsed_paste() {
            return Vec::new();
        }
        let buffer = self.input.text();
        let Some((_, query)) = mention_trigger(&buffer) else {
            return Vec::new();
        };
        rank_mentions(
            query,
            &self.workspace_files,
            &self.mention_history.frecency_scores(),
            10,
        )
    }

    /// The workspace path at the current mention-hint cursor, if any.
    pub(super) fn mention_hint_selected_path(&self) -> Option<String> {
        let cursor = self.hints.mention_cursor?;
        self.mention_hint_suggestions().into_iter().nth(cursor)
    }

    /// Number of [`RenderBlock`]s this App has drained from `rx`.
    #[must_use]
    pub const fn blocks_drained(&self) -> usize {
        self.blocks_drained
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_finds_at_token_at_word_start() {
        assert_eq!(mention_trigger("fix @src/ma"), Some((4, "src/ma")));
        assert_eq!(mention_trigger("@lib"), Some((0, "lib")));
        // Bare `@` (just opened) yields an empty query, still a trigger.
        assert_eq!(mention_trigger("see @"), Some((4, "")));
        // Email-like: `@` not at a word start → no trigger.
        assert_eq!(mention_trigger("user@host"), None);
        // Completed mention (whitespace after the token) → no trigger.
        assert_eq!(mention_trigger("see @a.rs next"), None);
        // No `@` at all.
        assert_eq!(mention_trigger("plain text"), None);
    }

    #[test]
    fn rank_prefers_basename_prefix_then_frecency() {
        let files = vec![
            "src/conversation/mod.rs".to_string(),
            "src/convert.rs".to_string(),
            "docs/conv.md".to_string(),
        ];
        let mut frec = HashMap::new();
        frec.insert("src/convert.rs".to_string(), 5.0);
        let ranked = rank_mentions("conv", &files, &frec, 10);
        // convert.rs: basename prefix (4.0) × (1+5) = 24 → clear top.
        assert_eq!(ranked[0], "src/convert.rs");
        assert_eq!(ranked.len(), 3, "all three match 'conv'");
    }

    #[test]
    fn empty_query_lists_by_frecency() {
        let files = vec!["a.rs".to_string(), "b.rs".to_string()];
        let mut frec = HashMap::new();
        frec.insert("b.rs".to_string(), 9.0);
        let ranked = rank_mentions("", &files, &frec, 10);
        assert_eq!(ranked[0], "b.rs", "highest frecency first on empty query");
    }

    #[test]
    fn apply_replaces_token_with_path_and_trailing_space() {
        assert_eq!(
            apply_mention("fix @con", 4, "src/convert.rs"),
            "fix @src/convert.rs "
        );
        assert_eq!(apply_mention("@l", 0, "lib.rs"), "@lib.rs ");
    }

    #[test]
    fn subsequence_matches_scattered_chars_in_order() {
        assert!(is_subsequence("cnv", "convert"));
        assert!(!is_subsequence("xyz", "convert"));
        // Order matters: "tc" is not a subsequence of "convert".
        assert!(!is_subsequence("tc", "convert"));
    }
}
