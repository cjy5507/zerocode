//! `/diff` interactive full-screen diff viewer modal.
//!
//! A keyboard-navigable review surface (the Cursor `Ctrl+R` / opencode
//! diff-tree lesson): a file selector across the top, the selected file's
//! hunks scrollable in the body. Read-only, so it is the lowest-risk
//! interactive modal — `Esc`/`q` closes, arrows navigate.
//!
//! Diff data is the provider-neutral [`DiffView`] already used by tool
//! results, rendered by [`crate::tui::blocks::diff`]; this module adds the
//! `git diff` unified-text → `DiffView` parser and the navigation state.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};

use super::super::cards::{CardFrame, SurfaceKind};
use runtime::message_stream::{DiffHunk, DiffLine, DiffLineKind, DiffView};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::super::theme::Theme;
use super::draw_scrollbar;

/// opencode `diff-viewer.tsx` parity (`MIN_SPLIT_WIDTH = 100`): the
/// before/after split is only offered when the diff pane is at least this
/// wide (columns). Two ~50-column panes need the room; below it even a
/// split *preference* renders unified — matching opencode's
/// `splitAvailable = patchPaneWidth >= MIN_SPLIT_WIDTH`.
const MIN_SPLIT_WIDTH: u16 = 100;

/// Outcome of a single key handled by [`DiffViewerModal`].
///
/// Navigation keys return `None` (the viewer stays open); these variants
/// are the events the host app must act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffViewerAction {
    /// `Esc`/`q`/`Ctrl+C` — close the viewer.
    Close,
    /// `r` — revert the currently selected file to `HEAD`. Carries the
    /// repo-relative path the host should `git checkout`.
    RevertFile(String),
}

/// Interactive diff viewer over one or more changed files.
#[derive(Debug, Clone)]
pub struct DiffViewerModal {
    files: Vec<DiffView>,
    selected: usize,
    scroll: u16,
    /// User's view *preference*: `true` = before/after split (the default,
    /// opencode-style "auto"), `false` = force unified. The split actually
    /// renders only when the pane also clears [`MIN_SPLIT_WIDTH`] — see
    /// [`DiffViewerModal::renders_split`].
    side_by_side: bool,
}

impl DiffViewerModal {
    /// Build a viewer over the given per-file diffs.
    #[must_use]
    pub fn new(files: Vec<DiffView>) -> Self {
        Self {
            files,
            selected: 0,
            scroll: 0,
            // opencode parity: prefer split; the width gate downgrades to
            // unified on narrow panes, making this "auto" in practice.
            side_by_side: true,
        }
    }

    /// Replace the file set (e.g. after a revert refreshed `git diff`),
    /// keeping the view mode but clamping the selection into range and
    /// resetting scroll to the top of the newly selected file.
    pub fn set_files(&mut self, files: Vec<DiffView>) {
        self.files = files;
        self.selected = self.selected.min(self.files.len().saturating_sub(1));
        self.scroll = 0;
    }

    /// The user's split *preference* (not necessarily what renders — a
    /// narrow pane forces unified). See [`Self::renders_split`].
    #[must_use]
    pub const fn is_side_by_side(&self) -> bool {
        self.side_by_side
    }

    /// opencode parity: the body renders as a before/after split only when
    /// the user prefers it *and* the pane clears [`MIN_SPLIT_WIDTH`]. On a
    /// narrow pane this is `false` so the two columns never get crushed.
    #[must_use]
    const fn renders_split(&self, body_width: u16) -> bool {
        self.side_by_side && body_width >= MIN_SPLIT_WIDTH
    }

    /// Repo-relative path of the currently selected file, if any.
    /// Prefers the new path, falling back to the old (deletions).
    #[must_use]
    pub fn selected_path(&self) -> Option<&str> {
        self.files
            .get(self.selected)
            .and_then(|v| v.new_path.as_deref().or(v.old_path.as_deref()))
    }

    /// `true` when there is nothing to show (clean tree) — the caller
    /// should fall back to a "no changes" card instead of opening.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Number of files in the viewer.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Index of the currently selected file.
    #[must_use]
    pub const fn selected(&self) -> usize {
        self.selected
    }

    fn next_file(&mut self) {
        if self.selected + 1 < self.files.len() {
            self.selected += 1;
            self.scroll = 0;
        }
    }

    fn prev_file(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.scroll = 0;
        }
    }

    /// Handle one key. Returns `Some(Close)` to dismiss, `Some(RevertFile)`
    /// to request a file revert; `None` while navigating / toggling.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<DiffViewerAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(DiffViewerAction::Close);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return Some(DiffViewerAction::Close),
            KeyCode::Left | KeyCode::Char('h' | '[') => self.prev_file(),
            KeyCode::Right | KeyCode::Char('l' | ']') => self.next_file(),
            KeyCode::Up | KeyCode::Char('k') => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll = self.scroll.saturating_add(1),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            KeyCode::Home | KeyCode::Char('g') => self.scroll = 0,
            // `s` toggles unified ↔ side-by-side; scroll resets so the
            // re-flowed body starts at the top.
            KeyCode::Char('s') => {
                self.side_by_side = !self.side_by_side;
                self.scroll = 0;
            }
            // `r` requests reverting the selected file to HEAD. The host
            // app runs the git checkout and rebuilds the viewer.
            KeyCode::Char('r') => {
                if let Some(path) = self.selected_path() {
                    return Some(DiffViewerAction::RevertFile(path.to_string()));
                }
            }
            _ => {}
        }
        None
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(" /diff ", theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }

        // header (file selector, 1 row) · body (scrollable diff) · footer.
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        frame.render_widget(
            Paragraph::new(self.header_line(theme, header.width)),
            header,
        );

        let view = self.files.get(self.selected);
        let split = self.renders_split(body.width);
        let content_rows = if split {
            self.draw_side_by_side(frame, body, theme, view)
        } else {
            // `diff::lines` leads with metadata + path headers; the fixed
            // selector row above already shows them, so drop the duplicate
            // rows inside the scroll body.
            // Lines borrow from the selected `DiffView`, which outlives this
            // draw call — no `'static` clone needed.
            let mut body_lines = view
                .map(|view| {
                    crate::tui::blocks::diff::lines(view, theme, true)
                        .into_iter()
                        .skip(crate::tui::blocks::diff::header_rows())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            crate::tui::blocks::diff::pad_changed_rows(
                &mut body_lines,
                usize::from(body.width),
            );
            let content_rows = body_lines.len();
            let scroll = clamped_scroll(self.scroll, content_rows, body.height);
            frame.render_widget(Paragraph::new(body_lines).scroll((scroll, 0)), body);
            content_rows
        };
        draw_scrollbar(frame, body, self.scroll, content_rows, theme);

        frame.render_widget(
            Paragraph::new(footer_line(
                theme,
                split,
                body.width >= MIN_SPLIT_WIDTH,
                self.scroll,
                content_rows,
                body.height,
                footer.width,
            )),
            footer,
        );
    }

    /// Render the selected file as a left=before / right=after split.
    ///
    /// The body is laid out as two equal columns separated by a 1-cell
    /// gutter. Both columns scroll together (`self.scroll` rows), so a
    /// removed line on the left always sits beside its replacement on the
    /// right. Rows are paired per hunk: context lines appear on both
    /// sides; removed lines fill the left, added lines the right, and the
    /// shorter side is padded with blank rows so the pairing stays aligned.
    fn draw_side_by_side(
        &self,
        frame: &mut Frame<'_>,
        body: Rect,
        theme: &Theme,
        view: Option<&DiffView>,
    ) -> usize {
        let Some(view) = view else { return 0 };
        // [ left | 1-cell separator | right ]
        let [left, sep, right] = Layout::horizontal([
            Constraint::Percentage(50),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .areas(body);

        let (left_lines, right_lines) = split_lines_for_width(view, theme, left.width, right.width);
        let content_rows = left_lines.len().max(right_lines.len());
        let scroll = clamped_scroll(self.scroll, content_rows, body.height);
        let sep_style = theme.typography.dim;
        let sep_lines: Vec<Line<'static>> = (0..body.height)
            .map(|_| Line::from(Span::styled("\u{2502}".to_string(), sep_style)))
            .collect();

        frame.render_widget(Paragraph::new(left_lines).scroll((scroll, 0)), left);
        frame.render_widget(Paragraph::new(sep_lines), sep);
        frame.render_widget(Paragraph::new(right_lines).scroll((scroll, 0)), right);
        content_rows
    }

    /// `‹ 2/5 ›  modified · path/to/file.rs · 2 hunks  +12 -3` selector row.
    fn header_line(&self, theme: &Theme, width: u16) -> Line<'static> {
        let Some(view) = self.files.get(self.selected) else {
            return Line::from(Span::styled("no changes".to_string(), theme.typography.dim));
        };
        let (adds, rems) = crate::tui::blocks::diff::tally(view);
        let kind = crate::tui::blocks::diff::change_kind(view);
        let hunk_label = crate::tui::blocks::diff::hunk_label(view);
        let tally = format!("+{adds} -{rems}");
        let arrows = theme.no_color;
        let (prev, next) = if arrows {
            ("< ", " >")
        } else {
            ("\u{2039} ", " \u{203A}")
        };
        let index = format!("{}/{}", self.selected + 1, self.files.len());
        let fixed_width = display_width(prev)
            + display_width(&index)
            + display_width(next)
            + display_width("  ")
            + display_width(kind)
            + display_width(" · ")
            + display_width(" · ")
            + display_width(&hunk_label)
            + display_width("  ")
            + display_width(&tally);
        let path_budget = usize::from(width).saturating_sub(fixed_width).max(1);
        let path =
            truncate_middle_to_cells(&crate::tui::blocks::diff::path_label(view), path_budget);
        Line::from(vec![
            Span::styled(prev.to_string(), theme.typography.key_hint),
            Span::styled(
                index,
                Style::new()
                    .fg(theme.palette.bright)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(next.to_string(), theme.typography.key_hint),
            Span::raw("  "),
            Span::styled(kind.to_string(), theme.diff_file_header_style()),
            Span::styled(" · ".to_string(), theme.typography.dim),
            Span::styled(
                path,
                Style::new()
                    .fg(theme.palette.cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" · ".to_string(), theme.typography.dim),
            Span::styled(hunk_label, theme.typography.dim),
            Span::raw("  "),
            Span::styled(format!("+{adds}"), theme.diff_add_style()),
            Span::raw(" "),
            Span::styled(format!("-{rems}"), theme.diff_del_style()),
        ])
    }
}

fn truncate_middle_to_cells(text: &str, max_cells: usize) -> String {
    if UnicodeWidthStr::width(text) <= max_cells {
        return text.to_string();
    }
    if max_cells == 0 {
        return String::new();
    }
    let ellipsis = "\u{2026}";
    let ellipsis_width = display_width(ellipsis);
    if max_cells <= ellipsis_width {
        return ellipsis.to_string();
    }

    let budget = max_cells.saturating_sub(ellipsis_width);
    let head_budget = budget / 2;
    let tail_budget = budget.saturating_sub(head_budget);
    let head = take_prefix_cells(text, head_budget);
    let tail = take_suffix_cells(text, tail_budget);
    format!("{head}{ellipsis}{tail}")
}

fn take_prefix_cells(text: &str, max_cells: usize) -> String {
    let mut out = String::new();
    let mut width = 0usize;
    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width.saturating_add(ch_width) > max_cells {
            break;
        }
        out.push(ch);
        width = width.saturating_add(ch_width);
    }
    out
}

fn take_suffix_cells(text: &str, max_cells: usize) -> String {
    let mut chars = Vec::new();
    let mut width = 0usize;
    for ch in text.chars().rev() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width.saturating_add(ch_width) > max_cells {
            break;
        }
        chars.push(ch);
        width = width.saturating_add(ch_width);
    }
    chars.into_iter().rev().collect()
}

use crate::tui::text_metrics::display_width;

fn footer_line(
    theme: &Theme,
    split: bool,
    can_split: bool,
    scroll: u16,
    content_rows: usize,
    viewport_height: u16,
    width: u16,
) -> Line<'static> {
    use super::FooterSegment;

    // The `s` hint shows the mode you'd switch *to*. On a pane too narrow
    // for split the toggle is inert, so surface *why* instead of a label
    // that wouldn't change anything.
    let (s_key_style, view_label) = if can_split {
        (theme.typography.key_hint, if split { "unified" } else { "split" })
    } else {
        (theme.typography.dim, "split ≥100 cols")
    };
    let mode = if split { "split" } else { "unified" };
    let progress = scroll_progress_label(scroll, content_rows, viewport_height);
    let full = super::modal_footer(
        theme,
        &[
            FooterSegment::label(mode),
            FooterSegment::label(progress.as_str()),
            FooterSegment::hint("←/→", "file"),
            FooterSegment::hint("↑/↓", "scroll"),
            FooterSegment::hint("PgUp/PgDn", "page"),
            FooterSegment::hint_with_key_style("s", view_label, s_key_style),
            FooterSegment::hint("r", "revert"),
            FooterSegment::hint("Esc", "close"),
        ],
        " · ",
    );
    if line_width(&full) <= usize::from(width) {
        return full;
    }

    let remaining = scroll_remaining_label(scroll, content_rows, viewport_height);
    let compact = super::modal_footer(
        theme,
        &[
            FooterSegment::label(mode),
            FooterSegment::label(remaining.as_str()),
            FooterSegment::hint("←/→", "file"),
            FooterSegment::hint("↑/↓", "scroll"),
            FooterSegment::hint_with_key_style("s", view_label, s_key_style),
            FooterSegment::hint("Esc", "close"),
        ],
        " · ",
    );
    if line_width(&compact) <= usize::from(width) {
        return compact;
    }

    super::modal_footer(
        theme,
        &[
            FooterSegment::label(mode),
            FooterSegment::label(remaining.as_str()),
            FooterSegment::hint("Esc", "close"),
        ],
        " · ",
    )
}

fn scroll_progress_label(scroll: u16, content_rows: usize, viewport_height: u16) -> String {
    if content_rows == 0 {
        return "no rows".to_string();
    }
    let viewport = usize::from(viewport_height).max(1);
    let clamped = usize::from(clamped_scroll(scroll, content_rows, viewport_height));
    let first = clamped.saturating_add(1);
    let last = (clamped + viewport).min(content_rows);
    let max_scroll = content_rows.saturating_sub(viewport);
    if max_scroll == 0 {
        return format!("all {content_rows} rows");
    }
    let pct_done = clamped.saturating_mul(100) / max_scroll;
    let pct_left = 100usize.saturating_sub(pct_done);
    if clamped == 0 {
        format!("top · rows {first}-{last}/{content_rows} · {pct_left}% left")
    } else if clamped >= max_scroll {
        format!("end · rows {first}-{last}/{content_rows} · 0% left")
    } else {
        format!("rows {first}-{last}/{content_rows} · {pct_left}% left")
    }
}

fn scroll_remaining_label(scroll: u16, content_rows: usize, viewport_height: u16) -> String {
    if content_rows == 0 {
        return "no rows".to_string();
    }
    let viewport = usize::from(viewport_height).max(1);
    let max_scroll = content_rows.saturating_sub(viewport);
    if max_scroll == 0 {
        return format!("all {content_rows}");
    }
    let clamped = usize::from(clamped_scroll(scroll, content_rows, viewport_height));
    let pct_done = clamped.saturating_mul(100) / max_scroll;
    let pct_left = 100usize.saturating_sub(pct_done);
    if clamped == 0 {
        format!("top · {pct_left}% left")
    } else if clamped >= max_scroll {
        "end · 0% left".to_string()
    } else {
        format!("{pct_left}% left")
    }
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

fn clamped_scroll(scroll: u16, content_rows: usize, viewport_height: u16) -> u16 {
    let max_scroll = content_rows.saturating_sub(usize::from(viewport_height));
    let max_scroll = u16::try_from(max_scroll).unwrap_or(u16::MAX);
    scroll.min(max_scroll)
}

/// Build the before/after column line lists for the side-by-side view.
///
/// Each hunk emits a `@@ … @@` marker on both sides, then its lines are
/// paired: a run of removed lines is zipped against the following run of
/// added lines (the classic replacement case), context lines flush both
/// pending runs and then appear on both columns. Padding blank rows keep
/// the two columns the same height so they scroll in lockstep.
#[cfg(test)]
fn split_lines<'a>(view: &'a DiffView, theme: &Theme) -> (Vec<Line<'a>>, Vec<Line<'a>>) {
    split_lines_for_width(view, theme, u16::MAX, u16::MAX)
}

fn split_lines_for_width<'a>(
    view: &'a DiffView,
    theme: &Theme,
    left_width: u16,
    right_width: u16,
) -> (Vec<Line<'a>>, Vec<Line<'a>>) {
    let mut left: Vec<Line<'a>> = vec![split_column_header(view, true, theme, left_width)];
    let mut right: Vec<Line<'a>> = vec![split_column_header(view, false, theme, right_width)];
    let hunk_style = Style::new()
        .fg(theme.palette.violet)
        .add_modifier(Modifier::BOLD);
    let add_style = theme.diff_add_style();
    let del_style = theme.diff_del_style();
    let ctx_style = theme.diff_context_style();
    let gutter_style = theme.typography.dim;

    // Pad whichever column is shorter up to `target` rows with blanks.
    let pad = |left: &mut Vec<Line<'a>>, right: &mut Vec<Line<'a>>| {
        let target = left.len().max(right.len());
        left.resize_with(target, || Line::from(""));
        right.resize_with(target, || Line::from(""));
    };

    for (idx, hunk) in view.hunks.iter().enumerate() {
        let hunk_no = idx + 1;
        let old_width = split_line_number_width(hunk.old_start, hunk.old_lines);
        let new_width = split_line_number_width(hunk.new_start, hunk.new_lines);
        let (hunk_adds, hunk_rems) = crate::tui::blocks::diff::hunk_tally(hunk);
        let left_hdr = format!(
            "Hunk {hunk_no} · old {} · -{}",
            split_range_label(hunk.old_start, hunk.old_lines),
            hunk_rems
        );
        let right_hdr = format!(
            "Hunk {hunk_no} · new {} · +{}",
            split_range_label(hunk.new_start, hunk.new_lines),
            hunk_adds
        );
        left.push(Line::styled(left_hdr, hunk_style));
        right.push(Line::styled(right_hdr, hunk_style));

        // Buffer consecutive removed / added lines so a removed run lines
        // up beside the matching added run, then flush at the next context
        // line (or end of hunk).
        let mut pend_del: Vec<Line<'a>> = Vec::new();
        let mut pend_add: Vec<Line<'a>> = Vec::new();
        let mut old_line = hunk.old_start;
        let mut new_line = hunk.new_start;
        let flush = |left: &mut Vec<Line<'a>>,
                     right: &mut Vec<Line<'a>>,
                     dels: &mut Vec<Line<'a>>,
                     adds: &mut Vec<Line<'a>>| {
            left.append(dels);
            right.append(adds);
            pad(left, right);
        };

        for line in &hunk.lines {
            match line.kind {
                DiffLineKind::Removed => {
                    pend_del.push(split_code_line(
                        Some(old_line),
                        old_width,
                        '-',
                        line.text.as_str(),
                        del_style,
                        gutter_style,
                        theme.diff_del_bg(),
                    ));
                    old_line = old_line.saturating_add(1);
                }
                DiffLineKind::Added => {
                    pend_add.push(split_code_line(
                        Some(new_line),
                        new_width,
                        '+',
                        line.text.as_str(),
                        add_style,
                        gutter_style,
                        theme.diff_add_bg(),
                    ));
                    new_line = new_line.saturating_add(1);
                }
                DiffLineKind::Context => {
                    flush(&mut left, &mut right, &mut pend_del, &mut pend_add);
                    left.push(split_code_line(
                        Some(old_line),
                        old_width,
                        ' ',
                        line.text.as_str(),
                        ctx_style,
                        gutter_style,
                        None,
                    ));
                    right.push(split_code_line(
                        Some(new_line),
                        new_width,
                        ' ',
                        line.text.as_str(),
                        ctx_style,
                        gutter_style,
                        None,
                    ));
                    old_line = old_line.saturating_add(1);
                    new_line = new_line.saturating_add(1);
                }
            }
        }
        flush(&mut left, &mut right, &mut pend_del, &mut pend_add);
    }
    (left, right)
}

fn split_column_header<'a>(view: &DiffView, before: bool, theme: &Theme, width: u16) -> Line<'a> {
    let label = if before { "before" } else { "after" };
    let path = if before {
        view.old_path.as_deref().unwrap_or("/dev/null")
    } else {
        view.new_path.as_deref().unwrap_or("/dev/null")
    };
    let fixed_width = display_width(label) + display_width(" · ");
    let path_budget = usize::from(width).saturating_sub(fixed_width).max(1);
    let path = truncate_middle_to_cells(path, path_budget);
    Line::from(vec![
        Span::styled(
            label.to_string(),
            Style::new()
                .fg(theme.palette.bright)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · ".to_string(), theme.typography.dim),
        Span::styled(
            path,
            Style::new()
                .fg(theme.palette.cyan)
                .add_modifier(Modifier::BOLD),
        ),
    ])
}

fn split_code_line(
    line_no: Option<u32>,
    width: usize,
    marker: char,
    text: &str,
    text_style: Style,
    gutter_style: Style,
    bg: Option<ratatui::style::Color>,
) -> Line<'_> {
    let apply_bg = |style: Style| match bg {
        Some(bg) => style.bg(bg),
        None => style,
    };
    let number = line_no.map_or_else(String::new, |value| value.to_string());
    Line::from(vec![
        Span::styled(format!("{number:>width$} "), apply_bg(gutter_style)),
        Span::styled(marker.to_string(), apply_bg(text_style)),
        Span::styled("\u{2502} ".to_string(), apply_bg(gutter_style)),
        Span::styled(text, apply_bg(text_style)),
    ])
}

fn split_line_number_width(start: u32, len: u32) -> usize {
    let end = start.saturating_add(len.saturating_sub(1)).max(start);
    end.to_string().len().max(4)
}

fn split_range_label(start: u32, len: u32) -> String {
    if len <= 1 {
        start.to_string()
    } else {
        format!("{}-{}", start, start.saturating_add(len.saturating_sub(1)))
    }
}

// ============================================================================
// Unified `git diff` text → DiffView parser
// ============================================================================

/// Parse `git diff` unified output into one [`DiffView`] per file.
///
/// Recognises `diff --git a/… b/…` file boundaries, `--- a/…` / `+++ b/…`
/// path lines (mapping `/dev/null` to "no path" for adds/deletes), and
/// `@@ -a,b +c,d @@` hunk headers. Body lines are classified by their
/// leading `+`/`-`/space marker (stripped from the stored text). Anything
/// outside a hunk (index lines, mode changes, "\ No newline…") is ignored.
#[must_use]
pub fn parse_unified_diff(text: &str) -> Vec<DiffView> {
    let mut files: Vec<DiffView> = Vec::new();
    let mut cur: Option<DiffView> = None;
    let mut hunk: Option<DiffHunk> = None;

    let flush_hunk = |cur: &mut Option<DiffView>, hunk: &mut Option<DiffHunk>| {
        if let (Some(view), Some(h)) = (cur.as_mut(), hunk.take()) {
            view.hunks.push(h);
        }
    };

    for line in text.lines() {
        if line.starts_with("diff --git") {
            flush_hunk(&mut cur, &mut hunk);
            if let Some(view) = cur.take() {
                files.push(view);
            }
            cur = Some(DiffView {
                old_path: None,
                new_path: None,
                language: None,
                hunks: Vec::new(),
            });
        } else if let Some(path) = line.strip_prefix("--- ") {
            if let Some(view) = cur.as_mut() {
                view.old_path = clean_diff_path(path);
            }
        } else if let Some(path) = line.strip_prefix("+++ ") {
            if let Some(view) = cur.as_mut() {
                let p = clean_diff_path(path);
                view.language = p.as_deref().and_then(detect_language);
                view.new_path = p;
            }
        } else if let Some(header) = line.strip_prefix("@@") {
            flush_hunk(&mut cur, &mut hunk);
            hunk = parse_hunk_header(header);
        } else if let Some(h) = hunk.as_mut() {
            match line.as_bytes().first() {
                Some(b'+') => h.lines.push(DiffLine {
                    kind: DiffLineKind::Added,
                    text: line[1..].to_string(),
                }),
                Some(b'-') => h.lines.push(DiffLine {
                    kind: DiffLineKind::Removed,
                    text: line[1..].to_string(),
                }),
                Some(b' ') => h.lines.push(DiffLine {
                    kind: DiffLineKind::Context,
                    text: line[1..].to_string(),
                }),
                // "\ No newline at end of file" and blank separators.
                _ => {}
            }
        }
    }
    flush_hunk(&mut cur, &mut hunk);
    if let Some(view) = cur.take() {
        files.push(view);
    }
    // Drop files that ended up with no hunks (pure mode/rename changes).
    files.retain(|v| !v.hunks.is_empty());
    files
}

/// `a/src/main.rs` / `b/src/main.rs` → `src/main.rs`; `/dev/null` → None.
/// Trailing tab-separated metadata (timestamps) is dropped.
fn clean_diff_path(raw: &str) -> Option<String> {
    let path = raw.split('\t').next().unwrap_or(raw).trim();
    if path == "/dev/null" {
        return None;
    }
    let stripped = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path);
    Some(stripped.to_string())
}

/// Parse `-a,b +c,d @@ …` (the part after the leading `@@`).
fn parse_hunk_header(header: &str) -> Option<DiffHunk> {
    let inner = header.trim_start_matches(['@', ' ']);
    let mut parts = inner.split_whitespace();
    let old = parts.next()?.strip_prefix('-')?;
    let new = parts.next()?.strip_prefix('+')?;
    let (old_start, old_lines) = parse_range(old);
    let (new_start, new_lines) = parse_range(new);
    Some(DiffHunk {
        old_start,
        old_lines,
        new_start,
        new_lines,
        lines: Vec::new(),
    })
}

/// `12,5` → `(12, 5)`; `12` → `(12, 1)`.
fn parse_range(s: &str) -> (u32, u32) {
    let mut it = s.split(',');
    let start = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let count = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
    (start, count)
}

fn detect_language(path: &str) -> Option<String> {
    let ext = std::path::Path::new(path).extension()?.to_str()?;
    Some(
        match ext {
            "rs" => "rust",
            "ts" | "tsx" => "typescript",
            "js" | "jsx" => "javascript",
            "py" => "python",
            "go" => "go",
            "md" => "markdown",
            "toml" => "toml",
            "json" => "json",
            "sh" => "bash",
            other => other,
        }
        .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        DiffViewerAction, DiffViewerModal, clamped_scroll, display_width, footer_line,
        parse_unified_diff, scroll_progress_label, split_lines, split_lines_for_width,
    };
    use crate::tui::theme::Theme;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;
    use ratatui::text::Line;
    use runtime::message_stream::{DiffHunk, DiffLine, DiffLineKind, DiffView};

    const SAMPLE: &str = "diff --git a/src/main.rs b/src/main.rs\n\
index abc..def 100644\n\
--- a/src/main.rs\n\
+++ b/src/main.rs\n\
@@ -1,3 +1,4 @@\n\
 fn main() {\n\
-    old();\n\
+    new();\n\
+    extra();\n\
 }\n\
diff --git a/README.md b/README.md\n\
--- a/README.md\n\
+++ b/README.md\n\
@@ -10 +10 @@\n\
-old title\n\
+new title\n";

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    fn dump_buffer(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut dump = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        dump
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn parses_multiple_files_and_hunks() {
        let files = parse_unified_diff(SAMPLE);
        assert_eq!(files.len(), 2, "two files");
        assert_eq!(files[0].new_path.as_deref(), Some("src/main.rs"));
        assert_eq!(files[0].language.as_deref(), Some("rust"));
        assert_eq!(files[0].hunks.len(), 1);
        let lines = &files[0].hunks[0].lines;
        let added = lines
            .iter()
            .filter(|l| l.kind == DiffLineKind::Added)
            .count();
        let removed = lines
            .iter()
            .filter(|l| l.kind == DiffLineKind::Removed)
            .count();
        assert_eq!((added, removed), (2, 1));
        // Markers are stripped from the stored text.
        assert!(lines.iter().any(|l| l.text == "    new();"));
        // Single-number range `@@ -10 +10 @@` defaults the count to 1.
        assert_eq!(files[1].hunks[0].old_lines, 1);
    }

    #[test]
    fn dev_null_paths_become_none() {
        let add =
            "diff --git a/new.txt b/new.txt\n--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+hello\n";
        let files = parse_unified_diff(add);
        assert_eq!(files[0].old_path, None);
        assert_eq!(files[0].new_path.as_deref(), Some("new.txt"));
    }

    #[test]
    fn navigation_switches_files_and_resets_scroll() {
        let mut modal = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        assert_eq!(modal.file_count(), 2);
        modal.handle_key(press(KeyCode::Down));
        assert!(modal.scroll > 0);
        modal.handle_key(press(KeyCode::Right));
        assert_eq!(modal.selected(), 1);
        assert_eq!(modal.scroll, 0, "switching files resets scroll");
        modal.handle_key(press(KeyCode::Right)); // clamp at last
        assert_eq!(modal.selected(), 1);
        assert_eq!(
            modal.handle_key(press(KeyCode::Esc)),
            Some(DiffViewerAction::Close)
        );
    }

    #[test]
    fn side_by_side_toggles_and_resets_scroll() {
        let mut modal = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        assert!(
            modal.is_side_by_side(),
            "defaults to split preference (opencode auto)"
        );

        // Scroll, then toggle: preference flips and scroll resets to top so
        // the re-flowed body starts clean.
        modal.handle_key(press(KeyCode::Down));
        assert!(modal.scroll > 0);
        assert_eq!(modal.handle_key(press(KeyCode::Char('s'))), None);
        assert!(!modal.is_side_by_side(), "s forces unified");
        assert_eq!(modal.scroll, 0, "toggling resets scroll");

        // Toggling again returns to split; navigation/close keys behave the
        // same in either mode.
        modal.handle_key(press(KeyCode::Char('s')));
        assert!(modal.is_side_by_side(), "s toggles back to split");
    }

    #[test]
    fn revert_emits_selected_path() {
        let mut modal = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        assert_eq!(
            modal.handle_key(press(KeyCode::Char('r'))),
            Some(DiffViewerAction::RevertFile("src/main.rs".to_string())),
            "r reverts the first file"
        );
        // After moving to the second file, revert targets it instead.
        modal.handle_key(press(KeyCode::Right));
        assert_eq!(
            modal.handle_key(press(KeyCode::Char('r'))),
            Some(DiffViewerAction::RevertFile("README.md".to_string())),
        );
    }

    #[test]
    fn split_lines_align_columns_and_pair_changes() {
        let files = parse_unified_diff(SAMPLE);
        let theme = Theme::zo();
        let (left, right) = split_lines(&files[0], &theme);
        // Columns are padded to equal height so they scroll in lockstep.
        assert_eq!(left.len(), right.len(), "columns stay aligned");
        // The single removed `old()` pairs beside the first added `new()`;
        // the extra added line falls on the next right-column row while the
        // left side is blank-padded.
        let left_text: Vec<String> = left
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let right_text: Vec<String> = right
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            left_text[0].contains("before · src/main.rs"),
            "left column should identify the old side first: {left_text:?}"
        );
        assert!(
            right_text[0].contains("after · src/main.rs"),
            "right column should identify the new side first: {right_text:?}"
        );
        assert!(
            left_text
                .iter()
                .any(|t| t.contains("Hunk 1 · old 1-3 · -1")),
            "left hunk header explains old range and removals: {left_text:?}"
        );
        assert!(
            right_text
                .iter()
                .any(|t| t.contains("Hunk 1 · new 1-4 · +2")),
            "right hunk header explains new range and additions: {right_text:?}"
        );
        assert!(
            left_text.iter().any(|t| t.contains("   1 -│     old();")),
            "left column should show old line number and removal marker: {left_text:?}"
        );
        assert!(
            right_text.iter().any(|t| t.contains("   1 +│     new();")),
            "right column should show new line number and addition marker: {right_text:?}"
        );
        assert!(right_text.iter().any(|t| t.contains("new();")));
        assert!(right_text.iter().any(|t| t.contains("extra();")));
    }

    #[test]
    fn split_column_headers_truncate_long_paths_to_column_width() {
        let theme = Theme::no_color();
        let view = DiffView {
            old_path: Some(
                "crates/zo-cli/src/tui/very/deeply/nested/before_component.rs"
                    .to_string(),
            ),
            new_path: Some(
                "crates/zo-cli/src/tui/very/deeply/nested/after_component.rs".to_string(),
            ),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 1,
                new_start: 1,
                new_lines: 1,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "old();".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "new();".to_string(),
                    },
                ],
            }],
        };

        let (left, right) = split_lines_for_width(&view, &theme, 28, 28);
        let left_header = line_text(&left[0]);
        let right_header = line_text(&right[0]);

        assert!(
            display_width(&left_header) <= 28,
            "left split header should fit its column: {left_header:?}"
        );
        assert!(
            display_width(&right_header) <= 28,
            "right split header should fit its column: {right_header:?}"
        );
        assert!(
            left_header.starts_with("before · ") && right_header.starts_with("after · "),
            "side labels should remain visible: {left_header:?} / {right_header:?}"
        );
        assert!(
            left_header.contains('\u{2026}') && right_header.contains('\u{2026}'),
            "long paths should truncate with an ellipsis: {left_header:?} / {right_header:?}"
        );
    }

    #[test]
    fn draws_both_modes_without_panic() {
        let theme = Theme::zo();
        // 120 cols clears MIN_SPLIT_WIDTH so both render paths are exercised.
        let backend = TestBackend::new(120, 24);
        let mut term = Terminal::new(backend).expect("backend");

        // Default preference is split; on a wide pane it renders split.
        let split = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        term.draw(|f| split.draw(f, Rect::new(0, 0, 120, 24), &theme))
            .expect("draw side-by-side");

        // `s` forces unified.
        let mut unified = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        unified.handle_key(press(KeyCode::Char('s')));
        term.draw(|f| unified.draw(f, Rect::new(0, 0, 120, 24), &theme))
            .expect("draw unified");
    }

    /// Dump the `TestBackend` buffer and assert the split view actually
    /// renders a vertical separator column (the "eye" the render-dump
    /// convention replaces) and the before/after text on both sides.
    #[test]
    fn side_by_side_dump_has_separator_and_both_sides() {
        let theme = Theme::zo();
        // 120 cols (≥ MIN_SPLIT_WIDTH) so the default split preference renders.
        let backend = TestBackend::new(120, 24);
        let mut term = Terminal::new(backend).expect("backend");
        let split = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        term.draw(|f| split.draw(f, Rect::new(0, 0, 120, 24), &theme))
            .expect("draw");

        let buf = term.backend().buffer();
        let mut dump = String::new();
        // The rounded modal border draws a `│` on the far-left and
        // far-right of every row, so a row with the interior separator has
        // *three* verticals — more than any unified row could show.
        let mut max_verticals_per_row = 0usize;
        for y in 0..buf.area.height {
            let mut row_verticals = 0usize;
            for x in 0..buf.area.width {
                let sym = buf[(x, y)].symbol();
                dump.push_str(sym);
                if sym == "\u{2502}" {
                    row_verticals += 1;
                }
            }
            dump.push('\n');
            max_verticals_per_row = max_verticals_per_row.max(row_verticals);
        }
        assert!(
            max_verticals_per_row >= 3,
            "split view adds an interior separator column (got {max_verticals_per_row} verticals)"
        );
        // before (removed `old()`) on the left, after (`new()`) on the right.
        assert!(dump.contains("before"), "before column label present");
        assert!(dump.contains("after"), "after column label present");
        assert!(dump.contains("old();"), "before text present");
        assert!(dump.contains("new();"), "after text present");
    }

    /// Locks the visible `/diff` contract: the first viewport must make the
    /// file identity, changed code, and navigation/footer controls distinct.
    #[test]
    fn render_dump_keeps_header_body_and_footer_visible() {
        let theme = Theme::no_color();
        let backend = TestBackend::new(120, 14);
        let mut term = Terminal::new(backend).expect("backend");
        let modal = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 14), &theme))
            .expect("draw");

        let dump = dump_buffer(&term);
        assert!(dump.contains("/diff"), "modal title missing: {dump}");
        assert!(dump.contains("modified"), "change kind missing: {dump}");
        assert!(dump.contains("src/main.rs"), "file path missing: {dump}");
        assert!(dump.contains("1 hunk"), "hunk count missing: {dump}");
        assert!(dump.contains("+2 -1"), "tally missing: {dump}");
        assert!(dump.contains("old();"), "removed code missing: {dump}");
        assert!(dump.contains("new();"), "added code missing: {dump}");
        assert!(dump.contains("PgUp/PgDn"), "page hint missing: {dump}");
        assert!(dump.contains("Esc close"), "close hint missing: {dump}");
    }

    #[test]
    fn header_truncates_long_path_but_keeps_review_metrics() {
        let view = DiffView {
            old_path: Some(
                "crates/zo-cli/src/tui/very/deeply/nested/component.rs".to_string(),
            ),
            new_path: Some(
                "crates/zo-cli/src/tui/very/deeply/nested/component.rs".to_string(),
            ),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 1,
                new_start: 1,
                new_lines: 1,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "old();".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "new();".to_string(),
                    },
                ],
            }],
        };
        let modal = DiffViewerModal::new(vec![view]);
        let line = modal.header_line(&Theme::no_color(), 54);
        let text = line_text(&line);

        assert!(
            display_width(&text) <= 54,
            "header should fit the target width: {text:?}"
        );
        assert!(text.contains("modified"), "change kind missing: {text:?}");
        assert!(text.contains("1 hunk"), "hunk count missing: {text:?}");
        assert!(text.contains("+1 -1"), "tally missing: {text:?}");
        assert!(
            text.contains('\u{2026}'),
            "long path should be visibly truncated, not silently dropped: {text:?}"
        );
    }

    /// opencode parity: the split renders only when the pane clears
    /// `MIN_SPLIT_WIDTH` (100). A narrow pane downgrades to unified even
    /// with the split preference on; a user-forced unified ignores width.
    #[test]
    fn narrow_pane_forces_unified_despite_split_preference() {
        let modal = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        assert!(modal.is_side_by_side(), "prefers split by default (auto)");
        assert!(!modal.renders_split(99), "99 < 100 → unified");
        assert!(modal.renders_split(100), "100 clears the gate");
        assert!(modal.renders_split(140), "wide pane splits");

        // A user who forced unified stays unified even on a wide pane.
        let mut unified = DiffViewerModal::new(parse_unified_diff(SAMPLE));
        unified.handle_key(press(KeyCode::Char('s')));
        assert!(!unified.renders_split(140), "forced-unified ignores width");
    }

    #[test]
    fn overflowing_unified_diff_draws_scrollbar() {
        let mut diff = String::from(
            "diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1,32 @@\n-old\n",
        );
        for line in 0..32 {
            use std::fmt::Write as _;
            let _ = writeln!(diff, "+new line {line}");
        }

        let mut modal = DiffViewerModal::new(parse_unified_diff(&diff));
        modal.handle_key(press(KeyCode::Char('s'))); // force unified
        let theme = Theme::no_color();
        let backend = TestBackend::new(90, 12);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| modal.draw(f, Rect::new(0, 0, 90, 12), &theme))
            .expect("draw with scrollbar");

        let dump = dump_buffer(&term);
        assert!(dump.contains('#'), "scrollbar thumb should render: {dump}");
        assert!(dump.contains('.'), "scrollbar track should render: {dump}");
    }

    #[test]
    fn scroll_is_clamped_to_available_diff_rows() {
        assert_eq!(clamped_scroll(99, 40, 10), 30);
        assert_eq!(clamped_scroll(4, 40, 10), 4);
        assert_eq!(clamped_scroll(8, 4, 10), 0);
    }

    #[test]
    fn footer_advertises_page_navigation() {
        let theme = Theme::no_color();
        let text: String = footer_line(&theme, true, true, 0, 40, 10, 120)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("split"), "footer: {text}");
        assert!(text.contains("top"), "footer: {text}");
        assert!(text.contains("rows 1-10/40"), "footer: {text}");
        assert!(text.contains("100% left"), "footer: {text}");
        assert!(text.contains("PgUp/PgDn"), "footer: {text}");
        assert!(text.contains("page"), "footer: {text}");
    }

    #[test]
    fn footer_compacts_before_it_drops_progress_or_close() {
        let theme = Theme::no_color();
        let line = footer_line(&theme, true, true, 0, 40, 10, 52);
        let text: String = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();

        assert!(
            super::line_width(&line) <= 52,
            "footer should fit compact width: {text:?}"
        );
        assert!(text.contains("100% left"), "progress survives: {text}");
        assert!(text.contains("Esc close"), "close hint survives: {text}");
        assert!(
            !text.contains("PgUp/PgDn"),
            "low-priority page hint should be dropped first: {text}"
        );
    }

    #[test]
    fn footer_reports_scroll_progress() {
        assert_eq!(
            scroll_progress_label(0, 4, 10),
            "all 4 rows",
            "short content should not pretend to have remaining rows"
        );
        assert_eq!(
            scroll_progress_label(15, 40, 10),
            "rows 16-25/40 · 50% left"
        );
        assert_eq!(
            scroll_progress_label(99, 40, 10),
            "end · rows 31-40/40 · 0% left"
        );
    }
}
