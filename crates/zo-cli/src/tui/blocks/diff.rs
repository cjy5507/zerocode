//! Unified-diff viewer for `ToolResultBody::Diff` with syntax highlighting.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use runtime::message_stream::{DiffHunk, DiffLineKind, DiffView};
use unicode_width::UnicodeWidthStr;

use super::compact_path_label;
use super::text::syntect_assets;
use crate::tui::markdown::syntax::SyntaxHighlighter;
use crate::tui::theme::Theme;

/// Count `(additions, removals)` across every hunk in `view`.
#[must_use]
pub fn tally(view: &DiffView) -> (usize, usize) {
    let mut add = 0;
    let mut rem = 0;
    for hunk in &view.hunks {
        for line in &hunk.lines {
            match line.kind {
                DiffLineKind::Added => add += 1,
                DiffLineKind::Removed => rem += 1,
                DiffLineKind::Context => {}
            }
        }
    }
    (add, rem)
}

/// Count hunks in a diff view.
#[must_use]
pub fn hunk_count(view: &DiffView) -> usize {
    view.hunks.len()
}

/// User-facing change kind for a file-level diff.
#[must_use]
pub fn change_kind(view: &DiffView) -> &'static str {
    match (view.old_path.as_deref(), view.new_path.as_deref()) {
        (None, Some(_)) => "added",
        (Some(_), None) => "deleted",
        (Some(old), Some(new)) if old != new => "renamed",
        _ => "modified",
    }
}

/// Compact one-line label for the diff header.
#[must_use]
pub fn file_header(view: &DiffView) -> String {
    let (adds, rems) = tally(view);
    format!("{} (+{adds} -{rems})", path_label(view))
}

/// Number of leading rows emitted by [`lines`] before hunk content.
#[must_use]
pub const fn header_rows() -> usize {
    1
}

/// Number of rows [`lines`] emits when `expanded == true`, without styling.
#[must_use]
pub fn rendered_line_count(view: &DiffView) -> usize {
    header_rows()
        + view.hunks.iter().map(|hunk| hunk.lines.len()).sum::<usize>()
        + view.hunks.len().saturating_sub(1)
}

/// Render the diff as a flat list of styled lines.
///
/// When `expanded == true`, emits the full hunks; when `false`,
/// emits only a summary line (caller typically omits calling this
/// in the collapsed case).
#[must_use]
#[allow(clippy::too_many_lines)] // one pass over hunks; gutter/marker/cell styling belongs together
pub fn lines<'a>(view: &'a DiffView, theme: &Theme, expanded: bool) -> Vec<Line<'a>> {
    let mut out: Vec<Line<'_>> = Vec::new();
    out.push(Line::styled(
        file_header(view),
        theme.diff_file_header_style(),
    ));
    if !expanded {
        return out;
    }
    let (ss, _syn_theme) = syntect_assets();
    let file_path = view
        .new_path
        .as_deref()
        .or(view.old_path.as_deref())
        .unwrap_or("");
    let ext = std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str());
    let syntax = ext
        .and_then(|e| ss.find_syntax_by_extension(e))
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut highlighter = SyntaxHighlighter::new(syntax);

    let number_width = view
        .hunks
        .iter()
        .map(|hunk| {
            line_number_width(hunk.old_start, hunk.old_lines)
                .max(line_number_width(hunk.new_start, hunk.new_lines))
        })
        .max()
        .unwrap_or(1);

    for (idx, hunk) in view.hunks.iter().enumerate() {
        if idx > 0 {
            out.push(Line::default());
        }
        let emphasis = word_emphasis_ranges(hunk);
        let mut old_line = hunk.old_start;
        let mut new_line = hunk.new_start;
        for (line_idx, line) in hunk.lines.iter().enumerate() {
            let (gutter_char, gutter_style) = match line.kind {
                DiffLineKind::Added => ('+', theme.diff_add_style()),
                DiffLineKind::Removed => ('-', theme.diff_del_style()),
                DiffLineKind::Context => (' ', theme.diff_context_style()),
            };
            let display_line = match line.kind {
                DiffLineKind::Added | DiffLineKind::Context => new_line,
                DiffLineKind::Removed => old_line,
            };

            // Semantic row-wide background tint for changed lines so an add/remove
            // reads as a band, not just a colored gutter. Restored on indexed
            // palettes too (roadmap ⑦, quantized into the indexed space); only a
            // `NO_COLOR`/Reset palette has no blend → foreground-only there.
            let line_bg = match line.kind {
                DiffLineKind::Added => theme.diff_add_bg(),
                DiffLineKind::Removed => theme.diff_del_bg(),
                DiffLineKind::Context => None,
            };
            let apply_bg = |style: Style| match line_bg {
                Some(bg) => style.bg(bg),
                None => style,
            };
            let gutter_dim = apply_bg(Style::new().fg(theme.palette.dim));
            let separator = " ";

            let mut spans = vec![
                Span::styled(format!("{display_line:>number_width$}"), gutter_dim),
                Span::styled(" ", gutter_dim),
                Span::styled(String::from(gutter_char), apply_bg(gutter_style)),
                Span::styled(separator, gutter_dim),
            ];

            let emphasis_bg = match line.kind {
                DiffLineKind::Added => theme.diff_add_emphasis_bg(),
                DiffLineKind::Removed => theme.diff_del_emphasis_bg(),
                DiffLineKind::Context => None,
            };
            let changed = emphasis.get(line_idx).and_then(Option::as_deref);

            let mut byte_off = 0usize;
            for (role, segment) in highlighter.highlight_line(&line.text, ss) {
                let raw = segment.trim_end_matches('\n');
                if raw.is_empty() {
                    continue;
                }
                // Color the token from the zo palette (via its scope role)
                // instead of syntect's base16 RGB, so a diff card highlights in
                // the same palette as prose code fences and degrades on
                // 256-color / NO_COLOR terminals.
                let rs = theme.syntax_style(role);
                spans.extend(emphasized_code_spans(
                    raw,
                    byte_off,
                    changed,
                    rs,
                    line_bg,
                    emphasis_bg,
                ));
                byte_off += raw.len();
            }

            out.push(Line::from(spans));
            match line.kind {
                DiffLineKind::Added => new_line = new_line.saturating_add(1),
                DiffLineKind::Removed => old_line = old_line.saturating_add(1),
                DiffLineKind::Context => {
                    old_line = old_line.saturating_add(1);
                    new_line = new_line.saturating_add(1);
                }
            }
        }
    }
    out
}

/// Per-line byte ranges to emphasize within a hunk: word-level intra-line
/// diffing (Claude-Code parity). For each maximal run of removed lines
/// immediately followed by added lines, each removed line is paired with an
/// added line by position and the differing word tokens get their byte ranges
/// returned so the renderer can paint a stronger background only on the words
/// that actually changed. `None` for a line means "no intra-line emphasis"
/// (whole-line add/remove, or context). The returned vec is parallel to
/// `hunk.lines`.
fn word_emphasis_ranges(hunk: &DiffHunk) -> Vec<Option<Vec<(usize, usize)>>> {
    let mut ranges: Vec<Option<Vec<(usize, usize)>>> = vec![None; hunk.lines.len()];
    let mut i = 0;
    while i < hunk.lines.len() {
        if hunk.lines[i].kind != DiffLineKind::Removed {
            i += 1;
            continue;
        }
        let rem_start = i;
        while i < hunk.lines.len() && hunk.lines[i].kind == DiffLineKind::Removed {
            i += 1;
        }
        let add_start = i;
        while i < hunk.lines.len() && hunk.lines[i].kind == DiffLineKind::Added {
            i += 1;
        }
        let rem_count = add_start - rem_start;
        let add_count = i - add_start;
        // Only pair when the block is a clean N→N replacement; mismatched
        // counts read better as whole-line bands than as guessed word pairs.
        if rem_count == 0 || rem_count != add_count {
            continue;
        }
        for k in 0..rem_count {
            let old_text = &hunk.lines[rem_start + k].text;
            let new_text = &hunk.lines[add_start + k].text;
            let (old_ranges, new_ranges) = word_diff_ranges(old_text, new_text);
            // Suppress emphasis when essentially the whole line differs — the
            // line-wide band already conveys that; word marks would be noise.
            if !word_ranges_cover_most(&old_ranges, old_text) {
                ranges[rem_start + k] = Some(old_ranges);
            }
            if !word_ranges_cover_most(&new_ranges, new_text) {
                ranges[add_start + k] = Some(new_ranges);
            }
        }
    }
    ranges
}

/// True when emphasized ranges cover most of the line's non-whitespace bytes
/// (≥ 85%), i.e. the lines barely share tokens, so word emphasis is just noise.
fn word_ranges_cover_most(ranges: &[(usize, usize)], text: &str) -> bool {
    let content: usize = text.chars().filter(|c| !c.is_whitespace()).count();
    if content == 0 {
        return true;
    }
    let covered: usize = ranges
        .iter()
        .map(|&(s, e)| {
            text.get(s..e).map_or(0, |slice| {
                slice.chars().filter(|c| !c.is_whitespace()).count()
            })
        })
        .sum();
    covered * 100 >= content * 85
}

/// Byte ranges within one line that are unique to the old side and the new
/// side respectively (parallel halves of a word-level diff).
type WordDiffRanges = (Vec<(usize, usize)>, Vec<(usize, usize)>);

/// Token-level diff of two lines, returning the byte ranges that are unique to
/// `old` and unique to `new` respectively. Tokenizes into word / non-word runs
/// (so identifiers and operators align) and runs an LCS so shared tokens are
/// skipped and only genuine edits are marked.
fn word_diff_ranges(old: &str, new: &str) -> WordDiffRanges {
    let old_tokens = tokenize_words(old);
    let new_tokens = tokenize_words(new);
    let lcs = lcs_token_table(&old_tokens, &new_tokens);

    let mut old_ranges = Vec::new();
    let mut new_ranges = Vec::new();
    let (mut oi, mut ni) = (0usize, 0usize);
    while oi < old_tokens.len() && ni < new_tokens.len() {
        if old_tokens[oi].2 == new_tokens[ni].2 {
            oi += 1;
            ni += 1;
        } else if lcs[oi + 1][ni] >= lcs[oi][ni + 1] {
            push_range(&mut old_ranges, old_tokens[oi].0, old_tokens[oi].1);
            oi += 1;
        } else {
            push_range(&mut new_ranges, new_tokens[ni].0, new_tokens[ni].1);
            ni += 1;
        }
    }
    while oi < old_tokens.len() {
        push_range(&mut old_ranges, old_tokens[oi].0, old_tokens[oi].1);
        oi += 1;
    }
    while ni < new_tokens.len() {
        push_range(&mut new_ranges, new_tokens[ni].0, new_tokens[ni].1);
        ni += 1;
    }
    (old_ranges, new_ranges)
}

/// Append `(start, end)`, merging into the previous range when adjacent so
/// runs of changed tokens become one span (fewer, cleaner emphasis spans).
fn push_range(ranges: &mut Vec<(usize, usize)>, start: usize, end: usize) {
    if let Some(last) = ranges.last_mut() {
        if last.1 == start {
            last.1 = end;
            return;
        }
    }
    ranges.push((start, end));
}

/// Tokenize into `(start, end, text)` byte-range runs: maximal runs of
/// word characters (alphanumeric + `_`) or single non-word, non-space chars,
/// with whitespace as its own token so alignment stays stable.
fn tokenize_words(text: &str) -> Vec<(usize, usize, &str)> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        if bytes[i] >= 0x80 {
            // Multi-byte UTF-8 scalar: consume the whole char as one token.
            let ch_len = text[i..].chars().next().map_or(1, char::len_utf8);
            i += ch_len;
        } else if is_word(bytes[i]) {
            while i < bytes.len() && bytes[i] < 0x80 && is_word(bytes[i]) {
                i += 1;
            }
        } else if bytes[i].is_ascii_whitespace() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
        } else {
            i += 1;
        }
        tokens.push((start, i, &text[start..i]));
    }
    tokens
}

/// LCS length table over token text. `table[i][j]` = LCS length of
/// `old[i..]` and `new[j..]`, so the diff walk above can pick the direction
/// that preserves the most shared tokens.
fn lcs_token_table(old: &[(usize, usize, &str)], new: &[(usize, usize, &str)]) -> Vec<Vec<u16>> {
    let mut table = vec![vec![0u16; new.len() + 1]; old.len() + 1];
    for i in (0..old.len()).rev() {
        for j in (0..new.len()).rev() {
            table[i][j] = if old[i].2 == new[j].2 {
                table[i + 1][j + 1].saturating_add(1)
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    table
}

/// Split a syntect-highlighted code segment (already styled with `base`) into
/// spans, applying the stronger `emphasis_bg` to any byte sub-range that
/// `changed` marks as a word-level edit, and `line_bg` elsewhere. `seg_off` is
/// the segment's byte offset within the full line, so the line-relative
/// `changed` ranges line up.
fn emphasized_code_spans<'a>(
    text: &str,
    seg_off: usize,
    changed: Option<&[(usize, usize)]>,
    base: Style,
    line_bg: Option<Color>,
    emphasis_bg: Option<Color>,
) -> Vec<Span<'a>> {
    let with_line_bg = |style: Style| match line_bg {
        Some(bg) => style.bg(bg),
        None => style,
    };
    let Some(changed) = changed.filter(|ranges| !ranges.is_empty()) else {
        return vec![Span::styled(text.to_string(), with_line_bg(base))];
    };
    let emphasis_style = match emphasis_bg.or(line_bg) {
        Some(bg) => base.bg(bg),
        None => base.add_modifier(Modifier::BOLD),
    };

    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut buf_emph = false;
    for (rel, ch) in text.char_indices() {
        let abs = seg_off + rel;
        let emph = changed.iter().any(|&(s, e)| abs >= s && abs < e);
        if emph != buf_emph && !buf.is_empty() {
            let style = if buf_emph {
                emphasis_style
            } else {
                with_line_bg(base)
            };
            spans.push(Span::styled(std::mem::take(&mut buf), style));
        }
        buf_emph = emph;
        buf.push(ch);
    }
    if !buf.is_empty() {
        let style = if buf_emph {
            emphasis_style
        } else {
            with_line_bg(base)
        };
        spans.push(Span::styled(buf, style));
    }
    spans
}

/// Render only hunk content, omitting the file metadata rows.
///
/// `ToolResult` summaries already carry file identity and change tallies, so
/// repeating the same header row before the hunk body delays the useful
/// code changes. The full `/diff` modal still uses [`lines`] directly.
#[must_use]
pub fn body_lines<'a>(view: &'a DiffView, theme: &Theme) -> Vec<Line<'a>> {
    let mut out = lines(view, theme, true);
    let drop = header_rows().min(out.len());
    out.drain(..drop);
    out
}

/// Extend changed-line backgrounds to the full review width. The tree rail or
/// modal chrome stays untouched; only the diff row becomes a continuous band.
pub(crate) fn pad_changed_rows(lines: &mut [Line<'_>], width: usize) {
    for line in lines {
        let Some(bg) = line.spans.iter().find_map(|span| span.style.bg) else {
            continue;
        };
        let used = line
            .spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum::<usize>();
        if used < width {
            line.spans.push(Span::styled(
                " ".repeat(width - used),
                Style::new().bg(bg),
            ));
        }
    }
}

#[must_use]
pub fn path_label(view: &DiffView) -> String {
    match (view.old_path.as_deref(), view.new_path.as_deref()) {
        (Some(old), Some(new)) if old != new => {
            format!("{} -> {}", compact_path_label(old), compact_path_label(new))
        }
        (_, Some(new)) => compact_path_label(new),
        (Some(old), None) => compact_path_label(old),
        (None, None) => "<unknown>".to_string(),
    }
}

fn line_number_width(start: u32, len: u32) -> usize {
    let end = start.saturating_add(len.saturating_sub(1)).max(start);
    end.to_string().len().max(1)
}

#[must_use]
pub fn hunk_label(view: &DiffView) -> String {
    let hunks = hunk_count(view);
    if hunks == 1 {
        "1 hunk".to_string()
    } else {
        format!("{hunks} hunks")
    }
}

/// Count added/removed lines in a single hunk. Shared with the full-screen
/// diff viewer so the inline summary and the modal report the same tallies.
pub(crate) fn hunk_tally(hunk: &DiffHunk) -> (usize, usize) {
    let mut add = 0;
    let mut rem = 0;
    for line in &hunk.lines {
        match line.kind {
            DiffLineKind::Added => add += 1,
            DiffLineKind::Removed => rem += 1,
            DiffLineKind::Context => {}
        }
    }
    (add, rem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::Theme;
    use runtime::message_stream::{DiffHunk, DiffLine};

    fn sample_view() -> DiffView {
        DiffView {
            old_path: Some("src/lib.rs".to_string()),
            new_path: Some("src/lib.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 1,
                new_start: 1,
                new_lines: 2,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Context,
                        text: "fn main() {".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "    let x = 1;".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "    let x = 0;".to_string(),
                    },
                ],
            }],
        }
    }

    #[test]
    fn diff_rows_match_editor_review_shape() {
        let rendered = lines(&sample_view(), &Theme::no_color(), true)
            .into_iter()
            .map(|line| {
                line.spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert_eq!(rendered[0], "src/lib.rs (+1 -1)");
        assert_eq!(rendered[1], "1   fn main() {");
        assert_eq!(rendered[2], "2 +     let x = 1;");
        assert_eq!(rendered[3], "2 -     let x = 0;");
    }

    /// Every span on an added line carries the add background tint on a
    /// true-color palette; removed lines carry the remove tint; context stays
    /// on the normal editor surface.
    #[test]
    fn changed_lines_get_background_tint_on_true_color() {
        let theme = Theme::zo();
        let add_bg = theme.diff_add_bg().expect("zo has add bg");
        let del_bg = theme.diff_del_bg().expect("zo has del bg");
        let view = sample_view();
        let out = lines(&view, &theme, true);

        let context = &out[1];
        let added = &out[2];
        let removed = &out[3];
        assert!(
            added.spans.iter().all(|s| s.style.bg == Some(add_bg)),
            "all added-line spans must carry the add bg tint"
        );
        assert!(
            removed.spans.iter().all(|s| s.style.bg == Some(del_bg)),
            "all removed-line spans must carry the del bg tint"
        );
        assert!(
            context.spans.iter().all(|s| s.style.bg.is_none()),
            "context rows should stay on the normal editor surface"
        );
    }

    /// Context remains background-free even next to a changed row, matching
    /// editor review views where only additions and removals form color bands.
    #[test]
    fn context_rows_stay_background_free() {
        let theme = Theme::zo();
        let ctx = |text: &str| DiffLine {
            kind: DiffLineKind::Context,
            text: text.to_string(),
        };
        let view = DiffView {
            old_path: Some("src/lib.rs".to_string()),
            new_path: Some("src/lib.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 4,
                new_start: 1,
                new_lines: 4,
                lines: vec![
                    ctx("far context"),
                    ctx("near context above"),
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "    let x = 1;".to_string(),
                    },
                    ctx("near context below"),
                    ctx("far context tail"),
                ],
            }],
        };
        let out = lines(&view, &theme, true);
        let bg_of = |idx: usize| out[1 + idx].spans.last().and_then(|s| s.style.bg);
        assert_eq!(bg_of(0), None);
        assert_eq!(bg_of(1), None);
        assert!(bg_of(2).is_some(), "the added row keeps its color band");
        assert_eq!(bg_of(3), None);
        assert_eq!(bg_of(4), None);
    }

    /// Roadmap ⑦: an indexed palette used to lose the diff line wash entirely
    /// (`diff_add_bg == None`). It is now restored *in the indexed color space*,
    /// so changed lines carry a background that still renders without truecolor.
    /// (A true NO_COLOR / Reset palette still has no blend → stays bg-less.)
    #[test]
    fn changed_lines_have_background_restored_on_indexed_palette() {
        let theme = Theme::default_dark();
        assert!(
            matches!(theme.diff_add_bg(), Some(ratatui::style::Color::Indexed(..))),
            "indexed palette now carries an indexed add bg"
        );
        let view = sample_view();
        let out = lines(&view, &theme, true);
        assert!(
            out[2..]
                .iter()
                .any(|line| line.spans.iter().any(|s| s.style.bg.is_some())),
            "a changed diff line must carry the restored wash on an indexed palette"
        );
    }

    /// The compact file header routes its color through the theme helper.
    #[test]
    fn headers_use_theme_helper_styles() {
        let theme = Theme::zo();
        let view = sample_view();
        let out = lines(&view, &theme, true);
        assert_eq!(
            out[0].style.fg,
            Some(theme.palette.cyan),
            "file header must use the cyan file-header style"
        );
        assert_eq!(out[0].spans.len(), 1);
    }

    #[test]
    fn file_header_names_hunks_and_renames() {
        let mut view = sample_view();
        view.old_path = Some("src/old.rs".to_string());
        view.new_path = Some("src/new.rs".to_string());
        view.hunks.push(DiffHunk {
            old_start: 20,
            old_lines: 1,
            new_start: 21,
            new_lines: 1,
            lines: vec![DiffLine {
                kind: DiffLineKind::Context,
                text: "tail".to_string(),
            }],
        });

        let header = file_header(&view);
        assert_eq!(header, "src/old.rs -> src/new.rs (+1 -1)");
    }

    #[test]
    fn rendered_line_count_matches_expanded_output_rows() {
        let theme = Theme::no_color();
        let view = sample_view();
        assert_eq!(rendered_line_count(&view), lines(&view, &theme, true).len());
    }

    #[test]
    fn file_header_names_added_deleted_and_modified_kinds() {
        let mut added = sample_view();
        added.old_path = None;
        added.new_path = Some("src/new.rs".to_string());
        assert_eq!(change_kind(&added), "added");
        assert_eq!(file_header(&added), "src/new.rs (+1 -1)");

        let mut deleted = sample_view();
        deleted.old_path = Some("src/old.rs".to_string());
        deleted.new_path = None;
        assert_eq!(change_kind(&deleted), "deleted");
        assert_eq!(file_header(&deleted), "src/old.rs (+1 -1)");

        let modified = sample_view();
        assert_eq!(change_kind(&modified), "modified");
        assert_eq!(file_header(&modified), "src/lib.rs (+1 -1)");
    }

    #[test]
    fn path_label_compacts_absolute_paths() {
        let mut view = sample_view();
        view.old_path =
            Some("/Users/joe/2026/zo/crates/zo-cli/src/tui/old.rs".to_string());
        view.new_path =
            Some("/Users/joe/2026/zo/crates/zo-cli/src/tui/new.rs".to_string());

        let path = path_label(&view);
        assert!(
            path.contains(
                "crates/zo-cli/src/tui/old.rs -> crates/zo-cli/src/tui/new.rs"
            ),
            "absolute workspace prefixes should be compacted in diff paths: {path}"
        );
        assert!(
            !path.contains("/Users/joe/2026/zo"),
            "absolute workspace prefix should not dominate the diff header: {path}"
        );
    }

    #[test]
    fn diff_lines_include_old_and_new_line_numbers() {
        let theme = Theme::no_color();
        let view = sample_view();
        let out = lines(&view, &theme, true);
        let rendered = out
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("1   fn main() {"));
        assert!(rendered.contains("2 +     let x = 1;"));
        assert!(rendered.contains("2 -     let x = 0;"));
    }

    /// Word-level emphasis: on a paired remove→add replacement, only the
    /// changed tokens carry the stronger emphasis background; the shared prefix
    /// keeps the line-wide tint. Zo is a true-color palette so the tints are
    /// defined (distinct from `None`).
    #[test]
    fn paired_replacement_emphasizes_only_changed_words() {
        let theme = Theme::zo();
        let view = DiffView {
            old_path: Some("src/lib.rs".to_string()),
            new_path: Some("src/lib.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 1,
                new_start: 1,
                new_lines: 1,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "let value = compute_old(x);".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "let value = compute_new(x);".to_string(),
                    },
                ],
            }],
        };
        let add_bg = theme.diff_add_bg().expect("zo add bg");
        let add_emph = theme.diff_add_emphasis_bg().expect("zo add emphasis bg");
        assert_ne!(
            add_bg, add_emph,
            "emphasis tint must be stronger than the line band"
        );

        let out = lines(&view, &theme, true);
        // Header, removed line, added line.
        let added = &out[2];
        let has_emphasis = added.spans.iter().any(|s| s.style.bg == Some(add_emph));
        let has_plain_band = added.spans.iter().any(|s| s.style.bg == Some(add_bg));
        assert!(has_emphasis, "changed word must carry the emphasis tint");
        assert!(has_plain_band, "shared text must keep the line-wide band");

        let emphasized: String = added
            .spans
            .iter()
            .filter(|s| s.style.bg == Some(add_emph))
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            emphasized.contains("new"),
            "emphasis covers the edit: {emphasized:?}"
        );
        assert!(
            !emphasized.contains("value"),
            "shared token not emphasized: {emphasized:?}"
        );
    }

    /// `word_diff_ranges` returns only the differing tokens' byte ranges,
    /// skipping the shared longest common token subsequence.
    #[test]
    fn word_diff_ranges_marks_only_changed_tokens() {
        let old = "alpha beta gamma";
        let new = "alpha BETA gamma";
        let (old_r, new_r) = word_diff_ranges(old, new);
        let old_marked: Vec<&str> = old_r.iter().map(|&(s, e)| &old[s..e]).collect();
        let new_marked: Vec<&str> = new_r.iter().map(|&(s, e)| &new[s..e]).collect();
        assert_eq!(
            old_marked,
            vec!["beta"],
            "only old's changed token: {old_marked:?}"
        );
        assert_eq!(
            new_marked,
            vec!["BETA"],
            "only new's changed token: {new_marked:?}"
        );
    }

    /// A whole-line replacement that shares almost no tokens suppresses word
    /// emphasis (the line band already conveys it) rather than marking
    /// everything.
    #[test]
    fn near_total_change_suppresses_word_emphasis() {
        let hunk = DiffHunk {
            old_start: 1,
            old_lines: 1,
            new_start: 1,
            new_lines: 1,
            lines: vec![
                DiffLine {
                    kind: DiffLineKind::Removed,
                    text: "completely different old line".to_string(),
                },
                DiffLine {
                    kind: DiffLineKind::Added,
                    text: "wholly unrelated replacement text".to_string(),
                },
            ],
        };
        let ranges = word_emphasis_ranges(&hunk);
        assert!(
            ranges.iter().all(Option::is_none),
            "near-total change should fall back to the line band: {ranges:?}"
        );
    }

    #[test]
    fn diff_line_number_gutter_expands_for_large_files() {
        let theme = Theme::no_color();
        let mut view = sample_view();
        view.hunks[0].old_start = 12_000;
        view.hunks[0].new_start = 12_500;

        let rendered = lines(&view, &theme, true)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("12500   fn main() {"));
        assert!(rendered.contains("12501 +     let x = 1;"));
        assert!(rendered.contains("12001 -     let x = 0;"));
    }
}
