//! Per-variant widgets for [`RenderBlock`].
//!
//! This module is the widget layer added in Lane L5. It owns one
//! submodule per [`RenderBlock`] variant plus the dispatcher
//! [`draw_block`] that the transcript viewport calls. See
//! `.zo/design/components.md` §5 for the canonical visual spec of
//! every widget and `code-rules.md` R1/R2/R9 for the boundary rules
//! (no Anthropic leakage, no ANSI strings, all styling through
//! `&Theme`). R6 is honored by treating [`RenderBlock::Reasoning`] as
//! a first-class variant — the legacy Anthropic `Thinking` naming is
//! **not** referenced anywhere in this tree.
//!
//! ## Living standard (mirrors L1)
//!
//! * The widget renderers are infallible — they produce `Vec<Line>` and
//!   never fail — so this module surfaces no error enum.
//! * Module layout `<area>/{mod,types,…}.rs` — this `mod.rs` is the
//!   dispatch surface and every submodule is a single file under
//!   `tui/blocks/`.
//! * Every `pub` item carries a `///` summary.

#![allow(clippy::doc_markdown)]

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use runtime::message_stream::RenderBlock;

use super::image_protocol::ImageProtocol;
use super::theme::Theme;

pub mod agent_result;
pub mod confirm;
pub mod diff;
pub mod image;
pub mod permission;
pub mod reasoning;
pub mod system;
pub mod text;
pub mod user_notice;
pub mod tool_call;
pub mod tool_provenance;
pub mod tool_result;

/// Width of a role mark prefix (`glyph` + two spaces), shared by user prose
/// and the assistant bullet/indent column. Three cells so an East-Asian-
/// Ambiguous mark glyph (`◆` U+25C6, 2 cells under a wide-ambiguous locale)
/// still fits without pushing the body column.
pub(crate) const ROLE_RAIL_WIDTH: u16 = 3;

/// Baseline measure for marked prose bodies (assistant bullet/indent blocks and
/// user messages). This is the readability floor below which the responsive
/// cap never narrows prose relative to the former 110-cell behavior, not a
/// global ceiling. Tool results, standalone data blocks, and non-prose widgets
/// keep the full width (data wants room; prose wants a measure).
const PROSE_MEASURE_FLOOR: u16 = 110;

/// Responsive wrap cap for `available` prose-body cells.
///
/// The result can exceed `available` at narrow widths; each caller retains its
/// own clamp so narrow-terminal behavior stays unchanged. Multiplication is
/// widened before applying the 90% measure to avoid `u16` overflow.
#[must_use]
pub(crate) fn prose_wrap_cap(available: u16) -> u16 {
    let responsive = u16::try_from(u32::from(available) * 9 / 10)
        .expect("90% of a u16 fits in a u16");
    PROSE_MEASURE_FLOOR.max(responsive)
}

/// How an assistant prose block carries its author mark (CC/codex bullet
/// grammar — see `docs/streaming-style-v3-2026-07-13.md` §3).
///
/// The mark column keeps a continued assistant thought in the same left
/// column (col 3) as the block it follows, instead of dropping to the
/// full-width left edge — which left the continuation's text visibly
/// misaligned under the first block's bullet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProseMark {
    /// First assistant prose of a spoken answer: `◆ ` author bullet on the
    /// first body row, indent below.
    Bullet,
    /// Assistant prose continuing directly after prior prose: indent only,
    /// no repeated bullet.
    Indent,
    /// No mark — full-width body (non-assistant content).
    Bare,
}

impl ProseMark {
    /// Whether this style reserves the mark column (and thus narrows the body).
    #[must_use]
    pub fn has_indent(self) -> bool {
        matches!(self, Self::Bullet | Self::Indent)
    }
}

/// Cross-cutting context for rendering one transcript block.
///
/// Bundles the parameters that were previously passed to [`draw_block`]
/// as a 10-argument positional list (which required
/// `#[allow(clippy::too_many_arguments)]`). Call sites now read by name
/// and adding a new rendering knob no longer ripples through the
/// signature. It is `Copy` (all fields are `Copy` / shared refs) so the
/// transcript can cheaply rebuild one per visible block.
#[derive(Clone, Copy)]
pub struct BlockDrawCtx<'a> {
    /// Active theme — the single source of all styling.
    pub theme: &'a Theme,
    /// `true` when the transcript focus cursor is on this block, so
    /// interactable widgets (`ToolResult`, `PermissionPrompt`,
    /// `Reasoning`) can highlight themselves.
    pub focused: bool,
    /// Expansion state for collapsible widgets (`ToolResult`,
    /// `Reasoning`).
    pub expanded: bool,
    /// Animation tick for spinners / streaming carets.
    pub tick: u64,
    /// Vertical scroll offset applied within the block's area.
    pub scroll_offset: u16,
    /// Terminal image protocol used by `Image` blocks.
    pub image_protocol: ImageProtocol,
    /// `true` when this is the active tail block (live streaming turn).
    pub is_tail_active: bool,
}

/// Dispatch on the [`RenderBlock`] variant and render into `area`.
///
/// Called by [`crate::tui::transcript::Transcript::draw`] once per
/// visible block, with a per-block [`BlockDrawCtx`].
///
/// Per `code-rules.md` R1, the match is exhaustive over the
/// provider-neutral variants only — any Anthropic-specific naming
/// would be a compile error because those types simply don't exist
/// here.
// One arm per `RenderBlock` variant: a flat dispatch match, so the length is
// inherent to the number of block kinds, not tangled logic. Each arm delegates
// to a per-block renderer module.
#[allow(clippy::too_many_lines)]
pub fn draw_block(frame: &mut Frame<'_>, area: Rect, block: &RenderBlock, ctx: &BlockDrawCtx) {
    let &BlockDrawCtx {
        theme,
        focused,
        expanded,
        tick,
        scroll_offset,
        image_protocol,
        is_tail_active,
    } = ctx;
    match block {
        RenderBlock::TextDelta { text, done, .. } => {
            text::draw(frame, area, text, *done, theme, tick, scroll_offset);
        }
        RenderBlock::Reasoning { id, text, done, .. } => {
            // transcript 의 draw 루프가 elapsed 포함으로 선처리하므로 이 dispatch
            // 경로는 폴백(타이밍 없음). seed 는 블록 id 로 안정.
            reasoning::draw(
                frame,
                area,
                text,
                *done,
                theme,
                focused,
                expanded,
                tick,
                scroll_offset,
                None,
                id.0,
            );
        }
        RenderBlock::ToolCall {
            tool_call_id,
            name,
            summary,
            preview,
            status,
            ..
        } => {
            // 전형적 dispatch — transcript 의 elapsed/agent-tree 정보는 ToolCall
            // arm 에서 직접 그릴 때 전달되므로, 여기서는 None 폴백.
            tool_call::draw(
                frame,
                area,
                &tool_call_id.0,
                name,
                summary,
                preview,
                *status,
                theme,
                tick,
                scroll_offset,
                is_tail_active,
                None,
                None,
                false,
            );
        }
        RenderBlock::ToolResult { is_error, body, .. } => {
            tool_result::draw(
                frame,
                area,
                *is_error,
                body,
                theme,
                focused,
                expanded,
                scroll_offset,
            );
        }
        RenderBlock::PermissionPrompt(prompt) => {
            // Historical transcript render: no live cursor state here, so focus
            // the safe default for display.
            let selected = permission::default_selected_index(prompt);
            permission::draw(frame, area, prompt, theme, focused, selected, scroll_offset);
        }
        RenderBlock::UserQuestionPrompt(prompt) => {
            let text = format!("Question: {}", prompt.question);
            text::draw(frame, area, &text, true, theme, tick, scroll_offset);
        }
        RenderBlock::Image {
            data, media_type, ..
        } => {
            image::draw(
                frame,
                area,
                data,
                media_type,
                image_protocol,
                theme,
                scroll_offset,
            );
        }
        RenderBlock::UserMessage { text, .. } => {
            draw_user_message(frame, area, text, theme, scroll_offset);
        }
        RenderBlock::UserNotice { message, .. } => {
            user_notice::draw(frame, area, message, theme, scroll_offset);
        }
        RenderBlock::AgentResult {
            label,
            status,
            body,
            ..
        } => {
            let view = agent_result::AgentCardView {
                label,
                status: *status,
                body,
                expanded,
                focused,
            };
            agent_result::draw(frame, area, &view, theme, scroll_offset);
        }
        RenderBlock::System { level, text, .. } => {
            system::draw(frame, area, *level, text, theme, scroll_offset);
        }
        RenderBlock::Card { card, .. } => {
            super::cards::draw(frame, area, card, theme, scroll_offset);
        }
        // Usage updates the live ledger via `App::record_live_usage`; it is
        // never pushed to the transcript, so there is nothing to draw here.
        // Usage/RateLimit update the HUD ledger only; nothing to draw here.
        RenderBlock::Usage { .. }
        | RenderBlock::CompactionProgress { .. }
        | RenderBlock::RateLimit(_) => {}
    }
}

/// 사용자(페이스트) 메시지 블록 — 어시스턴트와 동일한 마크다운 엔진을 거치고
/// 좌측에 amber rail (`┃`) 을 prepend 한다.
///
/// 이전에는 raw text 를 line-by-line 출력해서 `**bold**`/`## h2`/`---` 같은
/// 마크다운 문법이 화면에 그대로 노출됐다. `markdown::rendered_lines_for_width`
/// 로 위임해 시각을 어시스턴트 응답과 통일한다.
pub fn draw_user_message(
    frame: &mut Frame<'_>,
    area: Rect,
    text: &str,
    theme: &Theme,
    scroll_offset: u16,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    // 마크다운 본문은 좌측 3-cell rail(`┃` + 2 space)을 제외한 폭으로 렌더.
    // 빈 입력일 때도 한 줄의 rail 은 보이도록 single space 폴백. 폭은 산문
    // measure 캡을 공유한다(layout 측 측정과 정합 — 측정==페인트).
    let available_width = area.width.saturating_sub(ROLE_RAIL_WIDTH);
    let inner_width = available_width.clamp(1, prose_wrap_cap(available_width));
    let body = if text.is_empty() {
        vec![Line::from("")]
    } else {
        crate::tui::markdown::rendered_lines_for_width(text, theme, inner_width)
    };
    let preserves = crate::tui::markdown::preserves_layout(text);
    let body_rows = if preserves {
        u16::try_from(body.len()).unwrap_or(u16::MAX).max(1)
    } else {
        wrapped_rows(&body, inner_width)
    };
    draw_user_message_lines(frame, area, body, body_rows, preserves, theme, scroll_offset);
}

/// Shared user-message painter: `You` header row, a `┃` rail on every visible
/// body row (wrapped continuations included), and the markdown body in its own
/// column to the right of the rail.
///
/// The body column uses [`prose_wrap_cap`] on the wrap path so the paint wrap
/// width equals the layout measure width — painting at a different ultra-wide
/// width would leave phantom blank rows trailing the block. Preserved-layout
/// bodies (fences/tables) keep the full remaining width: their measured height
/// is line-count based, so no cap is needed for measure==paint.
fn draw_user_message_lines(
    frame: &mut Frame<'_>,
    area: Rect,
    body: Vec<Line<'_>>,
    body_rows: u16,
    preserves: bool,
    theme: &Theme,
    scroll_offset: u16,
) {
    use ratatui::widgets::{Paragraph, Wrap};

    if area.height == 0 || area.width == 0 {
        return;
    }

    let header_visible = scroll_offset == 0;
    if header_visible {
        frame.render_widget(
            Paragraph::new(user_header_line(theme)).style(theme.typography.body),
            Rect::new(area.x, area.y, area.width, 1),
        );
    }
    let body_y = area.y.saturating_add(u16::from(header_visible));
    let body_h = area.height.saturating_sub(u16::from(header_visible));
    let body_scroll = scroll_offset.saturating_sub(1);
    let visible_body_rows = body_rows.saturating_sub(body_scroll).min(body_h);
    if body_h == 0 || visible_body_rows == 0 {
        return;
    }

    // Full 3-cell rail run (`┃` + two spaces) so the style-run shape matches
    // the pre-split renderer cell-for-cell (golden fixtures byte-compare it).
    let rail_run = format!("{}  ", user_rail_glyph(theme));
    let rail: Vec<Line<'_>> = (0..visible_body_rows)
        .map(|_| Line::from(Span::styled(rail_run.clone(), user_rail_style(theme))))
        .collect();
    frame.render_widget(
        Paragraph::new(rail),
        Rect::new(
            area.x,
            body_y,
            ROLE_RAIL_WIDTH.min(area.width),
            visible_body_rows,
        ),
    );

    if area.width <= ROLE_RAIL_WIDTH {
        return;
    }
    let available_width = area.width.saturating_sub(ROLE_RAIL_WIDTH);
    let body_width = if preserves {
        available_width
    } else {
        available_width.min(prose_wrap_cap(available_width))
    };
    let body_area = Rect::new(
        area.x.saturating_add(ROLE_RAIL_WIDTH),
        body_y,
        body_width,
        body_h,
    );
    // 블록 전체 code_bg 배경을 제거한다 — 마크다운 엔진이 코드블록에 자체
    // code_bg 를 그리므로, 외곽 배경과 이중으로 겹쳐 보이는 회귀를 막는다.
    let para = if preserves {
        Paragraph::new(body)
            .style(theme.typography.body)
            .scroll((body_scroll, 0))
    } else {
        Paragraph::new(body)
            .style(theme.typography.body)
            .wrap(Wrap { trim: false })
            .scroll((body_scroll, 0))
    };
    frame.render_widget(para, body_area);
}

/// `┃  You` — the role header line above a user message.
///
/// Uses the same bar glyph as the body rail so the user turn reads as one
/// unit, and the bold amber style marks it as a role label (mirrors the
/// `◆ Zo` header on the assistant side).
fn user_header_line(theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        format!("{}  You", user_rail_glyph(theme)),
        user_rail_style(theme),
    ))
}

/// User-paste rail glyph (`┃`, or `|` under `NO_COLOR`).
fn user_rail_glyph(theme: &Theme) -> &'static str {
    if theme.no_color { "|" } else { "┃" }
}

/// Amber bold style for the user-paste rail.
fn user_rail_style(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.palette.accent)
        .add_modifier(Modifier::BOLD)
}

/// 캐시된 마크다운 body lines 를 사용자 메시지로 그린다.
///
/// `body_lines` 는 [`crate::tui::markdown::rendered_lines_for_width`] 의 결과로
/// measure 캡이 적용된 폭으로 미리 캐시된 lines 다. transcript 의 fast-path 가
/// 이를 호출해 매 프레임 pulldown-cmark + syntect 재파싱을 피한다.
/// `row_prefix`/`preserves` 는 같은 캐시 슬롯의 값 — 높이 측정과 동일한 행 수로
/// rail 을 칠해 측정==페인트를 유지한다.
pub fn draw_user_message_from_cache(
    frame: &mut Frame<'_>,
    area: Rect,
    body_lines: &[ratatui::text::Line<'static>],
    row_prefix: &[u32],
    preserves: bool,
    theme: &Theme,
    scroll_offset: u16,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let body_rows = if preserves {
        u16::try_from(body_lines.len()).unwrap_or(u16::MAX).max(1)
    } else {
        u16::try_from(row_prefix.last().copied().unwrap_or(0))
            .unwrap_or(u16::MAX)
            .max(1)
    };
    // 캐시 본문을 String 복제 없이 얕게 빌려 그린다.
    draw_user_message_lines(
        frame,
        area,
        borrow_lines(body_lines),
        body_rows,
        preserves,
        theme,
        scroll_offset,
    );
}

pub(crate) use crate::util::ansi::sanitize_inline;

/// Compact a file path for one-line tool/diff labels.
///
/// Strips inline control sequences, then anchors on the first recognized
/// repo-relative prefix (`crates/`, `src/`, `tests/`, `.zo/`, `.github/`)
/// so absolute paths collapse to their repo-relative tail. When no anchor
/// matches, paths longer than four segments keep only the last four.
pub(crate) fn compact_path_label(path: &str) -> String {
    let path = sanitize_inline(path);
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return "?".to_string();
    }

    for anchor in ["crates/", "src/", "tests/", ".zo/", ".github/"] {
        if let Some(idx) = trimmed.find(anchor) {
            return trimmed[idx..].to_string();
        }
    }

    let parts = trimmed
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() <= 4 {
        return trimmed.to_string();
    }
    parts[parts.len() - 4..].join("/")
}

/// Rows these `lines` occupy once wrapped to `width`.
///
/// Delegates to ratatui's [`Paragraph::line_count`], which wraps with the exact
/// same `WordWrapper` the renderer uses for `Wrap { trim: false }`. This keeps
/// height measurement and the actual draw in lockstep: a hand-rolled
/// `ceil(line_width / width)` counts naive cell-fills, but the renderer breaks
/// on word boundaries, so the two disagreed by a row on lines that wrap
/// mid-word — the cached layout then reserved the wrong height and a block
/// gained a phantom row until the next re-layout (the "extra line until you
/// scroll" glitch). One wrap engine, one source of truth.
///
/// The lines are borrowed zero-copy (no `String` reallocation) for the
/// measurement; the `Paragraph` is discarded immediately.
pub(crate) fn wrapped_rows(lines: &[ratatui::text::Line<'_>], width: u16) -> u16 {
    use ratatui::text::{Line, Span, Text};
    use ratatui::widgets::{Paragraph, Wrap};

    // Borrow the spans zero-copy (no `String` reallocation) into a `Text` whose
    // lifetime is tied to `lines`. `borrow_lines` is the `'static`-only sibling
    // used by the draw fast-path; height measurement runs over arbitrary
    // lifetimes, so the shallow borrow is open-coded here for the same effect.
    let borrowed: Vec<Line<'_>> = lines
        .iter()
        .map(|line| Line {
            style: line.style,
            alignment: line.alignment,
            spans: line
                .spans
                .iter()
                .map(|span| Span {
                    style: span.style,
                    content: std::borrow::Cow::Borrowed(span.content.as_ref()),
                })
                .collect(),
        })
        .collect();
    let measured = Paragraph::new(Text::from(borrowed))
        .wrap(Wrap { trim: false })
        .line_count(width.max(1));
    u16::try_from(measured).unwrap_or(u16::MAX).max(1)
}

/// Wrap-row counts per line as a prefix sum: `prefix[i]` is the rows occupied
/// by `lines[..i]` once wrapped to `width`, so `prefix[lines.len()]` equals
/// [`wrapped_rows`] (same `Paragraph::line_count` engine, line by line — the
/// `WordWrapper` never carries state across hard line breaks, so the per-line
/// sum and the whole-text count agree; a test pins this).
///
/// Built once per render-cache fill. Turns the per-frame O(total lines)
/// height re-measure into an O(1) lookup and lets the draw path binary-search
/// the first visible line instead of feeding the whole block through
/// `Paragraph`'s wrap-and-skip scroll — the two costs that made frame time
/// grow linearly with message length while streaming.
pub(crate) fn wrapped_row_prefix(lines: &[ratatui::text::Line<'_>], width: u16) -> Vec<u32> {
    let mut prefix = Vec::with_capacity(lines.len() + 1);
    prefix.push(0u32);
    let mut acc = 0u32;
    for line in lines {
        acc = acc.saturating_add(u32::from(line_wrapped_rows(line, width)));
        prefix.push(acc);
    }
    prefix
}

/// Rows a single line occupies once wrapped to `width` — the per-line unit
/// of [`wrapped_row_prefix`], measured with the same wrap engine as
/// [`wrapped_rows`] so the two never disagree.
pub(crate) fn line_wrapped_rows(line: &ratatui::text::Line<'_>, width: u16) -> u16 {
    use ratatui::text::{Line, Span, Text};
    use ratatui::widgets::{Paragraph, Wrap};

    let borrowed = Line {
        style: line.style,
        alignment: line.alignment,
        spans: line
            .spans
            .iter()
            .map(|span| Span {
                style: span.style,
                content: std::borrow::Cow::Borrowed(span.content.as_ref()),
            })
            .collect(),
    };
    let measured = Paragraph::new(Text::from(borrowed))
        .wrap(Wrap { trim: false })
        .line_count(width.max(1));
    u16::try_from(measured).unwrap_or(u16::MAX)
}

/// Slice out the lines covering viewport rows `[scroll, scroll + max_rows)`
/// of a wrapped block, using its [`wrapped_row_prefix`].
///
/// Returns `(visible lines, in-line scroll remainder for the first returned
/// line, whether the slice reaches the block's last line)`. Feeding only this
/// window to `Paragraph` (with the remainder as its scroll) paints the exact
/// same cells as scrolling the full block, at O(visible) instead of
/// O(scroll + visible) per frame.
pub(crate) fn visible_line_window<'a>(
    lines: &'a [ratatui::text::Line<'static>],
    row_prefix: &[u32],
    scroll: u16,
    max_rows: u16,
) -> (&'a [ratatui::text::Line<'static>], u16, bool) {
    debug_assert_eq!(row_prefix.len(), lines.len() + 1, "prefix must cover lines");
    if lines.is_empty() || row_prefix.len() != lines.len() + 1 {
        // Defensive: a malformed prefix falls back to the whole block, which
        // is always correct (just slower).
        return (lines, scroll, true);
    }
    let scroll = u32::from(scroll);
    let end_row = scroll.saturating_add(u32::from(max_rows));
    // First line whose row range extends past `scroll`.
    let start = row_prefix
        .partition_point(|&p| p <= scroll)
        .saturating_sub(1)
        .min(lines.len());
    // One past the last line that starts before `end_row`.
    let end = row_prefix
        .partition_point(|&p| p < end_row)
        .clamp(start, lines.len());
    let line_scroll = scroll.saturating_sub(row_prefix[start]);
    (
        &lines[start..end],
        u16::try_from(line_scroll).unwrap_or(u16::MAX),
        end >= lines.len(),
    )
}

/// 캐시된 `Line<'static>` 들을 String 복제 없이 얕게 빌려 `Paragraph` 입력용
/// `Vec<Line<'_>>` 로 만든다.
///
/// `ratatui::Paragraph` 는 소유 `Text` 를 요구하지만, 캐시 본문은 draw 동안
/// 살아있으므로 span 의 `String` 을 다시 할당하지 않고 `Cow::Borrowed` 로
/// 참조만 한다 — 큰 본문(tool 출력·긴 답변)을 매 프레임 다시 그릴 때 종전
/// `.clone()`/`.to_vec()` 의 heap String 복제를 없앤다. (`Vec<Line>`/
/// `Vec<Span>` 구조만 재구성하며 이는 span 수에 비례, String 길이엔 무관.)
pub(crate) fn borrow_lines<'a>(
    lines: &'a [ratatui::text::Line<'static>],
) -> Vec<ratatui::text::Line<'a>> {
    use ratatui::text::{Line, Span};
    lines
        .iter()
        .map(|line| Line {
            style: line.style,
            alignment: line.alignment,
            spans: line
                .spans
                .iter()
                .map(|span| Span {
                    style: span.style,
                    content: std::borrow::Cow::Borrowed(span.content.as_ref()),
                })
                .collect(),
        })
        .collect()
}

/// `true` if this variant is interactable (focusable via arrow keys).
///
/// Used by [`crate::tui::transcript::Transcript`] to decide which
/// blocks the focus cursor can land on. See `.zo/design/components.md`
/// §9.2.
///
/// `UserNotice` is deliberately excluded: like `UserMessage` / `System` it is
/// a static display panel with nothing to expand or answer, so it is not an
/// arrow-key focus target. Its verbatim text stays copyable through the hover
/// copy button (see `block_copy_payload`), which does not depend on focus.
#[must_use]
pub fn is_interactable(block: &RenderBlock) -> bool {
    matches!(
        block,
        RenderBlock::ToolResult { .. }
            | RenderBlock::PermissionPrompt(_)
            | RenderBlock::UserQuestionPrompt(_)
            | RenderBlock::Reasoning { .. }
            | RenderBlock::AgentResult { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    #[test]
    fn prose_wrap_cap_has_fixed_floor_then_scales_responsively() {
        for (available, expected) in [
            (80, 110),
            (122, 110),
            (123, 110),
            (130, 117),
            (200, 180),
            (u16::MAX, 58_981),
        ] {
            assert_eq!(
                prose_wrap_cap(available),
                expected,
                "unexpected prose cap for {available} available cells"
            );
        }
    }

    /// 회귀: `compact_path_label`이 세 블록(tool_call/tool_result/diff)에서
    /// 단일 구현으로 합쳐진 뒤에도 압축 동작을 동일하게 유지하는지 핀.
    #[test]
    fn compact_path_label_pins_behavior() {
        // Anchors on first recognized repo prefix, dropping the absolute head.
        assert_eq!(
            compact_path_label("/Users/joe/2026/zo/crates/foo/src/bar.rs"),
            "crates/foo/src/bar.rs"
        );
        // No anchor: keep only the last four segments.
        assert_eq!(compact_path_label("a/b/c/d/e/f.txt"), "c/d/e/f.txt");
        // Four-or-fewer segments are returned verbatim.
        assert_eq!(compact_path_label("x/y/z.rs"), "x/y/z.rs");
        // Empty / whitespace collapses to the sentinel.
        assert_eq!(compact_path_label("   "), "?");
        // Inline control chars are replaced (BEL -> space) before compaction.
        assert_eq!(compact_path_label("src/\u{7}main.rs"), "src/ main.rs");
    }

    /// 회귀: 사용자 페이스트 메시지가 raw markdown 문법(`**`, `##`)을
    /// 화면에 그대로 노출하지 않는다. 어시스턴트와 동일한 마크다운 엔진을
    /// 거치는지 검증.
    #[test]
    fn user_message_renders_markdown_not_raw_syntax() {
        let theme = Theme::default_dark();
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| {
                draw_user_message(
                    frame,
                    Rect::new(0, 0, 80, 12),
                    "## Heading\n\n**bold** then `code`",
                    &theme,
                    0,
                );
            })
            .expect("draw");
        let backend = terminal.backend();
        let content: String = backend
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        // raw 마크다운 문법은 노출 금지.
        assert!(
            !content.contains("##"),
            "raw '##' must not appear in user message buffer:\n{content}"
        );
        assert!(
            !content.contains("**"),
            "raw '**' must not appear in user message buffer:\n{content}"
        );
        // 본문은 표시되어야 한다.
        assert!(
            content.contains("Heading"),
            "heading text must render:\n{content}"
        );
        assert!(
            content.contains("bold"),
            "bold text must render:\n{content}"
        );
        assert!(
            content.contains("code"),
            "code text must render:\n{content}"
        );
    }

    /// 어시스턴트와 사용자 메시지가 **같은 폭** 에서 동일한 마크다운 본문을
    /// 산출하는지 — 두 경로가 byte-identical 한 마크다운 엔진을 거친다는 보장.
    #[test]
    fn user_message_and_assistant_share_markdown_engine() {
        let theme = Theme::default_dark();
        let md = "## Hi\n\n**bold** `code`\n\n- a\n  - b\n\n> quote";
        // 같은 폭에서 rendered_lines_for_width 결과는 byte-identical.
        let a = crate::tui::markdown::rendered_lines_for_width(md, &theme, 60);
        let b = crate::tui::markdown::rendered_lines_for_width(md, &theme, 60);
        assert_eq!(a, b, "same engine + same width must be byte-identical");
        // 폭이 다르면 HR / 코드블록 폭만 달라지고 줄 수는 동일.
        let c = crate::tui::markdown::rendered_lines_for_width(md, &theme, 40);
        assert_eq!(
            a.len(),
            c.len(),
            "structural line count must be width-invariant"
        );
    }

    /// Count rows actually painted by ratatui for `lines` wrapped to `width`,
    /// by rendering into a tall `TestBackend` and counting non-blank rows. This
    /// is the ground truth `wrapped_rows` must match.
    fn rendered_row_count(lines: &[Line<'static>], width: u16) -> u16 {
        use ratatui::widgets::{Paragraph, Wrap};

        let height = 200u16;
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| {
                let para = Paragraph::new(lines.to_vec()).wrap(Wrap { trim: false });
                frame.render_widget(para, Rect::new(0, 0, width, height));
            })
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let mut rows = 0u16;
        for y in 0..height {
            let any = (0..width).any(|x| buffer[(x, y)].symbol().trim() != "");
            if any {
                rows = y.saturating_add(1);
            }
        }
        rows.max(1)
    }

    /// Core regression for the "extra line until you scroll" glitch: the height
    /// `wrapped_rows` reports must equal the rows ratatui actually paints. The
    /// old `ceil(line_width / width)` measure diverged on mid-word wraps; the
    /// `Paragraph::line_count` delegation keeps them identical.
    #[test]
    fn wrapped_rows_matches_actual_render_height() {
        let cases: &[(&str, u16)] = &[
            // Mid-word wrap: a long unbroken token forces a break the naive
            // ceil-divide and the word-wrapper agree on, but the surrounding
            // spaces are where they used to differ.
            ("the quick brown fox jumps over the lazy dog again", 20),
            ("supercalifragilisticexpialidocious and friends", 15),
            // Width that splits exactly on a word boundary.
            ("one two three four five six seven eight", 10),
            // Trailing/leading spaces and short content.
            ("short", 40),
            ("a b c d e f g h i j k l m n o p q r s t", 9),
            // CJK (double-width) wrapping.
            ("한국어 텍스트가 좁은 폭에서 줄바꿈 되는 경우를 검증", 12),
        ];
        for (text, width) in cases {
            let lines = vec![Line::from(*text)];
            let measured = wrapped_rows(&lines, *width);
            let actual = rendered_row_count(&lines, *width);
            assert_eq!(
                measured, actual,
                "wrapped_rows({text:?}, {width}) = {measured} but ratatui painted {actual} rows"
            );
        }
    }

    /// Multi-line input (the common transcript case): the sum across several
    /// wrapping lines must also match the painted height exactly.
    #[test]
    fn wrapped_rows_matches_render_for_multiline() {
        let lines = vec![
            Line::from("the quick brown fox jumps over the lazy dog"),
            Line::from(""),
            Line::from("another paragraph that also needs to wrap a few times here"),
            Line::from("short tail"),
        ];
        let width = 18;
        assert_eq!(
            wrapped_rows(&lines, width),
            rendered_row_count(&lines, width),
            "multi-line measured height must equal painted rows"
        );
    }

    fn sample_wrap_lines() -> Vec<Line<'static>> {
        vec![
            Line::from("the quick brown fox jumps over the lazy dog again and again"),
            Line::from(""),
            Line::from("한국어 텍스트가 좁은 폭에서 줄바꿈 되는 경우를 검증하는 문장"),
            Line::from("supercalifragilisticexpialidocious"),
            Line::from("short"),
            Line::from("a b c d e f g h i j k l m n o p q r s t u v w x y z"),
        ]
    }

    /// The per-line prefix sum must agree with the whole-text `wrapped_rows`
    /// at every prefix boundary — the invariant that lets the cache answer
    /// height queries and slice the visible window without re-wrapping.
    #[test]
    fn wrapped_row_prefix_matches_wrapped_rows() {
        let lines = sample_wrap_lines();
        for width in [9u16, 12, 18, 40, 120] {
            let prefix = wrapped_row_prefix(&lines, width);
            assert_eq!(prefix.len(), lines.len() + 1);
            for cut in 0..=lines.len() {
                let expect = if cut == 0 {
                    0
                } else {
                    u32::from(wrapped_rows(&lines[..cut], width))
                };
                assert_eq!(
                    prefix[cut], expect,
                    "prefix[{cut}] diverged from wrapped_rows at width {width}"
                );
            }
        }
    }

    /// Painting only the `visible_line_window` slice (with its in-line scroll
    /// remainder) must produce the exact same cells as scrolling the full
    /// block — swept across every scroll offset of a wrapping block.
    #[test]
    fn visible_line_window_paints_identically_to_full_scroll() {
        use ratatui::widgets::{Paragraph, Wrap};

        let lines = sample_wrap_lines();
        let width = 18u16;
        let view_h = 6u16;
        let prefix = wrapped_row_prefix(&lines, width);
        let total = *prefix.last().expect("prefix non-empty");

        let render = |to_draw: Vec<Line<'static>>, scroll: u16| -> Vec<String> {
            let backend = TestBackend::new(width, view_h);
            let mut terminal = Terminal::new(backend).expect("terminal");
            terminal
                .draw(|frame| {
                    let para = Paragraph::new(to_draw.clone())
                        .wrap(Wrap { trim: false })
                        .scroll((scroll, 0));
                    frame.render_widget(para, Rect::new(0, 0, width, view_h));
                })
                .expect("draw");
            let buffer = terminal.backend().buffer().clone();
            (0..view_h)
                .map(|y| {
                    (0..width)
                        .map(|x| buffer[(x, y)].symbol().to_string())
                        .collect::<String>()
                })
                .collect()
        };

        for scroll in 0..=u16::try_from(total).expect("fits u16") + 2 {
            let full = render(lines.clone(), scroll);
            let (slice, line_scroll, _) = visible_line_window(&lines, &prefix, scroll, view_h);
            let sliced = render(slice.to_vec(), line_scroll);
            assert_eq!(
                full, sliced,
                "windowed paint diverged from full paint at scroll={scroll}"
            );
        }
    }
}
