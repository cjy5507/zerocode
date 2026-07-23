//! Markdown → ratatui `Line<'static>` 렌더링 엔진.
//!
//! 단일 책임: pulldown-cmark Event 스트림을 styled `Line` 으로 변환한다.
//! 두 호출자가 같은 엔진을 거치므로 어시스턴트(`TextDelta`)와 사용자
//! (`UserMessage`) 메시지의 마크다운 시각은 byte-identical 이다:
//!
//! * [`crate::tui::blocks::text`] — 어시스턴트 스트리밍 텍스트 위젯
//! * [`crate::tui::blocks::draw_user_message`] — 사용자 페이스트 위젯
//!
//! `code-rules.md` R1 (render-block only), R2 (no ANSI — `Line`/`Span`/
//! `Style` 만), R9 (`&Theme` 경유 스타일링), R10 (`NO_COLOR` 폴백)
//! 준수.

#![allow(clippy::doc_markdown)]

use std::borrow::Cow;
use std::sync::OnceLock;

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::highlighting::{Theme as SynTheme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;
use unicode_width::UnicodeWidthChar;

use crate::tui::theme::{CalloutKind, Theme};

/// `---` 가로선의 최소 폭(cells).
const HR_MIN_WIDTH: usize = 20;

// ============================================================================
// Public API
// ============================================================================

/// 마크다운을 폭에 맞춰 styled lines 로 변환한다.
///
/// 처리 분기:
/// 1. GFM 표 — [`render_table_markdown_for_width`] 가 정렬된 텍스트 산출
/// 2. 고밀도 터미널/TUI 캡처는 preformatted 처리. 이런 입력은 마크다운
///    문서가 아니라 화면 복사본이라 `**`, `#`, 4-space indent 같은
///    우연한 문자를 문법으로 해석하면 레이아웃이 망가진다.
/// 3. 명백한 markdown 시그널(heading / fenced code) 이 있으면 markdown 처리.
///    본문에 ASCII tree 같은 box-drawing 문자가 섞여 있어도 헤딩/코드펜스가
///    있으면 markdown 으로 가야 `##`, `` ` ``, `**` 가 raw 로 새지 않는다.
/// 4. box-drawing 문자만 있고 markdown 시그널 없음 → preformatted (기존 동작).
/// 5. 그 외 → [`Renderer`] 가 pulldown-cmark 이벤트를 styled spans 로 변환.
#[must_use]
pub fn rendered_lines_for_width(text: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    rendered_with_highlight(text, theme, width, true)
}

/// 스트리밍 중 "열린(open) 꼬리 블록" 전용 렌더 — syntect 하이라이트를 끈다.
///
/// 완료된 블록은 [`rendered_lines_for_width`] 로 한 번만(하이라이트 포함)
/// 렌더해 캐시하고, 매 프레임 다시 그리는 작은 꼬리만 이 경로로 처리한다.
/// 꼬리는 토큰이 도착할 때마다(=프레임마다) 새 텍스트로 다시 렌더되는데,
/// syntect 는 stateful per-line 이라 프레임마다 다시 칠하면 긴 답변에서
/// draw 루프가 끊긴다(스피너/캐럿이 멈춰 "행" 처럼 보임). 그래서 꼬리는
/// 하이라이트 없이(prose 스타일만) 그리고, 코드펜스가 닫혀 안정 구간으로
/// 넘어가는 순간 [`rendered_lines_for_width`] 의 1회 패스가 syntect 색을
/// 입힌다. prose(대다수)는 syntect 대상이 아니라 시각 변화가 없다.
#[must_use]
pub fn rendered_tail_for_width(text: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    rendered_with_highlight(text, theme, width, false)
}

mod streaming;
pub use streaming::{
    clip_tail_for_display, rendered_bounded_streaming_tail_for_width,
    rendered_fence_interior_lines, repair_unclosed_inline_markers,
};

/// Turn-end self-confidence marker (grammar: `[zo:turn-confidence]
/// low|medium|high — <one line why>`). This is the SSOT for the literal —
/// the session-side cascade parser re-exports it — and the renderer swaps
/// the raw bracket line for a dim chip so the routing signal doesn't read
/// as debug noise in the transcript.
pub const TURN_CONFIDENCE_MARKER: &str = "[zo:turn-confidence]";

/// Split a trailing confidence-marker line off `text`. Mirrors the parse
/// contract exactly: only the LAST non-empty line counts, so a marker quoted
/// mid-text renders as ordinary prose.
fn split_trailing_confidence_marker(text: &str) -> Option<(&str, &str)> {
    let trimmed_end = text.trim_end();
    let last_line = trimmed_end.lines().next_back()?;
    let rest = last_line.trim_start().strip_prefix(TURN_CONFIDENCE_MARKER)?;
    Some((&trimmed_end[..trimmed_end.len() - last_line.len()], rest))
}

/// The dim one-line chip a trailing confidence marker renders as.
fn confidence_chip_line(rest: &str, theme: &Theme) -> Line<'static> {
    let style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::ITALIC);
    let glyph = if theme.no_color { "*" } else { "◈" };
    let content = rest.trim();
    let text = if content.is_empty() {
        format!("{glyph} confidence")
    } else {
        format!("{glyph} confidence {content}")
    };
    Line::from(Span::styled(text, style))
}

fn rendered_with_highlight(
    text: &str,
    theme: &Theme,
    width: u16,
    highlight: bool,
) -> Vec<Line<'static>> {
    if let Some((body, rest)) = split_trailing_confidence_marker(text) {
        let mut lines = if body.trim().is_empty() {
            Vec::new()
        } else {
            rendered_with_highlight(body, theme, width, highlight)
        };
        lines.push(confidence_chip_line(rest, theme));
        return lines;
    }
    let strong_markdown = has_strong_markdown_signal(text);
    if looks_like_terminal_capture(text)
        && (!strong_markdown
            || (looks_like_dense_terminal_capture(text)
                && !has_authored_markdown_block_structure(text)))
    {
        return preformatted_lines(text, theme, width);
    }
    if has_markdown_table(text) {
        // A table mixed with other markdown (headings, fenced code, mermaid,
        // `**bold**`) must NOT route the *whole* block through the table-only
        // renderer — that path passes non-table lines through verbatim, so the
        // surrounding markdown leaks out as raw `##`/```` ``` ````/`**`. Segment
        // the block: tables go to the box renderer, everything else through the
        // full markdown `Renderer`.
        if strong_markdown {
            return render_mixed_table_markdown(text, theme, width, highlight);
        }
        // Pure table block (no other markdown): fast preformatted path.
        let rendered = render_table_markdown_for_width(text, width);
        return preformatted_lines(&rendered, theme, width);
    }
    if contains_box_drawing(text) && !strong_markdown {
        return preformatted_lines(text, theme, width);
    }
    let mut lines = Renderer::new(theme, width, highlight).render(text);
    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

/// Render a block that mixes GFM tables with other markdown. Splits the source
/// at table boundaries: contiguous non-table runs are rendered by the full
/// [`Renderer`] (so headings / code fences / emphasis are styled, not raw),
/// while table blocks are rendered by [`render_table_block`] (box-drawing,
/// pre-fit to `width`). The result is drawn *wrapped* (see [`preserves_layout`],
/// which returns `false` here) — table rows already fit `width`, so wrapping
/// only affects long prose, never the table alignment.
fn render_mixed_table_markdown(
    text: &str,
    theme: &Theme,
    width: u16,
    highlight: bool,
) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut prose: Vec<&str> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let starts_table = lines
            .peek()
            .copied()
            .is_some_and(|next| looks_like_table_row(line) && looks_like_table_separator(next));
        if !starts_table {
            prose.push(line);
            continue;
        }
        flush_prose_segment(&mut prose, theme, width, highlight, &mut out);
        // A box-rendered table needs one blank line of air on each side so it
        // does not collide with an adjacent heading/paragraph (the closed box
        // already owns its own top/bottom edge — the blank is the *gap*).
        push_blank_if_content(&mut out);
        // Consume the header + separator + any following body rows.
        let Some(separator) = lines.next() else {
            out.push(Line::from(line.to_string()));
            break;
        };
        let mut table_lines = vec![line, separator];
        while let Some(next) = lines.peek().copied() {
            if !looks_like_table_row(next) {
                break;
            }
            let Some(row) = lines.next() else { break };
            table_lines.push(row);
        }
        let rendered = render_table_block(&table_lines, width).join("\n");
        out.append(&mut preformatted_lines(&rendered, theme, width));
        out.push(Line::from(""));
    }
    flush_prose_segment(&mut prose, theme, width, highlight, &mut out);
    if out.is_empty() {
        out.push(Line::from(""));
    }
    out
}

/// Push a single blank line when `out` currently ends with real content, so
/// callers can guarantee a one-line gap without ever stacking double blanks
/// (no-op when `out` is empty or already ends blank).
fn push_blank_if_content(out: &mut Vec<Line<'static>>) {
    let ends_with_content = out.last().is_some_and(|line| {
        line.spans
            .iter()
            .any(|span| !span.content.trim().is_empty())
    });
    if ends_with_content {
        out.push(Line::from(""));
    }
}

/// Render and drain the accumulated non-table lines through the full markdown
/// [`Renderer`], appending the styled output to `out`.
fn flush_prose_segment(
    prose: &mut Vec<&str>,
    theme: &Theme,
    width: u16,
    highlight: bool,
    out: &mut Vec<Line<'static>>,
) {
    if prose.is_empty() {
        return;
    }
    let segment = prose.join("\n");
    out.append(&mut Renderer::new(theme, width, highlight).render(&segment));
    prose.clear();
}

/// Promote short standalone bold labels (`**답변**`, `**결론**`) into small
/// section headings. Models often emit these as visual section labels, but
/// CommonMark treats the following newline as a soft break, making the label
/// melt into the paragraph. Rendering them as H3 restores readable rhythm
/// without changing ordinary inline bold text.
fn promote_standalone_bold_labels(text: &str) -> Cow<'_, str> {
    if !text.contains('\n') {
        return Cow::Borrowed(text);
    }

    let chunks = text.split_inclusive('\n').collect::<Vec<_>>();
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    for (idx, chunk) in chunks.iter().enumerate() {
        let chunk = *chunk;
        let (line, ending) = chunk
            .strip_suffix('\n')
            .map_or((chunk, ""), |line| (line, "\n"));
        let has_following_body = chunks[idx + 1..].iter().any(|next| {
            let line = next.strip_suffix('\n').unwrap_or(next);
            !line.trim().is_empty()
        });
        if let Some(label) = standalone_bold_label(line).filter(|_| has_following_body) {
            changed = true;
            out.push_str("### ");
            out.push_str(label);
            out.push_str(ending);
        } else {
            out.push_str(chunk);
        }
    }
    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(text)
    }
}

fn standalone_bold_label(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix("**")?.strip_suffix("**")?.trim();
    if inner.is_empty() || inner.contains("**") {
        return None;
    }
    // Keep this to short label-like rows; long bold paragraphs should remain
    // ordinary emphasis.
    if cell_display_width(inner) > 48 {
        return None;
    }
    Some(inner)
}

/// 본문이 layout-preserving (표 / pure ASCII tree) 인지 — 캐시·wrap 결정에 사용.
/// heading / fenced code 가 함께 있으면 markdown 으로 빠지므로 layout 보존 대상 아님.
#[must_use]
pub fn preserves_layout(text: &str) -> bool {
    let strong_markdown = has_strong_markdown_signal(text);
    if looks_like_terminal_capture(text) {
        return !strong_markdown
            || (looks_like_dense_terminal_capture(text)
                && !has_authored_markdown_block_structure(text));
    }
    if has_markdown_table(text) {
        // Pure table → preformatted, pre-fit to width → draw no-wrap.
        // Table mixed with headings/fences is segment-rendered and drawn
        // *wrapped* (table rows still fit width), so it is not layout-preserving.
        return !strong_markdown;
    }
    contains_box_drawing(text) && !strong_markdown
}

fn is_list_item_or_blockquote_marker(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("> ") || trimmed == ">" {
        return true;
    }
    for marker in ["- [ ] ", "- [x] ", "- [X] ", "- ", "* ", "+ "] {
        if trimmed.starts_with(marker) {
            return true;
        }
    }
    let digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0
        && trimmed.as_bytes().get(digits) == Some(&b'.')
        && trimmed.as_bytes().get(digits + 1) == Some(&b' ')
    {
        return true;
    }
    false
}

/// 완료된 마크다운 블록 `[..n]` 과 스트리밍 중인 열린 꼬리 블록 `[n..]` 의
/// 분기 바이트 오프셋을 돌려준다 — 증분 스트리밍 렌더의 핵심.
///
/// CommonMark 블록은 빈 줄로 구분되므로, 최상위(코드펜스 바깥) 빈 줄 뒤에서
/// 끊으면 각 완료 세그먼트는 단독으로 파싱해도 동일한 결과가 나온다. 덕분에
/// 완료 블록은 한 번만 스타일링해 캐시하고, 매 프레임 작은 꼬리만 다시 그릴
/// 수 있다 (총 O(n), 프레임당 O(꼬리) → draw stall 없음, realtime-ux).
///
/// 규칙: 마지막 "비-빈줄 세그먼트"가 빈 줄로 닫히지 않았으면 그 세그먼트는
/// 열린(streaming) 블록 → 그 시작 오프셋을 반환. 텍스트가 빈 줄로 끝나
/// 열린 세그먼트가 없으면 전부 완료된 것이므로 `text.len()` 을 반환한다.
/// 코드펜스(```` ``` ````/`~~~`) 내부의 빈 줄은 경계로 보지 않는다.
#[must_use]
pub fn stable_prefix_len(text: &str) -> usize {
    let mut fence: Option<(u8, usize)> = None;
    let mut in_segment = false;
    let mut open_start = 0usize;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let len = line.len();
        let trimmed_start = line.trim_start();
        let run = leading_fence(trimmed_start);
        match fence {
            None => {
                if let Some(marker) = run {
                    fence = Some(marker);
                }
            }
            Some((fc, fn_len)) => {
                if let Some((c, n)) = run {
                    // 닫힘 펜스: 같은 문자, 길이 이상, 뒤에 정보문자열 없음.
                    if c == fc && n >= fn_len && trimmed_start[n..].trim().is_empty() {
                        fence = None;
                    }
                }
            }
        }
        let blank = line.trim().is_empty();
        let is_marker = fence.is_none() && is_list_item_or_blockquote_marker(line);
        if fence.is_some() {
            // 코드펜스 내부 — 빈 줄이라도 블록 본문으로 취급.
            if !in_segment {
                open_start = offset;
                in_segment = true;
            }
        } else if blank {
            in_segment = false;
        } else if is_marker || !in_segment {
            open_start = offset;
            in_segment = true;
        }
        offset += len;
    }
    if in_segment { open_start } else { text.len() }
}

/// 스트리밍 증분 렌더용 안정 경계 + 펜스 컨텍스트를 **단일 O(text) 스캔**으로.
///
/// 반환 `(boundary, fence_at_stable, fence_at_boundary, open_fence_lang)`:
/// - `boundary`: [`stable_prefix_len`] 과 동일하되, **마지막 열린 세그먼트가
///   `large_fence_threshold` 를 넘는 코드펜스**일 때만 경계를 그 펜스 안의 마지막
///   *완료된 줄* 뒤로 전진시킨다. 그러면 프레임당 다시 그리는 열린 꼬리가 펜스
///   전체가 아니라 마지막 미완성 줄 하나로 줄어든다(긴 코드답변 O(n²)→O(꼬리)).
/// - `fence_at_stable`: 바이트 `stable_len` 위치에서 열려 있던 펜스 마커 — 호출부가
///   승격 fragment `[stable_len..boundary]` 를 펜스 내부(코드)로 렌더하도록.
/// - `fence_at_boundary`: `boundary` 위치에서 열려 있던 펜스 마커 — 열린 꼬리
///   `[boundary..]` 를 펜스 내부로 렌더하도록.
/// - `open_fence_lang`: 큰 펜스가 열린 채 경계가 전진한 경우, 그 펜스 여는 줄의
///   info string 첫 단어(언어). 스트리밍 인테리어가 done 과 동일한 카드 상단
///   보더 라벨(`╭─ rust ─`)을 그리도록(roadmap ⑨). 큰-펜스 분기에서만 Some.
///
/// 펜스만 경계를 전진시킨다(코드 줄은 서로 독립이라 완료 줄이 재렌더되지 않음).
/// 산문/표는 열린 세그먼트 전체를 꼬리로 둔다(완료된 산문 줄도 이후 텍스트가
/// 도착하면 다시 wrap 될 수 있으므로). 임계 미만 펜스는 기존 경계 그대로(무변경).
/// A code-fence marker: `(fence char, run length)` — e.g. `(b'`', 3)` for ```` ``` ````.
pub type FenceMarker = (u8, usize);

/// Resumable scan cursor at a byte boundary, cached by the streaming layout so
/// the next frame can continue [`streaming_stable_prefix`] from the previous
/// boundary instead of re-scanning the entire accumulated text. This turns the
/// per-frame cost from O(total length) into O(newly streamed suffix) — the fix
/// for a long streamed answer freezing the TUI as it scans more text each frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StreamScanState {
    /// Byte offset this cursor describes — the boundary returned last frame.
    at: usize,
    fence: Option<FenceMarker>,
    open_fence_lang: Option<String>,
    in_segment: bool,
    open_start: usize,
    last_newline_end: usize,
}

/// Compute the stable-prefix boundary of streaming markdown text. See
/// [`streaming_stable_prefix_resumed`] for the incremental, cache-backed variant
/// the live layout uses; this full-scan form is kept for callers (and tests)
/// that have no prior cursor.
#[must_use]
pub fn streaming_stable_prefix(
    text: &str,
    stable_len: usize,
    large_fence_threshold: usize,
) -> (usize, Option<FenceMarker>, Option<FenceMarker>, Option<String>) {
    let (boundary, at_stable, at_boundary, lang, _state) =
        streaming_stable_prefix_resumed(text, stable_len, large_fence_threshold, None);
    (boundary, at_stable, at_boundary, lang)
}

/// [`streaming_stable_prefix`], resumable. When `resume` describes the cursor at
/// exactly `stable_len`, the scan starts there rather than at byte 0; otherwise
/// it falls back to a full scan (correctness over speed). Returns the usual
/// 4-tuple plus the cursor at the returned boundary — cache it and pass it back
/// next frame, where `resume.at == stable_len` will then hold.
///
/// The returned cursor is only meaningful (and `cursor.at == boundary`) when the
/// boundary advances past `stable_len`; the caller must only persist it then,
/// mirroring its own stable-region promotion.
#[must_use]
#[allow(clippy::too_many_lines)] // one cohesive scan; splitting it would scatter the cursor state
pub(crate) fn streaming_stable_prefix_resumed(
    text: &str,
    stable_len: usize,
    large_fence_threshold: usize,
    resume: Option<&StreamScanState>,
) -> (
    usize,
    Option<FenceMarker>,
    Option<FenceMarker>,
    Option<String>,
    StreamScanState,
) {
    // Resume only when the cached cursor sits exactly at this frame's stable
    // boundary and that boundary is within the text; else re-scan from scratch.
    let resume = resume.filter(|s| s.at == stable_len && stable_len <= text.len());

    let mut fence: Option<FenceMarker>;
    let mut open_fence_lang: Option<String>;
    let mut in_segment: bool;
    let mut open_start: usize;
    let mut last_newline_end: usize;
    let mut offset: usize;
    let mut fence_at_stable: Option<FenceMarker>;
    let mut stable_captured: bool;
    // A huge promoted fence's boundary is the last completed newline, so seed the
    // post-newline snapshot from `resume` — a frame that adds no new newline then
    // still has a valid cursor exactly at `stable_len`.
    let mut post_newline_state: Option<StreamScanState>;

    if let Some(s) = resume {
        fence = s.fence;
        open_fence_lang = s.open_fence_lang.clone();
        in_segment = s.in_segment;
        open_start = s.open_start;
        last_newline_end = s.last_newline_end;
        offset = stable_len;
        fence_at_stable = s.fence;
        stable_captured = true;
        post_newline_state = Some(s.clone());
    } else {
        fence = None;
        open_fence_lang = None;
        in_segment = false;
        open_start = 0;
        last_newline_end = 0;
        offset = 0;
        fence_at_stable = None;
        stable_captured = false;
        post_newline_state = None;
    }

    // The cursor at the start of the current open segment (pre-line state), the
    // resume point when the boundary is `open_start`.
    let mut seg_open_state: Option<StreamScanState> = None;

    let scan_from = if resume.is_some() { stable_len } else { 0 };
    for line in text[scan_from..].split_inclusive('\n') {
        // Fence state at `stable_len` = the state *before* the line that starts
        // there (boundaries are always line starts). Capture it once.
        if !stable_captured && offset >= stable_len {
            fence_at_stable = fence;
            stable_captured = true;
        }
        // Pre-line snapshot, kept in case this line opens the boundary segment.
        let pre = StreamScanState {
            at: offset,
            fence,
            open_fence_lang: open_fence_lang.clone(),
            in_segment,
            open_start,
            last_newline_end,
        };
        let len = line.len();
        let trimmed_start = line.trim_start();
        let run = leading_fence(trimmed_start);
        match fence {
            None => {
                if let Some(marker) = run {
                    fence = Some(marker);
                    // Capture the FULL info string (trimmed) exactly as the done
                    // pass does (`on_code_block_start` keeps the whole CodeBlockKind
                    // ::Fenced lang), so the streaming card's top-border label is
                    // byte-identical to done — not just the first word.
                    let n = marker.1;
                    let info = trimmed_start[n..].trim();
                    open_fence_lang = (!info.is_empty()).then(|| info.to_string());
                }
            }
            Some((fc, fn_len)) => {
                if let Some((c, n)) = run {
                    if c == fc && n >= fn_len && trimmed_start[n..].trim().is_empty() {
                        fence = None;
                        // Clear the lang on close so it is never read stale; it is
                        // only consulted while a fence is open, so this is
                        // behavior-preserving and keeps the cursor minimal.
                        open_fence_lang = None;
                    }
                }
            }
        }
        let blank = line.trim().is_empty();
        let is_marker = fence.is_none() && is_list_item_or_blockquote_marker(line);
        if fence.is_some() {
            if !in_segment {
                open_start = offset;
                in_segment = true;
                seg_open_state = Some(pre);
            }
        } else if blank {
            in_segment = false;
        } else if is_marker || !in_segment {
            open_start = offset;
            in_segment = true;
            seg_open_state = Some(pre);
        }
        offset += len;
        if line.ends_with('\n') {
            last_newline_end = offset;
            post_newline_state = Some(StreamScanState {
                at: offset,
                fence,
                open_fence_lang: open_fence_lang.clone(),
                in_segment,
                open_start,
                last_newline_end,
            });
        }
    }
    if !stable_captured {
        // stable_len at/after EOF — the fence state at the end applies.
        fence_at_stable = fence;
    }
    let eof_state = StreamScanState {
        at: text.len(),
        fence,
        open_fence_lang: open_fence_lang.clone(),
        in_segment,
        open_start,
        last_newline_end,
    };
    let normal_boundary = if in_segment { open_start } else { text.len() };
    // A large open code fence: promote its completed lines so the open tail is
    // just the trailing partial line. `open_start` is where the fence segment
    // began; everything up to `last_newline_end` is completed code.
    if let Some(marker) = fence {
        if text.len().saturating_sub(open_start) > large_fence_threshold
            && last_newline_end > normal_boundary
        {
            let state = post_newline_state.unwrap_or_else(|| eof_state.clone());
            return (
                last_newline_end,
                fence_at_stable,
                Some(marker),
                open_fence_lang,
                state,
            );
        }
    }
    // `seg_open_state.at == open_start == normal_boundary` whenever the boundary
    // advanced (a segment opened in this scan); `eof_state` covers the not-in-
    // segment boundary at EOF. The caller only persists this when the boundary
    // moves past `stable_len`, so the fallback never feeds a stale cursor back.
    let boundary_state = if in_segment {
        seg_open_state.unwrap_or(eof_state)
    } else {
        eof_state
    };
    (normal_boundary, fence_at_stable, None, None, boundary_state)
}

/// 줄 시작의 연속된 `` ` `` / `~` 펜스 run 을 `(문자, 길이)` 로. 3 미만은 무시.
pub(super) fn leading_fence(s: &str) -> Option<(u8, usize)> {
    let bytes = s.as_bytes();
    let first = *bytes.first()?;
    if first != b'`' && first != b'~' {
        return None;
    }
    let mut n = 0;
    while n < bytes.len() && bytes[n] == first {
        n += 1;
    }
    (n >= 3).then_some((first, n))
}

// ============================================================================
// Renderer — per-event 상태머신
// ============================================================================

// A pulldown-cmark event sink: each bool is an independent parse-state flag
// (highlight on/off, marker sniffing, callout break, code-pad defer), not a
// configuration record, so collapsing them into a struct adds no clarity.
#[allow(clippy::struct_excessive_bools)]
struct Renderer<'t> {
    theme: &'t Theme,
    width: u16,
    /// `false` 이면 코드블록에 syntect 하이라이트를 적용하지 않는다 —
    /// 스트리밍 꼬리 렌더가 프레임당 비용을 낮추려고 끈다.
    highlight: bool,
    lines: Vec<Vec<Span<'static>>>,
    bold: u8,
    italic: u8,
    strike: u8,
    code: u8,
    code_buffer: String,
    code_lang: Option<String>,
    heading: Option<HeadingLevel>,
    blockquote: u8,
    /// `Some` once the outermost blockquote is recognised as a GitHub
    /// admonition (`> [!NOTE]` …). Drives the rail color + label.
    blockquote_kind: Option<CalloutKind>,
    /// `true` immediately after entering an outermost blockquote, until
    /// the first text tokens are sniffed for an admonition marker.
    blockquote_marker_pending: bool,
    /// Accumulates the leading blockquote text while sniffing — pulldown
    /// splits `[!NOTE]` into `[`, `!NOTE`, `]` tokens.
    blockquote_marker_buf: String,
    /// `true` right after a callout header is emitted so the next soft
    /// break drops the body onto its own line under the label.
    callout_break_pending: bool,
    /// `true` right after an inline-code span, while we wait to see what
    /// follows. The trailing 1-cell pad is only emitted before a Latin
    /// word char — never before punctuation (`,` `.` `)` …) or a Korean
    /// particle, which attach directly. Eagerly padding produced the
    /// stray gaps "`sqlgate` ," / "`1.26.3` 로".
    code_pad_pending: bool,
    link_url: Option<String>,
    link_text_acc: String,
    /// `None` == unordered, `Some(n)` == ordered counter starting at n.
    list_stack: Vec<Option<u64>>,
    in_list_item: u8,
    /// Display-cell width of the current list item's marker (including depth
    /// indent). Used to pre-wrap long items so continuation rows align under
    /// the item text rather than at column 0. Reset to 0 outside a list item.
    list_item_marker_width: usize,
    /// `self.lines` index at which the current list item began. Used by
    /// `pre_wrap_item_lines` to limit back-scan to the current item only.
    item_start_line_idx: usize,
}

impl<'t> Renderer<'t> {
    fn new(theme: &'t Theme, width: u16, highlight: bool) -> Self {
        Self {
            theme,
            width,
            highlight,
            lines: vec![Vec::new()],
            bold: 0,
            italic: 0,
            strike: 0,
            code: 0,
            code_buffer: String::new(),
            code_lang: None,
            heading: None,
            blockquote: 0,
            blockquote_kind: None,
            blockquote_marker_pending: false,
            blockquote_marker_buf: String::new(),
            callout_break_pending: false,
            code_pad_pending: false,
            link_url: None,
            link_text_acc: String::new(),
            list_stack: Vec::new(),
            in_list_item: 0,
            list_item_marker_width: 0,
            item_start_line_idx: 0,
        }
    }

    fn render(mut self, text: &str) -> Vec<Line<'static>> {
        let mut opts = Options::empty();
        opts.insert(Options::ENABLE_STRIKETHROUGH);
        opts.insert(Options::ENABLE_TASKLISTS);
        // `$…$` / `$$…$$` parse as math events (rendered verbatim) instead of
        // the `*`/`_` inside them toggling emphasis and eating operators.
        opts.insert(Options::ENABLE_MATH);
        let readable = polish_dense_prose_for_display(text);
        let normalized = promote_standalone_bold_labels(readable.as_ref());
        let parser = Parser::new_ext(normalized.as_ref(), opts);
        for ev in parser {
            self.handle(ev);
        }
        self.lines
            .into_iter()
            .map(|spans| {
                if spans.is_empty() {
                    Line::from("")
                } else {
                    Line::from(spans)
                }
            })
            .collect()
    }

    fn handle(&mut self, ev: Event<'_>) {
        match ev {
            Event::Start(Tag::Strong) => self.bold = self.bold.saturating_add(1),
            Event::End(TagEnd::Strong) => self.bold = self.bold.saturating_sub(1),
            Event::Start(Tag::Emphasis) => self.italic = self.italic.saturating_add(1),
            Event::End(TagEnd::Emphasis) => self.italic = self.italic.saturating_sub(1),
            Event::Start(Tag::Strikethrough) => self.strike = self.strike.saturating_add(1),
            Event::End(TagEnd::Strikethrough) => self.strike = self.strike.saturating_sub(1),
            Event::Code(c) => self.on_inline_code(&c),
            Event::Start(Tag::CodeBlock(kind)) => self.on_code_block_start(kind),
            Event::End(TagEnd::CodeBlock) => self.on_code_block_end(),
            Event::Start(Tag::Heading { level, .. }) => self.on_heading_start(level),
            Event::End(TagEnd::Heading(_)) => self.on_heading_end(),
            Event::Start(Tag::BlockQuote(_)) => self.on_blockquote_start(),
            Event::End(TagEnd::BlockQuote(_)) => self.on_blockquote_end(),
            Event::Rule => self.on_rule(),
            Event::Start(Tag::Link { dest_url, .. }) => self.on_link_start(&dest_url),
            Event::End(TagEnd::Link) => self.on_link_end(),
            Event::Start(Tag::List(start)) => self.list_stack.push(start),
            Event::End(TagEnd::List(_)) => self.on_list_end(),
            Event::Start(Tag::Item) => self.on_item_start(),
            Event::End(TagEnd::Item) => self.on_item_end(),
            Event::TaskListMarker(checked) => self.on_task_list_marker(checked),
            // LaTeX math renders verbatim with its delimiters (a terminal has
            // no math layout). Without ENABLE_MATH the `$a * b$` operators
            // parsed as emphasis and vanished from the rendered line — worst
            // for GPT models, which emit math notation freely.
            Event::InlineMath(math) => self.on_inline_math(&math, false),
            Event::DisplayMath(math) => self.on_inline_math(&math, true),
            Event::Text(t) => self.on_text(&t),
            Event::SoftBreak => self.on_soft_break(),
            Event::HardBreak => self.push_blank_line(),
            Event::End(TagEnd::Paragraph) => {
                self.flush_pending_marker();
                if self.in_list_item == 0 {
                    self.ensure_visible_blank_line();
                }
            }
            _ => {}
        }
    }

    // ---- text & inline ---------------------------------------------------

    fn on_text(&mut self, text: &str) {
        if self.code > 0 {
            self.code_buffer.push_str(text);
            return;
        }
        // Leading text of an outermost blockquote is buffered until we can
        // tell whether it is a GitHub admonition marker (`[!NOTE]` …),
        // which pulldown-cmark splits into `[`, `!NOTE`, `]` tokens.
        if self.blockquote_marker_pending {
            match self.feed_marker(text) {
                MarkerFeed::Pending | MarkerFeed::Callout => return,
                MarkerFeed::NotMarker(flushed) => {
                    self.render_inline_text(&flushed);
                    return;
                }
            }
        }
        self.render_inline_text(text);
    }

    /// Render a run of inline text with the active emphasis / heading /
    /// link / blockquote style.
    fn render_inline_text(&mut self, text: &str) {
        if self.link_url.is_some() {
            self.link_text_acc.push_str(text);
        }
        let style = if let Some(level) = self.heading {
            heading_text_style(self.theme, level)
        } else if self.link_url.is_some() {
            link_style(self.theme)
        } else {
            inline_style(
                self.theme,
                InlineFlags {
                    bold: self.bold > 0,
                    italic: self.italic > 0,
                    strike: self.strike > 0,
                    in_blockquote: self.blockquote > 0,
                },
            )
        };
        self.push_multiline(text, style);
    }

    /// Feed one text token to the admonition recogniser, accumulating
    /// until a `[…]` marker resolves (or is ruled out).
    fn feed_marker(&mut self, text: &str) -> MarkerFeed {
        self.blockquote_marker_buf.push_str(text);
        let trimmed = self.blockquote_marker_buf.trim_start();
        if trimmed.is_empty() {
            return MarkerFeed::Pending;
        }
        if !trimmed.starts_with('[') {
            self.blockquote_marker_pending = false;
            return MarkerFeed::NotMarker(std::mem::take(&mut self.blockquote_marker_buf));
        }
        let Some(close) = trimmed.find(']') else {
            return MarkerFeed::Pending;
        };
        let marker_part = trimmed[..=close].to_string();
        let rest = trimmed[close + 1..].trim_start().to_string();
        self.blockquote_marker_pending = false;
        let buf = std::mem::take(&mut self.blockquote_marker_buf);
        let Some((kind, _)) = parse_callout_marker(&marker_part) else {
            // Bracketed but not a known admonition → ordinary text.
            return MarkerFeed::NotMarker(buf);
        };
        self.blockquote_kind = Some(kind);
        self.apply_callout_header(kind);
        if !rest.is_empty() {
            self.callout_break_pending = false;
            self.push_blank_line();
            self.render_inline_text(&rest);
        }
        MarkerFeed::Callout
    }

    /// Flush any buffered (unresolved) marker text as ordinary inline text.
    fn flush_pending_marker(&mut self) {
        if self.blockquote_marker_pending {
            self.blockquote_marker_pending = false;
            let flushed = std::mem::take(&mut self.blockquote_marker_buf);
            if !flushed.is_empty() {
                self.render_inline_text(&flushed);
            }
        }
    }

    /// LaTeX math, verbatim with its `$`/`$$` delimiters restored: a terminal
    /// cannot typeset it, but showing the raw notation beats the pre-math
    /// behavior where `*`/`_` inside `$…$` parsed as emphasis and dropped
    /// operators from formulas.
    fn on_inline_math(&mut self, math: &str, display: bool) {
        let delim = if display { "$$" } else { "$" };
        let style = code_inline_style(self.theme);
        self.last_line()
            .push(Span::styled(format!("{delim}{math}{delim}"), style));
        if display {
            self.push_blank_line();
        }
    }

    fn on_inline_code(&mut self, text: &str) {
        // A previous code span's deferred pad is resolved against this one's
        // first glyph (handled below by the leading-pad check), so drop it.
        self.code_pad_pending = false;
        let style = code_inline_style(self.theme);
        // Phase 2.3 — 인접 글자에 붙지 않게 1셀 패딩. 직전 span 이 이미
        // 공백으로 끝나면 생략.
        let prev_ends_with_ws = self
            .last_line()
            .last()
            .and_then(|span| span.content.chars().next_back())
            .is_none_or(char::is_whitespace);
        if !prev_ends_with_ws {
            self.last_line().push(Span::raw(" "));
        }
        self.last_line().push(Span::styled(text.to_string(), style));
        // 뒤쪽 패딩은 다음 인라인이 올 때 첫 글자를 보고 결정한다(지연).
        // 코드 바로 뒤에 구두점이나 한글 조사가 붙으면 공백을 넣지 않는다.
        self.code_pad_pending = true;
    }

    /// Resolve a deferred inline-code trailing pad against the glyph that is
    /// about to follow. A 1-cell pad keeps prose like "`flag`text" legible,
    /// but punctuation, whitespace and CJK (Korean particles) attach with no
    /// gap — so "`sqlgate`," stays tight and "`1.26.3`로" reads correctly.
    fn flush_code_pad(&mut self, next: Option<char>) {
        if !std::mem::take(&mut self.code_pad_pending) {
            return;
        }
        if next.is_some_and(|c| c.is_ascii_alphanumeric()) {
            self.last_line().push(Span::raw(" "));
        }
    }

    // ---- code blocks -----------------------------------------------------

    fn on_code_block_start(&mut self, kind: CodeBlockKind<'_>) {
        self.code = self.code.saturating_add(1);
        self.code_buffer.clear();
        self.code_lang = match kind {
            CodeBlockKind::Fenced(lang) => {
                let lang = lang.into_string();
                if lang.is_empty() { None } else { Some(lang) }
            }
            CodeBlockKind::Indented => None,
        };
        self.ensure_blank_line();
    }

    fn on_code_block_end(&mut self) {
        self.code = self.code.saturating_sub(1);

        // Mermaid fences render as an actual box-and-arrow diagram. On parse
        // failure we fall through to the normal code-block render so the source
        // is never lost.
        if self
            .code_lang
            .as_deref()
            .is_some_and(|l| l.eq_ignore_ascii_case("mermaid"))
            && self.render_mermaid_diagram()
        {
            self.code_buffer.clear();
            self.code_lang = None;
            return;
        }

        // A text-family fence carries prose, not code (v3 readability): the
        // model routinely wraps a long natural-language answer in ```text,
        // and the no-wrap monotone code card turns it into a wall whose
        // overlong rows spill past the frame. Render it as a wrapped quote
        // block instead — dim rail, body at full reading contrast, physical
        // wrap to width so nothing ever crosses the edge. Real code fences
        // (rust/json/…, and bare ``` which conventionally holds code/logs)
        // are untouched.
        if self
            .code_lang
            .as_deref()
            .is_some_and(is_prose_fence_lang)
        {
            let rows = prose_fence_lines(&self.code_buffer, self.width, self.theme);
            self.lines.extend(rows);
            self.ensure_visible_blank_line();
            self.code_buffer.clear();
            self.code_lang = None;
            return;
        }

        let is_diff = self
            .code_lang
            .as_deref()
            .is_some_and(|l| l.eq_ignore_ascii_case("diff") || l.eq_ignore_ascii_case("patch"));

        // A ```diff / ```patch fence gets semantic +/- coloring (matching
        // the native `ToolResultBody::Diff` viewer) instead of generic
        // syntect highlighting, which left additions and removals an
        // undifferentiated gray.
        let highlighted = if is_diff {
            diff_code_rows(&self.code_buffer, self.theme)
        } else if self.highlight {
            highlight_code(&self.code_buffer, self.code_lang.as_deref(), self.theme)
        } else {
            plain_code_rows(&self.code_buffer, self.theme)
        };
        // The card frame (top border + "  │ " rail + bottom border) is the single
        // source shared with the giant-fence streaming interior (roadmap ⑨), so
        // a big code block streams with the exact frame it settles into.
        let frame = code_card_frame_lines(
            highlighted,
            self.code_lang.as_deref(),
            is_diff,
            self.width,
            self.theme,
            true,
            true,
        );
        self.lines.extend(frame);
        self.ensure_visible_blank_line();
        self.code_buffer.clear();
        self.code_lang = None;
    }

    /// Render the buffered mermaid source as a diagram. Returns `false` (without
    /// touching `self.lines`) when the source is unsupported, so the caller can
    /// fall back to the raw code-block render.
    fn render_mermaid_diagram(&mut self) -> bool {
        let Some(rows) = super::mermaid_layout::render(&self.code_buffer, self.width, self.theme)
        else {
            return false;
        };
        self.ensure_blank_line();
        for row in rows {
            self.lines.push(row);
        }
        self.push_blank_line();
        true
    }

    // ---- headings --------------------------------------------------------

    fn on_heading_start(&mut self, level: HeadingLevel) {
        self.ensure_blank_line();
        self.heading = Some(level);
        let (glyph, style) = heading_glyph(self.theme, level);
        self.last_line()
            .push(Span::styled(glyph.to_string(), style));
    }

    fn on_heading_end(&mut self) {
        let level = self.heading.take();
        // Document-section headings (H1/H2) close with one visible blank row —
        // body glued to the very next line reads cramped. H3+ (incl. promoted
        // bold labels and dense numbered sections) keep the compact next-row
        // rhythm so structured lists don't balloon the transcript.
        if matches!(level, Some(HeadingLevel::H1 | HeadingLevel::H2)) {
            self.ensure_visible_blank_line();
        } else {
            self.push_blank_line();
        }
    }

    // ---- blockquote ------------------------------------------------------

    fn on_blockquote_start(&mut self) {
        self.blockquote = self.blockquote.saturating_add(1);
        // Only the outermost quote sniffs for an admonition marker.
        if self.blockquote == 1 {
            self.blockquote_kind = None;
            self.blockquote_marker_pending = true;
        }
        self.ensure_blank_line();
        self.prepend_blockquote_rail_if_empty();
    }

    fn on_blockquote_end(&mut self) {
        self.flush_pending_marker();
        self.blockquote = self.blockquote.saturating_sub(1);
        if self.blockquote == 0 {
            self.blockquote_kind = None;
            self.blockquote_marker_pending = false;
            self.callout_break_pending = false;
            // Closing the outermost quote, strip any trailing rail-only rows
            // (a `▌`/`▎` glyph with no body) so a callout does not leave dangling
            // colored rail stubs below its last line of text.
            self.strip_trailing_rail_only_lines();
        }
        self.ensure_visible_blank_line();
    }

    /// Drop trailing lines that contain only a blockquote/callout rail glyph
    /// and whitespace. Called when the outermost quote closes so the rail does
    /// not extend past the quote's actual content as empty colored stubs.
    fn strip_trailing_rail_only_lines(&mut self) {
        while self
            .lines
            .last()
            .is_some_and(|line| line_is_rail_only(line))
        {
            self.lines.pop();
        }
    }

    /// A soft break inside a callout header drops the body onto a fresh
    /// rail-prefixed line; everywhere else it is the usual space.
    fn on_soft_break(&mut self) {
        // The break itself separates, so drop any deferred code pad.
        self.code_pad_pending = false;
        // A break before the marker resolved means it was never a marker.
        if self.blockquote_marker_pending {
            self.flush_pending_marker();
            self.last_line().push(Span::raw(" "));
            return;
        }
        if self.callout_break_pending {
            self.callout_break_pending = false;
            self.push_blank_line();
        } else {
            self.last_line().push(Span::raw(" "));
        }
    }

    /// Recolor the just-pushed quote rail into a callout rail and append
    /// the bold admonition label (e.g. `Note`, `Warning`).
    fn apply_callout_header(&mut self, kind: CalloutKind) {
        let rail = blockquote_rail_span(self.theme, Some(kind));
        let label_style = self.theme.callout_style(kind);
        let label = callout_label(kind).to_string();
        {
            let line = self.last_line();
            if line.first().is_some_and(|s| is_rail_glyph(&s.content)) {
                line[0] = rail;
            } else {
                line.insert(0, rail);
            }
        }
        self.last_line().push(Span::styled(label, label_style));
        self.callout_break_pending = true;
    }

    // ---- horizontal rule -------------------------------------------------

    fn on_rule(&mut self) {
        self.ensure_blank_line();
        let target = usize::from(self.width).saturating_sub(4).max(HR_MIN_WIDTH);
        let style = Style::new().fg(self.theme.palette.dim);
        self.last_line()
            .push(Span::styled("─".repeat(target), style));
        self.ensure_visible_blank_line();
    }

    // ---- links -----------------------------------------------------------

    fn on_link_start(&mut self, dest: &str) {
        self.link_url = Some(dest.to_string());
        self.link_text_acc.clear();
    }

    fn on_link_end(&mut self) {
        let Some(url) = self.link_url.take() else {
            return;
        };
        // Phase 2.7 — 화살표 글리프.
        let arrow = if self.theme.no_color {
            " *"
        } else {
            " \u{2197}"
        };
        let dim_style = Style::new().fg(self.theme.palette.dim);
        self.last_line()
            .push(Span::styled(arrow.to_string(), dim_style));
        // text 와 url 이 동일하면 ( url ) 접미 생략 — 중복 noise 제거.
        let text_eq_url = self.link_text_acc.trim() == url.trim();
        if !text_eq_url {
            self.last_line()
                .push(Span::styled(format!(" ({url})"), dim_style));
        }
        self.link_text_acc.clear();
    }

    // ---- lists -----------------------------------------------------------

    fn on_list_end(&mut self) {
        self.list_stack.pop();
        if self.list_stack.is_empty() {
            self.ensure_visible_blank_line();
        }
    }

    fn on_item_start(&mut self) {
        self.in_list_item = self.in_list_item.saturating_add(1);
        self.ensure_blank_line();
        let depth = self.list_stack.len().saturating_sub(1);
        let indent = "  ".repeat(depth);
        let (marker, marker_style) = if let Some(Some(n)) = self.list_stack.last_mut() {
            let text = format!("{indent}{n}. ");
            *n = n.saturating_add(1);
            // Ordered numbers are markers, not content: give them the same
            // recessive treatment as unordered bullets (depth-quiet neutral, no
            // hue, no BOLD) so a numbered list stops being a "색 이벤트". The
            // ordinal value is carried by the digits, not by color weight.
            let style = Style::new().fg(bullet_color_for_depth(self.theme, depth));
            (text, style)
        } else {
            let glyph = bullet_glyph_for_depth(depth, self.theme.no_color);
            let color = bullet_color_for_depth(self.theme, depth);
            (format!("{indent}{glyph} "), Style::new().fg(color))
        };
        // Record where this item starts (the blank line we just ensured) and
        // what the marker's display width is, so `on_item_end` can pre-wrap.
        self.item_start_line_idx = self.lines.len().saturating_sub(1);
        self.list_item_marker_width = cell_display_width(&marker);
        self.last_line().push(Span::styled(marker, marker_style));
    }

    fn on_item_end(&mut self) {
        // Pre-wrap any list-item lines that are wider than `self.width` so
        // ratatui's wrap is a no-op on them and height==draw stays consistent.
        if self.width > 0 {
            self.pre_wrap_item_lines();
        }
        self.in_list_item = self.in_list_item.saturating_sub(1);
        self.list_item_marker_width = 0;
    }

    /// GFM task item: the parser consumes the literal `[x]`/`[ ]`, so render a
    /// real checkbox. Unordered items swap their bullet for the box
    /// (GitHub-style); ordered task items keep the number and get the box
    /// appended after it.
    fn on_task_list_marker(&mut self, checked: bool) {
        let depth = self.list_stack.len().saturating_sub(1);
        let unordered = matches!(self.list_stack.last(), Some(None));
        if unordered {
            // Drop the freshly pushed `{indent}{bullet} ` span; the checkbox
            // takes the bullet's place at the same indent.
            self.last_line().pop();
        }
        let indent = if unordered {
            "  ".repeat(depth)
        } else {
            String::new()
        };
        let (glyph, style) = if self.theme.no_color {
            let text = if checked { "[x] " } else { "[ ] " };
            (text.to_string(), Style::new())
        } else if checked {
            (
                "☑ ".to_string(),
                Style::new().fg(self.theme.palette.success),
            )
        } else {
            ("☐ ".to_string(), Style::new().fg(self.theme.palette.dim))
        };
        let full_marker = format!("{indent}{glyph}");
        // Update the marker width so pre_wrap_item_lines uses the checkbox width.
        self.list_item_marker_width = cell_display_width(&full_marker);
        self.last_line().push(Span::styled(full_marker, style));
    }

    // ---- helpers ---------------------------------------------------------

    /// Post-process the lines added for the current list item (from
    /// `item_start_line_idx` to the end of `self.lines`). Any line whose
    /// display width exceeds `self.width` is split into multiple lines, where
    /// each continuation line begins with a blank span of exactly
    /// `list_item_marker_width` cells — the hanging indent. This keeps height
    /// measurement (which counts `self.lines`) equal to what ratatui draws
    /// (which sees already-split lines and has nothing left to wrap).
    fn pre_wrap_item_lines(&mut self) {
        let w = usize::from(self.width);
        let hanging = self.list_item_marker_width;
        if w == 0 || hanging == 0 || hanging >= w {
            return;
        }
        let start = self.item_start_line_idx;
        // Drain the item's lines, re-wrap each one, and collect the results.
        // The first line carries the marker span as its leading span — pass
        // `keep_first_span = true` so it is emitted verbatim without being
        // split on the spaces inside its depth-indent prefix.
        let item_lines: Vec<Vec<Span<'static>>> = self.lines.drain(start..).collect();
        let mut is_first = true;
        for spans in item_lines {
            let wrapped = wrap_spans_with_hanging(spans, w, hanging, is_first);
            is_first = false;
            self.lines.extend(wrapped);
        }
    }

    fn last_line(&mut self) -> &mut Vec<Span<'static>> {
        if self.lines.is_empty() {
            self.lines.push(Vec::new());
        }
        self.lines
            .last_mut()
            .expect("lines is non-empty by construction above")
    }

    /// 현재 마지막 라인이 비어있으면 그대로, 비어있지 않으면 새 빈 라인 추가.
    fn ensure_blank_line(&mut self) {
        if !self.lines.last().is_some_and(Vec::is_empty) {
            self.push_blank_line();
        }
    }

    /// 빈 라인 push. blockquote 활성 시 자동으로 레일 prepend (callout 이면
    /// 종류별 색의 `▌`, 일반 인용이면 `▎`).
    fn push_blank_line(&mut self) {
        // A line break supersedes any deferred inline-code trailing pad.
        self.code_pad_pending = false;
        let mut spans = Vec::new();
        if self.blockquote > 0 {
            spans.push(blockquote_rail_span(self.theme, self.blockquote_kind));
        }
        self.lines.push(spans);
    }

    /// Ensure a future block starts after one visible blank row. A single blank
    /// line is consumed by the next block's first text/marker, so block-level
    /// separators need two trailing blank-ish rows while building.
    fn ensure_visible_blank_line(&mut self) {
        while self.trailing_blankish_lines() < 2 {
            self.push_blank_line();
        }
    }

    fn trailing_blankish_lines(&self) -> usize {
        self.lines
            .iter()
            .rev()
            .take_while(|line| spans_are_blankish(line))
            .count()
    }

    /// 현재 라인이 빈 상태이고 blockquote 가 활성이면 레일을 추가한다.
    fn prepend_blockquote_rail_if_empty(&mut self) {
        if self.blockquote > 0
            && self
                .lines
                .last()
                .is_some_and(|l| l.iter().all(|s| s.content.is_empty()))
        {
            let rail = blockquote_rail_span(self.theme, self.blockquote_kind);
            self.last_line().push(rail);
        }
    }

    /// `text` 가 `\n` 을 포함할 때 라인 단위로 분할해 push.
    ///
    /// pulldown-cmark 는 fenced code 본문을 단일 `Event::Text` 로 보내므로
    /// 임베디드 `\n` 을 splat 해주지 않으면 ratatui 가 그대로 출력해 staircase
    /// 가 발생한다.
    fn push_multiline(&mut self, text: &str, style: Style) {
        self.flush_code_pad(text.chars().next());
        let mut parts = text.split('\n');
        if let Some(first) = parts.next() {
            if !first.is_empty() {
                self.last_line()
                    .push(Span::styled(first.to_string(), style));
            }
        }
        for rest in parts {
            self.push_blank_line();
            if !rest.is_empty() {
                self.last_line().push(Span::styled(rest.to_string(), style));
            }
        }
    }
}

mod prose;
use prose::polish_dense_prose_for_display;

pub(crate) mod syntax;
use syntax::SyntaxHighlighter;

/// Outcome of feeding a text token to the admonition-marker recogniser.
enum MarkerFeed {
    /// Still buffering — the run could yet become a `[…]` marker.
    Pending,
    /// Recognised and emitted a callout header.
    Callout,
    /// Not a marker — the caller must render the returned text normally.
    NotMarker(String),
}

mod style;
use style::{
    InlineFlags, blockquote_rail_span, bullet_color_for_depth, bullet_glyph_for_depth,
    callout_label, code_inline_style, heading_glyph, heading_text_style, inline_style,
    is_rail_glyph, line_is_rail_only, link_style, parse_callout_marker, spans_are_blankish,
};

// ============================================================================
// Layout-preserving 감지 (table / box-drawing)
// ============================================================================

fn preformatted_lines(rendered: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    let style = Style::new().fg(theme.palette.fg);
    let dim_style = Style::new().fg(theme.palette.dim);
    let mut lines: Vec<Line<'static>> = Vec::new();

    if width == 0 {
        for raw in rendered.lines() {
            lines.push(Line::from(vec![Span::styled(raw.to_string(), style)]));
        }
        if lines.is_empty() {
            lines.push(Line::from(""));
        }
        return lines;
    }

    let max = usize::from(width);
    for raw in rendered.lines() {
        let mut remaining = raw;
        let mut first = true;
        let continuation_prefix = preformatted_continuation_prefix(raw, theme);
        let continuation_width = cell_display_width(&continuation_prefix);
        let cont_max = max.saturating_sub(continuation_width).max(8);
        loop {
            let limit = if first { max } else { cont_max };
            let (head, tail) = split_at_display_width(remaining, limit);
            if first {
                lines.push(Line::from(vec![Span::styled(head.to_string(), style)]));
                first = false;
            } else {
                lines.push(Line::from(vec![
                    Span::styled(continuation_prefix.clone(), dim_style),
                    Span::styled(head.to_string(), style),
                ]));
            }
            if tail.is_empty() {
                break;
            }
            remaining = tail;
        }
    }
    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

fn preformatted_continuation_prefix(raw: &str, theme: &Theme) -> String {
    let leading = raw
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .map(char_display_width)
        .sum::<usize>();
    let indent = leading.max(2);
    let marker = if theme.no_color { "> " } else { "\u{21B3} " };
    format!("{}{}", " ".repeat(indent), marker)
}

/// 누적 셀폭이 `max_width` 를 초과하지 않는 가장 긴 prefix 와 나머지를 반환.
/// CJK / 박스 문자는 셀폭 2 로 계산. `(head, tail)` 모두 원본 슬라이스라 alloc 0.
/// Split `s` at the byte offset where its cumulative display width first
/// exceeds `max_width`, returning `(head, tail)`. A leading char wider than
/// `max_width` is forced into `head` so the split always makes progress.
/// Shared with the tool-result block so wrapping is uniform.
pub(crate) fn split_at_display_width(s: &str, max_width: usize) -> (&str, &str) {
    if max_width == 0 {
        return ("", s);
    }
    let mut acc: usize = 0;
    for (i, ch) in s.char_indices() {
        let w = char_display_width(ch);
        if acc + w > max_width {
            if i == 0 {
                let end = i + ch.len_utf8();
                return (&s[..end], &s[end..]);
            }
            return (&s[..i], &s[i..]);
        }
        acc += w;
    }
    (s, "")
}

/// Split a list-item's spans into multiple rows with a hanging indent.
///
/// Words are split on whitespace boundaries within each span's text. The
/// first row is emitted as-is up to `max_width` cells; every continuation row
/// starts with a blank span of `hanging` cells (the marker width) so the text
/// aligns under the item text, not the bullet.
///
/// Blank / empty span-lists pass through unchanged (they are inter-item blank
/// rows that must not gain an indent). Lines that already fit in `max_width`
/// are also emitted unchanged.
#[allow(clippy::too_many_lines)] // cohesive single-pass hanging-indent wrapper
fn wrap_spans_with_hanging(
    spans: Vec<Span<'static>>,
    max_width: usize,
    hanging: usize,
    // When `true`, the first span in `spans` is the list-item marker and is
    // emitted verbatim onto the first row (its depth-indent spaces must not
    // be split on word boundaries). The first row's remaining budget is
    // reduced by the marker's display width before content spans are wrapped.
    keep_first_span: bool,
) -> Vec<Vec<Span<'static>>> {
    // Empty lines (inter-item blank rows) pass through unchanged.
    if spans.iter().all(|s| s.content.is_empty()) {
        return vec![spans];
    }
    // Fast-path: measure the total display width; if it fits, skip wrapping.
    let total: usize = spans.iter().map(|s| cell_display_width(&s.content)).sum();
    if total <= max_width {
        return vec![spans];
    }

    let cont_width = max_width.saturating_sub(hanging).max(1);

    let mut rows: Vec<Vec<Span<'static>>> = Vec::new();
    let mut current: Vec<Span<'static>> = Vec::new();
    let mut current_width: usize = 0;
    // First row uses `max_width`; continuation rows use `cont_width`.
    let mut is_continuation = false;

    // Emit the marker span verbatim onto the first row, accounting for its
    // display width, so the word-splitter below starts correctly positioned.
    let mut spans_iter = spans.into_iter();
    if keep_first_span {
        if let Some(marker) = spans_iter.next() {
            let mw = cell_display_width(&marker.content);
            current.push(marker);
            current_width = mw;
        }
    }
    let spans: Vec<Span<'static>> = spans_iter.collect();

    // Flush `current` into `rows` and begin a new continuation row.
    let flush = |current: &mut Vec<Span<'static>>,
                 rows: &mut Vec<Vec<Span<'static>>>,
                 current_width: &mut usize,
                 is_continuation: &mut bool| {
        // Trim trailing whitespace spans before flushing.
        while let Some(last) = current.last_mut() {
            let trimmed = last.content.trim_end_matches(' ');
            if trimmed == last.content.as_ref() {
                break;
            }
            if trimmed.is_empty() {
                current.pop();
            } else {
                last.content = std::borrow::Cow::Owned(trimmed.to_string());
                break;
            }
        }
        rows.push(std::mem::take(current));
        *current_width = 0;
        *is_continuation = true;
        if hanging > 0 {
            current.push(Span::raw(" ".repeat(hanging)));
            *current_width = hanging;
        }
    };

    for span in spans {
        let style = span.style;
        // Clone the content so `remaining` can borrow a stable &str.
        let text = span.content.to_string();
        let mut remaining: &str = &text;

        // Walk through the span's text word by word (split on single spaces).
        while !remaining.is_empty() {
            let budget = if is_continuation {
                cont_width
            } else {
                max_width
            };

            // Split at the next space.
            let (word, rest) = match remaining.find(' ') {
                Some(pos) => (&remaining[..pos], &remaining[pos + 1..]),
                None => (remaining, ""),
            };

            if word.is_empty() {
                // Leading space token: emit one space if there is room and we
                // already have content (skip leading spaces on continuation rows
                // that already have the indent).
                if current_width < budget
                    && current_width > if is_continuation { hanging } else { 0 }
                {
                    current.push(Span::styled(" ".to_string(), style));
                    current_width += 1;
                }
                remaining = rest;
                continue;
            }

            let word_width = cell_display_width(word);

            // If the word alone exceeds the budget, force-break it character
            // by character.
            if word_width > budget {
                for ch in word.chars() {
                    let cw = char_display_width(ch);
                    let bud = if is_continuation {
                        cont_width
                    } else {
                        max_width
                    };
                    let min_on_row = if is_continuation { hanging } else { 0 };
                    if current_width + cw > bud && current_width > min_on_row {
                        flush(
                            &mut current,
                            &mut rows,
                            &mut current_width,
                            &mut is_continuation,
                        );
                    }
                    let mut s = String::with_capacity(ch.len_utf8());
                    s.push(ch);
                    current.push(Span::styled(s, style));
                    current_width += cw;
                }
                remaining = rest;
                continue;
            }

            // Normal word: wrap before it if it doesn't fit.
            let min_on_row = if is_continuation { hanging } else { 0 };
            if current_width + word_width > budget && current_width > min_on_row {
                flush(
                    &mut current,
                    &mut rows,
                    &mut current_width,
                    &mut is_continuation,
                );
            }

            current.push(Span::styled(word.to_string(), style));
            current_width += word_width;

            // Add a trailing space between this word and the next (unless the
            // rest is empty). Emit it if it fits; if not, it will be skipped
            // at the start of the next iteration.
            if !rest.is_empty() {
                let bud = if is_continuation {
                    cont_width
                } else {
                    max_width
                };
                if current_width < bud {
                    current.push(Span::styled(" ".to_string(), style));
                    current_width += 1;
                }
            }
            remaining = rest;
        }
    }

    // Flush the last row.
    if !current.is_empty() {
        rows.push(current);
    }

    if rows.is_empty() {
        rows.push(Vec::new());
    }
    rows
}

/// 터미널 셀폭: `unicode-width` 기반 (CJK / 전각 = 2, 그 외 = 1).
///
/// [`cell_display_width`] 는 `Line::width()`(= ratatui = `unicode-width`
/// 의 str 구현) 를 쓰므로 char 단위도 같은 크레이트로 맞춰 표/wrap 정렬이
/// 어긋나지 않게 한다. 제어문자는 0 (str 구현과 동일하게 `unwrap_or(0)`).
fn char_display_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn contains_box_drawing(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(
            ch,
            '│' | '┃'
                | '─'
                | '━'
                | '┼'
                | '┌'
                | '┐'
                | '└'
                | '┘'
                | '┬'
                | '┴'
                | '├'
                | '┤'
                | '╭'
                | '╮'
                | '╰'
                | '╯'
                | '╞'
                | '╡'
                | '╪'
        )
    })
}

fn looks_like_terminal_capture(text: &str) -> bool {
    let mut structural_lines = 0usize;
    let mut screen_glyph_lines = 0usize;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if starts_like_terminal_row(trimmed) {
            structural_lines += 1;
        }
        if raw.chars().any(is_terminal_screen_glyph) {
            screen_glyph_lines += 1;
        }
    }

    structural_lines >= 2 || screen_glyph_lines >= 2
}

fn looks_like_dense_terminal_capture(text: &str) -> bool {
    let mut nonblank_lines = 0usize;
    let mut terminalish_lines = 0usize;
    let mut screen_glyph_lines = 0usize;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        nonblank_lines += 1;

        let starts_terminal = starts_like_terminal_row(trimmed);
        let has_screen_glyph = raw.chars().any(is_terminal_screen_glyph);
        if starts_terminal || has_screen_glyph {
            terminalish_lines += 1;
        }
        if has_screen_glyph {
            screen_glyph_lines += 1;
        }
    }

    let dense_terminal_rows = terminalish_lines >= 4 && terminalish_lines * 2 >= nonblank_lines;
    let pasted_screen_with_sidebar = screen_glyph_lines >= 2 && terminalish_lines >= 3;
    dense_terminal_rows || pasted_screen_with_sidebar
}

fn starts_like_terminal_row(trimmed: &str) -> bool {
    trimmed.starts_with("└")
        || trimmed.starts_with("├")
        || trimmed.starts_with("│")
        || trimmed.starts_with("┃")
        || trimmed.starts_with("┌")
        || trimmed.starts_with("┐")
        || trimmed.starts_with("╭")
        || trimmed.starts_with("╰")
        || trimmed.starts_with("▸")
        || trimmed.starts_with("▾")
}

fn is_terminal_screen_glyph(ch: char) -> bool {
    matches!(
        ch,
        '░' | '▒' | '▓' | '█' | '■' | '▸' | '▾' | '●' | '○' | '◐' | '◒'
    )
}

/// heading (`#` ~ `######` + space), fenced code (` ``` `, `~~~`), or
/// obvious strong-emphasis labels require full markdown handling. This is the
/// escape hatch that keeps surrounding prose styled when a block also contains
/// box-drawing text or a markdown table.
pub(crate) fn has_strong_markdown_signal(text: &str) -> bool {
    for raw in text.lines() {
        let line = raw.trim_start();
        if line_has_strong_markdown_signal(line) {
            return true;
        }
    }
    false
}

fn line_has_strong_markdown_signal(line: &str) -> bool {
    if leading_fence(line).is_some() || is_atx_heading(line) {
        return true;
    }
    has_balanced_inline_marker(line, "**") || has_balanced_inline_marker(line, "__")
}

/// ATX heading recognizer (`#`–`######` + space) — the one lexicon entry for
/// heading detection, shared by the strong-signal and authored-block sniffers
/// (the pulldown Renderer parses headings itself and never consults this).
fn is_atx_heading(line: &str) -> bool {
    let bytes = line.as_bytes();
    let hashes = bytes.iter().take_while(|b| **b == b'#').count();
    (1..=6).contains(&hashes) && bytes.get(hashes) == Some(&b' ')
}

/// Authored markdown *block* structure — a fenced code block opener (` ``` ` /
/// `~~~`) or an ATX heading (`#`–`######` + space). A genuine pasted terminal
/// capture never contains these (zo has already rendered such markers away
/// before they reach a transcript dump), so their presence vetoes the
/// dense-capture override that would otherwise dump a long markdown answer as
/// raw text whenever it embeds box-drawing examples or echoes zo's own
/// heading glyphs (the "마지막에서 화면이 깨지네" raw-render bug). Distinct from
/// [`has_strong_markdown_signal`], which also fires on incidental inline
/// `**bold**` that a real screen dump legitimately contains.
fn has_authored_markdown_block_structure(text: &str) -> bool {
    text.lines().any(|raw| {
        let line = raw.trim_start();
        leading_fence(line).is_some() || is_atx_heading(line)
    })
}

fn has_balanced_inline_marker(line: &str, marker: &str) -> bool {
    let Some(start) = line.find(marker) else {
        return false;
    };
    let after_start = start + marker.len();
    let Some(end_offset) = line[after_start..].find(marker) else {
        return false;
    };
    let inner = &line[after_start..after_start + end_offset];
    !inner.trim().is_empty()
}

fn has_markdown_table(text: &str) -> bool {
    let mut lines = text.lines();
    let Some(mut current) = lines.next() else {
        return false;
    };
    for next in lines {
        if looks_like_table_row(current) && looks_like_table_separator(next) {
            return true;
        }
        current = next;
    }
    false
}

fn looks_like_table_row(line: &str) -> bool {
    line.matches('|').count() >= 2
}

fn looks_like_table_separator(line: &str) -> bool {
    if !line.contains('|') {
        return false;
    }
    let cells = split_table_cells(line);
    !cells.is_empty() && cells.iter().all(|cell| is_table_separator_cell(cell))
}

/// Single source of truth for "is this a GFM table separator *cell*", used by the
/// done-pass detector ([`looks_like_table_separator`]). Since the streaming tail
/// now renders through the same pulldown path as `done` (the naive line-parser
/// and its separate `>= 3`-dash detector were deleted in the third-parser
/// removal), there is no longer a second streaming detector to keep in lock-step
/// — GFM accepts `>= 1` dash and the done path has always rendered single-dash
/// separators as tables, so `| - | - |` is a table at every frame.
pub(super) fn is_table_separator_cell(cell: &str) -> bool {
    let trimmed = cell.trim();
    !trimmed.is_empty()
        && trimmed.contains('-')
        && trimmed.chars().all(|ch| matches!(ch, '-' | ':' | ' '))
}

// ============================================================================
// Markdown table rendering
// ============================================================================

/// GFM 표 컬럼 정렬 — 구분행의 `:` 위치로 결정.
/// `:---`=Left, `---:`=Right, `:---:`=Center, `---`=Left(기본).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColAlign {
    Left,
    Center,
    Right,
}

/// 구분행(`| :--- | ---: |`)에서 컬럼별 정렬을 파싱한다.
fn parse_alignments(separator: &str) -> Vec<ColAlign> {
    split_table_cells(separator)
        .iter()
        .map(|cell| {
            let c = cell.trim();
            match (c.starts_with(':'), c.ends_with(':')) {
                (true, true) => ColAlign::Center,
                (false, true) => ColAlign::Right,
                _ => ColAlign::Left,
            }
        })
        .collect()
}

fn render_table_markdown_for_width(text: &str, max_width: u16) -> String {
    let mut output = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        if lines
            .peek()
            .copied()
            .is_some_and(|next| looks_like_table_row(line) && looks_like_table_separator(next))
        {
            let Some(separator) = lines.next() else {
                output.push(line.to_owned());
                continue;
            };
            let mut table_lines = vec![line, separator];
            while let Some(next) = lines.peek().copied() {
                if !looks_like_table_row(next) {
                    break;
                }
                let Some(row) = lines.next() else { break };
                table_lines.push(row);
            }
            output.extend(render_table_block(&table_lines, max_width));
            continue;
        }
        output.push(line.to_owned());
    }
    output.join("\n")
}

fn render_table_block(lines: &[&str], max_width: u16) -> Vec<String> {
    let rows = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            if idx == 1 {
                None
            } else {
                Some(split_table_cells(line))
            }
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Vec::new();
    }
    let column_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let natural_widths = (0..column_count)
        .map(|column| {
            rows.iter()
                .filter_map(|row| row.get(column))
                .map(|cell| cell_display_width(cell))
                .max()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let widths = fit_table_widths(&natural_widths, max_width);
    // 구분행(lines[1])에서 컬럼 정렬을 파싱 — 숫자/우측정렬 컬럼 보존.
    let aligns = lines
        .get(1)
        .map(|s| parse_alignments(s))
        .unwrap_or_default();

    let mut rendered = Vec::new();
    rendered.push(render_table_hrule(&widths, TableRule::Top));
    rendered.extend(render_wrapped_table_row(&rows[0], &widths, &aligns));
    rendered.push(render_table_hrule(&widths, TableRule::Mid));
    for row in rows.iter().skip(1) {
        rendered.extend(render_wrapped_table_row(row, &widths, &aligns));
    }
    rendered.push(render_table_hrule(&widths, TableRule::Bottom));
    rendered
}

fn fit_table_widths(natural_widths: &[usize], max_width: u16) -> Vec<usize> {
    let mut widths = natural_widths.to_vec();
    let column_count = widths.len();
    if column_count == 0 || max_width == 0 {
        return widths;
    }
    let max_width = usize::from(max_width);
    let overhead = 1 + (column_count * 3);
    if max_width <= overhead + column_count {
        return vec![1; column_count];
    }
    let available = max_width - overhead;
    if widths.iter().sum::<usize>() <= available {
        return widths;
    }
    let min_widths = widths
        .iter()
        .map(|width| (*width).clamp(1, 6))
        .collect::<Vec<_>>();
    while widths.iter().sum::<usize>() > available {
        let Some((index, _)) = widths
            .iter()
            .enumerate()
            .filter(|(idx, width)| **width > min_widths[*idx])
            .max_by_key(|(_, width)| **width)
        else {
            break;
        };
        widths[index] = widths[index].saturating_sub(1);
    }
    widths
}

fn render_wrapped_table_row(row: &[String], widths: &[usize], aligns: &[ColAlign]) -> Vec<String> {
    let wrapped = widths
        .iter()
        .enumerate()
        .map(|(index, width)| {
            let cell = row.get(index).map_or("", String::as_str);
            wrap_cell(cell, *width)
        })
        .collect::<Vec<_>>();
    let row_height = wrapped.iter().map(Vec::len).max().unwrap_or(1).max(1);
    (0..row_height)
        .map(|line_idx| {
            let cells = wrapped
                .iter()
                .map(|lines| lines.get(line_idx).cloned().unwrap_or_default())
                .collect::<Vec<_>>();
            render_table_row(&cells, widths, aligns)
        })
        .collect()
}

fn render_table_row(row: &[String], widths: &[usize], aligns: &[ColAlign]) -> String {
    let mut line =
        String::with_capacity(widths.iter().sum::<usize>() + (widths.len() * 3).saturating_add(1));
    line.push('│');
    for (index, width) in widths.iter().enumerate() {
        let cell = row.get(index).map_or("", String::as_str);
        let cell_width = cell_display_width(cell);
        let pad = width.saturating_sub(cell_width);
        // 정렬에 따라 좌/우 패딩 분배. Left 는 (0, pad) 라 종전 동작과 동일.
        let (left, right) = match aligns.get(index).copied().unwrap_or(ColAlign::Left) {
            ColAlign::Left => (0, pad),
            ColAlign::Right => (pad, 0),
            ColAlign::Center => (pad / 2, pad - pad / 2),
        };
        line.push(' ');
        line.push_str(&" ".repeat(left));
        line.push_str(cell);
        line.push_str(&" ".repeat(right + 1));
        line.push('│');
    }
    line
}

fn wrap_cell(cell: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in cell.chars() {
        let ch_width = char_display_width(ch);
        if ch == '\n' {
            lines.push(current.trim_end().to_string());
            current.clear();
            current_width = 0;
            continue;
        }
        if current_width > 0 && current_width.saturating_add(ch_width) > width {
            lines.push(current.trim_end().to_string());
            current.clear();
            current_width = 0;
            if ch.is_whitespace() {
                continue;
            }
        }
        current.push(ch);
        current_width = current_width.saturating_add(ch_width);
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current.trim_end().to_string());
    }
    lines
}

pub(super) fn cell_display_width(s: &str) -> usize {
    Line::from(s.to_string()).width()
}

/// Which horizontal box rule to draw for a table row boundary.
#[derive(Clone, Copy)]
enum TableRule {
    /// Top edge above the header (`╭┬╮`).
    Top,
    /// Header/body separator, doubling as the GFM `---` row (`├┼┤`).
    Mid,
    /// Bottom edge below the last data row (`╰┴╯`).
    Bottom,
}

/// Render one horizontal box-drawing rule sized to `widths`.
///
/// Rounded corners (`╭ ╮ ╰ ╯`) match the code-block card (`╭─ … ╰─`) so
/// tables and code read as the same family of boxes. The `Mid` variant is the
/// header separator; `Top`/`Bottom` close the box so it never visually bleeds
/// into an adjacent heading or paragraph.
fn render_table_hrule(widths: &[usize], rule: TableRule) -> String {
    let (left, junction, right) = match rule {
        TableRule::Top => ('╭', '┬', '╮'),
        TableRule::Mid => ('├', '┼', '┤'),
        TableRule::Bottom => ('╰', '┴', '╯'),
    };
    let mut line = String::with_capacity(
        widths
            .iter()
            .map(|width| width + 3)
            .sum::<usize>()
            .saturating_add(1),
    );
    line.push(left);
    for (index, width) in widths.iter().enumerate() {
        line.push_str(&"─".repeat(width + 2));
        line.push(if index + 1 < widths.len() {
            junction
        } else {
            right
        });
    }
    line
}

fn split_table_cells(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or(trimmed);
    inner
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

// ============================================================================
// Syntect 코드 하이라이트
// ============================================================================

/// 공유 SyntaxSet + Theme (lazy `OnceLock`).
///
/// 최초 호출은 syntect 기본 신택스/테마 세트를 디스크 없이 임베드 바이너리에서
/// 역직렬화하므로 수십 ms 가 든다(측정). 이 비용이 첫 코드블록을 그리는
/// `draw()` 안에서 처음 치러지면 그 `draw` 가 TUI select! 스레드를 그만큼
/// 블로킹해 "첫 출력 멈춤"으로 보인다. 그래서 세션 부트가 입력 대기 전
/// 유휴 시간에 [`prewarm_syntect_assets`] 로 이 `OnceLock` 을 미리 채워,
/// 첫 렌더는 항상 캐시 히트가 되게 한다.
pub fn syntect_assets() -> &'static (SyntaxSet, SynTheme) {
    static ASSETS: OnceLock<(SyntaxSet, SynTheme)> = OnceLock::new();
    ASSETS.get_or_init(|| {
        let ss = SyntaxSet::load_defaults_newlines();
        let ts = ThemeSet::load_defaults();
        let theme = ts
            .themes
            .get("base16-ocean.dark")
            .cloned()
            .or_else(|| ts.themes.values().next().cloned())
            .unwrap_or_default();
        (ss, theme)
    })
}

/// 세션 부트가 유휴 시간에 호출하는 syntect 에셋 pre-warm.
///
/// [`syntect_assets`] 의 `OnceLock` 을 미리 채우는 것이 전부다. 블로킹
/// 스레드에서 호출하도록 의도됐고(부트의 `spawn_blocking`), 반환값을 버리는
/// 것은 캐시 적재만이 목적이기 때문이다. 별도 함수로 둬 호출부가 "무엇을 왜"
/// 하는지 한 줄로 읽히게 한다(SRP).
pub fn prewarm_syntect_assets() {
    let _ = syntect_assets();
}

/// 코드 카드 프레임을 그린다 — done 렌더([`Renderer::on_code_block_end`])와 거대
/// 코드펜스의 스트리밍 인테리어([`streaming::rendered_fence_interior_lines`])가
/// 공유하는 단일 출처(roadmap ⑨). 상단 보더(`  ╭─ {lang} ───`), 각 행 좌측 레일
/// (`  │ `) + 코드 배경, 그리고 `with_bottom_border` 면 하단 보더(`  ╰───`).
/// 스트리밍은 펜스가 아직 열려 있어 하단 보더를 생략하고 light 행을 넘기며, done
/// 은 syntect 행 + 하단 보더를 넘긴다 — 레일/래핑/줄수는 양쪽이 **구성상** 동일
/// (색만 다름)이라 settle 시 reflow/스크롤 점프가 없다. 입력은 `rows`(이미 색칠된
/// 코드 행)라 호출부가 highlighter(syntect / light / diff)를 고르고 이 fn 은
/// 모드 무관하게 프레임만 입힌다.
pub(super) fn code_card_frame_lines(
    rows: Vec<Vec<Span<'static>>>,
    lang: Option<&str>,
    is_diff: bool,
    width: u16,
    theme: &Theme,
    with_top_border: bool,
    with_bottom_border: bool,
) -> Vec<Vec<Span<'static>>> {
    let border_style = Style::new().fg(theme.palette.muted);
    let bg = theme.code_surface();

    // Phase 2.2 — 폭에 적응. 좌측 "  ╭─" 가 3 cells + 라벨 + 잔여 ─.
    let target = usize::from(width).saturating_sub(4).max(HR_MIN_WIDTH);

    let mut out: Vec<Vec<Span<'static>>> = Vec::with_capacity(rows.len() + 2);

    if with_top_border {
        let lang_label = if is_diff {
            " diff +/- ".to_string()
        } else {
            lang.map(|l| format!(" {l} ")).unwrap_or_default()
        };
        let label_visible = lang_label.chars().count();
        let dashes = target.saturating_sub(label_visible + 1);
        let mut top = vec![Span::styled("  ╭─".to_string(), border_style)];
        if !lang_label.is_empty() {
            let label_style = if is_diff {
                Style::new()
                    .fg(theme.palette.warn)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new()
                    .fg(theme.palette.dim)
                    .add_modifier(Modifier::ITALIC)
            };
            top.push(Span::styled(lang_label, label_style));
        }
        top.push(Span::styled("─".repeat(dashes), border_style));
        out.push(top);
    }

    for row in rows {
        let mut indented = vec![Span::styled("  │ ".to_string(), border_style)];
        indented.extend(row.into_iter().map(|mut span| {
            span.style = span.style.bg(bg);
            span
        }));
        out.push(indented);
    }

    if with_bottom_border {
        let bottom_dashes = target.saturating_sub(1);
        out.push(vec![Span::styled(
            format!("  ╰{}", "─".repeat(bottom_dashes)),
            border_style,
        )]);
    }

    out
}

/// syntect 없는 코드블록 렌더 — 스트리밍 꼬리 전용. 줄 단위로 body 색만
/// 입힌다 (bg 는 호출부 `on_code_block_end` 가 적용). [`highlight_code`] 와
/// 동일하게 `LinesWithEndings` 로 줄을 끊어 결과 행수가 일치한다.
fn plain_code_rows(code: &str, theme: &Theme) -> Vec<Vec<Span<'static>>> {
    LinesWithEndings::from(code)
        .map(|line| {
            let mut text = line.trim_end_matches('\n').to_string();
            if text.contains('\t') {
                text = text.replace('\t', "    ");
            }
            vec![Span::styled(text, Style::new().fg(theme.palette.fg))]
        })
        .collect()
}

/// Parse the old/new starting line numbers from a unified-diff hunk header
/// (`@@ -old,len +new,len @@`), returning `(old_start, new_start)`. `None` when
/// the header is malformed, so the caller can skip seeding the gutter counters.
fn parse_hunk_starts(header: &str) -> Option<(u32, u32)> {
    let inner = header.trim_start_matches('@');
    let mut parts = inner.split_whitespace();
    let old = parts.next()?.strip_prefix('-')?;
    let new = parts.next()?.strip_prefix('+')?;
    let old_start: u32 = old.split(',').next()?.parse().ok()?;
    let new_start: u32 = new.split(',').next()?.parse().ok()?;
    Some((old_start, new_start))
}

/// Whether a fence language tag marks prose content (v3 readability): the
/// text-family tags models use to wrap natural-language passages. Bare
/// fences (no tag) are deliberately NOT included — by convention they hold
/// code/log output that wants layout fidelity.
fn is_prose_fence_lang(lang: &str) -> bool {
    matches!(
        lang.to_ascii_lowercase().as_str(),
        "text" | "txt" | "plain" | "plaintext"
    )
}

/// Render a text-family fence as a wrapped quote block: a `│ ` rail in the
/// muted gradation with the body at normal reading contrast, physically
/// wrapped to `width` (CJK-aware, space-preferring) so every wrapped row
/// keeps the rail and no row can overflow the frame. Blank source lines keep
/// a bare rail row so the passage reads as one contained block.
fn prose_fence_lines(content: &str, width: u16, theme: &Theme) -> Vec<Vec<Span<'static>>> {
    const RAIL: &str = "\u{2502} ";
    // `dim`, not `muted`: on a dark theme the muted gradation is nearly
    // invisible at cell size, so the passage read as an uncontained indent
    // blob ("half-drawn"). The rail is the quote block's only container —
    // it must be quiet but *visible*.
    let rail_style = Style::new().fg(theme.palette.dim);
    let body_style = Style::new().fg(theme.palette.fg);
    let body_width = usize::from(width.saturating_sub(2).max(8));
    let mut out: Vec<Vec<Span<'static>>> = Vec::new();
    for src in content.lines() {
        if src.trim().is_empty() {
            out.push(vec![Span::styled(RAIL, rail_style)]);
            continue;
        }
        for chunk in wrap_to_display_cells(src, body_width) {
            out.push(vec![
                Span::styled(RAIL, rail_style),
                Span::styled(chunk, body_style),
            ]);
        }
    }
    if out.is_empty() {
        out.push(vec![Span::styled(RAIL, rail_style)]);
    }
    out
}

/// Greedy display-cell wrap for one source line: breaks at the last space
/// inside the budget when one exists (Latin words survive), else mid-run
/// (CJK breaks anywhere, per Korean line-break convention). Uses the shared
/// `text_metrics` char widths, so a `ko_KR` 2-cell hangul row never exceeds
/// its budget.
fn wrap_to_display_cells(line: &str, max_cells: usize) -> Vec<String> {
    let max_cells = max_cells.max(1);
    let mut rows = Vec::new();
    let mut row = String::new();
    let mut row_cells = 0usize;
    let mut last_space: Option<(usize, usize)> = None; // (byte idx in row, cells before space)
    for ch in line.chars() {
        let w = crate::tui::text_metrics::char_width(ch);
        if row_cells + w > max_cells && !row.is_empty() {
            if let Some((byte_idx, _)) = last_space {
                let rest = row.split_off(byte_idx);
                rows.push(std::mem::take(&mut row));
                row = rest.trim_start().to_string();
            } else {
                rows.push(std::mem::take(&mut row));
            }
            row_cells = row.chars().map(crate::tui::text_metrics::char_width).sum();
            last_space = None;
        }
        if ch == ' ' && !row.is_empty() {
            last_space = Some((row.len(), row_cells));
        }
        row.push(ch);
        row_cells += w;
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

/// Render a unified-diff code fence with semantic per-line coloring:
/// additions green, removals red, hunk headers violet, file headers cyan,
/// context muted. Mirrors the native [`super::blocks::diff`] viewer so
/// ```diff fences in assistant prose read the same as tool-result diffs.
fn diff_code_rows(code: &str, theme: &Theme) -> Vec<Vec<Span<'static>>> {
    // Track old/new line numbers across the fence so each body row can show a
    // dim line-number gutter, mirroring the native `super::blocks::diff` viewer.
    // The counters are only seeded once a `@@` hunk header is parsed; a bare
    // snippet without headers (`- old` / `+ new`) keeps the original gutter-less
    // rendering, so short illustrative diffs are unaffected.
    let gutter_style = Style::new().fg(theme.palette.dim);
    let mut old_line: Option<u32> = None;
    let mut new_line: Option<u32> = None;
    let mut seen_hunk = false;

    LinesWithEndings::from(code)
        .map(|line| {
            let mut text = line.trim_end_matches('\n').to_string();
            if text.contains('\t') {
                text = text.replace('\t', "    ");
            }
            // Order matters: `+++`/`---` file headers start with `+`/`-`, so
            // they must be matched before the single-char add/remove arms.
            if text.starts_with("@@") {
                if let Some((old_start, new_start)) = parse_hunk_starts(&text) {
                    old_line = Some(old_start);
                    new_line = Some(new_start);
                    seen_hunk = true;
                }
                let style = Style::new()
                    .fg(theme.palette.violet)
                    .add_modifier(Modifier::BOLD);
                return vec![Span::styled(text, style)];
            }
            if text.starts_with("+++") || text.starts_with("---") {
                let style = Style::new()
                    .fg(theme.palette.cyan)
                    .add_modifier(Modifier::BOLD);
                return vec![Span::styled(text, style)];
            }
            // `\ No newline at end of file` annotates the *preceding* row; it is
            // not a real line in either file, so it must NOT advance the gutter
            // counters (otherwise every row after it is numbered one off). Render
            // it dim with a blank gutter to stay column-aligned, like the context
            // arm but counter-neutral.
            if text.starts_with("\\ ") {
                let cell = blank_gutter_cell(seen_hunk);
                return styled_diff_row(cell, gutter_style, text, theme.diff_context_style());
            }

            let (gutter, body_style) = if text.starts_with('+') {
                let cell = diff_gutter_cell(None, new_line.as_mut(), seen_hunk);
                (cell, theme.diff_add_style())
            } else if text.starts_with('-') {
                let cell = diff_gutter_cell(old_line.as_mut(), None, seen_hunk);
                (cell, theme.diff_del_style())
            } else {
                let cell = diff_gutter_cell(old_line.as_mut(), new_line.as_mut(), seen_hunk);
                (cell, theme.diff_context_style())
            };

            styled_diff_row(gutter, gutter_style, text, body_style)
        })
        .collect()
}

/// Build one diff body row: an optional dim line-number gutter followed by the
/// styled diff text. Shared by the add/remove/context arms and the
/// `\ No newline` annotation so every row renders the gutter identically.
fn styled_diff_row(
    gutter: Option<String>,
    gutter_style: Style,
    text: String,
    body_style: Style,
) -> Vec<Span<'static>> {
    match gutter {
        Some(gutter) => vec![
            Span::styled(gutter, gutter_style),
            Span::styled(text, body_style),
        ],
        None => vec![Span::styled(text, body_style)],
    }
}

/// A counter-neutral gutter cell (all blanks) matching [`diff_gutter_cell`]'s
/// width, for rows that occupy a gutter column but belong to neither file
/// (`\ No newline at end of file`). `None` before any hunk header, like the
/// real gutter, so bare snippets stay gutter-less.
fn blank_gutter_cell(seen_hunk: bool) -> Option<String> {
    const NUM_WIDTH: usize = 4;
    seen_hunk.then(|| {
        let blank = " ".repeat(NUM_WIDTH);
        format!("{blank} {blank} ")
    })
}

/// Format the line-number gutter cell for one diff body row and advance the
/// supplied counters. Returns `None` before any hunk header is seen (so bare
/// snippets stay gutter-less). An added line has no old number; a removed line
/// has no new number; a context line shows both. The gutter is a fixed
/// `old new ` column so changed and context rows stay vertically aligned.
fn diff_gutter_cell(
    old_line: Option<&mut u32>,
    new_line: Option<&mut u32>,
    seen_hunk: bool,
) -> Option<String> {
    const NUM_WIDTH: usize = 4;
    if !seen_hunk {
        return None;
    }
    let old_cell = match old_line {
        Some(value) => {
            let rendered = format!("{value:>NUM_WIDTH$}");
            *value = value.saturating_add(1);
            rendered
        }
        None => " ".repeat(NUM_WIDTH),
    };
    let new_cell = match new_line {
        Some(value) => {
            let rendered = format!("{value:>NUM_WIDTH$}");
            *value = value.saturating_add(1);
            rendered
        }
        None => " ".repeat(NUM_WIDTH),
    };
    Some(format!("{old_cell} {new_cell} "))
}

/// Syntax-highlight `code` into framed rows. Each token is classified by its
/// syntect scope into a zo `SyntaxRole` and colored through
/// [`Theme::syntax_style`], so the code card stays inside the zo palette and
/// degrades with the terminal (256-color / `NO_COLOR`) instead of emitting a raw
/// base16 `Color::Rgb`.
fn highlight_code(code: &str, lang: Option<&str>, theme: &Theme) -> Vec<Vec<Span<'static>>> {
    let (ss, _syn_theme) = syntect_assets();
    let syntax = lang
        .and_then(|l| {
            ss.find_syntax_by_token(l)
                .or_else(|| ss.find_syntax_by_name(l))
        })
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut highlighter = SyntaxHighlighter::new(syntax);
    let mut rows: Vec<Vec<Span<'static>>> = Vec::new();
    for line in LinesWithEndings::from(code) {
        let regions = highlighter.highlight_line(line, ss);
        let mut row: Vec<Span<'static>> = Vec::with_capacity(regions.len());
        for (role, segment) in regions {
            let mut text = segment.trim_end_matches('\n').to_string();
            if text.is_empty() {
                continue;
            }
            if text.contains('\t') {
                text = text.replace('\t', "    ");
            }
            row.push(Span::styled(text, theme.syntax_style(role)));
        }
        rows.push(row);
    }
    rows
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;

#[cfg(test)]
mod resume_table_diff_tests {
    use super::*;


    /// The done-pass table-separator detector and the shared cell predicate agree
    /// (via the single shared [`is_table_separator_cell`]), so a `| - | - |` row
    /// and a `| --- | --- |` row are both separators on BOTH passes — a table
    /// never reflows from prose to table on completion. A dash-less cell (pure
    /// colon, or prose) is a separator on neither.
    #[test]
    fn table_separator_detectors_agree() {
        assert!(is_table_separator_cell("-"));
        assert!(is_table_separator_cell("---"));
        assert!(is_table_separator_cell(" :---: "));
        assert!(!is_table_separator_cell(":")); // no dash
        assert!(!is_table_separator_cell("")); // empty
        assert!(!is_table_separator_cell("ab")); // prose

        // Done-pass detector uses the shared predicate, matching the streaming
        // pass for both single-dash and long-dash separators.
        assert!(looks_like_table_separator("| --- | --- |"));
        assert!(looks_like_table_separator("| - | - |"));
        assert!(has_markdown_table("| a | b |\n| --- | --- |\n| 1 | 2 |"));
        assert!(has_markdown_table("| a | b |\n| - | - |\n| 1 | 2 |"));
        // A dash-less "separator" is not a table on the done pass either.
        assert!(!has_markdown_table("| a | b |\n| x | y |\n| 1 | 2 |"));
    }

    fn flatten_rows(rows: &[Vec<Span<'static>>]) -> Vec<String> {
        rows.iter()
            .map(|row| row.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    /// `\ No newline at end of file` must not advance the diff gutter counters —
    /// rows after it keep their correct line numbers (regression: the marker was
    /// treated as a context line, shifting every following number by one).
    #[test]
    fn diff_no_newline_marker_does_not_desync_gutter() {
        let theme = Theme::default_dark();
        // Lines explicit to avoid string-continuation ambiguity: a hunk header,
        // a context line, a removal, the `\ No newline` marker, then a context.
        let code = concat!(
            "@@ -1,2 +1,2 @@\n",
            " ctx a\n",
            "-old\n",
            "\\ No newline at end of file\n",
            "+new\n",
            " ctx b\n",
        );
        let rows = flatten_rows(&diff_code_rows(code, &theme));
        let joined = rows.join("\n");
        // The marker row renders verbatim and carries no line number.
        assert!(
            rows.iter().any(|r| r.trim() == "\\ No newline at end of file"),
            "marker must render verbatim with a blank gutter: {joined:?}"
        );
        // Counters: ctx a = old1/new1; -old = old2; +new = new2; ctx b = old3/new3.
        // The marker between -old and +new must NOT consume a number, so +new is
        // new line 2 and ctx b is old3/new3.
        assert!(
            rows.iter().any(|r| r.contains("ctx b") && r.contains('3')),
            "trailing context keeps numbers after the marker: {joined:?}"
        );
    }

    /// Without the fix the marker would advance the new-line counter, pushing the
    /// following `+new one` from new-number 1 to 2. Pin the exact numbering.
    #[test]
    fn diff_no_newline_marker_keeps_following_added_number_exact() {
        let theme = Theme::default_dark();
        let code = concat!(
            "@@ -1,1 +1,2 @@\n",
            "-old\n",
            "\\ No newline at end of file\n",
            "+new one\n",
            "+new two\n",
        );
        let rows = flatten_rows(&diff_code_rows(code, &theme));
        let joined = rows.join("\n");
        // new counter starts at 1; `+new one` is new line 1, `+new two` is new 2.
        let one = rows.iter().find(|r| r.contains("new one")).expect("row");
        let two = rows.iter().find(|r| r.contains("new two")).expect("row");
        assert!(one.contains('1'), "first added row is new line 1: {joined:?}");
        assert!(two.contains('2'), "second added row is new line 2: {joined:?}");
    }
}
