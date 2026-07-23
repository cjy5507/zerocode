//! `RenderBlock::TextDelta` widget — assistant 스트리밍 마크다운 위젯.
//!
//! 마크다운 파싱/스타일링은 [`crate::tui::markdown`] 모듈로 위임한다
//! (단일 책임 원칙). 이 모듈은 다음만 책임진다:
//!
//! * 캐시된 lines 를 받아 그대로 그리기 (`draw_cached`)
//! * 저자 불릿(`◆`) + 들여쓰기 마크 컬럼 (`draw_marked_paragraph`)
//! * Paragraph wrap 정책 + scroll offset 적용
//!
//! 스트리밍 중 본문에는 어떤 진행 장식도 없다(caret 폐기, v3 §4) — 진행
//! 표시는 하단 활동 라인(TurnActivity)의 몫이다.
//!
//! 사용자 페이스트 메시지(`UserMessage`)도 동일한
//! [`crate::tui::markdown::rendered_lines_for_width`] 를 거치도록 wire 되어
//! 있어 어시스턴트와 시각이 byte-identical 이다 (see `blocks/mod.rs`).
//!
//! `code-rules.md` R1, R2, R9 준수.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::tui::markdown;
use crate::tui::theme::Theme;

use super::wrapped_rows;

use crate::tui::glyphs;
/// Re-export the shared syntect assets so existing callers (e.g.
/// [`super::diff`]) that referenced `tui::blocks::text::syntect_assets`
/// before the `markdown` module split continue to compile.
pub(crate) use crate::tui::markdown::syntect_assets;

/// 텍스트 블록을 그린다.
///
/// `text` 는 누적된 본문 전체다 (transcript 가 deltas 를 미리 합쳐서 호출).
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    text: &str,
    done: bool,
    theme: &Theme,
    tick: u64,
    scroll_offset: u16,
) {
    draw_with_mark(
        frame,
        area,
        text,
        done,
        theme,
        tick,
        scroll_offset,
        super::ProseMark::Bare,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn draw_with_mark(
    frame: &mut Frame<'_>,
    area: Rect,
    text: &str,
    done: bool,
    theme: &Theme,
    tick: u64,
    scroll_offset: u16,
    prose: super::ProseMark,
) {
    let body_width = mark_body_width(area.width, prose);
    let lines = rendered_lines_for_width(text, done, theme, tick, body_width);
    let preserves = done && markdown::preserves_layout(text);
    match prose {
        super::ProseMark::Bullet | super::ProseMark::Indent => draw_marked_paragraph(
            frame,
            area,
            lines,
            preserves,
            theme,
            scroll_offset,
            prose == super::ProseMark::Bullet && scroll_offset == 0,
        ),
        super::ProseMark::Bare => {
            draw_paragraph(frame, area, lines, preserves, theme, scroll_offset);
        }
    }
}

/// 미리 렌더링된(캐시된) lines 를 그린다. transcript 가 스트리밍 중
/// pulldown-cmark + syntect 재파싱 비용을 피하기 위해 호출한다.
///
/// `row_prefix` 는 캐시가 함께 보관하는 줄별 wrap-행 prefix-sum
/// ([`super::wrapped_row_prefix`]). 이것으로 viewport 에 보이는 줄 구간만
/// 잘라 그린다 — 종전에는 블록 **전체** 라인을 매 프레임 얕은 복사로 빌리고
/// `Paragraph` 가 scroll 만큼 wrap-skip 해서, 큰 블록 위를 스크롤하거나 긴
/// 답변이 스트리밍될수록 프레임 비용이 본문 길이에 선형으로 늘었다.
/// `preserves_layout` 블록(wrap 없음)은 줄==행이라 prefix 없이 직접 자른다.
#[allow(clippy::too_many_arguments)]
pub fn draw_cached(
    frame: &mut Frame<'_>,
    area: Rect,
    cached_lines: &[Line<'static>],
    row_prefix: &[u32],
    preserves_layout: bool,
    theme: &Theme,
    scroll_offset: u16,
    prose: super::ProseMark,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let (window, line_scroll, _includes_last) = if preserves_layout {
        // No wrap: line index == row index.
        let start = usize::from(scroll_offset).min(cached_lines.len());
        let end = start
            .saturating_add(usize::from(area.height))
            .min(cached_lines.len());
        (&cached_lines[start..end], 0u16, end >= cached_lines.len())
    } else {
        super::visible_line_window(cached_lines, row_prefix, scroll_offset, area.height)
    };
    // String 복제 없이 얕게 빌린다 — 보이는 창만큼만이라 프레임당 O(visible).
    let lines = super::borrow_lines(window);
    match prose {
        super::ProseMark::Bullet | super::ProseMark::Indent => draw_marked_paragraph(
            frame,
            area,
            lines,
            preserves_layout,
            theme,
            line_scroll,
            prose == super::ProseMark::Bullet && scroll_offset == 0,
        ),
        super::ProseMark::Bare => {
            draw_paragraph(frame, area, lines, preserves_layout, theme, line_scroll);
        }
    }
}

fn mark_body_width(area_width: u16, prose: super::ProseMark) -> u16 {
    if prose.has_indent() {
        let available_width = area_width.saturating_sub(super::ROLE_RAIL_WIDTH);
        available_width.clamp(1, super::prose_wrap_cap(available_width))
    } else {
        area_width.max(1)
    }
}

/// 텍스트의 wrapped 높이 추정 — layout 캐시가 사용한다.
pub(crate) fn estimate_rows(text: &str, done: bool, theme: &Theme, width: u16) -> u16 {
    let lines = rendered_lines_for_width(text, done, theme, 0, width);
    if done && markdown::preserves_layout(text) {
        u16::try_from(lines.len()).unwrap_or(u16::MAX).max(1)
    } else {
        wrapped_rows(&lines, width)
    }
}

/// transcript 의 cache miss 경로에서 호출하는 entrypoint.
///
/// `done`/`tick` 은 caret blink 와 무관 — caret 은 항상 draw-time 에서만
/// 추가되므로 cache 본문에는 들어가지 않는다.
pub(crate) fn rendered_lines_for_width(
    text: &str,
    done: bool,
    theme: &Theme,
    tick: u64,
    width: u16,
) -> Vec<Line<'static>> {
    let _ = tick;
    if done {
        if let Some(lines) = verifier_json_summary_lines(text, theme, width) {
            return lines;
        }
        let mut lines = markdown::rendered_lines_for_width(text, theme, width);
        // 블록 끝의 빈 줄을 떼어낸다. 블록 사이 분리는 transcript 의
        // block_gap(빈 줄 1, 레일 없음)이 전담하므로, 마크다운 문단의
        // trailing blank 가 좌측 레일과 함께 그려져 "꼬리 레일 + 빈 줄"로
        // 이중 분리(OpenCode 대비 산만)되는 것을 막는다 — 이벤트 사이는
        // 정확히 빈 줄 1개. 문단 사이 내부 빈 줄은 보존(끝에서만 제거).
        trim_trailing_blank_lines(&mut lines);
        lines
    } else {
        // Streaming cache-miss fallback: keep the cheap no-syntect path, but
        // still run the markdown event renderer when the live text visibly
        // contains markdown. Plain output here made long Gemini-style answers
        // suddenly expose raw `##`/`**`/list markers whenever draw missed the
        // transcript cache for a frame.
        streaming_markdown_or_plain_lines(text, theme, width)
    }
}

struct VerifierJsonSummary {
    accepted: bool,
    issues: Vec<String>,
}

fn parse_verifier_json_summary(text: &str) -> Option<VerifierJsonSummary> {
    let trimmed = text.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    // This runs on every `done` block, so only treat text that LOOKS like a
    // verifier verdict as one: the old single-`accepted` shape, or the per-lens
    // shape (spec/regression/security). Otherwise a normal JSON-ish answer would
    // render as a spurious "✗ rejected".
    if !looks_like_verifier_object(trimmed) {
        return None;
    }
    // Reuse the deep-gate's own reader so the displayed ✓/✗ always matches the
    // accept/reject the loop acted on: the per-lens AnyReject fold, or the old
    // `{accepted,issues}` contract via its fallback. One source of truth, no
    // third copy of the lens logic.
    let verdict = decision_core::parse_lens_verifier(trimmed);
    Some(VerifierJsonSummary {
        accepted: verdict.accepted,
        issues: verdict.issues,
    })
}

/// Whether a JSON-object slice carries a verifier verdict: the legacy single
/// `accepted` field, or the per-lens `spec`+`regression`+`security` trio.
fn looks_like_verifier_object(obj: &str) -> bool {
    obj.contains("\"accepted\"") || has_lens_trio(obj)
}

fn has_lens_trio(obj: &str) -> bool {
    obj.contains("\"spec\"") && obj.contains("\"regression\"") && obj.contains("\"security\"")
}

/// Find the LAST balanced top-level `{…}` in `text` that looks like a per-lens
/// verifier verdict. The VERIFY sub-turn is told to emit ONLY the single-line
/// JSON, but models sometimes explain-then-conclude or wrap it in a ```json
/// fence — the deep-gate parser is robust to that, so the TUI should be too.
/// Returns the object's byte range. String contents (an `issues` entry may hold
/// braces) are skipped so the matcher stays balanced; the trailing `}` is ASCII
/// so the returned range lands on char boundaries.
fn find_embedded_verdict_span(text: &str) -> Option<(usize, usize)> {
    let mut depth: u32 = 0;
    let mut start: Option<usize> = None;
    let mut in_str = false;
    let mut escaped = false;
    let mut best: Option<(usize, usize)> = None;
    for (i, b) in text.bytes().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        if has_lens_trio(&text[s..=i]) {
                            best = Some((s, i + 1));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    best
}

/// Prose surrounding an embedded verdict object, with an enclosing ```json …
/// ``` code fence stripped so excising the JSON cannot strand a dangling fence.
fn surrounding_prose(text: &str, start: usize, end: usize) -> (String, String) {
    let mut before = &text[..start];
    let after_raw = &text[end..];
    // A fence opener (```/```json) on the line directly above the JSON.
    let trimmed = before.trim_end();
    if let Some(nl) = trimmed.rfind('\n') {
        if trimmed[nl + 1..].trim_start().starts_with("```") {
            before = &before[..nl];
        }
    } else if trimmed.trim_start().starts_with("```") {
        before = "";
    }
    // A fence closer on the line directly below the JSON.
    let mut after = after_raw;
    if let Some(rest) = after_raw.trim_start().strip_prefix("```") {
        after = rest.find('\n').map_or("", |nl| &rest[nl + 1..]);
    }
    (before.trim().to_string(), after.trim().to_string())
}

fn verifier_json_summary_lines(
    text: &str,
    theme: &Theme,
    width: u16,
) -> Option<Vec<Line<'static>>> {
    // 1. The whole message is the verdict JSON (the bare line the VERIFY
    //    sub-turn is asked to emit): swap it wholesale for the pretty verdict.
    if let Some(verdict) = parse_verifier_json_summary(text) {
        return Some(verdict_lines(verdict.accepted, &verdict.issues, theme));
    }
    // 2. The verdict JSON is embedded in surrounding prose / a ```json fence
    //    (the model explained-then-concluded). Keep the prose — it carries the
    //    per-lens reasoning — but render the raw JSON object as the pretty
    //    verdict block beneath it instead of leaking `{"spec":…}` as code.
    let (start, end) = find_embedded_verdict_span(text)?;
    let verdict = decision_core::parse_lens_verifier(&text[start..end]);
    let (before, after) = surrounding_prose(text, start, end);
    let mut lines = Vec::new();
    if !before.is_empty() {
        lines.extend(markdown::rendered_lines_for_width(&before, theme, width));
    }
    if !after.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::raw(String::new()));
        }
        lines.extend(markdown::rendered_lines_for_width(&after, theme, width));
    }
    if !lines.is_empty() {
        lines.push(Line::raw(String::new()));
    }
    lines.extend(verdict_lines(verdict.accepted, &verdict.issues, theme));
    Some(lines)
}

/// The compact pretty verdict block: a ✓/✗ headline plus any itemized issues.
/// Shared by the whole-message and embedded-in-prose paths.
fn verdict_lines(accepted: bool, issues: &[String], theme: &Theme) -> Vec<Line<'static>> {
    let (glyph, color, label) = if accepted {
        (
            if theme.no_color { "[ok]" } else { "✓" },
            theme.palette.success,
            "Verification accepted",
        )
    } else {
        (
            if theme.no_color { "[x]" } else { "✗" },
            theme.palette.error,
            "Verification rejected",
        )
    };
    let verdict_style = Style::new().fg(color).add_modifier(Modifier::BOLD);
    let mut lines = vec![Line::from(vec![
        Span::styled(format!("{glyph} "), verdict_style),
        Span::styled(label.to_string(), verdict_style),
    ])];

    if issues.is_empty() {
        if !accepted {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    "No specific issues were itemized.".to_string(),
                    theme.typography.dim,
                ),
            ]));
        }
        return lines;
    }

    let issue_style = Style::new().fg(theme.palette.fg);
    for issue in issues {
        lines.push(Line::from(vec![
            Span::styled("- ".to_string(), Style::new().fg(theme.palette.warn)),
            Span::styled(issue.clone(), issue_style),
        ]));
    }
    lines
}

/// Split streamed text into one [`Line`] per source line, unstyled.
/// [`draw_paragraph`] applies the body style and wraps. No markdown or
/// syntax-highlighting work — that is deferred to the `done` frame.
fn plain_streaming_lines(text: &str) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![Line::raw(String::new())];
    }
    text.split('\n')
        .map(|line| Line::raw(line.to_string()))
        .collect()
}

fn streaming_markdown_or_plain_lines(text: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    // Temporarily close any unclosed inline marker so a half-typed `**bold` or
    // `` `code `` streams as styled text instead of flashing the raw marker
    // until its closer arrives. Completed markers borrow the source (zero-copy).
    let repaired = markdown::repair_unclosed_inline_markers(text);
    let mut lines = markdown::rendered_bounded_streaming_tail_for_width(&repaired, theme, width)
        .unwrap_or_else(|| plain_streaming_lines(&repaired));
    trim_stream_trailing_blanks(&mut lines, text);
    lines
}

/// Match the done path's unconditional trailing-blank trim (the `trim_trailing_
/// blank_lines` call on the `done` branch) so a dense answer streams at the SAME
/// height it settles into (roadmap ⑧): the markdown `Renderer` appends trailing
/// blank-line artifacts after the last block that the done pass removes, and
/// keeping them while streaming made a section-style answer shrink by a line or
/// two at settle. Trim them whenever the source does NOT literally end with a
/// blank line. The one case we KEEP a trailing blank is `"\n\n"`: there the
/// blank is an intentional, just-typed paragraph break, and showing it
/// immediately reads better mid-stream — it becomes an internal break (which the
/// done pass keeps) as soon as the next paragraph's text arrives. Internal blank
/// lines are never trailing, so this never collapses a real paragraph gap.
fn trim_stream_trailing_blanks(lines: &mut Vec<Line<'static>>, source: &str) {
    if !source.ends_with("\n\n") {
        trim_trailing_blank_lines(lines);
    }
}

/// 열린 코드펜스의 버퍼 내용이 이보다 크면 완료된 코드 줄을 안정 영역으로
/// 승격해 열린 꼬리를 마지막 미완성 줄로 묶는다([`markdown::streaming_stable_prefix`]).
/// 이 미만의 펜스는 기존 경로(펜스 시작을 경계로) 그대로 — 꼬리 전체 재렌더 비용이
/// 아직 작아(≤~1ms) 무변경이 안전하다. 큰 코드답변에서만 O(n²)→O(꼬리) 로 전환.
const LARGE_OPEN_FENCE_LIMIT: usize = 16 * 1024;

/// 안정 영역으로 승격되는 fragment 가 이보다 크고 코드 펜스를 포함하면 syntect
/// 하이라이트를 생략한다 — 펜스가 닫히는 프레임의 1회 syntect 패스가 입력/렌더
/// 루프를 블로킹해 스트리밍이 "뚝뚝" 끊겨 보이기 때문이다. 측정(warm syntect):
/// 코드 syntect 는 **~1.5ms/KB** (4KB≈7.5ms · 8KB≈12ms · 12KB≈17ms>16ms예산 ·
/// 24KB≈33ms=프레임 2개 드랍), light(하이라이트 off) 경로는 크기 무관 ~40µs.
/// 8KB 로 잡으면 close-frame 의 syntect 가 ~12ms 로 프레임 예산(16ms) 아래에
/// 묶여 드랍 프레임이 사라지고, 8KB 초과 펜스는 스트리밍 중 light 로 그렸다가
/// `done` 패스([`layout::should_keep_incremental_done_cache`] 의
/// `FINAL_DONE_CODE_RENDER_LIMIT` = 24KB 미만)가 권위적 색을 입힌다 — 즉 색은
/// 잃지 않고 mid-stream 끊김만 제거. 8KB =
/// "프레임당 풀렌더하기엔 큰 fragment" 의 단일 임계로 일관.
const FRAGMENT_SYNTECT_SKIP_LIMIT: usize = 8 * 1024;

// syntect on code is ~1.5ms/KB, so a fragment over this threshold must skip the
// authoritative pass on the close frame or the streaming loop drops a frame.
// Enforced at compile time so a future bump can't silently reintroduce the
// 12–33ms mid-stream stutter (the `done` pass recolors below
// `FINAL_DONE_CODE_RENDER_LIMIT`, so color is deferred, never lost).
const _: () = assert!(FRAGMENT_SYNTECT_SKIP_LIMIT <= 8 * 1024);

/// 코드가 많은 대형 fragment 인지 — 그러면 streaming 승격 시 syntect 를 건너뛴다.
fn fragment_skips_syntect(fragment: &str) -> bool {
    fragment.len() > FRAGMENT_SYNTECT_SKIP_LIMIT
        && (fragment.contains("```") || fragment.contains("~~~"))
}

/// 스트리밍 중 증분 마크다운 렌더.
///
/// 직전 프레임의 `prev_lines[..prev_stable_count]` (이미 스타일링된 완료
/// 블록들) 을 **이동(zero-copy)** 으로 재사용한다. 마지막 프레임 이후 새로
/// 완료된 블록만 [`markdown::rendered_lines_for_width`] 로 한 번(syntect 포함)
/// 렌더해 안정 영역에 append 하고, 열린 꼬리만 syntect 없이 다시 그린다.
///
/// [버퍼 튜닝 계획: 스트리밍 마크다운 Pop-in 완화]
/// 스트리밍 종료 시점에 전체 블록이 plain에서 하이라이트 색상으로 일시에 바뀌는 Pop-in 현상을
/// 완화하기 위해 다음과 같은 증분 병합 방식의 튜닝 계획을 적용할 수 있다:
/// 1. 문단 및 코드 펜스의 경계를 감지하여 이미 완성된 단락은 즉시 백그라운드 스레드에서
///    하이라이트 처리를 비동기로 병합(coalescing).
/// 2. 꼬리 부분(unclosed tail)에 대해서만 임시 스타일을 입히고 완성 상태가 되는 단락만
///    동기식으로 캐시에 커밋하여, done 시점의 대규모 재파싱 연산을 완전히 방지함.
///
/// `row_prefix` 는 lines 와 한 몸으로 유지되는 줄별 wrap-행 prefix-sum
/// ([`super::wrapped_row_prefix`] 과 동일 계약, `len == lines.len() + 1`).
/// 안정 영역의 행 수는 재계산하지 않고 그대로 두며, 새로 붙는 라인만 측정해
/// 누적한다 — 높이 질의(O(1))와 draw 의 가시창 슬라이싱이 이를 사용한다.
///
/// 반환: `(그릴 lines, row_prefix, 새 stable_len(byte), 새 stable_line_count)`.
///
/// 비용: 프레임당 O(꼬리), 블록 완료 시 1회 O(해당 블록). 안정 영역을 절대
/// 복제/재파싱하지 않으므로 메시지가 길어져도 draw 가 멈추지 않는다
/// (realtime-ux). 최종 권위 렌더는 `done` 프레임에서 한 번만 수행된다.
pub(crate) fn streaming_incremental(
    text: &str,
    theme: &Theme,
    width: u16,
    lines: Vec<Line<'static>>,
    row_prefix: Vec<u32>,
    stable_len: usize,
    stable_line_count: usize,
) -> (Vec<Line<'static>>, Vec<u32>, usize, usize) {
    let (lines, row_prefix, stable_len, stable_line_count, _scan) = streaming_incremental_resumed(
        text,
        theme,
        width,
        lines,
        row_prefix,
        stable_len,
        stable_line_count,
        None,
    );
    (lines, row_prefix, stable_len, stable_line_count)
}

/// [`streaming_incremental`], threading the resumable stable-prefix scan cursor.
/// The live layout passes the previous frame's cursor (cached in
/// `RenderCache::Text`) so the per-frame stable-prefix scan is O(new suffix)
/// rather than O(whole accumulated text) — the streaming-freeze fix. Returns the
/// cursor at the new boundary, to cache for the next frame (only meaningful when
/// the boundary advanced; `None` when nothing new was promoted).
#[allow(clippy::too_many_arguments)]
pub(crate) fn streaming_incremental_resumed(
    text: &str,
    theme: &Theme,
    width: u16,
    mut lines: Vec<Line<'static>>,
    mut row_prefix: Vec<u32>,
    mut stable_len: usize,
    mut stable_line_count: usize,
    mut prev_scan: Option<markdown::StreamScanState>,
) -> (
    Vec<Line<'static>>,
    Vec<u32>,
    usize,
    usize,
    Option<markdown::StreamScanState>,
) {
    // 커서가 어긋났으면(폭 변경 등으로 호출부가 빈 상태를 넘기지 않은 경우의
    // 방어) 전부 버리고 처음부터 다시 쌓는다. row_prefix 는 lines 와 항상
    // `len + 1` 로 묶여야 하므로 어긋남도 같은 리셋으로 흡수한다.
    if stable_len > text.len()
        || stable_line_count > lines.len()
        || row_prefix.len() != lines.len() + 1
    {
        lines.clear();
        row_prefix.clear();
        stable_len = 0;
        stable_line_count = 0;
        // The cached cursor is keyed to the discarded stable region; drop it so
        // the scan restarts from byte 0 instead of resuming from a stale point.
        prev_scan = None;
    }
    if row_prefix.is_empty() {
        row_prefix.push(0);
    }
    // Stable boundary + fence context in one scan. Inside a *huge* open code
    // fence the boundary advances to the last completed line (so the per-frame
    // open tail is just the trailing partial line, not the whole fence — the
    // O(n²)-per-turn re-wrap that froze long coding answers); `fence_at_*` tell us
    // to render the fence interior as plain code instead of as standalone markdown.
    // Resume the scan from the cached cursor at `stable_len` so it costs O(new
    // suffix), not O(whole accumulated text) — the per-frame streaming-freeze fix.
    let (boundary, fence_at_stable, fence_at_boundary, fence_lang, scan_at_boundary) =
        markdown::streaming_stable_prefix_resumed(
            text,
            stable_len,
            LARGE_OPEN_FENCE_LIMIT,
            prev_scan.as_ref(),
        );
    let boundary = boundary.min(text.len());
    // Persist the cursor only when the boundary advances (i.e. new content was
    // promoted) — mirroring the `stable_len` promotion below, so the cached
    // cursor's `at` always equals the next frame's `stable_len`.
    let new_scan = if boundary > stable_len {
        Some(scan_at_boundary)
    } else {
        prev_scan
    };
    // 직전 꼬리는 버리고 이미 스타일링된 안정 영역(라인 + 행 prefix)만 남긴다.
    lines.truncate(stable_line_count);
    row_prefix.truncate(stable_line_count + 1);
    // 새로 완료된 블록들을 한 번만 스타일링(syntect 포함)해 안정 영역에 붙인다.
    if boundary > stable_len {
        let fragment_src = &text[stable_len..boundary];
        // Promoting a newly-completed block into the stable region normally runs
        // the authoritative markdown + syntect pass once. For a very large code
        // fence that single pass is 30ms+ on the TUI loop (measured) — it lands
        // on the exact frame the fence closes and reads as the stream freezing.
        // Skip syntect for such fragments; their final color is reconstructed by
        // the `done` full pass for normal-sized blocks, while huge code blocks
        // already keep the incremental cache on `done`
        // (see `should_keep_incremental_done_cache`), so the policy stays
        // consistent either way.
        let mut fragment = if let Some(marker) = fence_at_stable {
            // The fragment begins INSIDE a huge open code fence (a continuation):
            // render its completed lines as the SAME card interior (rail + bg) the
            // done pass uses (roadmap ⑨), so settle does not reflow. No top border
            // here — it was drawn once on the opening-promotion fragment below.
            markdown::rendered_fence_interior_lines(
                fragment_src,
                theme,
                marker,
                width,
                fence_lang.as_deref(),
                false, // emit_top_border: continuation, border already drawn
                false, // skip_opener: mid-fence, no opener line present
            )
        } else if let Some(marker) = fence_at_boundary {
            // The fragment OPENS a huge fence that is still open at the boundary
            // (the one-time promotion when the fence first crosses the size
            // threshold). Render it as the card interior and draw the top border
            // ONCE here; strip the ```lang opener so it is not a code row. The
            // body is light (no syntect) — the multi-ms full pass over a 16 KB+
            // fence is the stall we avoid; `done` keeps this cache for huge fences.
            markdown::rendered_fence_interior_lines(
                fragment_src,
                theme,
                marker,
                width,
                fence_lang.as_deref(),
                true, // emit_top_border: this fragment opens the fence
                true, // skip_opener: fragment_src starts with the ```lang line
            )
        } else if fragment_skips_syntect(fragment_src) {
            markdown::rendered_tail_for_width(fragment_src, theme, width)
        } else {
            markdown::rendered_lines_for_width(fragment_src, theme, width)
        };
        if stable_line_count > 0 {
            trim_leading_blank(&mut fragment);
        }
        stable_line_count += fragment.len();
        append_with_rows(&mut lines, &mut row_prefix, fragment, width);
        stable_len = boundary;
    }
    // 열린 꼬리만 매 프레임 다시 그린다 (작고, syntect 없음 → 값쌈).
    // 구조물(펜스/확인된 표) 안에서는 완성 줄만 표시한다(v3 §5 라인 게이트) —
    // 이 클립이 캐시되는 lines 자체에 적용되므로 높이(layout)와 draw 가 항상
    // 같은 것을 본다(측정==페인트).
    let tail = markdown::clip_tail_for_display(&text[boundary..], fence_at_boundary.is_some());
    if !tail.is_empty() {
        let mut tail_lines = if let Some(marker) = fence_at_boundary {
            // The open tail is the trailing slice of a huge code fence — render it
            // as the SAME card interior (rail + bg) as done (roadmap ⑨). It starts
            // mid-fence (no opener) and the top border was already drawn by the
            // opening-promotion fragment, so emit neither here.
            markdown::rendered_fence_interior_lines(
                tail,
                theme,
                marker,
                width,
                fence_lang.as_deref(),
                false, // emit_top_border
                false, // skip_opener
            )
        } else {
            streaming_markdown_or_plain_lines(tail, theme, width)
        };
        if stable_line_count > 0 {
            trim_leading_blank(&mut tail_lines);
        }
        append_with_rows(&mut lines, &mut row_prefix, tail_lines, width);
    }
    if lines.is_empty() {
        let placeholder = vec![Line::raw(String::new())];
        append_with_rows(&mut lines, &mut row_prefix, placeholder, width);
    }
    (lines, row_prefix, stable_len, stable_line_count, new_scan)
}

/// `extra` 라인들을 `lines` 로 옮겨 붙이면서 각 라인의 wrap-행 수를
/// `row_prefix` 에 누적한다 — `row_prefix.len() == lines.len() + 1` 불변식을
/// 한 곳에서 유지한다.
fn append_with_rows(
    lines: &mut Vec<Line<'static>>,
    row_prefix: &mut Vec<u32>,
    extra: Vec<Line<'static>>,
    width: u16,
) {
    lines.reserve(extra.len());
    row_prefix.reserve(extra.len());
    let mut acc = row_prefix.last().copied().unwrap_or(0);
    for line in extra {
        acc = acc.saturating_add(u32::from(super::line_wrapped_rows(&line, width)));
        row_prefix.push(acc);
        lines.push(line);
    }
}

/// 선두의 빈(공백뿐인) 라인들을 제거 — 프래그먼트/꼬리를 안정 영역 뒤에
/// 이어 붙일 때 블록 구분 빈 줄이 이중으로 쌓이는 것을 막는다. 직전 블록이
/// 이미 trailing 빈 줄로 끝나므로 구분은 유지된다.
fn trim_leading_blank(lines: &mut Vec<Line<'static>>) {
    let drop = lines
        .iter()
        .take_while(|l| l.spans.iter().all(|s| s.content.trim().is_empty()))
        .count();
    if drop > 0 {
        lines.drain(..drop);
    }
}

/// 블록 끝의 빈(공백뿐) 라인들을 제거 — 블록 사이 수직 리듬은 transcript 의
/// `block_gap`(레일 없는 빈 줄 1)이 담당하므로, 문단 trailing blank 가 거기에
/// 겹쳐 분리가 이중이 되는 것을 막는다. 내부 문단 구분 빈 줄은 보존하기 위해
/// 끝에서만 제거하며, 빈 메시지가 0-height 가 되지 않도록 최소 1줄은 남긴다.
fn trim_trailing_blank_lines(lines: &mut Vec<Line<'static>>) {
    while lines.len() > 1
        && lines
            .last()
            .is_some_and(|l| l.spans.iter().all(|s| s.content.trim().is_empty()))
    {
        lines.pop();
    }
}

/// transcript 가 캐시 무효화 정책에 쓰는 layout-preserving 검사.
pub(crate) fn preserves_layout_pub(text: &str) -> bool {
    markdown::preserves_layout(text)
}

// ============================================================================
// Internal helpers
// ============================================================================

fn draw_paragraph(
    frame: &mut Frame<'_>,
    area: Rect,
    lines: Vec<Line<'_>>,
    preserves_layout: bool,
    theme: &Theme,
    scroll_offset: u16,
) {
    let para = if preserves_layout {
        Paragraph::new(lines)
            .style(theme.typography.body)
            .scroll((scroll_offset, 0))
    } else {
        Paragraph::new(lines)
            .style(theme.typography.body)
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0))
    };
    frame.render_widget(para, area);
}

/// `body_scroll` 은 `lines` 에 적용할 Paragraph 스크롤 — 호출자가 전체 본문을
/// 넘기면 블록 스크롤과 같고, 가시창 슬라이스를 넘기면 그 창의 줄 내 잔여
/// 오프셋이다.
///
/// `bullet_visible` 이면 첫 행의 마크 컬럼에 `◆` 저자 불릿을 그린다 (CC/codex
/// 불릿 문법) — 호출자가 "Bullet 마크이고 블록 스크롤 0(첫 행이 보임)"을
/// 판정해 넘긴다. 이후 행과 `Indent` 연속 블록은 마크 없이 같은 본문 컬럼만
/// 유지한다. 완료(`done`) 전환에 따른 시각 변화는 없다 — 진행 표시는
/// transcript 가 아니라 하단 활동 라인의 몫이다 (v3 §4).
fn draw_marked_paragraph(
    frame: &mut Frame<'_>,
    area: Rect,
    mut lines: Vec<Line<'_>>,
    preserves_layout: bool,
    theme: &Theme,
    body_scroll: u16,
    bullet_visible: bool,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    if bullet_visible {
        let mark = glyphs::pick(
            !theme.no_color,
            glyphs::ZO_DIAMOND,
            glyphs::ZO_DIAMOND_NC,
        );
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                mark,
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            ))),
            Rect::new(area.x, area.y, super::ROLE_RAIL_WIDTH.min(area.width), 1),
        );
        // Answer-lead emphasis (v3 readability): promote the bullet row's
        // default-fg spans one gradation to `bright`, so the first line of a
        // spoken answer pops out of the surrounding meta/tool noise when
        // scanning. Draw-time restyle only (span styles are by-value even on
        // borrowed content) — cached lines and heights are untouched, and
        // already-emphasized spans (bold=bright, code=cyan, links) keep their
        // own colors.
        if let Some(first) = lines.first_mut() {
            for span in &mut first.spans {
                if span.style.fg == Some(theme.palette.fg) {
                    span.style.fg = Some(theme.palette.bright);
                }
            }
        }
    }

    if area.width <= super::ROLE_RAIL_WIDTH {
        return;
    }
    // Wrap-path body width must equal the measure width (`mark_body_width` /
    // layout's `prose_wrap_cap` clamp). Painting the wrap at a different
    // ultra-wide width would leave phantom blank rows trailing every paragraph
    // (the "여백" bug).
    // Preserved-layout blocks (fences/tables) keep the full width: their height
    // is line-count based (width-independent) and their card interiors were
    // already rendered at the measure width, so no cap is needed to agree.
    let available_width = area.width.saturating_sub(super::ROLE_RAIL_WIDTH);
    let body_width = if preserves_layout {
        available_width
    } else {
        available_width.min(super::prose_wrap_cap(available_width))
    };
    let text_area = Rect::new(
        area.x.saturating_add(super::ROLE_RAIL_WIDTH),
        area.y,
        body_width,
        area.height,
    );
    draw_paragraph(
        frame,
        text_area,
        lines,
        preserves_layout,
        theme,
        body_scroll,
    );
}


#[cfg(test)]
mod tests {
    use super::{rendered_lines_for_width, streaming_incremental, streaming_incremental_resumed};
    use crate::tui::glyphs;
    use crate::tui::theme::Theme;
    use ratatui::text::Line;

    /// Manual profiling harness for the streaming render hot path. Run with:
    /// `cargo test -p zo-cli --release --lib \
    ///   perf_streaming_render_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "perf measurement, run manually with --ignored --nocapture"]
    fn perf_streaming_render_cost() {
        use crate::tui::markdown;
        use std::fmt::Write as _;
        use std::time::Instant;
        let theme = Theme::default_dark();
        let width = 100u16;

        // Worst case: a long code fence with no interior blank lines, so the
        // streaming boundary cannot advance until the fence closes.
        let mut text = String::from("# 분석 결과\n\n다음은 구현입니다.\n\n```rust\n");
        for i in 0..600 {
            let _ = writeln!(text, "fn function_{i}(value: i32) -> i32 {{ value * {i} + 1 }}");
        }
        text.push_str("```\n\n## 요약\n\n- 항목 1\n- 항목 2\n");
        eprintln!("[perf] text len = {} bytes", text.len());

        let t = Instant::now();
        let lines = markdown::rendered_lines_for_width(&text, &theme, width);
        eprintln!(
            "[perf] DONE full render: {:?} ({} lines)",
            t.elapsed(),
            lines.len()
        );

        let chars: Vec<char> = text.chars().collect();
        let t = Instant::now();
        let (mut l, mut r, mut sl, mut sc) = (Vec::new(), Vec::new(), 0usize, 0usize);
        let (mut shown, mut frames, mut worst) = (0usize, 0u32, std::time::Duration::ZERO);
        while shown < chars.len() {
            shown = (shown + 256).min(chars.len());
            let partial: String = chars[..shown].iter().collect();
            let ft = Instant::now();
            let out = streaming_incremental(&partial, &theme, width, l, r, sl, sc);
            worst = worst.max(ft.elapsed());
            (l, r, sl, sc) = out;
            frames += 1;
        }
        eprintln!(
            "[perf] STREAMING {frames} frames total: {:?}, worst frame: {:?}",
            t.elapsed(),
            worst
        );
    }

    /// Scaling measurement: stream a code answer at 8/32/128/256 KB and report
    /// the worst single frame plus the cumulative `streaming_stable_prefix` scan.
    /// Run with:
    /// `cargo test -p zo-cli --release --lib \
    ///   perf_streaming_scaling -- --ignored --nocapture`
    ///
    /// This is a manual harness, NOT a wall-clock CI gate: a single frame can
    /// spike past any tight budget under concurrent test load, so a wall-clock
    /// assert here flakes. The always-on, deterministic protection is
    /// `large_code_fence_close_skips_syntect_under_frame_budget` plus the
    /// compile-time `FRAGMENT_SYNTECT_SKIP_LIMIT` assertion — those pin the
    /// mechanism (a big fence skips syntect on close) without timing flakiness.
    #[test]
    #[ignore = "perf measurement, run manually with --ignored --nocapture"]
    fn perf_streaming_scaling() {
        use crate::tui::markdown;
        use std::fmt::Write as _;
        use std::time::{Duration, Instant};
        let theme = Theme::default_dark();
        let width = 100u16;
        // Real sessions pre-warm syntect at startup (session/mod.rs); mirror that
        // so the first size isn't charged the one-time syntax-set load.
        let _ = markdown::rendered_lines_for_width("```rust\nfn warm() {}\n```\n", &theme, width);
        for target_kb in [8usize, 32, 128, 256] {
            let mut text = String::from("# 분석\n\n구현:\n\n```rust\n");
            let mut i = 0;
            while text.len() < target_kb * 1024 {
                let _ = writeln!(text, "fn function_{i}(value: i32) -> i32 {{ value * {i} + 1 }}");
                i += 1;
            }
            text.push_str("```\n\n## 요약\n\n- 항목 1\n- 항목 2\n");

            let chars: Vec<char> = text.chars().collect();
            let (mut l, mut r, mut sl, mut sc) = (Vec::new(), Vec::new(), 0usize, 0usize);
            let (mut shown, mut frames, mut worst) = (0usize, 0u32, Duration::ZERO);
            let mut scan_total = Duration::ZERO;
            while shown < chars.len() {
                shown = (shown + 256).min(chars.len());
                let partial: String = chars[..shown].iter().collect();
                // Isolate the scan that does not yet resume from stable_len.
                let st = Instant::now();
                let _ = markdown::streaming_stable_prefix(&partial, sl, super::LARGE_OPEN_FENCE_LIMIT);
                scan_total += st.elapsed();
                let ft = Instant::now();
                let out = streaming_incremental(&partial, &theme, width, l, r, sl, sc);
                worst = worst.max(ft.elapsed());
                (l, r, sl, sc) = out;
                frames += 1;
            }
            eprintln!(
                "[perf] {:>4}KB ({} bytes) · {frames} frames · worst frame {:?} · scan cumulative {:?}",
                target_kb,
                text.len(),
                worst,
                scan_total,
            );
        }
    }

    /// Cost of the three render paths on a closed code fence by size — pins where
    /// the authoritative syntect pass becomes a dropped streaming frame.
    #[test]
    #[ignore = "perf measurement, run manually with --ignored --nocapture"]
    fn perf_fence_render_paths_by_size() {
        use crate::tui::markdown;
        use std::fmt::Write as _;
        use std::time::Instant;
        let theme = Theme::default_dark();
        let width = 100u16;
        // Warm up syntect's lazy syntax-set / theme load so the first measured
        // size isn't charged the one-time init (~20ms).
        let _ = markdown::rendered_lines_for_width("```rust\nfn warm() {}\n```\n", &theme, width);
        for kb in [1usize, 2, 3, 4, 6, 8, 12, 16] {
            let mut body = String::new();
            let mut i = 0;
            while body.len() < kb * 1024 {
                let _ = writeln!(body, "    let v_{i} = compute({i}) + offset * {i};");
                i += 1;
            }
            let fragment = format!("```rust\n{body}```\n");
            let t = Instant::now();
            let _ = markdown::rendered_lines_for_width(&fragment, &theme, width);
            let syntect = t.elapsed();
            let t = Instant::now();
            let _ = super::streaming_markdown_or_plain_lines(&fragment, &theme, width);
            let light = t.elapsed();
            eprintln!(
                "[perf] fence {:>2}KB ({} bytes) · syntect {:?} · light {:?}",
                kb,
                fragment.len(),
                syntect,
                light,
            );
        }
    }

    #[test]
    fn repair_styles_unclosed_inline_markers_while_streaming() {
        use crate::tui::markdown;
        use ratatui::style::Modifier;
        let theme = Theme::default_dark();

        // 미완성 **bold 는 raw 별표 없이 즉시 굵게.
        let bold = super::streaming_markdown_or_plain_lines("값은 **중요", &theme, 80);
        assert!(
            bold.iter().any(|l| l.spans.iter().any(
                |s| s.content.contains("중요") && s.style.add_modifier.contains(Modifier::BOLD)
            )),
            "미완성 **bold 는 즉시 굵게: {:?}",
            flatten(&bold)
        );
        assert!(!flatten(&bold).contains('*'), "raw 별표 노출 금지");

        // 미완성 `code 는 raw 백틱 없이 렌더.
        let code = super::streaming_markdown_or_plain_lines("실행 `zo buil", &theme, 80);
        assert!(!flatten(&code).contains('`'), "raw 백틱 노출 금지");

        // 미완성 ~~strike 는 raw 물결 없이 취소선.
        let strike = super::streaming_markdown_or_plain_lines("옛 ~~삭제됨", &theme, 80);
        assert!(!flatten(&strike).contains('~'), "raw 물결 노출 금지");

        // 짝이 맞는 마커는 zero-copy(원본 그대로) — 회귀 가드.
        assert!(
            matches!(
                markdown::repair_unclosed_inline_markers("값은 **중요** 합니다"),
                std::borrow::Cow::Borrowed(_)
            ),
            "완성 마커는 원본 그대로"
        );

        // 열린 코드펜스 본문의 `**` 등은 인라인 문법이 아니므로 건드리지 않음.
        assert!(
            matches!(
                markdown::repair_unclosed_inline_markers("```rust\nlet x = a ** b"),
                std::borrow::Cow::Borrowed(_)
            ),
            "코드펜스 안은 repair 스킵"
        );
    }

    #[test]
    fn verifier_summary_renders_per_lens_verdict() {
        // Regression for the adversarial-review blocker: the TUI verdict summary
        // must recognize the per-lens VERIFY shape, not only the old `accepted`
        // one — otherwise every deep-gate verify turn renders as plain markdown.
        let theme = Theme::default_dark();
        let accept = super::verifier_json_summary_lines(
            r#"{"spec": true, "regression": true, "security": true, "issues": []}"#,
            &theme,
            80,
        )
        .expect("per-lens accept must render a verifier summary");
        assert!(flatten(&accept).contains("accepted"));

        let reject = super::verifier_json_summary_lines(
            r#"{"spec": false, "regression": true, "security": true, "issues": ["missing test"]}"#,
            &theme,
            80,
        )
        .expect("per-lens reject must render a verifier summary");
        let reject_text = flatten(&reject);
        assert!(reject_text.contains("rejected"));
        assert!(reject_text.contains("missing test"));

        // Old single-`accepted` shape still renders (backward compatible).
        assert!(
            super::verifier_json_summary_lines(r#"{"accepted": true, "issues": []}"#, &theme, 80)
                .is_some()
        );
        // A normal JSON-ish answer is not mistaken for a verdict.
        assert!(super::verifier_json_summary_lines(r#"{"foo": 1}"#, &theme, 80).is_none());
    }

    #[test]
    fn verifier_summary_extracts_verdict_embedded_in_prose() {
        // The model is told to emit ONLY the JSON, but sometimes it explains
        // each lens first and concludes with the verdict (the reported "raw
        // JSON leaking into the transcript" case). The per-lens prose must be
        // kept, and the trailing `{…}` rendered as the pretty verdict block —
        // never shown raw.
        let theme = Theme::default_dark();
        let text = "I've confirmed:\n\n- spec: edits are behavior-preserving\n- regression: only intended files changed\n- security: no new surface\n\n{\"spec\": true, \"regression\": true, \"security\": true, \"issues\": []}";
        let lines = super::verifier_json_summary_lines(text, &theme, 80)
            .expect("embedded verdict must render");
        let rendered = flatten(&lines);
        // Pretty verdict present…
        assert!(rendered.contains("accepted"), "verdict rendered: {rendered}");
        // …prose kept…
        assert!(
            rendered.contains("behavior-preserving"),
            "per-lens prose kept: {rendered}"
        );
        // …and the raw JSON braces are gone.
        assert!(
            !rendered.contains("\"spec\""),
            "raw JSON must not leak: {rendered}"
        );
    }

    #[test]
    fn verifier_summary_extracts_verdict_from_json_fence() {
        // Same, but the verdict is wrapped in a ```json fence: strip the fence
        // and render the verdict, leaving no dangling code block.
        let theme = Theme::default_dark();
        let text = "```json\n{\"spec\": true, \"regression\": true, \"security\": true, \"issues\": []}\n```";
        let lines = super::verifier_json_summary_lines(text, &theme, 80)
            .expect("fenced verdict must render");
        let rendered = flatten(&lines);
        assert!(rendered.contains("accepted"));
        assert!(!rendered.contains('`'), "fence stripped: {rendered}");
        assert!(!rendered.contains("\"spec\""), "raw JSON gone: {rendered}");
    }

    #[test]
    fn fragment_skips_syntect_only_for_large_code_fences() {
        // Large code fence → skip syntect (the 30ms+ single-frame freeze case).
        let big_code = format!("```rust\n{}```", "fn f() { let _ = 0; }\n".repeat(2000));
        assert!(big_code.len() > super::FRAGMENT_SYNTECT_SKIP_LIMIT);
        assert!(super::fragment_skips_syntect(&big_code));

        // Small code fence → keep syntect (cheap, colored).
        assert!(!super::fragment_skips_syntect("```rust\nfn f() {}\n```"));

        // Large prose with no fence → keep the normal markdown path.
        let big_prose = "단락 ".repeat(20 * 1024);
        assert!(big_prose.len() > super::FRAGMENT_SYNTECT_SKIP_LIMIT);
        assert!(!super::fragment_skips_syntect(&big_prose));
    }

    /// Always-on structural perf gate (covers debug, where wall-clock gates are
    /// too noisy). syntect on code is ~1.5ms/KB, so the threshold that routes a
    /// closing fence to the light (no-syntect) path must stay low enough that no
    /// single streaming frame exceeds the 16ms budget. Regression: the limit was
    /// 24KB, so an 8–24KB fence took a 12–33ms syntect pass on the frame it closed
    /// (the measured "뚝뚝" mid-stream stutter). The `done` pass recolors below
    /// `FINAL_DONE_CODE_RENDER_LIMIT`, so color is deferred, never lost.
    #[test]
    fn large_code_fence_close_skips_syntect_under_frame_budget() {
        // The constant relationship is enforced at compile time, see the
        // `const _` assertion next to FRAGMENT_SYNTECT_SKIP_LIMIT.
        // ~13KB fence — squarely in the old valley of death (didn't promote, took
        // a ~17ms syntect frame on close). Must now skip syntect.
        let valley = format!("```rust\n{}```\n", "fn f() { let _ = 0; }\n".repeat(620));
        assert!(valley.len() > 12 * 1024 && valley.len() < super::LARGE_OPEN_FENCE_LIMIT);
        assert!(
            super::fragment_skips_syntect(&valley),
            "a 13KB fence must take the light path on close, not a 17ms syntect pass"
        );
    }

    #[test]
    fn streaming_incremental_large_code_fence_stays_structured_without_syntect() {
        use std::fmt::Write as _;
        let theme = Theme::default_dark();
        let mut text = String::from("## 구현\n\n```rust\n");
        for i in 0..800 {
            let _ = writeln!(text, "fn function_{i}(value: i32) -> i32 {{ value }}");
        }
        text.push_str("```\n");
        assert!(text.len() > super::FRAGMENT_SYNTECT_SKIP_LIMIT);

        // A large code fence renders through the syntect-free pulldown path while
        // streaming (fragment promotion or the bounded open tail ≤64KB), so the
        // fence markers become a code card and never leak as raw text.
        let (lines, row_prefix, _stable_len, _stable_count) =
            streaming_incremental(&text, &theme, 100, Vec::new(), Vec::new(), 0, 0);
        assert_eq!(row_prefix.len(), lines.len() + 1);
        let flat = flatten(&lines);
        // Code body is visible…
        assert!(flat.contains("function_0") && flat.contains("function_799"));
        // …and the raw ``` fence markers never leak into the rendered output.
        assert!(
            !flat.contains("```"),
            "fence markers must not leak: tail of {flat:?}"
        );
    }

    /// Pins the bullet grammar (v3 §3): a `Bullet` block carries the `◆`
    /// author mark on its first row with the body starting at the mark-column
    /// offset, and the mark adds no extra height rows.
    #[test]
    fn bullet_block_draws_author_mark_on_first_body_row() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::layout::Rect;

        let theme = Theme::default_dark();
        let (width, height) = (32u16, 4u16);
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                super::draw_with_mark(
                    frame,
                    Rect::new(0, 0, width, height),
                    "first line\n\nsecond line",
                    true,
                    &theme,
                    0,
                    0,
                    super::super::ProseMark::Bullet,
                );
            })
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let row = |y: u16| -> String {
            (0..width)
                .map(|x| buffer[(x, y)].symbol().to_string())
                .collect()
        };
        assert_eq!(
            buffer[(0u16, 0u16)].symbol(),
            glyphs::ZO_DIAMOND,
            "bullet mark missing at row 0 col 0: {:?}",
            row(0)
        );
        // Body rides the same row as the bullet (no header rows) and starts at
        // the mark-column offset shared with continuation rows.
        assert!(
            row(0).contains("first line"),
            "first body line must share the bullet row: {:?}",
            row(0)
        );
        assert!(
            row(2).contains("second line") && row(2).starts_with("   "),
            "continuation rows keep the indent column without a mark: {:?}",
            row(2)
        );
    }

    /// An `Indent` continuation never repeats the author mark — its rows are
    /// indent-only so the prose reads as one flow under the first bullet.
    #[test]
    fn indent_block_has_no_mark_glyph() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::layout::Rect;

        let theme = Theme::default_dark();
        let (width, height) = (32u16, 3u16);
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                super::draw_with_mark(
                    frame,
                    Rect::new(0, 0, width, height),
                    "continued thought",
                    true,
                    &theme,
                    0,
                    0,
                    super::super::ProseMark::Indent,
                );
            })
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let row0: String = (0..width).map(|x| buffer[(x, 0u16)].symbol().to_string()).collect();
        assert!(
            row0.starts_with("   ") && row0.contains("continued thought"),
            "indent rows must keep the body column without any mark glyph: {row0:?}"
        );
        assert!(
            !row0.contains(glyphs::ZO_DIAMOND),
            "continuation must not repeat the author bullet: {row0:?}"
        );
    }

    #[test]
    fn streaming_cache_miss_path_keeps_markdown_structured() {
        let theme = Theme::default_dark();
        let md = "## 요약\n\n- **상태**: 진행 중\n- 다음 항목";
        let lines = rendered_lines_for_width(md, false, &theme, 0, 72);
        let text = flatten(&lines);

        assert!(text.contains("요약") && text.contains("상태"));
        assert!(
            !text.contains("## ") && !text.contains("**"),
            "live cache-miss rendering must not expose raw markdown: {text:?}"
        );
        assert!(
            text.contains('\u{258C}') || text.contains('\u{258E}'),
            "heading glyph should be applied on the live fallback path: {text:?}"
        );
    }

    #[test]
    fn streaming_single_trailing_newline_does_not_create_phantom_body_line() {
        let theme = Theme::default_dark();
        let lines = rendered_lines_for_width("단일 줄\n", false, &theme, 0, 80);
        assert_eq!(
            lines.len(),
            1,
            "single trailing newline must not add a phantom body row: {lines:?}"
        );
        assert!(flatten(&lines).contains("단일 줄"));
    }

    /// 라인들의 보이는 텍스트를 합친다 (마커 누수 검증용).
    fn flatten(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 누적 prefix 로 스트리밍을 시뮬레이션하며, 직전 반환값을 다음 호출의
    /// prev 로 넘긴다 (transcript 캐시와 동일한 계약).
    #[test]
    fn streaming_incremental_styles_completed_blocks_without_leaking_markers() {
        let theme = Theme::default_dark();
        let full =
            "## 개요\n\n- **규모**: 10개 크레이트\n- 품질: clippy\n\n### 다음\n\n본문 타이핑 중";
        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut rows: Vec<u32> = Vec::new();
        let mut stable_len = 0usize;
        let mut stable_count = 0usize;
        let mut last_stable_len = 0usize;
        let mut end = 1;
        while end <= full.len() {
            if !full.is_char_boundary(end) {
                end += 1;
                continue;
            }
            let (l, rp, sl, sc) = streaming_incremental(
                &full[..end],
                &theme,
                60,
                lines,
                rows,
                stable_len,
                stable_count,
            );
            assert!(
                sl >= last_stable_len,
                "stable_len regressed {last_stable_len}->{sl}"
            );
            assert!(sc <= l.len(), "stable_count {sc} exceeds lines {}", l.len());
            // 증분 누적된 행 prefix 는 매 step 전체 재계산과 동일해야 한다 —
            // 높이/슬라이싱이 이 벡터를 신뢰하는 근거.
            assert_eq!(
                rp,
                crate::tui::blocks::wrapped_row_prefix(&l, 60),
                "incremental row prefix diverged from full recompute at {end} bytes"
            );
            // 완료된(안정) 영역에는 raw 마크다운 마커가 남으면 안 된다.
            let stable_text = flatten(&l[..sc]);
            assert!(
                !stable_text.contains("## "),
                "stable region leaked heading marker: {stable_text:?}"
            );
            assert!(
                !stable_text.contains("**"),
                "stable region leaked bold marker: {stable_text:?}"
            );
            last_stable_len = sl;
            lines = l;
            rows = rp;
            stable_len = sl;
            stable_count = sc;
            end += 1;
        }
        // 최종 누적: 본문 글자는 보존되고 헤딩 글리프(▌/▎)가 적용돼 있다.
        let all = flatten(&lines);
        assert!(
            all.contains("개요") && all.contains("다음"),
            "content preserved: {all:?}"
        );
        assert!(
            all.contains('\u{258C}') || all.contains('\u{258E}'),
            "heading glyph applied during streaming: {all:?}"
        );
    }

    #[test]
    fn streaming_incremental_keeps_long_markdown_tail_structured() {
        let theme = Theme::default_dark();
        let mut text = String::from("```js\n");
        // Exceed the plain-streaming fallback threshold. The open fence must
        // still be rendered as markdown while streaming; otherwise users see
        // raw ``` markers and a broken block until the final done=true pass.
        for i in 0..5200 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "if (values[{i}] > 0) {{ sum += values[{i}]; }}");
        }

        let (lines, row_prefix, stable_len, stable_count) =
            streaming_incremental(&text, &theme, 96, Vec::new(), Vec::new(), 0, 0);
        // P1: a huge open fence (> LARGE_OPEN_FENCE_LIMIT) now promotes its
        // completed code lines into the stable region so the per-frame re-render
        // is O(trailing line) instead of O(whole fence). The completed lines are
        // styled once (here, all of them — the buffer ends on a line boundary).
        assert!(
            stable_len > 0,
            "huge open fence must promote completed code lines, not re-render the whole tail each frame"
        );
        assert!(stable_count > 0, "promoted fence lines are cached stable");
        assert_eq!(row_prefix.len(), lines.len() + 1);
        let all = flatten(&lines);
        assert!(all.contains("values[0]"), "code content preserved");
        assert!(all.contains("values[5199]"), "latest streamed code line present");
        assert!(
            !all.contains("```"),
            "streaming markdown tail must not fall back to raw fenced text"
        );
    }

    #[test]
    fn streaming_incremental_huge_fence_promotes_lines_incrementally_across_frames() {
        // P1: across streaming frames a huge open fence reuses its already-styled
        // stable region and promotes only the newly-completed lines — the boundary
        // advances instead of re-rendering the whole fence each frame (the O(n²)
        // that froze long coding answers). The fence interior never corrupts and
        // never leaks raw ``` markers.
        let theme = Theme::default_dark();
        let mut text = String::from("```js\n");
        for i in 0..2000 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "const x{i} = compute({i});");
        }
        assert!(text.len() > super::LARGE_OPEN_FENCE_LIMIT);

        // Frame 1.
        let (lines1, rows1, stable_len1, count1) =
            streaming_incremental(&text, &theme, 96, Vec::new(), Vec::new(), 0, 0);
        assert!(stable_len1 > 0, "completed fence lines promoted to stable");
        assert_eq!(rows1.len(), lines1.len() + 1);

        // Frame 2: more code lines arrive; the cached stable region is reused and
        // the boundary advances over only the new lines.
        for i in 2000..2010 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "const x{i} = compute({i});");
        }
        let (lines2, rows2, stable_len2, _count2) =
            streaming_incremental(&text, &theme, 96, lines1, rows1, stable_len1, count1);
        assert!(
            stable_len2 > stable_len1,
            "boundary advances as more fence lines complete (incremental promotion)"
        );
        assert_eq!(rows2.len(), lines2.len() + 1);
        let all = flatten(&lines2);
        assert!(
            all.contains("compute(0)") && all.contains("compute(2009)"),
            "first and latest streamed code lines both present"
        );
        assert!(!all.contains("```"), "no raw fence markers leak");
    }

    #[test]
    fn resumed_incremental_matches_full_scan_wiring_across_frames() {
        // The cache-backed `streaming_incremental_resumed` (resumes the stable-
        // prefix scan from the cached cursor) must produce byte-identical output
        // to threading `streaming_incremental` (which full-scans every frame),
        // frame by frame. This guards the production wiring of the streaming-
        // freeze fix: same promotion decisions, same lines, same boundary.
        let theme = Theme::default_dark();
        let mut full = String::from("intro paragraph one.\n\nsecond paragraph here.\n\n```rust\n");
        for i in 0..40 {
            use std::fmt::Write as _;
            let _ = writeln!(full, "let v{i} = {i} * 2;");
        }
        full.push_str("```\n\n- bullet one\n- bullet two\n\ntrailing partial line");

        // Two parallel simulations threaded across frames: `inc` resumes the
        // cached scan cursor; `whole` full-scans every frame via the wrapper.
        let (mut inc_lines, mut inc_rows, mut inc_stable, mut inc_count): (
            Vec<Line<'static>>,
            Vec<u32>,
            _,
            _,
        ) = (Vec::new(), Vec::new(), 0usize, 0usize);
        let mut inc_scan: Option<super::markdown::StreamScanState> = None;
        let (mut whole_lines, mut whole_rows, mut whole_stable, mut whole_count) =
            (Vec::new(), Vec::new(), 0usize, 0usize);

        let mut end = 0usize;
        while end <= full.len() {
            if !full.is_char_boundary(end) {
                end += 1;
                continue;
            }
            let prefix = &full[..end];
            let resumed = streaming_incremental_resumed(
                prefix, &theme, 96, inc_lines, inc_rows, inc_stable, inc_count, inc_scan,
            );
            let wrapped = streaming_incremental(
                prefix,
                &theme,
                96,
                whole_lines,
                whole_rows,
                whole_stable,
                whole_count,
            );
            assert_eq!(resumed.2, wrapped.2, "stable_len diverged at end={end}");
            assert_eq!(resumed.0.len(), wrapped.0.len(), "line count diverged at end={end}");
            assert_eq!(resumed.1, wrapped.1, "row prefix diverged at end={end}");
            assert_eq!(
                flatten(&resumed.0),
                flatten(&wrapped.0),
                "rendered text diverged at end={end}"
            );
            inc_lines = resumed.0;
            inc_rows = resumed.1;
            inc_stable = resumed.2;
            inc_count = resumed.3;
            inc_scan = resumed.4;
            whole_lines = wrapped.0;
            whole_rows = wrapped.1;
            whole_stable = wrapped.2;
            whole_count = wrapped.3;
            end += 7; // stride keeps the test fast while still crossing boundaries
        }
    }

    /// roadmap ⑨: a GIANT (>16KB) code fence must STREAM with the SAME card frame
    /// (top border + "  │ " rail) it SETTLES into at done, so the block does not
    /// reflow / scroll-jump when the turn ends. The only allowed differences are
    /// the bottom border (legitimately added when the fence actually closes) and
    /// color. Compares PLAIN per-line text (rail + code), never span count/style —
    /// syntect (done) splits one logical line into many colored spans whose
    /// concatenation still equals the light streaming row.
    #[test]
    fn giant_fence_streams_with_same_card_frame_as_done_no_reflow() {
        let theme = Theme::default_dark();
        let w = 80u16;
        // Long lines that WILL wrap under the 4-col rail, so the rail's effect on
        // the wrap point is exercised — the load-bearing byte-identity.
        let mut body = String::new();
        for i in 0..300 {
            use std::fmt::Write as _;
            let _ = writeln!(
                body,
                "let very_long_identifier_{i} = some_function_call_with_a_long_name({i}, {i} + 1, {i} + 2); // trailing comment forcing a wrap at this width"
            );
        }
        assert!(body.len() > super::LARGE_OPEN_FENCE_LIMIT);
        let open = format!("```rust\n{body}"); // fence still OPEN (streaming)
        let closed = format!("```rust\n{body}```\n"); // fence CLOSED (done)

        let (s_lines, s_prefix, _, _) =
            streaming_incremental(&open, &theme, w, Vec::new(), Vec::new(), 0, 0);
        let d_lines = rendered_lines_for_width(&closed, true, &theme, 0, w);
        assert_eq!(s_prefix.len(), s_lines.len() + 1, "row prefix stays paired");

        let plain = |l: &Line<'static>| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        };

        // Exactly one top border in each, byte-identical (rail + " rust " + dashes).
        let s_top: Vec<String> = s_lines
            .iter()
            .map(&plain)
            .filter(|t| t.starts_with("  ╭"))
            .collect();
        let d_top: Vec<String> = d_lines
            .iter()
            .map(&plain)
            .filter(|t| t.starts_with("  ╭"))
            .collect();
        assert_eq!(
            s_top.len(),
            1,
            "streaming draws the card top border exactly once"
        );
        assert_eq!(d_top.len(), 1, "done draws the card top border once");
        assert_eq!(
            s_top[0], d_top[0],
            "top border byte-identical (rail + label + dashes)"
        );

        // Every body row carries the identical rail + identical code text.
        let s_body: Vec<String> = s_lines
            .iter()
            .map(&plain)
            .filter(|t| t.starts_with("  │ "))
            .collect();
        let d_body: Vec<String> = d_lines
            .iter()
            .map(&plain)
            .filter(|t| t.starts_with("  │ "))
            .collect();
        assert!(!s_body.is_empty(), "streaming has railed code rows");
        assert_eq!(
            s_body.len(),
            d_body.len(),
            "same code-row count → no reflow at settle"
        );
        for (s, d) in s_body.iter().zip(d_body.iter()) {
            assert_eq!(s, d, "streaming vs done code row must be byte-identical");
        }

        // Done has the bottom border; the still-open streaming fence does not.
        assert!(
            d_lines.iter().map(&plain).any(|t| t.starts_with("  ╰")),
            "done draws the bottom border"
        );
        assert!(
            !s_lines.iter().map(&plain).any(|t| t.starts_with("  ╰")),
            "streaming open fence omits the bottom border"
        );

        // Height parity through the SAME wrap engine proves identical wrap points.
        let s_body_lines: Vec<Line<'static>> = s_lines
            .iter()
            .filter(|l| plain(l).starts_with("  │ "))
            .cloned()
            .collect();
        let d_body_lines: Vec<Line<'static>> = d_lines
            .iter()
            .filter(|l| plain(l).starts_with("  │ "))
            .cloned()
            .collect();
        let s_h = *crate::tui::blocks::wrapped_row_prefix(&s_body_lines, w)
            .last()
            .unwrap();
        let d_h = *crate::tui::blocks::wrapped_row_prefix(&d_body_lines, w)
            .last()
            .unwrap();
        assert_eq!(
            s_h, d_h,
            "wrapped body height identical → settle does not scroll-jump"
        );
    }

    /// roadmap ⑨ scope: giant ```diff and ```patch fences keep the LEGACY light
    /// interior (no card) while streaming, because the done render prepends a
    /// line-number gutter the light path can't match — framing either alias would
    /// reflow. This guards the deliberate fallback: no card rail/border, no crash,
    /// no raw ``` leak.
    #[test]
    fn giant_diff_and_patch_fences_stream_legacy_without_a_card_frame() {
        let theme = Theme::default_dark();
        let mut body = String::from("@@ -1,2 +1,2 @@\n context line before changes\n");
        for i in 0..400 {
            use std::fmt::Write as _;
            let _ = writeln!(body, "+added line {i} with enough text to exceed the limit");
        }
        assert!(body.len() > super::LARGE_OPEN_FENCE_LIMIT);

        let plain_line = |line: &Line<'static>| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        };

        for lang in ["diff", "patch"] {
            let open = format!("```{lang}\n{body}");
            let (lines, prefix, _, _) =
                streaming_incremental(&open, &theme, 80, Vec::new(), Vec::new(), 0, 0);
            assert_eq!(prefix.len(), lines.len() + 1, "row prefix stays paired");
            let plain_lines: Vec<String> = lines.iter().map(plain_line).collect();
            let all = plain_lines.join("\n");
            assert!(all.contains("@@ -1,2 +1,2 @@"), "{lang} hunk header preserved");
            assert!(all.contains("added line 0"), "{lang} content preserved");
            assert!(!all.contains("```"), "{lang} raw fence markers do not leak");
            assert!(
                plain_lines
                    .iter()
                    .all(|line| !line.starts_with("  ╭") && !line.starts_with("  │ ")),
                "a {lang} fence keeps the legacy light interior (no card) while streaming"
            );
        }
    }

    /// 폭 변경 시 호출부가 빈 prev 를 넘기는 계약 — 처음부터 안전하게 재구성.
    #[test]
    fn streaming_incremental_keeps_long_table_tail_structured() {
        let theme = Theme::default_dark();
        let mut text = String::from(
            "| name | value |
| --- | ---: |
",
        );
        // No heading/bold/fence/list marker: the large open table tail renders
        // through pulldown, which structures GFM rows instead of leaking pipes.
        for i in 0..2000 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "| metric-{i} | {i} |");
        }
        assert!(text.len() > 8 * 1024);

        let (lines, row_prefix, stable_len, stable_count) =
            streaming_incremental(&text, &theme, 96, Vec::new(), Vec::new(), 0, 0);
        assert_eq!(
            stable_len, 0,
            "dense table tail remains open while streaming"
        );
        assert_eq!(stable_count, 0, "open table tail must not be cached stable");
        assert_eq!(row_prefix.len(), lines.len() + 1);
        let all = flatten(&lines);
        assert!(all.contains("name") && all.contains("metric-0"));
        assert!(
            !all.contains("| ---") && !all.contains("---: |") && !all.contains('|'),
            "large streaming tables must not expose raw GFM table syntax: {all:?}"
        );
    }

    #[test]
    fn streaming_incremental_keeps_empty_cell_table_tail_structured() {
        let theme = Theme::default_dark();
        let mut text = String::from(
            "| name | value |
| --- | --- |
",
        );
        for i in 0..2000 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "| metric-{i} | |");
        }
        assert!(text.len() > 8 * 1024);

        let (lines, row_prefix, stable_len, stable_count) =
            streaming_incremental(&text, &theme, 96, Vec::new(), Vec::new(), 0, 0);
        assert_eq!(stable_len, 0);
        assert_eq!(stable_count, 0);
        assert_eq!(row_prefix.len(), lines.len() + 1);
        let all = flatten(&lines);
        assert!(all.contains("name") && all.contains("metric-0"));
        assert!(
            !all.contains("| metric-0 |") && !all.contains('|'),
            "empty-cell GFM table rows must not leak raw pipe syntax: {all:?}"
        );
    }

    #[test]
    fn streaming_incremental_does_not_tableize_non_table_pipe_lines() {
        let theme = Theme::default_dark();
        let mut text = String::from(
            "## Commands
",
        );
        // A long markdown tail can contain shell pipelines; those are not table
        // rows unless a header is immediately followed by a GFM delimiter row.
        for i in 0..1000 {
            use std::fmt::Write as _;
            let _ = writeln!(text, "cat file{i}.log | grep ERROR | sort");
        }
        assert!(text.len() > 8 * 1024);

        let (lines, _, stable_len, stable_count) =
            streaming_incremental(&text, &theme, 120, Vec::new(), Vec::new(), 0, 0);
        assert_eq!(stable_len, 0);
        assert_eq!(stable_count, 0);
        let all = flatten(&lines);
        assert!(
            all.contains("cat file0.log | grep ERROR | sort"),
            "non-table pipe text must stay intact: {all:?}"
        );
        assert!(
            !all.contains("cat file0.log  ·  grep ERROR  ·  sort"),
            "non-table pipe text must not be rendered as a table row: {all:?}"
        );
    }

    #[test]
    fn streaming_incremental_single_trailing_newline_height_is_stable() {
        let theme = Theme::default_dark();
        let text = "단일 줄\n";
        let (lines, rows, stable_len, stable_count) =
            streaming_incremental(text, &theme, 80, Vec::new(), Vec::new(), 0, 0);
        assert_eq!(lines.len(), 1);
        assert_eq!(rows.last().copied(), Some(1));

        let (lines_again, rows_again, stable_len_again, stable_count_again) =
            streaming_incremental(text, &theme, 80, lines, rows, stable_len, stable_count);
        assert_eq!(lines_again.len(), 1);
        assert_eq!(rows_again.last().copied(), Some(1));
        assert_eq!(stable_len_again, stable_len);
        assert_eq!(stable_count_again, stable_count);
    }

    /// roadmap ⑧ residual: a section-style dense answer (numbered rows promoted to
    /// headings) must STREAM at the same height it SETTLES into. The headings
    /// already promote live (small tails take the full-pulldown streaming path),
    /// so the only divergence was the renderer's trailing blank-line artifacts,
    /// which the done pass trims but streaming kept — a line-or-two shrink at
    /// settle. Now both trim, so the line counts match.
    #[test]
    fn dense_prose_streams_at_the_height_it_settles_into() {
        let theme = Theme::default_dark();
        let dense = "1. 인증 흐름\n토큰을 먼저 확인한다\n2. 갱신 경로\n만료 전 재발급한다";
        let stream = rendered_lines_for_width(dense, false, &theme, 0, 60);
        let done = rendered_lines_for_width(dense, true, &theme, 0, 60);
        assert_eq!(
            stream.len(),
            done.len(),
            "dense prose streams at the settle height (no trailing-blank shrink)\n\
             stream={stream:#?}\ndone={done:#?}"
        );
        let plain =
            |l: &Line<'static>| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>();
        // The promotion is live: the section rows are present while streaming.
        assert!(
            stream.iter().any(|l| plain(l).contains("인증 흐름"))
                && stream.iter().any(|l| plain(l).contains("갱신 경로")),
            "section rows present while streaming: {stream:#?}"
        );
        // A just-typed paragraph break ("\n\n") still shows immediately mid-stream.
        let mid = rendered_lines_for_width("para one\n\n", false, &theme, 0, 60);
        assert!(
            mid.last().is_some_and(|l| plain(l).trim().is_empty()),
            "an intentional just-typed blank line stays visible mid-stream: {mid:#?}"
        );
    }

    #[test]
    fn streaming_incremental_rebuilds_from_empty_prev() {
        let theme = Theme::default_dark();
        let text = "para a\n\npara b open";
        let (l, rp, sl, sc) = streaming_incremental(text, &theme, 40, Vec::new(), Vec::new(), 0, 0);
        assert!(!l.is_empty());
        assert_eq!(rp.len(), l.len() + 1, "row prefix must cover every line");
        assert!(sl <= text.len());
        assert!(sc <= l.len());
    }

    /// 커서가 어긋난(stable_len > text) 입력은 방어적으로 리셋해 패닉이 없다.
    #[test]
    fn streaming_incremental_defensive_reset_on_desync() {
        let theme = Theme::default_dark();
        let (l, rp, sl, sc) = streaming_incremental(
            "short",
            &theme,
            40,
            vec![Line::raw("stale")],
            vec![0, 1],
            9999,
            50,
        );
        assert!(!l.is_empty());
        assert_eq!(rp.len(), l.len() + 1);
        assert_eq!(sc, 0, "desync must reset stable_count");
        assert!(sl <= "short".len());
    }

    /// 캐시 경로(`draw_cached`: 가시창 슬라이스 + 줄 내 잔여 스크롤)와 비캐시
    /// 경로(`draw_with_label`: 전체 본문 + Paragraph wrap-skip)가 모든 스크롤
    /// 오프셋에서 byte-identical 한 화면을 그린다 — 슬라이싱 최적화의 핵심
    /// 등가성 보증. plain / labeled 양쪽 모두 sweep 한다.
    #[test]
    fn draw_cached_window_matches_full_draw_at_every_scroll() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::layout::Rect;

        let theme = Theme::default_dark();
        let md = "## 제목\n\n긴 문단이 좁은 폭에서 여러 행으로 wrap 되는 경우를 검증한다. \
                  `inline code` 와 **bold** 를 포함하고 한국어 CJK 폭 계산도 섞는다.\n\n\
                  - 항목 하나\n- 항목 둘\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n\n\
                  마지막 문단도 충분히 길어서 wrap 이 발생한다. one two three four five six.";
        let width = 32u16;
        let view_h = 7u16;

        for prose in [
            super::super::ProseMark::Bare,
            super::super::ProseMark::Indent,
            super::super::ProseMark::Bullet,
        ] {
            let body_width = super::mark_body_width(width, prose);
            let lines = super::rendered_lines_for_width(md, true, &theme, 0, body_width);
            let rows = crate::tui::blocks::wrapped_row_prefix(&lines, body_width);
            let total = rows.last().copied().unwrap_or(0) + 4;

            let paint = |use_cache: bool, scroll: u16| -> Vec<String> {
                let backend = TestBackend::new(width, view_h);
                let mut terminal = Terminal::new(backend).expect("terminal");
                terminal
                    .draw(|frame| {
                        let area = Rect::new(0, 0, width, view_h);
                        if use_cache {
                            super::draw_cached(
                                frame, area, &lines, &rows, false, &theme, scroll, prose,
                            );
                        } else {
                            super::draw_with_mark(frame, area, md, true, &theme, 0, scroll, prose);
                        }
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

            for scroll in 0..=u16::try_from(total).expect("fits") {
                assert_eq!(
                    paint(true, scroll),
                    paint(false, scroll),
                    "cached window draw diverged from full draw (prose={prose:?}, scroll={scroll})"
                );
            }
        }
    }
}
