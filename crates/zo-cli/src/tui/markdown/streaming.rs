//! 스트리밍/증분 마크다운 표시 경로 — 토큰이 도착하는 도중의 열린 꼬리를
//! 전체 pulldown/syntect 렌더 없이 저비용으로 그린다. 닫히지 않은 인라인
//! 마커 임시봉합(repair)과, 마크다운처럼 보이는 큰 열린 블록을 매 프레임
//! 가볍게 렌더하는 bounded tail 렌더러를 담당한다. 전체 렌더(`Renderer`)와는
//! 별개 책임이라 분리한다.

use std::borrow::Cow;

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use syntect::util::LinesWithEndings;

use crate::tui::theme::Theme;

use super::{code_card_frame_lines, rendered_tail_for_width};

/// 스트리밍 꼬리에서 아직 닫히지 않은 인라인 마커(`*`/`_` 기울임·`**`/`__`
/// 굵게·`` ` `` 코드·`~~` 취소선)를 임시로 닫아, 토큰이 도착하는 도중에도 raw
/// 마커가 잠깐 노출됐다가 스타일로 바뀌는 깜빡임 없이 곧장 styled 로 보이게
/// 한다(tigerabrodi repair). 마커가 이미 짝이 맞으면 원본을 그대로
/// 빌려준다(zero-copy). 인라인 마커는 한 줄 안에서 완결되므로 마지막 줄만
/// 검사하고, 열린 코드펜스 본문은 인라인 문법이 아니므로 건드리지 않는다.
///
/// 단일 `*`/`_` 기울임은 CommonMark 의 flanking 규칙을 적용해 진짜 강조
/// opener 만 닫는다 — `2 * 3` (앞뒤 공백) 이나 `snake_case` (intraword `_`) 의
/// 우연한 마커는 강조가 아니므로 건드리지 않는다(pulldown 도 강조로 보지 않아
/// 깜빡임 자체가 없다).
#[must_use]
pub fn repair_unclosed_inline_markers(text: &str) -> Cow<'_, str> {
    let last = text.rsplit('\n').next().unwrap_or(text);
    let trimmed = last.trim_start();
    if super::leading_fence(trimmed).is_some() || has_open_code_fence(text) {
        return Cow::Borrowed(text);
    }
    let suffix = unclosed_marker_suffix(last);
    if suffix.is_empty() {
        return Cow::Borrowed(text);
    }
    Cow::Owned(format!("{text}{suffix}"))
}

/// Closers needed to balance the open inline markers in `line`, in the order
/// they must be appended (innermost emphasis first, then code). Empty when the
/// line is already balanced.
fn unclosed_marker_suffix(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut in_code = false;
    // A stack of open inline delimiter runs as `(marker char, run length)`:
    // `('*', 1)` italic, `('_', 2)` bold, `('~', 2)` strikethrough. Pulldown
    // pairs runs by char + length, so unclosed runs are closed in reverse (LIFO)
    // order — preserving nesting (e.g. `**bold ~~strike` → `~~` then `**`).
    let mut open: Vec<(char, usize)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '`' => {
                in_code = !in_code;
                i += 1;
            }
            '~' if !in_code && chars.get(i + 1) == Some(&'~') => {
                toggle_paired_run(&mut open, '~', 2);
                i += 2;
            }
            '*' | '_' if !in_code => {
                let marker = chars[i];
                let run = run_length(&chars, i, marker);
                let before = if i == 0 { None } else { Some(chars[i - 1]) };
                let after = chars.get(i + run).copied();
                apply_emphasis_run(&mut open, marker, run, before, after);
                i += run;
            }
            _ => i += 1,
        }
    }
    let mut suffix = String::new();
    // Code is the innermost still-open span (emphasis inside code is inert), so
    // close it first, then unwind the delimiter stack (LIFO) so nested runs close
    // in the order pulldown expects.
    if in_code {
        suffix.push('`');
    }
    while let Some((marker, run)) = open.pop() {
        for _ in 0..run {
            suffix.push(marker);
        }
    }
    suffix
}

/// Toggle a paired-delimiter run (`~~`) on the open stack: pop if it matches the
/// top (a close), else push (an open). Strikethrough has no flanking subtlety in
/// GFM, so simple LIFO pairing matches pulldown.
fn toggle_paired_run(open: &mut Vec<(char, usize)>, marker: char, run: usize) {
    if open.last().is_some_and(|(c, _)| *c == marker) {
        open.pop();
    } else {
        open.push((marker, run));
    }
}

/// Length of the run of identical `marker` chars starting at `start`.
fn run_length(chars: &[char], start: usize, marker: char) -> usize {
    let mut n = 0;
    while chars.get(start + n) == Some(&marker) {
        n += 1;
    }
    n
}

/// Fold one `*`/`_` delimiter run into the open-emphasis stack using CommonMark
/// flanking: a run that can close and matches the top opener pops it; otherwise
/// a run that can open is pushed. `_` additionally cannot open/close intraword.
fn apply_emphasis_run(
    open: &mut Vec<(char, usize)>,
    marker: char,
    run: usize,
    before: Option<char>,
    after: Option<char>,
) {
    let left = is_left_flanking(before, after);
    let right = is_right_flanking(before, after);
    // Underscores never open/close inside a word (`foo_bar_baz`).
    let intraword = before.is_some_and(char_is_word) && after.is_some_and(char_is_word);
    let can_open = left && !(marker == '_' && intraword);
    let can_close = right && !(marker == '_' && intraword);
    if can_close && open.last().is_some_and(|(c, _)| *c == marker) {
        open.pop();
    } else if can_open {
        open.push((marker, run));
    }
}

/// CommonMark left-flanking: the run is immediately followed by non-whitespace,
/// and either not followed by punctuation or preceded by whitespace/punctuation.
fn is_left_flanking(before: Option<char>, after: Option<char>) -> bool {
    let Some(after) = after else { return false };
    if after.is_whitespace() {
        return false;
    }
    !is_punct(after) || before.is_none_or(|b| b.is_whitespace() || is_punct(b))
}

/// CommonMark right-flanking: the run is immediately preceded by non-whitespace,
/// and either not preceded by punctuation or followed by whitespace/punctuation.
fn is_right_flanking(before: Option<char>, after: Option<char>) -> bool {
    let Some(before) = before else { return false };
    if before.is_whitespace() {
        return false;
    }
    !is_punct(before) || after.is_none_or(|a| a.is_whitespace() || is_punct(a))
}

fn char_is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
}

/// 줄 시작 ```/~~~ 펜스 토글이 홀수면 마지막 줄이 열린 코드펜스 안이다.
fn has_open_code_fence(text: &str) -> bool {
    let mut open = false;
    for line in text.lines() {
        let mut t = line.trim_start();
        if t.starts_with('>') {
            t = t[1..].trim_start();
        }
        if super::leading_fence(t).is_some() {
            open = !open;
        }
    }
    open
}

/// v3 §5 구조물 라인 게이트 — 열린 꼬리의 **표시용** 서브슬라이스.
///
/// 산문은 char 단위 type-in 을 유지하되(무클립), **비원자적 구조물** 안에서는
/// 완성된 줄만 내보낸다(codex 의 newline-gated 커밋 동형): 미완성 코드 줄과
/// 미완성 표 행은 다음 문자가 도착할 때마다 재렌더/재정렬되므로, 줄이 완성될
/// 때까지 숨겨 지터를 없애고 프레임당 꼬리 재렌더도 줄인다. 데이터(누적
/// 텍스트)는 건드리지 않는다 — `done` 의 권위 렌더가 전량을 그린다.
///
/// 클립 조건:
/// * `in_large_fence` — 경계가 대형 펜스 내부로 전진한 경우(호출자가
///   `fence_at_boundary` 로 판정): 꼬리 전체가 코드 줄이다.
/// * 꼬리에 열린 펜스가 있음 — 소형 펜스는 통째로 꼬리에 살므로 여기서 잡는다.
///   부수효과로 부분 opener(<code>```ru</code>)도 완성 전 노출되지 않는다.
/// * **확인된** GFM 표 후보 — 헤더행 + 구분자행이 모두 완성된 뒤에만. 확인
///   전에는 클립하지 않는다(오탐의 최악이 "지터 잔존"이지 텍스트 실종이
///   아니게 — 보수 전략).
#[must_use]
pub fn clip_tail_for_display(tail: &str, in_large_fence: bool) -> &str {
    if !(in_large_fence || tail_holds_open_structure(tail)) {
        return tail;
    }
    match tail.rfind('\n') {
        Some(i) => &tail[..=i],
        None => "",
    }
}

/// 꼬리가 줄-게이트가 필요한 열린 구조물을 담고 있는가 —
/// [`clip_tail_for_display`] 전용 판정.
fn tail_holds_open_structure(tail: &str) -> bool {
    has_open_code_fence(tail) || confirmed_table_candidate(tail)
}

/// 꼬리가 "확인된" GFM 표 후보로 시작하는가: 완성된 헤더행 + 완성된
/// 구분자행. 술어는 done 패스와 같은 SSOT(`looks_like_table_*`)를 쓴다.
fn confirmed_table_candidate(tail: &str) -> bool {
    let mut lines = tail.lines();
    let (Some(first), Some(second)) = (lines.next(), lines.next()) else {
        return false;
    };
    // `lines()` 는 마지막 미완성 줄도 내주므로, 구분자행이 실제로 개행으로
    // 닫혔는지 확인한다 — `|---|` 까지만 온 프레임에 표로 확정하지 않는다.
    let second_complete = tail
        .splitn(3, '\n')
        .nth(2)
        .is_some();
    second_complete
        && super::looks_like_table_row(first)
        && super::looks_like_table_separator(second)
}

/// Streaming-only bounded renderer for an open tail.
///
/// Renders the open tail through the SAME pulldown path as the settle pass
/// ([`rendered_tail_for_width`], no syntect), so the streaming approximation and
/// the authoritative `done` render agree — there is no third markdown parser to
/// drift. Measured per-frame cost is ≤0.5ms even on a 16KB table (D-2'), well
/// inside frame budget; the naive line-parser this replaced was in fact *slower*
/// on the pathological blank-line-free paragraph it was built for.
///
/// Returns `None` only past [`TAIL_PLAIN_FALLBACK_LIMIT`] — a degenerate
/// blank-line-free block large enough that one pulldown pass per frame would grow
/// unbounded — so the caller drops to its cheapest plain-line path (the
/// pre-existing no-signal fallthrough shape).
#[must_use]
pub fn rendered_bounded_streaming_tail_for_width(
    text: &str,
    theme: &Theme,
    width: u16,
) -> Option<Vec<Line<'static>>> {
    (text.len() <= TAIL_PLAIN_FALLBACK_LIMIT).then(|| rendered_tail_for_width(text, theme, width))
}

/// Above this an open tail stops rendering through pulldown every frame and the
/// caller drops to its plain-line path. 64KB bounds the per-frame linear cost of
/// a degenerate blank-line-free block while sitting far above any realistic open
/// tail (D-2': a 16KB table pulldown is ≤0.5ms, a 16KB paragraph ~72µs). Below it
/// the tail always renders through the same pulldown path as the settle pass, so
/// streaming and `done` agree.
const TAIL_PLAIN_FALLBACK_LIMIT: usize = 64 * 1024;

/// Render an open code-fence interior (text that begins *inside* a fence opened
/// by `init_fence`) as plain styled code lines. Used by the streaming-incremental
/// renderer to promote completed lines of a huge open fence into the stable
/// region — and to draw the tail — without re-parsing the whole fence each frame.
/// Plain (no syntect), matching the large-fence streaming policy; the `done` pass
/// re-highlights authoritatively.
#[must_use]
/// 거대 코드펜스의 인테리어를 done 과 **동일한 카드**(상단 보더 + `  │ ` 레일 +
/// 코드 배경)로 그린다(roadmap ⑨) — 단, 펜스가 아직 열려 있어 하단 보더는 생략.
/// 레일이 양쪽 동일하므로 위젯 래핑이 같아 settle 시 reflow/스크롤 점프가 없다.
/// done 의 `plain_code_rows` 와 같은 `LinesWithEndings` 토큰화로 줄 수도 일치시킨다.
/// `lang` 이 diff/patch 면 done 은 줄번호 거터를 붙여 폭이 달라지므로 카드를 입히지
/// 않고 레거시 경량 인테리어로 폴백한다(거터 패리티는 후속 과제).
///
/// - `emit_top_border`: 펜스가 *처음 열리는* 승격 fragment 에서만 true — 상단 보더
///   를 딱 한 번 그린다. 이어지는 인테리어 fragment·열린 꼬리는 false.
/// - `skip_opener`: fragment 가 ```lang 여는 줄로 시작하면 true — 그 줄을 코드 행
///   으로 렌더하지 않고 버린다(상단 보더가 대신하고, 여는 줄을 닫는 줄로 오인하지
///   않도록).
pub fn rendered_fence_interior_lines(
    text: &str,
    theme: &Theme,
    init_fence: (u8, usize),
    width: u16,
    lang: Option<&str>,
    emit_top_border: bool,
    skip_opener: bool,
) -> Vec<Line<'static>> {
    // Strip the opening ```lang line when this fragment carries it, so it is not
    // rendered as a code row (the top border replaces it) and neither the card
    // loop nor the diff fallback mistakes the opener for a closer.
    let body = if skip_opener {
        text.find('\n').map_or("", |nl| &text[nl + 1..])
    } else {
        text
    };
    let is_diff =
        lang.is_some_and(|l| l.eq_ignore_ascii_case("diff") || l.eq_ignore_ascii_case("patch"));
    if is_diff {
        // A ```diff fence's done render prepends a line-number gutter the light
        // streaming path can't match — framing it would shift every row and
        // reflow at settle. Keep the legacy gutterless light interior for now
        // (opener already stripped, so init_fence renders body as code interior).
        return diff_fence_interior_lines(body, theme, init_fence);
    }
    // Tokenize EXACTLY like the done path (`plain_code_rows`) so a fragment's row
    // count matches done line-for-line — `LinesWithEndings` and `str::lines`
    // disagree on a trailing newline, and that delta would be a reflow.
    let rows: Vec<Vec<Span<'static>>> = LinesWithEndings::from(body)
        .map(|line| {
            let mut t = line.trim_end_matches('\n').to_string();
            if t.contains('\t') {
                t = t.replace('\t', "    ");
            }
            vec![Span::styled(t, Style::new().fg(theme.palette.cyan))]
        })
        .collect();
    code_card_frame_lines(rows, lang, false, width, theme, emit_top_border, false)
        .into_iter()
        .map(Line::from)
        .collect()
}

/// Render the interior of an open `diff`/`patch` code fence as the legacy
/// gutterless light interior: each line as cyan code text on the code background
/// (tabs → 4 spaces), fence markers hidden. This is the exact output the deleted
/// `large_streaming_markdown_lines` produced for its diff branch — a diff fence's
/// `done` render prepends a line-number gutter this streaming path can't match,
/// so framing it as a card would reflow every row at settle (gutter parity is a
/// follow-up). Only ever fed a still-OPEN interior (a closed fence promotes
/// through the full `rendered_lines_for_width`, never here), so `init_fence`
/// stays open and every non-marker line is a code row.
fn diff_fence_interior_lines(
    text: &str,
    theme: &Theme,
    init_fence: (u8, usize),
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut fence: Option<(u8, usize)> = Some(init_fence);
    for raw in text.lines() {
        if let Some((ch, len)) = super::leading_fence(raw.trim_start()) {
            match fence {
                None => fence = Some((ch, len)),
                Some((open_ch, open_len)) if ch == open_ch && len >= open_len => fence = None,
                Some(_) => {}
            }
            // Hide raw ```/~~~ markers; the done pass draws the authoritative card.
            continue;
        }
        if fence.is_some() {
            lines.push(Line::from(vec![Span::styled(
                raw.replace('\t', "    "),
                Style::new().fg(theme.palette.cyan).bg(theme.code_surface()),
            )]));
        }
    }
    if lines.is_empty() {
        lines.push(Line::raw(String::new()));
    }
    lines
}

#[cfg(test)]
mod streaming_repair_tests {
    use super::*;

    /// A trailing single `*`/`_` italic opener (the common `*기울임` case) must be
    /// temporarily closed so it streams styled, not as a raw marker that flashes.
    #[test]
    fn repair_closes_single_italic_markers() {
        assert_eq!(&*repair_unclosed_inline_markers("값은 *기울임"), "값은 *기울임*");
        assert_eq!(&*repair_unclosed_inline_markers("값은 _기울임"), "값은 _기울임_");
    }

    /// `__bold` underscore-bold opener gets `__` appended.
    #[test]
    fn repair_closes_underscore_bold() {
        assert_eq!(&*repair_unclosed_inline_markers("값은 __강조"), "값은 __강조__");
    }

    /// Flanking rules: an accidental `*` between spaces (`2 * 3`) is not emphasis,
    /// and an intraword `_` (`snake_case`) is not emphasis — neither is repaired.
    #[test]
    fn repair_respects_flanking_rules() {
        assert!(matches!(
            repair_unclosed_inline_markers("결과는 2 * 3"),
            std::borrow::Cow::Borrowed(_)
        ));
        assert!(matches!(
            repair_unclosed_inline_markers("함수 my_long_name 호출"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// A balanced single-italic span is borrowed unchanged (zero-copy).
    #[test]
    fn repair_leaves_balanced_single_italic_untouched() {
        assert!(matches!(
            repair_unclosed_inline_markers("이건 *기울임* 끝"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// Nested unclosed strike inside bold must close inner-first (`~~` then `**`)
    /// so pulldown pairs the delimiters correctly.
    #[test]
    fn repair_closes_nested_markers_inner_first() {
        assert_eq!(
            &*repair_unclosed_inline_markers("**굵게 ~~취소"),
            "**굵게 ~~취소~~**"
        );
    }

    /// Code is the innermost still-open span: an open backtick after an open bold
    /// closes the backtick first, then the bold.
    #[test]
    fn repair_closes_code_before_outer_emphasis() {
        assert_eq!(
            &*repair_unclosed_inline_markers("**굵게 `코드"),
            "**굵게 `코드`**"
        );
    }
}
