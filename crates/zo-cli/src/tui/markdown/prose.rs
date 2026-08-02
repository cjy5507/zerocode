//! 밀집 산문(dense prose) 표시 전용 정리 — 모델이 빈 줄 없이 쏟아낸 섹션형
//! 답변을 CommonMark 가 한 문단으로 뭉개지 않도록, 최상위 번호 행을 소제목으로
//! 승격하고 근거 라벨을 독립 문단으로 떼어낸다. 표시 직전의 휴리스틱일 뿐
//! 전체 렌더(`Renderer`)와는 별개 책임이라 분리한다.

use std::borrow::Cow;

use super::{cell_display_width, leading_fence};

/// Light display-only cleanup for answer blocks that arrive as dense prose.
///
/// Models often emit section-ish Korean answers as:
///
/// ```text
/// 1. 핵심 제목
/// 본문...
/// 근거: file.rs:10
/// 2. 다음 제목
/// ```
///
/// CommonMark sees the unblanked continuation lines as one paragraph/list item,
/// so the TUI collapses the whole answer into a wall of wrapped text. Promote
/// only top-level numbered rows that clearly have a following prose body into
/// small section headings, and give evidence labels their own paragraph. Plain
/// compact ordered lists (`1. A\n2. B`) and fenced code stay byte-for-byte.
pub(super) fn polish_dense_prose_for_display(text: &str) -> Cow<'_, str> {
    if !text.contains('\n') {
        return Cow::Borrowed(text);
    }

    let chunks = text.split_inclusive('\n').collect::<Vec<_>>();
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    let mut previous_blank = true;
    let mut colon_child_list_context = false;
    let mut fence: Option<(u8, usize)> = None;

    for (idx, chunk) in chunks.iter().enumerate() {
        let chunk = *chunk;
        let (line, ending) = split_line_ending(chunk);
        let trimmed_start = line.trim_start();

        if fence.is_none() {
            if let Some(marker) = leading_fence(trimmed_start) {
                out.push_str(chunk);
                fence = Some(marker);
                previous_blank = line.trim().is_empty();
                colon_child_list_context = false;
                continue;
            }
        } else {
            out.push_str(chunk);
            if fence_closes(fence, trimmed_start) {
                fence = None;
            }
            previous_blank = line.trim().is_empty();
            continue;
        }

        let list_like_child = colon_child_list_context
            .then(|| normalize_colon_child_list_line(line))
            .flatten();
        if let Some(child) = list_like_child {
            out.push_str("  ");
            out.push_str(child);
            out.push_str(ending);
            changed = true;
            previous_blank = false;
            colon_child_list_context = true;
            continue;
        }

        let section_label = dense_numbered_section_label(line)
            .filter(|_| next_nonblank_line(&chunks, idx).is_some_and(is_prose_continuation_line));
        let evidence_label = is_dense_evidence_label(line);

        if (section_label.is_some() || evidence_label) && !previous_blank {
            out.push('\n');
            changed = true;
        }

        if let Some(label) = section_label {
            out.push_str("### ");
            out.push_str(label);
            out.push_str(ending);
            changed = true;
        } else {
            out.push_str(chunk);
        }
        let is_blank = line.trim().is_empty();
        previous_blank = is_blank;
        if !is_blank {
            colon_child_list_context = opens_colon_child_list_context(line);
        }
    }

    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(text)
    }
}

fn normalize_colon_child_list_line(line: &str) -> Option<&str> {
    let leading = line.len().saturating_sub(line.trim_start().len());
    if leading < 4 {
        return None;
    }
    let trimmed = line.trim_start();
    is_markdown_list_marker(trimmed).then_some(trimmed)
}

fn opens_colon_child_list_context(line: &str) -> bool {
    let trimmed = line.trim_end();
    !trimmed.is_empty() && trimmed.ends_with(':')
}

fn is_markdown_list_marker(trimmed: &str) -> bool {
    trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || ordered_list_marker_len(trimmed).is_some()
}

fn ordered_list_marker_len(trimmed: &str) -> Option<usize> {
    let marker_digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if !(1..=3).contains(&marker_digits) {
        return None;
    }
    (trimmed.as_bytes().get(marker_digits) == Some(&b'.')
        && trimmed.as_bytes().get(marker_digits + 1) == Some(&b' '))
    .then_some(marker_digits + 2)
}

fn split_line_ending(chunk: &str) -> (&str, &str) {
    chunk
        .strip_suffix('\n')
        .map_or((chunk, ""), |line| (line, "\n"))
}

fn fence_closes(fence: Option<(u8, usize)>, trimmed_start: &str) -> bool {
    let Some((fc, fn_len)) = fence else {
        return false;
    };
    leading_fence(trimmed_start)
        .is_some_and(|(c, n)| c == fc && n >= fn_len && trimmed_start[n..].trim().is_empty())
}

fn next_nonblank_line<'a>(chunks: &'a [&str], idx: usize) -> Option<&'a str> {
    chunks[idx + 1..].iter().find_map(|chunk| {
        let (line, _) = split_line_ending(chunk);
        (!line.trim().is_empty()).then_some(line)
    })
}

fn dense_numbered_section_label(line: &str) -> Option<&str> {
    if line.len().saturating_sub(line.trim_start().len()) > 2 {
        return None;
    }
    let trimmed = line.trim_start();
    let marker_digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if !(1..=2).contains(&marker_digits) {
        return None;
    }
    if trimmed.as_bytes().get(marker_digits) != Some(&b'.')
        || trimmed.as_bytes().get(marker_digits + 1) != Some(&b' ')
    {
        return None;
    }
    let body = trimmed[marker_digits + 2..].trim();
    if body.is_empty() || cell_display_width(body) > 110 {
        return None;
    }
    Some(trimmed)
}

fn is_prose_continuation_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    !trimmed.starts_with('#')
        && !trimmed.starts_with("- ")
        && !trimmed.starts_with("* ")
        && !trimmed.starts_with("+ ")
        && dense_numbered_section_label(line).is_none()
}

fn is_dense_evidence_label(line: &str) -> bool {
    if line.len().saturating_sub(line.trim_start().len()) > 2 {
        return false;
    }
    let trimmed = line.trim_start();
    ["근거:", "테스트:", "검증:", "결론:"]
        .iter()
        .any(|label| trimmed.starts_with(label))
}
