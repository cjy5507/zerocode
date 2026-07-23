use super::{
    cell_display_width, char_display_width, clip_tail_for_display, has_markdown_table,
    preserves_layout, render_table_markdown_for_width,
    rendered_bounded_streaming_tail_for_width, rendered_lines_for_width, rendered_tail_for_width,
    stable_prefix_len, streaming_stable_prefix, streaming_stable_prefix_resumed,
};
use crate::tui::blocks::wrapped_rows;
use crate::tui::theme::Theme;
use ratatui::style::{Color, Modifier};
use ratatui::text::Line;

fn dark() -> Theme {
    Theme::default_dark()
}

fn flatten_lines(lines: &[Line<'static>]) -> String {
    lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The bounded streaming tail renders through pulldown up to
/// `TAIL_PLAIN_FALLBACK_LIMIT`; past it the caller drops to its plain path
/// (`None`). Guards the 64KB bound that replaced the deleted third parser.
#[test]
fn bounded_streaming_tail_renders_below_limit_and_falls_back_above() {
    let theme = dark();
    // A markdown-looking tail within the bound renders (Some).
    let within = "# heading\n\nsome **bold** prose\n";
    assert!(rendered_bounded_streaming_tail_for_width(within, &theme, 80).is_some());
    // A pathologically large blank-line-free block exceeds the bound → None, so
    // the caller uses its cheapest plain-line path.
    let huge = "x".repeat(64 * 1024 + 1);
    assert!(rendered_bounded_streaming_tail_for_width(&huge, &theme, 80).is_none());
}

// ---- 기존 테이블 테스트 (text.rs 에서 이전) -----------------------------

#[test]
fn detects_markdown_table_without_collecting_input_lines() {
    assert!(has_markdown_table("| a | b |\n| - | - |\n| 1 | 2 |"));
    assert!(!has_markdown_table("plain text\nwithout a table"));
}

#[test]
fn renders_markdown_table_and_preserves_surrounding_text() {
    let rendered =
        render_table_markdown_for_width("before\n| a | b |\n| - | - |\n| 1 | 2 |\nafter", 0);
    // Closed rounded box: ╭┬╮ top, ├┼┤ header separator, ╰┴╯ bottom.
    let expected = "before\n\
                        ╭───┬───╮\n\
                        │ a │ b │\n\
                        ├───┼───┤\n\
                        │ 1 │ 2 │\n\
                        ╰───┴───╯\n\
                        after";
    assert_eq!(rendered, expected);
}

#[test]
fn table_renders_as_closed_rounded_box() {
    let rendered = render_table_markdown_for_width("| a | b |\n| - | - |\n| 1 | 2 |", 0);
    assert!(
        rendered.contains("╭───┬───╮"),
        "rounded top border: {rendered}"
    );
    assert!(
        rendered.contains("├───┼───┤"),
        "header separator: {rendered}"
    );
    assert!(
        rendered.contains("╰───┴───╯"),
        "rounded bottom border: {rendered}"
    );
    // Old open-box separator (│ ends) must be gone.
    assert!(
        !rendered.contains("│───┼───│"),
        "stale open separator: {rendered}"
    );
}

#[test]
fn mixed_table_separated_from_following_heading_by_blank() {
    // A table directly followed by a heading must not collide: the box is
    // closed and one blank line of air sits between the two.
    let theme = dark();
    let lines = rendered_lines_for_width("| a | b |\n| - | - |\n| 1 | 2 |\n## Next", &theme, 40);
    let bottom = lines
        .iter()
        .position(|l| l.spans.iter().any(|s| s.content.contains('╰')))
        .expect("table must have a ╰ bottom border");
    let heading = lines
        .iter()
        .position(|l| l.spans.iter().any(|s| s.content.contains('\u{258C}')))
        .expect("## heading must render its ▌ glyph");
    assert!(
        heading > bottom + 1,
        "blank line must separate table from heading (bottom={bottom}, heading={heading})"
    );
    assert!(
        lines[bottom + 1]
            .spans
            .iter()
            .all(|s| s.content.trim().is_empty()),
        "row directly after the table must be blank: {:?}",
        lines[bottom + 1]
    );
}

#[test]
fn mixed_table_and_markdown_does_not_leak_raw_markers() {
    // 표 + 헤딩 + 굵게가 한 블록에 섞이면, 종전엔 블록 전체가 표-전용
    // 렌더러로 가서 `##`/`**` 가 raw 로 샜다. 세그먼트 렌더 후엔 표는
    // 박스로, 나머지는 정상 markdown 으로 처리돼야 한다.
    let theme = dark();
    let text = "## 크레이트별 역할\n\n\
                    일반 **굵은** 문단.\n\n\
                    | 레이어 | 크레이트 |\n\
                    | --- | --- |\n\
                    | L1 | core-types |\n\n\
                    마무리 문단.";
    let lines = rendered_lines_for_width(text, &theme, 60);
    let flat: String = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!flat.contains("## "), "heading marker leaked: {flat:?}");
    assert!(!flat.contains("**"), "bold marker leaked: {flat:?}");
    assert!(flat.contains('│'), "table not rendered as box: {flat:?}");
    assert!(
        flat.contains("크레이트별 역할") && flat.contains("core-types"),
        "content not preserved: {flat:?}"
    );
}

#[test]
fn mixed_table_with_bold_label_does_not_leak_raw_markers() {
    // A bold-only section label is enough markdown signal to avoid the
    // table-only path leaking raw `**` around a priority plan table.
    let theme = dark();
    let text = "**우선순위 개선안**\n\n\
                | 우선순위 | 개선안 | 검증 |\n\
                | --- | --- | --- |\n\
                | P0 | OpenAI cached_tokens 계측 복구 | cache_read_input_tokens 확인 |";
    let lines = rendered_lines_for_width(text, &theme, 72);
    let flat: String = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!flat.contains("**"), "bold marker leaked: {flat:?}");
    assert!(flat.contains('│'), "table not rendered as box: {flat:?}");
    assert!(
        flat.contains("우선순위 개선안") && flat.contains("cached_tokens"),
        "content not preserved: {flat:?}"
    );
}

#[test]
fn terminal_capture_with_incidental_markdown_stays_preformatted() {
    let theme = dark();
    let text = "└ ▸ Read: /Users/joe/2026/zo/crates/zo-cli/src/main.rs · 80 lines  ok  ░\n\
                └ /Users/joe/2026/zo/crates/zo-cli/src/main.rs  rust      ░\n\
                     1 mod attach;                                             ░\n\
                └ [+78 more lines · Enter to expand]                            ░\n\
                └ ▸ Result: {\"newString\":\"**bold-looking payload**\"} ok        ░";

    assert!(
        preserves_layout(text),
        "terminal captures should bypass document markdown parsing"
    );

    let lines = rendered_lines_for_width(text, &theme, 96);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(flat.contains("└ ▸ Read:"), "tree row disappeared: {flat:?}");
    assert!(
        flat.contains("**bold-looking payload**"),
        "incidental markdown inside a screen dump must stay raw: {flat:?}"
    );
    assert!(
        !flat.contains("╭─"),
        "screen dump must not be reinterpreted as a code block: {flat:?}"
    );
}

#[test]
fn pasted_zo_screen_dump_stays_preformatted() {
    let theme = dark();
    let text = "└ ▸ Read: /Users/joe/2026/zo/workspace/project/crates/zo-cli/src/main.rs · 80 lines  ok  ░  ~/2026/zo/workspace/project\n\
  └ /Users/joe/2026/zo/workspace/project/crates/zo-cli/src/main.rs  rust      ░\n\
       1 mod attach;                                                        ░  · session\n\
설정 파일이 저장되면서 작업트리에 잡히는 문제를 막기 위해, 생성되는 로컬 설정/세션 선호 디렉터리를 .gitignore와 zo init 기본 ignore 목록에     use  ■░░░░░░░░░ 11%\n\
추가하겠습니다.                                                             ░  ctx 16.8k new · 105.0k cached\n\
└ ▸ Result: {\"filePath\":\"/Users/joe/2026/zo/workspace/project/crates/zo-cli/src/init.rs\",\"gitDiff\":null,\"newString\":\"const GITIGNORE_ENTRIES\"}  ok  ░\n\
  └ [+749 more lines · Enter to expand]                                     ░";

    assert!(
        preserves_layout(text),
        "Zo screen dumps should bypass document markdown parsing"
    );

    let lines = rendered_lines_for_width(text, &theme, 120);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(flat.contains("└ ▸ Read:"), "tool row preserved: {flat:?}");
    assert!(
        flat.contains("use  ■░░░░░░░░░ 11%"),
        "sidebar gauge row preserved: {flat:?}"
    );
    assert!(
        flat.contains("\"newString\"") && flat.contains("GITIGNORE_ENTRIES"),
        "JSON-looking tool payload stays raw inside a screen dump: {flat:?}"
    );
    assert!(
        !flat.contains("╭─"),
        "screen dump must not be reinterpreted as a code block: {flat:?}"
    );
}

#[test]
fn pasted_zo_screen_dump_wraps_continuations_under_their_row() {
    let theme = dark();
    let text = "└ ▸ Result: {\"filePath\":\"/Users/joe/2026/zo/crates/zo-cli/src/init.rs\",\"gitDiff\":null,\"newString\":\"const GITIGNORE_ENTRIES: [&str; 4] = [\"}  ok  ░\n\
  └ {\"filePath\":\"/Users/joe/2026/zo/crates/zo-cli/src/init.rs\",\"gitDiff\":null,\"newString\":\"const GITIGNORE_ENTRIES: [&str; 4] = [\"}  ░";

    assert!(
        preserves_layout(text),
        "screen dump fixture should use the preformatted renderer"
    );

    let lines = rendered_lines_for_width(text, &theme, 72);
    let rows = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    let continuation_rows = rows
        .iter()
        .filter(|row| row.contains('\u{21B3}'))
        .collect::<Vec<_>>();
    assert!(
        !continuation_rows.is_empty(),
        "narrow screen dump should exercise wrapping: {rows:?}"
    );
    assert!(
        continuation_rows
            .iter()
            .all(|row| row.starts_with("  \u{21B3}")),
        "wrapped terminal dump continuations should stay nested, not start at column zero: {rows:?}"
    );
}

#[test]
fn assistant_markdown_with_tool_output_examples_does_not_become_terminal_capture() {
    let theme = dark();
    let text = "이번 라운드는 **tool 결과의 \"더보기\" 안내 라인**을 더 가볍게 다듬었습니다.\n\n\
                **Before**\n\
                ```\n\
                └ [+23 more lines · Enter to expand]\n\
                ```\n\
                **After**\n\
                ```\n\
                └ +23 more ↵\n\
                ```\n\
                - 장황한 `lines · Enter to expand` 제거\n\
                - `NO_COLOR` 환경에서는 클릭 표시 대신 `[enter]`로 풀백";

    assert!(
        !preserves_layout(text),
        "assistant markdown with small terminal examples must stay document markdown"
    );

    let lines = rendered_lines_for_width(text, &theme, 96);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!flat.contains("**"), "bold markers leaked: {flat:?}");
    assert!(!flat.contains("```"), "fence markers leaked: {flat:?}");
    assert!(flat.contains("Before"), "before label missing: {flat:?}");
    assert!(flat.contains("After"), "after label missing: {flat:?}");
    assert!(
        flat.contains("+23 more") && flat.contains("NO_COLOR"),
        "example content not preserved: {flat:?}"
    );
}

#[test]
fn long_markdown_answer_with_ui_mockup_glyphs_stays_markdown() {
    // Regression: a long assistant answer that embeds a zo-UI mockup (tree
    // rows `├│└` + block/gauge glyphs `█░`) used to trip
    // `looks_like_dense_terminal_capture` and render the WHOLE answer as raw
    // text — `##` / ``` / `**` markers leaked (the "마지막에서 화면이 깨지네"
    // bug). Authored markdown block structure (headings + fences) must veto that
    // dense-capture override so the answer stays styled document markdown.
    let theme = dark();
    let text = "## 추천 최종 형태\n\n\
                첫 실행 화면은 이런 구조가 좋습니다:\n\n\
                ```text\n\
                █ ZO\n\
                ├ Detected: Rust workspace · branch main\n\
                │ ██████░░░░ 60%\n\
                └ [ Claude ] [ OpenAI ] [ Skip ]\n\
                ```\n\n\
                ## 구현 우선순위\n\n\
                - **P0**: onboarding 카드를 checklist 스타일로\n\
                - **P1**: progress indicator 추가";

    assert!(
        !preserves_layout(text),
        "authored markdown with a UI mockup must stay document markdown, not a raw capture"
    );

    let lines = rendered_lines_for_width(text, &theme, 96);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!flat.contains("## "), "heading markers leaked: {flat:?}");
    assert!(!flat.contains("```"), "fence markers leaked: {flat:?}");
    assert!(!flat.contains("**P0**"), "bold markers leaked: {flat:?}");
    assert!(
        flat.contains("추천 최종 형태") && flat.contains("구현 우선순위"),
        "heading text missing: {flat:?}"
    );
    assert!(
        flat.contains("ZO") && flat.contains("onboarding 카드"),
        "answer content not preserved: {flat:?}"
    );
}

#[test]
fn gemini_indented_sublist_markers_render_as_markdown_not_code_blocks() {
    let theme = dark();
    let text = "* 개선안:\n\n    * **공유 필터 코어 추출**: `tools` 크레이트에 단일 코어 필터 함수를 정의합니다.\n\n* 문제점 (`rate_limit.rs:193-205`):\n\n    * **Provider별 RateGovernor / Cooldown 분리**: Provider별 독립된 Governor와 Cooldown 타이머를 관리합니다.";

    let lines = rendered_lines_for_width(text, &theme, 96);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        flat.contains("공유 필터 코어 추출") && flat.contains("Provider별 RateGovernor"),
        "Gemini-style nested items must remain visible: {flat:?}"
    );
    assert!(
        !flat.contains("╭─"),
        "nested items must not become code cards: {flat:?}"
    );
    assert!(
        !flat.contains("* **"),
        "raw markdown list/bold markers leaked: {flat:?}"
    );
    assert!(
        !flat.contains("`tools`"),
        "raw inline-code marker leaked: {flat:?}"
    );
}

#[test]
fn colon_followed_by_indented_non_list_stays_code_block() {
    let theme = dark();
    let text = "예시:\n\n    let value = compute();";
    let lines = rendered_lines_for_width(text, &theme, 80);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        flat.contains("╭─"),
        "ordinary indented code remains a code card: {flat:?}"
    );
    assert!(
        flat.contains("let value = compute();"),
        "code body preserved: {flat:?}"
    );
}

#[test]
fn paragraphs_keep_visible_air_between_blocks() {
    let theme = dark();
    let lines = rendered_lines_for_width("첫 문단입니다.\n\n둘째 문단입니다.", &theme, 60);
    let first = lines
        .iter()
        .position(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("첫 문단"))
        })
        .expect("first paragraph visible");
    let second = lines
        .iter()
        .position(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("둘째 문단"))
        })
        .expect("second paragraph visible");
    assert!(
        second > first + 1,
        "paragraphs need a visible blank row between them: {lines:?}"
    );
    assert!(
        lines[first + 1]
            .spans
            .iter()
            .all(|span| span.content.trim().is_empty()),
        "row between paragraphs must be blank: {:?}",
        lines[first + 1]
    );
}

#[test]
fn list_blocks_keep_air_before_following_paragraph() {
    let theme = dark();
    let lines = rendered_lines_for_width(
        "- 현재 프로젝트 기준\n- 백그라운드 확인\n\n정리입니다.",
        &theme,
        60,
    );
    let second_item = lines
        .iter()
        .position(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("백그라운드"))
        })
        .expect("second list item visible");
    let summary = lines
        .iter()
        .position(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("정리입니다"))
        })
        .expect("following paragraph visible");
    assert!(
        summary > second_item + 1,
        "list block and following paragraph need visible air: {lines:?}"
    );
}

#[test]
fn dense_numbered_sections_get_visible_air() {
    let theme = dark();
    let text = "이유는 다음과 같습니다.\n\
                1. 작업트리가 이미 큰 폭으로 더럽습니다.\n\
                현재 git diff --stat 기준으로 36개 파일이 걸려 있습니다.\n\
                근거: crates/zo-cli/src/tui/app/mod.rs:1\n\
                2. TUI가 아직 큰 중심 객체에 많은 책임을 들고 있습니다.\n\
                App은 상태 머신, 테마, 렌더 채널을 함께 들고 있습니다.";
    let lines = rendered_lines_for_width(text, &theme, 90);
    let find = |needle: &str| {
        lines
            .iter()
            .position(|line| line.spans.iter().any(|span| span.content.contains(needle)))
            .unwrap_or_else(|| panic!("{needle:?} not visible in {lines:?}"))
    };
    let first = find("1. 작업트리");
    let body = find("현재 git diff");
    let evidence = find("근거:");
    let second = find("2. TUI");

    assert!(
        lines[first]
            .spans
            .iter()
            .any(|span| span.content.contains('\u{258E}')),
        "dense numbered section should render as a small heading: {:?}",
        lines[first]
    );
    assert_eq!(
        body,
        first + 1,
        "body sits directly under the section heading (compact): {lines:?}"
    );
    assert!(
        evidence > body + 1,
        "evidence label needs its own paragraph: {lines:?}"
    );
    assert!(
        second > evidence + 1,
        "next dense section needs visible separation: {lines:?}"
    );
}

#[test]
fn compact_numbered_lists_do_not_promote_to_headings() {
    let theme = dark();
    let lines = rendered_lines_for_width("1. Alpha\n2. Beta\n3. Gamma", &theme, 60);
    let has_h3_glyph = lines.iter().any(|line| {
        line.spans
            .iter()
            .any(|span| span.content.contains('\u{258E}'))
    });
    assert!(
        !has_h3_glyph,
        "plain compact ordered list must stay a list: {lines:?}"
    );
}

#[test]
fn wraps_wide_korean_table_cells_to_fit_width() {
    let rendered = render_table_markdown_for_width(
        "| 우선순위 | 액션 | 이유 |\n\
             | --- | --- | --- |\n\
             | P1 | 신규 TUI 파일을 생성하고 runtime 핫스팟 실측 결과를 sidebar에 정렬 | 정렬 깨짐 방지 |",
        58,
    );
    let table_lines = rendered.lines().collect::<Vec<_>>();
    assert!(
        table_lines
            .iter()
            .all(|line| cell_display_width(line) <= 58),
        "all rendered table rows must fit viewport width:\n{rendered}"
    );
    assert!(
        table_lines
            .iter()
            .filter(|line| line.starts_with('│'))
            .count()
            > 3,
        "long Korean row should wrap into multiple aligned rows:\n{rendered}"
    );
}

#[test]
fn table_right_aligns_numeric_column_from_separator() {
    // `---:` 정렬 힌트가 셀 패딩에 반영돼 숫자가 우측에 붙는다.
    let rendered = render_table_markdown_for_width(
        "| name | qty |\n| :--- | ---: |\n| apple | 5 |\n| fig | 100 |",
        0,
    );
    let lines: Vec<&str> = rendered.lines().collect();
    let apple = lines
        .iter()
        .find(|l| l.contains("apple"))
        .expect("apple row");
    // qty 폭 3, "5" 우측정렬 → "5" 앞에 공백, 뒤에 " │".
    assert!(
        apple.contains("  5 │"),
        "qty must be right-aligned (got {apple:?})"
    );
    // name 은 좌측정렬 유지 — "apple" 바로 앞이 "│ ".
    assert!(
        apple.contains("│ apple "),
        "name stays left-aligned: {apple:?}"
    );
}

// ---- Phase 2.1 — Heading 위계 -------------------------------------------

#[test]
fn heading_levels_use_distinct_glyphs() {
    let theme = dark();
    let h1 = rendered_lines_for_width("# Hello", &theme, 60);
    let h2 = rendered_lines_for_width("## Hello", &theme, 60);
    let h3 = rendered_lines_for_width("### Hello", &theme, 60);
    // H1 → "█ ", H2 → "▌ ", H3 → "▎ "
    assert!(
        h1.iter()
            .any(|l| l.spans.iter().any(|s| s.content.contains('\u{2588}'))),
        "H1 must use █ glyph: {h1:?}"
    );
    assert!(
        h2.iter()
            .any(|l| l.spans.iter().any(|s| s.content.contains('\u{258C}'))),
        "H2 must use ▌ glyph: {h2:?}"
    );
    assert!(
        h3.iter()
            .any(|l| l.spans.iter().any(|s| s.content.contains('\u{258E}'))),
        "H3 must use ▎ glyph: {h3:?}"
    );
}

#[test]
fn char_width_matches_cell_width_for_korean() {
    // per-char 합 == str-level(ratatui Line::width). 어긋나면 표/wrap 정렬 깨짐.
    let s = "한국어 표 정렬 abc";
    let by_char: usize = s.chars().map(char_display_width).sum();
    assert_eq!(
        by_char,
        cell_display_width(s),
        "per-char and str-level width must agree (both unicode-width)"
    );
    assert_eq!(char_display_width('한'), 2, "Hangul syllable is 2 cells");
    assert_eq!(char_display_width('a'), 1, "ASCII is 1 cell");
}

#[test]
fn heading_levels_converge_to_brightness_hierarchy() {
    let theme = dark();
    let fg = |md: &str| -> Color {
        rendered_lines_for_width(md, &theme, 60)
            .iter()
            .flat_map(|l| l.spans.clone())
            .find(|s| s.content.contains("Hello"))
            .and_then(|s| s.style.fg)
            .expect("heading text span carries a fg color")
    };
    let (h1, h2, h3) = (fg("# Hello"), fg("## Hello"), fg("### Hello"));

    // v2 절제: 헤딩은 hue 가 아니라 명도(bright)+글리프 두께로 위계를 만든다.
    // 브랜드 앰버는 유저레일·포커스·라이브 순간 전용 — 헤딩에 쓰면 본문이
    // 색 이벤트가 된다 (CC "orange mess" 소음의 진원).
    assert_eq!(h1, theme.palette.bright, "H1 == bright");
    assert_eq!(h2, theme.palette.bright, "H2 == bright (converged)");
    assert_eq!(h3, theme.palette.bright, "H3 == bright (converged)");

    // 색 경쟁 회귀 가드 — 어떤 헤딩도 브랜드/세컨더리 hue 금지.
    for (name, c) in [("H1", h1), ("H2", h2), ("H3", h3)] {
        assert_ne!(c, theme.palette.accent, "{name} must not use the brand accent");
        assert_ne!(c, theme.palette.cyan, "{name} must not use cyan");
        assert_ne!(c, theme.palette.violet, "{name} must not use violet");
    }

    // 동색 H2·H3 의 위계는 글리프 두께(▌ vs ▎)가 마저 표현한다.
    let has_glyph = |md: &str, g: char| -> bool {
        rendered_lines_for_width(md, &theme, 60)
            .iter()
            .flat_map(|l| l.spans.clone())
            .any(|s| s.content.contains(g))
    };
    assert!(has_glyph("## Hello", '\u{258C}'), "H2 uses ▌ glyph");
    assert!(has_glyph("### Hello", '\u{258E}'), "H3 uses ▎ glyph");
}

#[test]
fn standalone_bold_label_promotes_to_section_heading() {
    let theme = dark();
    let lines = rendered_lines_for_width("**답변**\n먼저 확인한 내용입니다.", &theme, 60);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!flat.contains("**"), "raw bold marker leaked: {flat:?}");

    let heading = lines
        .iter()
        .position(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains('\u{258E}'))
        })
        .expect("promoted label must render as an H3 heading");
    let body = lines
        .iter()
        .position(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("먼저 확인"))
        })
        .expect("body paragraph must remain visible");
    assert_eq!(
        body,
        heading + 1,
        "body sits directly under the promoted label (compact): {flat:?}"
    );
}

// ---- Phase 2.2 — code block top + bottom border --------------------------

#[test]
fn code_block_emits_top_and_bottom_borders() {
    let theme = dark();
    let lines = rendered_lines_for_width("```rust\nfn x(){}\n```", &theme, 60);
    let has_top = lines
        .iter()
        .any(|l| l.spans.iter().any(|s| s.content.contains("╭─")));
    let has_bottom = lines
        .iter()
        .any(|l| l.spans.iter().any(|s| s.content.contains("╰")));
    assert!(has_top, "code block must have ╭─ top border: {lines:?}");
    assert!(
        has_bottom,
        "code block must have ╰ bottom border: {lines:?}"
    );
}

#[test]
fn diff_fence_has_visible_diff_badge() {
    let theme = dark();
    let lines = rendered_lines_for_width("```diff\n- old\n+ new\n```", &theme, 60);
    let flat = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        flat.contains("diff +/-"),
        "diff fence must advertise semantic diff rendering: {flat:?}"
    );
    assert!(flat.contains("- old"), "removal row missing: {flat:?}");
    assert!(flat.contains("+ new"), "addition row missing: {flat:?}");
}

/// A `diff` fence that carries a `@@` hunk header gets a line-number gutter so
/// the reader can see *where* in the file each change lands — the bug where
/// markdown diffs showed changes with no positional context. Old/new counters
/// advance independently: a context line shows both, an addition only the new
/// number, a removal only the old number.
#[test]
fn diff_fence_with_hunk_header_shows_line_number_gutter() {
    let theme = dark();
    let src = "```diff\n\
@@ -10,3 +20,4 @@\n\
 context\n\
-removed\n\
+added one\n\
+added two\n\
```";
    let lines = rendered_lines_for_width(src, &theme, 80);
    let rows = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let flat = rows.join("\n");

    // Context line carries both old (10) and new (20) numbers.
    assert!(
        rows.iter()
            .any(|r| r.contains("10") && r.contains("20") && r.contains("context")),
        "context row must show both old and new line numbers: {flat:?}"
    );
    // Removed line carries the old number (11) but no new number; the next
    // added line takes the new number (20) since context consumed 20.
    assert!(
        rows.iter()
            .any(|r| r.contains("11") && r.contains("removed")),
        "removed row must show the old line number: {flat:?}"
    );
    assert!(
        rows.iter()
            .any(|r| r.contains("21") && r.contains("added one")),
        "first added row must show the new line number 21: {flat:?}"
    );
    assert!(
        rows.iter()
            .any(|r| r.contains("22") && r.contains("added two")),
        "second added row must show the new line number 22: {flat:?}"
    );
}

/// A bare `diff` snippet *without* a `@@` header stays gutter-less: short
/// illustrative diffs in prose must not gain phantom `1`-based line numbers the
/// author never implied. Guards the backward-compatible path.
#[test]
fn diff_fence_without_hunk_header_has_no_line_number_gutter() {
    let theme = dark();
    let lines = rendered_lines_for_width("```diff\n-old\n+new\n```", &theme, 60);
    // The body rows are the framed `│ ` lines; none of them may carry a numeric
    // gutter before the `-old` / `+new` text.
    for line in &lines {
        let flat = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        if flat.contains("old") {
            assert!(
                flat.trim_start_matches(|c: char| !c.is_alphanumeric())
                    .starts_with("old"),
                "removal row must not gain a line-number gutter: {flat:?}"
            );
        }
    }
}

// ---- Streaming tail: syntect off so the live tail never re-highlights -----

/// Distinct foreground colors across every code-body row (the framed `\u{2502} `
/// lines) of a rendered code block.
fn code_body_fg_colors(lines: &[ratatui::text::Line<'static>]) -> Vec<Color> {
    let mut seen = Vec::new();
    for line in lines {
        let is_body = line
            .spans
            .first()
            .is_some_and(|s| s.content.contains('\u{2502}'));
        if !is_body {
            continue;
        }
        for span in &line.spans {
            if span.content.trim().is_empty() || span.content.contains('\u{2502}') {
                continue;
            }
            if let Some(fg) = span.style.fg {
                if !seen.contains(&fg) {
                    seen.push(fg);
                }
            }
        }
    }
    seen
}

/// The streaming-tail renderer must not run syntect on an open code block:
/// syntect is stateful per line, so re-highlighting the growing tail every
/// frame stalls the draw loop on long answers. The stable-prefix pass
/// (`rendered_lines_for_width`) still applies full color once the fence closes,
/// so the only difference is the *open* tail \u{2014} which renders in one uniform
/// code color instead of multi-color syntax highlighting.
#[test]
fn streaming_tail_skips_syntect_on_open_code_block() {
    let theme = dark();
    let code = "```rust\nfn main() { let x = 42; println!(\"{x}\"); }\n```";

    let stable = code_body_fg_colors(&rendered_lines_for_width(code, &theme, 60));
    let tail = code_body_fg_colors(&rendered_tail_for_width(code, &theme, 60));

    assert!(
        stable.len() > 1,
        "stable (closed) render must syntax-highlight code in multiple colors: {stable:?}"
    );
    assert!(
        tail.len() <= 1,
        "streaming tail must render code in one uniform color (no per-frame syntect): {tail:?}"
    );
}

/// For ordinary prose (the dominant streaming case) syntect never runs, so the
/// open-tail render and the stable render are byte-identical \u{2014} turning tail
/// highlighting off changes nothing the user sees while typing.
#[test]
fn streaming_tail_matches_stable_render_for_prose() {
    let theme = dark();
    let prose = "## 개요\n\n- **굵게** 항목\n- 일반 항목\n\n본문 문단입니다.";

    assert_eq!(
        rendered_tail_for_width(prose, &theme, 60),
        rendered_lines_for_width(prose, &theme, 60),
        "prose has no syntect work, so tail and stable renders must be identical"
    );
}

// ---- Phase 2.3 — 인라인 코드 패딩 --------------------------------------

#[test]
fn inline_code_has_chip_padding() {
    let theme = dark();
    // `c` 가 'x' 와 'y' 사이에 있어 양쪽 패딩이 들어가야 한다.
    let lines = rendered_lines_for_width("x`c`y", &theme, 60);
    let first = &lines[0];
    // padding span 이 있는지 — 인접한 raw 공백 span.
    let has_padding = first
        .spans
        .windows(2)
        .any(|w| w[0].content == " " || w[1].content == " ");
    assert!(
        has_padding,
        "inline code must have space padding: {first:?}"
    );
}

// ---- Phase 2.5 — Blockquote 연속 레일 -----------------------------------

#[test]
fn blockquote_rail_on_every_paragraph_line() {
    let theme = dark();
    let lines = rendered_lines_for_width("> first line\n>\n> second line", &theme, 60);
    let rail_count = lines
        .iter()
        .filter(|l| l.spans.iter().any(|s| s.content.contains('\u{258E}')))
        .count();
    assert!(
        rail_count >= 2,
        "blockquote rail must appear on at least 2 paragraph lines, got {rail_count}: {lines:?}"
    );
}

// ---- W6 — GitHub admonition callouts -----------------------------------

#[test]
fn callout_note_emits_colored_rail_and_bold_label() {
    let theme = dark();
    let lines = rendered_lines_for_width("> [!NOTE]\n> Body text here", &theme, 60);
    let label = lines
        .iter()
        .flat_map(|l| &l.spans)
        .find(|s| s.content == "Note")
        .expect("callout must emit a 'Note' label");
    assert_eq!(
        label.style.fg,
        Some(theme.palette.info),
        "label uses Note color"
    );
    assert!(
        label.style.add_modifier.contains(Modifier::BOLD),
        "callout label must be bold"
    );
    // The marker text itself must not leak into the output.
    let joined = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
        .collect::<String>();
    assert!(
        !joined.contains("[!NOTE]"),
        "marker must be stripped: {joined}"
    );
    // Rails are the heavier ▌ glyph tinted with the callout color.
    let rail = lines
        .iter()
        .flat_map(|l| &l.spans)
        .find(|s| s.content.contains('\u{258C}'))
        .expect("callout rail glyph ▌");
    assert_eq!(rail.style.fg, Some(theme.palette.info));
}

/// Regression: a callout must not leave dangling rail-only rows below its last
/// line of text. The blockquote-close spacing previously emitted `▌`-prefixed
/// blank rows (the `ensure_visible_blank_line` filler counted a lone rail as
/// blank), so a NOTE rendered two colored rail stubs under its body. The final
/// rows must be genuinely empty (no rail glyph) or carry real text.
#[test]
fn callout_does_not_leave_trailing_rail_only_rows() {
    let theme = dark();
    let lines = rendered_lines_for_width("> [!NOTE]\n> Body text here\n", &theme, 60);

    // Find the last row that carries real (non-rail, non-blank) text.
    let last_text_idx = lines
        .iter()
        .rposition(|l| {
            l.spans
                .iter()
                .any(|s| !s.content.trim().is_empty() && !s.content.contains('\u{258C}'))
        })
        .expect("callout has body text");

    // Every row *after* the body must be free of rail glyphs — no dangling stubs.
    for line in &lines[last_text_idx + 1..] {
        let has_rail = line.spans.iter().any(|s| s.content.contains('\u{258C}'));
        let flat = line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>();
        assert!(
            !has_rail,
            "callout must not trail rail-only rows after its body: {flat:?}"
        );
    }
}

#[test]
fn callout_warning_renders_inline_body() {
    let theme = dark();
    let lines = rendered_lines_for_width("> [!WARNING] Be careful here", &theme, 60);
    let label = lines
        .iter()
        .flat_map(|l| &l.spans)
        .find(|s| s.content == "Warning")
        .expect("Warning label");
    assert_eq!(label.style.fg, Some(theme.palette.warn));
    let has_body = lines.iter().any(|l| {
        l.spans
            .iter()
            .any(|s| s.content.contains("Be careful here"))
    });
    assert!(has_body, "inline body after marker must render: {lines:?}");
}

#[test]
fn plain_blockquote_keeps_thin_neutral_rail() {
    // Regression: a non-admonition quote must keep the ▎ accent_dim
    // rail and gain no callout label.
    let theme = dark();
    let lines = rendered_lines_for_width("> just a quote", &theme, 60);
    let rail = lines
        .iter()
        .flat_map(|l| &l.spans)
        .find(|s| s.content.contains('\u{258E}'))
        .expect("plain quote keeps ▎ rail");
    assert_eq!(rail.style.fg, Some(theme.palette.accent_dim));
    let has_callout_rail = lines
        .iter()
        .any(|l| l.spans.iter().any(|s| s.content.contains('\u{258C}')));
    assert!(
        !has_callout_rail,
        "plain quote must not use the callout ▌ rail"
    );
}

// ---- Compact heading rhythm ---------------------------------------------

#[test]
fn h1_renders_without_section_rule() {
    // 컴팩트 리듬: H1 위 전폭 가로줄은 노이즈 — 헤딩 글리프/색이 이미
    // 섹션을 구분하므로 rule 줄이 없어야 한다.
    let theme = dark();
    let lines = rendered_lines_for_width("# Title", &theme, 60);
    let rule_idx = lines.iter().position(|l| {
        !l.spans.is_empty()
            && l.spans
                .iter()
                .all(|s| !s.content.is_empty() && s.content.chars().all(|c| c == '─'))
    });
    assert!(rule_idx.is_none(), "no rule line above H1: {lines:?}");
    lines
        .iter()
        .position(|l| l.spans.iter().any(|s| s.content.contains('\u{2588}')))
        .expect("H1 █ glyph line");
}

#[test]
fn h2_body_separated_by_one_visible_blank_row() {
    // 문서 섹션 헤딩(H1/H2) 아래엔 가시 빈 줄 1행 — 본문이 바로 붙으면
    // 갑갑하게 읽힌다 (CC 패리티).
    let theme = dark();
    let lines = rendered_lines_for_width("## 결과\n본문이 바로 이어짐", &theme, 60);
    let find = |needle: &str| {
        lines
            .iter()
            .position(|line| line.spans.iter().any(|span| span.content.contains(needle)))
            .unwrap_or_else(|| panic!("{needle:?} not visible in {lines:?}"))
    };
    let heading = find("결과");
    let body = find("본문이 바로");
    assert_eq!(
        body,
        heading + 2,
        "one visible blank row must separate H2 from its body: {lines:?}"
    );
}

#[test]
fn h3_body_stays_on_next_row_compact() {
    // 소형 섹션 헤딩(H3, 승격 라벨 포함)은 컴팩트 유지 — dense 섹션이
    // 부풀지 않도록 본문이 바로 다음 행에 시작한다.
    let theme = dark();
    let lines = rendered_lines_for_width("### 근거\n본문이 바로 이어짐", &theme, 60);
    let find = |needle: &str| {
        lines
            .iter()
            .position(|line| line.spans.iter().any(|span| span.content.contains(needle)))
            .unwrap_or_else(|| panic!("{needle:?} not visible in {lines:?}"))
    };
    let heading = find("근거");
    let body = find("본문이 바로");
    assert_eq!(
        body,
        heading + 1,
        "H3 body must sit directly under the heading: {lines:?}"
    );
}

// ---- Phase 2.6 — HR width-aware ----------------------------------------

#[test]
fn hr_is_width_aware() {
    let theme = dark();
    let wide = rendered_lines_for_width("---", &theme, 80);
    let narrow = rendered_lines_for_width("---", &theme, 30);
    let wide_len = wide
        .iter()
        .flat_map(|l| l.spans.iter())
        .map(|s| s.content.chars().count())
        .max()
        .unwrap_or(0);
    let narrow_len = narrow
        .iter()
        .flat_map(|l| l.spans.iter())
        .map(|s| s.content.chars().count())
        .max()
        .unwrap_or(0);
    assert!(wide_len >= 70, "wide HR must be ≥70 chars, got {wide_len}");
    assert!(
        narrow_len >= 20,
        "narrow HR must be ≥20 chars, got {narrow_len}"
    );
    assert!(wide_len > narrow_len, "HR must grow with width");
}

// ---- Phase 2.7 — link arrow + dedupe -----------------------------------

#[test]
fn link_emits_arrow_glyph() {
    let theme = dark();
    let lines = rendered_lines_for_width("[docs](https://x.com)", &theme, 60);
    let has_arrow = lines
        .iter()
        .any(|l| l.spans.iter().any(|s| s.content.contains('\u{2197}')));
    assert!(has_arrow, "link must end with ↗ glyph: {lines:?}");
}

#[test]
fn link_dedupes_url_when_text_equals_dest() {
    let theme = dark();
    let lines = rendered_lines_for_width("[https://x.com](https://x.com)", &theme, 60);
    let has_paren_url = lines.iter().any(|l| {
        l.spans
            .iter()
            .any(|s| s.content.contains("(https://x.com)"))
    });
    assert!(
        !has_paren_url,
        "link text == dest URL must not duplicate ( url ): {lines:?}"
    );
}

#[test]
fn link_text_uses_cyan_with_underline() {
    let theme = dark();
    let lines = rendered_lines_for_width("[docs](https://x.com)", &theme, 60);
    let link = lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .find(|span| span.content == "docs")
        .expect("link text span must be rendered");

    assert_eq!(link.style.fg, Some(theme.palette.cyan));
    assert!(
        link.style.add_modifier.contains(Modifier::UNDERLINED),
        "links should be cyan and underlined so the affordance is unmistakable"
    );
}

// ---- Phase 2.8 — Strikethrough -----------------------------------------

#[test]
fn strikethrough_emits_crossed_out_modifier() {
    let theme = dark();
    let lines = rendered_lines_for_width("~~no~~", &theme, 60);
    let has_strike = lines.iter().any(|l| {
        l.spans
            .iter()
            .any(|s| s.style.add_modifier.contains(Modifier::CROSSED_OUT))
    });
    assert!(has_strike, "~~no~~ must emit CROSSED_OUT: {lines:?}");
}

#[test]
fn inline_math_renders_verbatim_with_operators_intact() {
    let theme = dark();
    // Pre-ENABLE_MATH, the `*` inside `$…$` toggled emphasis: the operator
    // vanished and rendered as "a  b". GPT emits math notation freely.
    let lines = rendered_lines_for_width("area $a * b$ done", &theme, 60);
    let text = flatten_lines(&lines);
    assert!(
        text.contains("$a * b$"),
        "math must render verbatim, delimiters and operators intact: {text:?}"
    );

    // Display math keeps its $$ fences too.
    let lines = rendered_lines_for_width("$$x_1 + x_2$$", &theme, 60);
    let text = flatten_lines(&lines);
    assert!(
        text.contains("$$x_1 + x_2$$"),
        "display math must render verbatim: {text:?}"
    );
}

// ---- Phase 2.9 — bold → bright -----------------------------------------

#[test]
fn bold_uses_bright_palette_color() {
    let theme = dark();
    let lines = rendered_lines_for_width("**bold**", &theme, 60);
    let has_bright_bold = lines.iter().any(|l| {
        l.spans.iter().any(|s| {
            s.style.add_modifier.contains(Modifier::BOLD)
                && s.style.fg == Some(theme.palette.bright)
        })
    });
    assert!(
        has_bright_bold,
        "**bold** must use bright palette + BOLD: {lines:?}"
    );
}

// ---- Phase 2.4 — nested list glyphs ------------------------------------

#[test]
fn nested_lists_use_depth_glyphs() {
    let theme = dark();
    let lines = rendered_lines_for_width("- a\n  - b\n    - c", &theme, 60);
    let joined = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
        .collect::<String>();
    assert!(joined.contains('\u{2022}'), "depth-0 must use •: {joined}");
    assert!(joined.contains('\u{25E6}'), "depth-1 must use ◦: {joined}");
    assert!(joined.contains('\u{25AA}'), "depth-2 must use ▪: {joined}");
}

// ---- R10 — NO_COLOR --------------------------------------------------

#[test]
fn markdown_under_no_color_produces_no_color_spans() {
    let theme = Theme::no_color();
    let lines = rendered_lines_for_width("# h1\n\n**bold**\n\n- item\n- item", &theme, 60);
    // 모든 span 의 fg 는 Color::Reset 이거나 Color::Rgb (syntect) 만 허용.
    // 이 입력에는 코드블록이 없으므로 Rgb 도 없어야 함.
    for line in &lines {
        for span in &line.spans {
            if let Some(fg) = span.style.fg {
                assert!(
                    matches!(fg, Color::Reset),
                    "NO_COLOR span has non-Reset fg: {fg:?} content={:?}",
                    span.content
                );
            }
        }
    }
}

// ---- 스트리밍 증분 — 블록 경계(stable_prefix_len) ----------------------

#[test]
fn stable_prefix_splits_after_last_top_level_blank_line() {
    // 마지막 빈 줄 뒤(열린 단락 앞)에서 끊긴다.
    let text = "para one\n\npara two still typing";
    let n = stable_prefix_len(text);
    assert_eq!(&text[..n], "para one\n\n");
    assert_eq!(&text[n..], "para two still typing");
}

#[test]
fn stable_prefix_is_full_len_when_text_ends_on_blank_line() {
    // 빈 줄로 끝나면 열린 세그먼트가 없으므로 전부 완료.
    let text = "## Heading\n\n- item a\n- item b\n\n";
    assert_eq!(stable_prefix_len(text), text.len());
}

#[test]
fn stable_prefix_keeps_open_segment_when_no_trailing_blank() {
    let text = "# Title\n\nbody line";
    let n = stable_prefix_len(text);
    assert_eq!(&text[..n], "# Title\n\n");
}

#[test]
fn stable_prefix_does_not_split_inside_code_fence() {
    // 코드펜스 내부의 빈 줄은 경계가 아니다 — 펜스가 닫히기 전까지 열림.
    let text = "intro\n\n```rust\nfn a() {\n\n}\n";
    let n = stable_prefix_len(text);
    // "intro\n\n" 만 안정, 미완성 펜스 전체는 열린 꼬리.
    assert_eq!(&text[..n], "intro\n\n");
    assert!(text[n..].starts_with("```rust"));
}

#[test]
fn stable_prefix_commits_closed_code_fence_followed_by_blank() {
    let text = "```rust\nfn a() {}\n```\n\nnext";
    let n = stable_prefix_len(text);
    assert_eq!(&text[..n], "```rust\nfn a() {}\n```\n\n");
    assert_eq!(&text[n..], "next");
}

#[test]
fn stable_prefix_zero_when_single_unbroken_block() {
    // 빈 줄이 없으면 전부 열린 꼬리 (안정 0).
    assert_eq!(stable_prefix_len("one long line still going"), 0);
}

#[test]
fn stable_prefix_is_monotonic_as_text_grows() {
    // 텍스트가 자라도 경계는 절대 뒤로 가지 않는다 (증분 재사용 전제).
    let full = "# A\n\nbody a\n\n## B\n\nbody b\n\n### C\n\nbody c tail";
    let mut prev = 0;
    for end in 1..=full.len() {
        if !full.is_char_boundary(end) {
            continue;
        }
        let n = stable_prefix_len(&full[..end]);
        assert!(
            n >= prev || n == 0,
            "boundary went backwards at end={end}: {prev} -> {n}"
        );
        // 빈 줄로 끝나는 prefix 가 아니면 경계는 prefix 길이를 넘지 않는다.
        assert!(n <= end);
        if n != 0 {
            prev = prev.max(n);
        }
    }
}

/// Flatten a rendered block into one string (span contents joined per line,
/// lines joined by `\n`).
fn flatten(text: &str, width: u16) -> String {
    rendered_lines_for_width(text, &dark(), width)
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

#[test]
fn gfm_task_list_renders_checkboxes_instead_of_literal_brackets() {
    let flat = flatten("- [ ] open item\n- [x] done item", 80);
    // The bullet is swapped for a real checkbox; the source `[ ]`/`[x]`
    // must not leak through as literal text.
    assert!(flat.contains("☐ open item"), "unchecked box: {flat}");
    assert!(flat.contains("☑ done item"), "checked box: {flat}");
    assert!(!flat.contains("[ ]"), "literal brackets leaked: {flat}");
    assert!(!flat.contains('•'), "bullet must be replaced: {flat}");
}

#[test]
fn nested_and_ordered_task_items_keep_their_structure() {
    let flat = flatten("- [x] top\n  - [ ] nested\n\n1. [x] ordered", 80);
    assert!(flat.contains("☑ top"), "{flat}");
    // Nested item keeps its 2-space indent in front of the box.
    assert!(flat.contains("  ☐ nested"), "{flat}");
    // Ordered task items keep the number and append the box.
    assert!(flat.contains("1. ☑ ordered"), "{flat}");
}

#[test]
fn no_color_task_list_falls_back_to_ascii_boxes() {
    let mut theme = dark();
    theme.no_color = true;
    let flat = rendered_lines_for_width("- [ ] a\n- [x] b", &theme, 80)
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(flat.contains("[ ] a"), "{flat}");
    assert!(flat.contains("[x] b"), "{flat}");
}

#[test]
fn inline_code_does_not_gap_before_punctuation_or_korean_particle() {
    // The eager trailing pad used to render "`sqlgate`," as "sqlgate ," and
    // "`1.26.3`로" as "1.26.3 로". Punctuation and Korean particles must
    // attach to the code span with no inserted space.
    let flat = flatten("module `sqlgate`, Go `1.26.3`로 구성", 80);
    assert!(
        flat.contains("sqlgate, Go"),
        "no gap before comma: {flat:?}"
    );
    assert!(flat.contains("1.26.3로"), "no gap before 로: {flat:?}");
    assert!(!flat.contains("sqlgate ,"), "stray pre-comma gap: {flat:?}");
    assert!(
        !flat.contains("1.26.3 로"),
        "stray pre-particle gap: {flat:?}"
    );
}

#[test]
fn inline_code_keeps_single_space_when_source_has_one() {
    // Source "`code` text" (a space already present) must stay a single
    // space — the old always-on trailing pad doubled it to "code  text".
    let flat = flatten("run `cargo` then build", 80);
    assert!(flat.contains("cargo then"), "single space kept: {flat:?}");
    assert!(!flat.contains("cargo  then"), "doubled space: {flat:?}");
}

#[test]
fn inline_code_pads_before_latin_word_for_legibility() {
    // When a Latin word abuts the code with no source space, a 1-cell pad
    // still keeps it readable ("`flag`value" → "flag value").
    let flat = flatten("pass `--flag`value here", 80);
    assert!(flat.contains("--flag value"), "latin pad kept: {flat:?}");
}

// ============================================================================
// List hanging-indent wrap tests
// ============================================================================

/// Helper: count the number of visual rows a list-item block produces when
/// pre-wrapped by the `Renderer` and then measured by the same wrap engine
/// used by `text_block_height`. Because `pre_wrap_item_lines` already splits
/// each line to ≤ `width` cells, `wrapped_rows` must agree with the actual
/// line count — if it doesn't the height/draw contract is broken.
fn list_rows_at_width(md: &str, width: u16) -> (Vec<String>, u16) {
    let theme = dark();
    let lines = rendered_lines_for_width(md, &theme, width);
    // Flatten for visual inspection.
    let rows: Vec<String> = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect();
    // Measure via the same `Paragraph::line_count` engine used by
    // `text_block_height` so we can assert height==draw.
    let measured = wrapped_rows(&lines, width);
    (rows, measured)
}

/// Bullet list item whose text is long enough to wrap must produce continuation
/// rows that begin with the hanging indent (spaces equal to the marker width)
/// rather than starting at column 0.
#[test]
fn bullet_list_long_item_wraps_with_hanging_indent() {
    // "• " is 2 cells (bullet + space). Render at width=20 so the full text
    // "The quick brown fox jumps over the lazy dog" cannot fit on one row.
    let md = "- The quick brown fox jumps over the lazy dog";
    let (rows, _measured) = list_rows_at_width(md, 20);

    // Find the marker row and continuation rows.
    let marker_row = rows
        .iter()
        .find(|r| r.contains('•'))
        .expect("must have a bullet row");
    let cont_rows: Vec<&String> = rows
        .iter()
        .filter(|r| !r.is_empty() && !r.contains('•') && r.trim_start() != r.as_str())
        .collect();

    assert!(
        !cont_rows.is_empty(),
        "text at width=20 must wrap onto continuation rows; rows={rows:?}"
    );
    for row in &cont_rows {
        // Continuation rows must start with at least 2 spaces (marker width).
        assert!(
            row.starts_with("  "),
            "continuation row must start with 2-space hanging indent; row={row:?}, all rows={rows:?}"
        );
    }
    // Sanity: the marker row itself starts with the bullet, not spaces.
    assert!(
        marker_row.starts_with('•'),
        "marker row must start with bullet; row={marker_row:?}"
    );
}

/// Height measured via `wrapped_rows` must equal the number of rendered lines
/// for a wrapping bullet list item (the height/draw agreement invariant).
#[test]
fn bullet_list_height_equals_draw_line_count() {
    let md = "- The quick brown fox jumps over the lazy dog";
    let theme = dark();
    let width = 20u16;
    let lines = rendered_lines_for_width(md, &theme, width);
    let line_count = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let measured = wrapped_rows(&lines, width);
    // After pre-wrapping, each emitted Line already fits ≤ width cells, so
    // `wrapped_rows` (which uses `Paragraph::line_count`) must match
    // `lines.len()` (no extra wrapping needed).
    assert_eq!(
        measured, line_count,
        "height({measured}) must equal line_count({line_count}) at width={width}; \
         pre-wrapping and draw must agree"
    );
    // Guard: the item must actually have wrapped (otherwise the test proves nothing).
    assert!(
        line_count > 3,
        "item at width={width} must produce more than 3 lines (got {line_count}): pre-wrap not working"
    );
}

/// Ordered list item: "1. " marker is 3 cells. Continuation rows must align
/// 3 spaces in.
#[test]
fn ordered_list_long_item_wraps_with_hanging_indent() {
    let md = "1. The quick brown fox jumps over the lazy dog and keeps running";
    let (rows, _measured) = list_rows_at_width(md, 20);

    let marker_row = rows
        .iter()
        .find(|r| r.contains("1."))
        .expect("must have an ordered marker row");
    let cont_rows: Vec<&String> = rows
        .iter()
        .filter(|r| !r.is_empty() && !r.contains("1.") && r.starts_with("   "))
        .collect();

    assert!(
        !cont_rows.is_empty(),
        "ordered item at width=20 must wrap; rows={rows:?}"
    );
    for row in &cont_rows {
        // "1. " is 3 cells, so continuation must start with 3 spaces.
        assert!(
            row.starts_with("   "),
            "ordered continuation must start with 3-space hanging indent; row={row:?}"
        );
    }
    assert!(
        marker_row.starts_with("1."),
        "marker row must start with '1.'; row={marker_row:?}"
    );
}

/// Ordered list height/draw agreement.
#[test]
fn ordered_list_height_equals_draw_line_count() {
    let md = "1. The quick brown fox jumps over the lazy dog and keeps running";
    let theme = dark();
    let width = 20u16;
    let lines = rendered_lines_for_width(md, &theme, width);
    let line_count = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let measured = wrapped_rows(&lines, width);
    assert_eq!(
        measured, line_count,
        "ordered list height({measured}) must equal draw line_count({line_count})"
    );
    assert!(
        line_count > 3,
        "ordered item at width={width} must wrap (got {line_count} lines)"
    );
}

/// Nested list: inner items have a 4-space depth indent + 2-cell marker = 6
/// cells total. Continuation rows of the inner item must start with 6 spaces.
#[test]
fn nested_list_inner_item_wraps_with_correct_hanging_indent() {
    // Depth 0 bullet (2 cells: "• "), depth 1 bullet (4 cells: "  ◦ ").
    let md = "- top\n  - The quick brown fox jumps over the lazy dog at great speed";
    let (rows, measured) = list_rows_at_width(md, 24);

    // Outer item row.
    assert!(
        rows.iter().any(|r| r.contains('•') && r.contains("top")),
        "must have outer bullet row; rows={rows:?}"
    );
    // Inner item row uses ◦ (depth-1 bullet).
    let inner_marker_row = rows
        .iter()
        .find(|r| r.contains('◦'))
        .expect("must have inner bullet row");
    assert!(
        inner_marker_row.starts_with("  "),
        "inner marker must start with 2-space depth indent; row={inner_marker_row:?}"
    );

    // Continuation rows of the inner item must start with 4 spaces
    // (2 depth indent + 2 marker = 4).
    let inner_cont_rows: Vec<&String> = rows
        .iter()
        .filter(|r| r.starts_with("    ") && !r.contains('◦'))
        .collect();
    assert!(
        !inner_cont_rows.is_empty(),
        "inner item at width=24 must wrap; rows={rows:?}"
    );
    for row in &inner_cont_rows {
        assert!(
            row.starts_with("    "),
            "inner continuation must start with 4-space indent; row={row:?}"
        );
    }

    // Height/draw agreement.
    let line_count = u16::try_from(rows.len()).unwrap_or(u16::MAX);
    assert_eq!(
        measured, line_count,
        "nested list height({measured}) must equal draw line_count({line_count})"
    );
}

/// Short items (fit within `width`) must NOT gain an extra indent: they pass
/// through the pre-wrap step unchanged (fast-path).
#[test]
fn short_list_item_does_not_gain_extra_indent() {
    let md = "- short";
    let (rows, _) = list_rows_at_width(md, 40);
    let bullet_row = rows
        .iter()
        .find(|r| r.contains('•'))
        .expect("must have a bullet row");
    // Must start with the bullet, not a blank hanging indent.
    assert!(
        bullet_row.starts_with('•'),
        "short item must not gain a hanging indent; row={bullet_row:?}"
    );
    // No continuation rows.
    let cont_rows: Vec<&String> = rows
        .iter()
        .filter(|r| !r.is_empty() && !r.contains('•') && r.starts_with("  "))
        .collect();
    assert!(
        cont_rows.is_empty(),
        "short item must have no continuation rows; cont={cont_rows:?}, all={rows:?}"
    );
}

/// `NO_COLOR` path: ASCII `-` marker (1 cell + space = 2 cells) still wraps
/// with a 2-space hanging indent.
#[test]
fn no_color_bullet_list_wraps_with_hanging_indent() {
    let mut theme = dark();
    theme.no_color = true;
    let md = "- The quick brown fox jumps over the lazy dog";
    let lines = rendered_lines_for_width(md, &theme, 20);
    let rows: Vec<String> = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect();

    let marker_row = rows
        .iter()
        .find(|r| r.starts_with("- "))
        .expect("must have ASCII bullet row");
    let cont_rows: Vec<&String> = rows
        .iter()
        .filter(|r| !r.is_empty() && !r.starts_with("- ") && r.starts_with("  "))
        .collect();
    assert!(
        !cont_rows.is_empty(),
        "NO_COLOR item must wrap at width=20; rows={rows:?}"
    );
    for row in &cont_rows {
        assert!(
            row.starts_with("  "),
            "NO_COLOR continuation must start with 2-space hanging indent; row={row:?}"
        );
    }
    let _ = marker_row;
}

#[test]
fn debug_print_nested_and_nocolor() {
    // Nested list
    let md = "- top\n  - The quick brown fox jumps over the lazy dog at great speed";
    let (rows, _) = list_rows_at_width(md, 24);
    println!("NESTED ROWS:");
    for (i, r) in rows.iter().enumerate() {
        println!("  [{i}] {r:?}");
    }

    // No color
    let mut theme = dark();
    theme.no_color = true;
    let md2 = "- The quick brown fox jumps over the lazy dog";
    let lines = rendered_lines_for_width(md2, &theme, 20);
    println!("NO_COLOR ROWS:");
    for (i, l) in lines.iter().enumerate() {
        let flat: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
        println!("  [{i}] {flat:?}");
    }
}

#[test]
fn stable_prefix_len_promotes_list_items_and_blockquotes() {
    // A regular paragraph has no boundary.
    assert_eq!(stable_prefix_len("line 1\nline 2"), 0);

    // List item starts a new segment.
    assert_eq!(stable_prefix_len("line 1\n- item 1"), 7);
    assert_eq!(stable_prefix_len("line 1\n* item 1"), 7);
    assert_eq!(stable_prefix_len("line 1\n1. item 1"), 7);

    // List items inside a code block must NOT promote segments.
    assert_eq!(stable_prefix_len("```rust\nline 1\n- item 1"), 0);

    // Blockquote starts a new segment.
    assert_eq!(stable_prefix_len("line 1\n> quote 1"), 7);
}

#[test]
fn streaming_stable_prefix_advances_only_inside_a_large_open_fence() {
    // Below the threshold: identical to `stable_prefix_len` — a small open fence
    // stays a whole unstable tail (boundary 0), no fence context promoted.
    let small = "```rust\nlet a = 1;\nlet b = 2;\n";
    let (boundary, at_stable, at_boundary, lang) = streaming_stable_prefix(small, 0, 16 * 1024);
    assert_eq!(boundary, 0, "small open fence keeps the whole tail open");
    assert_eq!(at_stable, None);
    assert_eq!(at_boundary, None);
    assert_eq!(lang, None, "no fence promoted below threshold → no lang");

    // A huge open fence: the boundary advances to the last completed line, and the
    // tail is reported as fence interior so the caller renders it as code.
    let mut huge = String::from("```rust\n");
    for i in 0..4000 {
        use std::fmt::Write as _;
        let _ = writeln!(huge, "let v{i} = {i};");
    }
    huge.push_str("let partial = "); // trailing partial line (no newline yet)
    let (boundary, _at_stable, at_boundary, lang) = streaming_stable_prefix(&huge, 0, 16 * 1024);
    assert!(boundary > 0, "huge open fence promotes completed code lines");
    assert!(
        boundary <= huge.len() - "let partial = ".len(),
        "the trailing partial line stays in the open tail"
    );
    assert_eq!(
        at_boundary,
        Some((b'`', 3)),
        "tail begins inside the still-open fence"
    );
    assert_eq!(
        lang.as_deref(),
        Some("rust"),
        "the open fence's language is captured for the streaming card label"
    );
    // Stable byte must land on a line boundary (just past a newline).
    assert!(huge.as_bytes()[boundary - 1] == b'\n');

    // Resuming with stable_len inside the fence reports fence-interior context for
    // the promoted fragment (so the caller renders it as code, not prose).
    let (_b2, at_stable2, _ab2, _lang2) = streaming_stable_prefix(&huge, boundary, 16 * 1024);
    assert_eq!(
        at_stable2,
        Some((b'`', 3)),
        "a fragment promoted from mid-fence is fence interior"
    );
}

/// The incremental, cache-backed `streaming_stable_prefix_resumed` MUST produce
/// byte-identical results to the full-scan `streaming_stable_prefix` for every
/// prefix of every input, driving `stable_len`/cache exactly as the live layout
/// (`streaming_incremental`) does. This is the safety net for the O(total)→
/// O(suffix) streaming-freeze fix: if the incremental scan ever diverges, the
/// streamed render would corrupt, so this asserts equivalence exhaustively.
#[test]
fn resumed_stable_prefix_matches_full_scan_for_every_prefix() {
    // A 50-line fenced block, large enough to cross a small fence threshold.
    let mut big_code = String::from("intro paragraph\n\n```rust\n");
    for i in 0..50 {
        use std::fmt::Write as _;
        let _ = writeln!(big_code, "let v{i} = {i} + {i};");
    }
    big_code.push_str("```\n\nclosing paragraph after the code.\n\nmore");

    let corpus = [
        "",
        "hello",
        "hello world\n",
        "para one\n\npara two\n\npara three",
        "line without trailing newline",
        "- item one\n- item two\n- item three\n",
        "> a quote\n> continues\n\nthen prose",
        "```\nplain fence\nno lang\n```\nafter",
        "```python\nprint(1)\nprint(2)\n```\n\nprose tail",
        "open fence never closes\n```rust\nlet a = 1;\nlet b = 2;\nlet c = 3;\n",
        "text\n```\ncode\n```\nmore text\n```\nsecond fence\n",
        "트래픽 한국어\n\n코드:\n```rust\nfn 함수() {}\n```\n끝",
        "a\n\n\n\nb\n\n\n\nc",
        &big_code,
    ];
    // Include a threshold small enough to exercise the huge-fence promotion path
    // on the modest corpus, plus the production limit.
    for threshold in [4usize, 32, 64, 16 * 1024] {
        for text in corpus {
            let mut stable_len = 0usize;
            let mut cache = None;
            let mut end = 0usize;
            while end <= text.len() {
                if !text.is_char_boundary(end) {
                    end += 1;
                    continue;
                }
                let prefix = &text[..end];
                let full = streaming_stable_prefix(prefix, stable_len, threshold);
                let (rb, rs, rab, rl, rstate) =
                    streaming_stable_prefix_resumed(prefix, stable_len, threshold, cache.as_ref());
                assert_eq!(
                    full,
                    (rb, rs, rab, rl),
                    "resumed != full at threshold={threshold} end={end} stable_len={stable_len} text={text:?}"
                );
                // Drive the boundary exactly like `streaming_incremental`: only
                // promote (and persist the cursor) when the boundary advances.
                if rb > stable_len {
                    stable_len = rb;
                    assert_eq!(
                        rstate.at, rb,
                        "persisted cursor must sit at the new boundary (end={end})"
                    );
                    cache = Some(rstate);
                }
                end += 1;
            }
        }
    }
}

// ── v3 §5 구조물 라인 게이트 (clip_tail_for_display) ────────────────────

/// 산문 꼬리는 클립하지 않는다 — char 단위 type-in 감성 유지 (사용자 결정).
#[test]
fn clip_tail_leaves_prose_untouched() {
    assert_eq!(clip_tail_for_display("스트리밍 중인 문장", false), "스트리밍 중인 문장");
    assert_eq!(
        clip_tail_for_display("첫 줄\n- 리스트 항목 하나\n- 미완성 항", false),
        "첫 줄\n- 리스트 항목 하나\n- 미완성 항",
        "lists are prose-class: completed items never re-render, so no gate"
    );
}

/// 열린 펜스 안에서는 완성 줄만 표시한다 — 미완성 코드 줄과 부분 opener 는
/// 개행이 도착할 때까지 숨는다 (codex newline-gated 커밋 동형).
#[test]
fn clip_tail_gates_open_fences_at_complete_lines() {
    assert_eq!(
        clip_tail_for_display("```rust\nfn main() {\n    prin", false),
        "```rust\nfn main() {\n",
        "the partial code line must be held back"
    );
    assert_eq!(
        clip_tail_for_display("```ru", false),
        "",
        "a partial fence opener must not flash raw"
    );
    // 닫힌 펜스 + 이어지는 산문 = 열린 구조물 아님 → 무클립.
    assert_eq!(
        clip_tail_for_display("```sh\nls\n```\n이어지는 산", false),
        "```sh\nls\n```\n이어지는 산"
    );
}

/// 경계가 대형 펜스 내부로 전진한 꼬리(호출자가 fence_at_boundary 로 판정)는
/// 항상 줄 게이트를 받는다.
#[test]
fn clip_tail_gates_large_fence_interior() {
    assert_eq!(clip_tail_for_display("    let x = 4", true), "");
    assert_eq!(
        clip_tail_for_display("let done = 1;\nlet part", true),
        "let done = 1;\n"
    );
}

/// 표는 헤더행+구분자행이 **완성**된 뒤에만 게이트 — 확인 전에는 클립하지
/// 않는다 (오탐의 최악이 지터 잔존이지 텍스트 실종이 아니게, 보수 전략).
#[test]
fn clip_tail_gates_only_confirmed_tables() {
    // 헤더만: 미확인 → 무클립.
    assert_eq!(clip_tail_for_display("| 이름 | 값 |", false), "| 이름 | 값 |");
    // 구분자행이 아직 개행으로 닫히지 않음: 미확인 → 무클립.
    assert_eq!(
        clip_tail_for_display("| 이름 | 값 |\n|---|---", false),
        "| 이름 | 값 |\n|---|---"
    );
    // 확인된 표 + 미완성 데이터 행: 완성 행까지만 표시.
    assert_eq!(
        clip_tail_for_display("| 이름 | 값 |\n|---|---|\n| a | 1 |\n| b | 2", false),
        "| 이름 | 값 |\n|---|---|\n| a | 1 |\n",
        "the partial table row must be held back so column widths stop jittering"
    );
}

// ── v3 가독성: text 펜스 프로즈화 + 물리 wrap ─────────────────────────

/// ```text 펜스는 코드 카드가 아니라 wrap 되는 인용 블록으로 렌더된다 —
/// 모든 행이 │ 레일을 갖고, 어떤 행도 폭을 넘지 않는다(선 넘어감 금지).
#[test]
fn text_fence_renders_as_wrapped_quote_block() {
    let theme = dark();
    let width = 40u16;
    let md = "설명입니다.\n\n```text\n수집 중단부터 조치했습니다. 디스크가 꽉 차서 클러스터 마스터가 안 붙는 상태였습니다. 자동삭제가 안 된 게 아니라 스케줄러가 꺼져 있었습니다.\n\n두 번째 문단입니다.\n```\n";
    let lines = rendered_lines_for_width(md, &theme, width);
    let flat: Vec<String> = lines
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
        .collect();
    let quote_rows: Vec<&String> = flat.iter().filter(|r| r.starts_with('\u{2502}')).collect();
    assert!(
        quote_rows.len() >= 4,
        "long CJK passage must wrap into multiple railed rows: {flat:?}"
    );
    for line in &lines {
        let cells: usize = line
            .spans
            .iter()
            .map(|s| super::cell_display_width(s.content.as_ref()))
            .sum();
        assert!(
            cells <= usize::from(width),
            "no row may cross the frame edge ({cells} > {width}): {line:?}"
        );
    }
    // 카드 프레임(╭─ text ─)이 아니어야 한다.
    assert!(
        !flat.iter().any(|r| r.contains('\u{256d}')),
        "text fence must not render as a code card: {flat:?}"
    );
    // 빈 소스 줄도 레일을 유지해 한 덩어리로 읽힌다.
    assert!(
        flat.iter().any(|r| r.trim_end() == "\u{2502}"),
        "blank source lines keep a bare rail row: {flat:?}"
    );
}

/// 진짜 코드 펜스(rust)는 프로즈화의 영향을 받지 않는다 — 카드 유지.
#[test]
fn code_fence_keeps_its_card_after_prose_fence_change() {
    let theme = dark();
    let md = "```rust\nfn main() {}\n```\n";
    let lines = rendered_lines_for_width(md, &theme, 60);
    let flat: String = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
        .collect();
    assert!(flat.contains('\u{256d}'), "rust fence keeps the card frame: {flat:?}");
}

/// wrap_to_display_cells: 라틴은 공백 우선 절단, CJK 는 셀 예산으로 절단 —
/// 어떤 청크도 예산을 넘지 않는다.
#[test]
fn wrap_to_display_cells_respects_budget_and_spaces() {
    let latin = super::wrap_to_display_cells("alpha beta gamma delta", 11);
    assert_eq!(latin, vec!["alpha beta", "gamma delta"]);
    let cjk = super::wrap_to_display_cells("가나다라마바사", 6);
    for chunk in &cjk {
        assert!(super::cell_display_width(chunk) <= 6, "chunk over budget: {chunk:?}");
    }
    assert_eq!(cjk.concat(), "가나다라마바사", "no character may be lost");
}

/// The turn-end confidence marker renders as a dim chip, and only when it is
/// the FINAL line (mirroring the cascade parse contract) — a marker quoted
/// mid-text stays literal prose.
#[test]
fn trailing_turn_confidence_marker_renders_as_dim_chip() {
    let theme = dark();
    let text = "결과를 정리했습니다.\n\n[zo:turn-confidence] low — verify를 못 돌렸음";
    let lines = rendered_lines_for_width(text, &theme, 80);
    let flat = flatten_lines(&lines);
    assert!(!flat.contains("[zo:turn-confidence]"), "{flat}");
    assert!(flat.contains("◈ confidence low — verify를 못 돌렸음"), "{flat}");
    let chip = lines.last().expect("chip line");
    assert!(
        chip.spans
            .iter()
            .all(|span| span.style.add_modifier.contains(Modifier::ITALIC)),
        "chip must render dim-italic"
    );

    // Marker-only block: just the chip.
    let only = rendered_lines_for_width("[zo:turn-confidence] low — x", &theme, 80);
    assert_eq!(only.len(), 1);

    // Quoted mid-text (confident close afterwards) stays raw.
    let quoted = "예시:\n[zo:turn-confidence] low — 이유\n이번 턴은 확신 있습니다.";
    let flat = flatten_lines(&rendered_lines_for_width(quoted, &theme, 80));
    assert!(flat.contains("[zo:turn-confidence]"), "{flat}");
}
