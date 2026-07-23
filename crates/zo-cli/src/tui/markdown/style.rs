//! Markdown 스타일/글리프 헬퍼 — heading·inline·bullet·blockquote·callout 의
//! 색/모디파이어/글리프 결정. 순수 함수 모음(테마만 의존). `Renderer`(부모
//! 모듈)와 스트리밍 fast-path 가 이 헬퍼들을 공유한다.

use pulldown_cmark::HeadingLevel;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

use crate::tui::theme::{CalloutKind, Theme};

/// Heading 레벨을 시맨틱 접근자용 1–6 정수로 환원.
pub(super) fn heading_level_num(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// Phase 2.1 — heading 레벨별 글리프 + 스타일.
///
/// 색은 전적으로 [`Theme::heading_style`] 가 결정한다(R9 단일 소스). 글리프만
/// 레벨별 위계를 담당: 컬러 모드는 블록 글리프가 두께로 얇아지고(█ ▌ ▎ ·),
/// NO_COLOR 모드는 `#` 개수로 위계를 표현한다.
pub(super) fn heading_glyph(theme: &Theme, level: HeadingLevel) -> (&'static str, Style) {
    let style = theme.heading_style(heading_level_num(level));
    let glyph = if theme.no_color {
        match level {
            HeadingLevel::H1 => "# ",
            HeadingLevel::H2 => "## ",
            HeadingLevel::H3 => "### ",
            HeadingLevel::H4 => "#### ",
            HeadingLevel::H5 => "##### ",
            HeadingLevel::H6 => "###### ",
        }
    } else {
        match level {
            // █  full block — 강한 위계
            HeadingLevel::H1 => "\u{2588} ",
            // ▌  left half block — 중간 위계
            HeadingLevel::H2 => "\u{258C} ",
            // ▎  3/8 block — 약한 위계 강조
            HeadingLevel::H3 => "\u{258E} ",
            // ·  middle dot — H4-H6 공통
            _ => "\u{00B7} ",
        }
    };
    (glyph, style)
}

/// Heading 본문 텍스트의 스타일 — 글리프와 동일한 [`Theme::heading_style`].
///
/// v2 절제: H1-H3=`bright`, H4-H6=`dim` — 위계는 hue 가 아니라 명도와
/// 글리프 두께(█/▌/▎/·)가 담당한다. 브랜드 앰버는 유저레일·포커스·라이브
/// 순간 전용이라 헤딩에서 퇴출됐다. NO_COLOR 는 BOLD 로만 위계 표현.
pub(super) fn heading_text_style(theme: &Theme, level: HeadingLevel) -> Style {
    theme.heading_style(heading_level_num(level))
}

/// 인라인 스타일 결정에 필요한 플래그 묶음.
///
/// 4개의 bool 을 개별 인자로 받으면 clippy `fn_params_excessive_bools` 가
/// 경고하고 호출부 가독성도 떨어지므로 구조체로 묶는다.
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(super) struct InlineFlags {
    pub(super) bold: bool,
    pub(super) italic: bool,
    pub(super) strike: bool,
    pub(super) in_blockquote: bool,
}

pub(super) fn inline_style(theme: &Theme, flags: InlineFlags) -> Style {
    let InlineFlags {
        bold,
        italic,
        strike,
        in_blockquote,
    } = flags;
    let mut style = if in_blockquote {
        // Phase 2.5 — blockquote 안은 dim + italic 강제.
        Style::new()
            .fg(theme.palette.dim)
            .add_modifier(Modifier::ITALIC)
    } else if bold {
        // Phase 2.9 — bold 일 때 fg 를 bright 로 승격 (단순 BOLD modifier 이상).
        Style::new()
            .fg(theme.palette.bright)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(theme.palette.fg)
    };
    if bold && in_blockquote {
        style = style.add_modifier(Modifier::BOLD);
    }
    if italic {
        style = style.add_modifier(Modifier::ITALIC);
    }
    // Phase 2.8 — strikethrough modifier (현재 누락되어 있던 것).
    if strike {
        style = style.add_modifier(Modifier::CROSSED_OUT);
    }
    style
}

pub(super) fn code_inline_style(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.palette.cyan)
        .bg(theme.code_surface())
}

pub(super) fn link_style(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.palette.cyan)
        .add_modifier(Modifier::UNDERLINED)
}

/// Phase 2.4 — 리스트 깊이별 글리프.
pub(super) fn bullet_glyph_for_depth(depth: usize, no_color: bool) -> &'static str {
    if no_color {
        return "-";
    }
    match depth {
        0 => "\u{2022}", // •
        1 => "\u{25E6}", // ◦
        _ => "\u{25AA}", // ▪
    }
}

/// Bullets are punctuation, not content: they stay in the quiet neutral
/// steps (deeper = quieter) so list bodies read at full contrast while the
/// markers recede. Depth is already carried by the glyph shape (• ◦ ▪);
/// spending brand/secondary hues here made every list a color event.
pub(super) fn bullet_color_for_depth(theme: &Theme, depth: usize) -> Color {
    match depth {
        0 | 1 => theme.palette.dim,
        _ => theme.palette.muted,
    }
}

/// Phase 2.5 / W6 — blockquote 의 좌측 레일 span.
///
/// 일반 인용은 가는 `▎` + `accent_dim`. GitHub admonition (callout) 으로
/// 인식되면 종류별 시맨틱 색(`Theme::callout_color`)의 두꺼운 `▌` 레일로
/// 승격해 Note/Tip/Warning/Important 가 한눈에 구분된다.
pub(super) fn blockquote_rail_span(theme: &Theme, kind: Option<CalloutKind>) -> Span<'static> {
    match kind {
        Some(k) => Span::styled(
            "\u{258C} ".to_string(),
            Style::new().fg(theme.callout_color(k)),
        ),
        None => Span::styled(
            "\u{258E} ".to_string(),
            Style::new().fg(theme.palette.accent_dim),
        ),
    }
}

/// `▎`/`▌` 레일 글리프로 시작하는 span 인지 — callout 헤더가 직전에 깔린
/// 일반 인용 레일을 식별해 교체할 때 쓴다.
pub(super) fn is_rail_glyph(content: &str) -> bool {
    let t = content.trim_start();
    t.starts_with('\u{258E}') || t.starts_with('\u{258C}')
}

pub(super) fn spans_are_blankish(spans: &[Span<'_>]) -> bool {
    spans
        .iter()
        .all(|span| span.content.trim().is_empty() || is_rail_glyph(&span.content))
}

/// `true` when a line carries a blockquote/callout rail glyph but no body text
/// — i.e. a dangling rail stub. A genuinely empty line (no spans / only
/// whitespace) returns `false` so the normal blank-line spacing is preserved.
pub(super) fn line_is_rail_only(spans: &[Span<'_>]) -> bool {
    let mut saw_rail = false;
    for span in spans {
        if is_rail_glyph(&span.content) {
            saw_rail = true;
        } else if !span.content.trim().is_empty() {
            return false;
        }
    }
    saw_rail
}

/// blockquote 첫 텍스트에서 GitHub admonition 마커를 파싱한다.
/// `[!NOTE]`, `[!TIP]`, `[!WARNING]`/`[!CAUTION]`, `[!IMPORTANT]` 지원
/// (대소문자 무시). 마커 뒤 같은 줄 잔여 텍스트를 함께 돌려준다.
pub(super) fn parse_callout_marker(text: &str) -> Option<(CalloutKind, String)> {
    const MARKERS: &[(&str, CalloutKind)] = &[
        ("[!note]", CalloutKind::Note),
        ("[!tip]", CalloutKind::Tip),
        ("[!warning]", CalloutKind::Warning),
        ("[!caution]", CalloutKind::Warning),
        ("[!important]", CalloutKind::Important),
    ];
    let trimmed = text.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    for (marker, kind) in MARKERS {
        if lower.starts_with(marker) {
            let rest = trimmed[marker.len()..].trim_start().to_string();
            return Some((*kind, rest));
        }
    }
    None
}

/// callout 헤더 라벨.
pub(super) const fn callout_label(kind: CalloutKind) -> &'static str {
    match kind {
        CalloutKind::Note => "Note",
        CalloutKind::Tip => "Tip",
        CalloutKind::Warning => "Warning",
        CalloutKind::Important => "Important",
    }
}
