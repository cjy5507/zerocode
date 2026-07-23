//! Tool call 의 origin (local runtime vs. MCP server) 분류.
//!
//! 도구 이름 `mcp__<server>__<tool>` 패턴을 인식해 MCP 서버로 분류하고,
//! 그 외는 모두 로컬 runtime 도구로 본다. `tool_call.rs` 가 이 분류를
//! 사용해 `@server` 칩을 violet 으로 표시한다.
//!
//! 로컬 도구는 silent default — `@local` 칩은 표시하지 않는다.
//! Antigravity 스타일 디자인 원칙: "차이를 만드는 메타데이터만 표시".

use ratatui::style::{Modifier, Style};

use crate::tui::theme::Theme;

/// 도구 호출의 출처.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin<'a> {
    /// 로컬 runtime 도구 (Bash, Read, Write, Grep 등).
    Local,
    /// MCP 서버 도구. inner 는 서버 이름 (예: `"almanac"`, `"context7"`).
    Mcp(&'a str),
}

/// 도구 이름에서 origin 추론.
///
/// `mcp__almanac__search_pages` → `Origin::Mcp("almanac")`.
/// 그 외 모든 이름 → `Origin::Local`.
#[must_use]
pub fn classify(name: &str) -> Origin<'_> {
    if let Some(rest) = name.strip_prefix("mcp__") {
        if let Some((server, _tool)) = rest.split_once("__") {
            return Origin::Mcp(server);
        }
    }
    Origin::Local
}

/// 도구의 표시 이름 — MCP prefix 와 server segment 제거.
///
/// `mcp__almanac__search_pages` → `"search_pages"`.
#[must_use]
pub fn display_name(name: &str) -> &str {
    if let Some(rest) = name.strip_prefix("mcp__") {
        if let Some((_, tool)) = rest.split_once("__") {
            return tool;
        }
    }
    name
}

/// origin 칩의 스타일.
///
/// MCP → `violet + italic` (테마 팔레트에서 "Reasoning/MCP indicator"
/// 슬롯). Local 은 호출자가 칩 표시 자체를 skip 한다.
#[must_use]
pub fn chip_style(origin: Origin<'_>, theme: &Theme) -> Style {
    match origin {
        Origin::Local => Style::new()
            .fg(theme.palette.muted)
            .add_modifier(Modifier::ITALIC),
        Origin::Mcp(_) => Style::new()
            .fg(theme.palette.violet)
            .add_modifier(Modifier::ITALIC),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_recognises_mcp_prefix() {
        assert_eq!(
            classify("mcp__almanac__search_pages"),
            Origin::Mcp("almanac")
        );
        assert_eq!(
            classify("mcp__context7__query-docs"),
            Origin::Mcp("context7")
        );
    }

    #[test]
    fn classify_local_for_runtime_tools() {
        assert_eq!(classify("bash"), Origin::Local);
        assert_eq!(classify("Read"), Origin::Local);
        assert_eq!(classify("Edit"), Origin::Local);
    }

    #[test]
    fn classify_local_when_mcp_prefix_lacks_tool_segment() {
        // mcp__ 뒤에 segment 가 하나뿐이면 Local 폴백.
        assert_eq!(classify("mcp__incomplete"), Origin::Local);
    }

    #[test]
    fn display_name_strips_mcp_prefix() {
        assert_eq!(display_name("mcp__almanac__search_pages"), "search_pages");
        assert_eq!(display_name("Read"), "Read");
        assert_eq!(display_name("mcp__only_two__rest_part"), "rest_part");
    }

    #[test]
    fn display_name_preserves_tool_with_dashes() {
        assert_eq!(display_name("mcp__context7__query-docs"), "query-docs");
    }
}
