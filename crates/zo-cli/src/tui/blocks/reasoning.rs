//! `RenderBlock::Reasoning` widget — quiet collapsible reasoning text.
//!
//! Per `.zo/design/components.md` §5.4 and `code-rules.md` R6,
//! this widget renders the provider-neutral `Reasoning` variant
//! (Anthropic "thinking", `OpenAI` "reasoning summary"). The word
//! `Thinking` does **not** appear anywhere in the dispatch surface —
//! R1 enforcement.
//!
//! Visual:
//! * No border.
//! * Collapsed by default (`expanded == false`): hidden from the transcript so
//!   live progress stays in the status area instead of leaking internal steps.
//! * Expanded state keeps a subtle left rail for body scanning.
//! * No streaming caret — the live affordance is the `✦ Thinking…` line.

use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::tui::glyphs;
use crate::tui::term::reduce_motion_enabled;
use crate::tui::theme::Theme;

use super::wrapped_rows;

const GUTTER_WIDTH: u16 = 2;

/// Minimum measured thinking time for a *settled, collapsed* reasoning block to
/// leave a permanent one-line trace (`<summary> · N.Ns`) instead of
/// vanishing to height 0. A short think is noise and stays hidden; a long one
/// kept the user waiting, so its duration + topic is worth a quiet line.
///
/// This threshold is the single SSOT shared by every site that decides a
/// collapsed-done reasoning block's visibility/height — [`suppress_collapsed`]
/// (called from [`draw`], [`estimate_rows`], and transcript's
/// `is_reasoning_visually_suppressed`). Measure and draw MUST agree on it, or the
/// transcript's row prefix-sums underflow and scroll math drifts.
pub(crate) const COLLAPSED_DONE_MIN_ELAPSED: Duration = Duration::from_secs(3);

pub const NO_COLOR_PREFIX: &str = "[step] ";

/// Neutral live-reasoning verbs shown before the animated `…` cue. One is
/// chosen per reasoning block via [`zo_reveal_verb`]; this array is the single
/// source of truth for the set.
pub const ZO_REVEAL_VERBS: [&str; 6] = [
    "Thinking",
    "Planning",
    "Exploring",
    "Solving",
    "Reviewing",
    "Working",
];

/// Pick a live-reasoning verb deterministically from `seed` (the reasoning
/// block's stable id). Seeding by the block — not the render tick — keeps one
/// word fixed for the life of a block (no per-frame flicker, and width-stable so
/// the dot animation never reflows the line), while different reasoning steps in
/// a turn surface different words. The set is small and the modulo is uniform
/// over it, so the variety reads as deliberate rather than random noise.
#[must_use]
pub fn zo_reveal_verb(seed: u64) -> &'static str {
    let idx = usize::try_from(seed % ZO_REVEAL_VERBS.len() as u64).unwrap_or(0);
    ZO_REVEAL_VERBS[idx]
}

/// Whether a collapsed reasoning block is hidden from the transcript.
///
/// A *streaming* reasoning block (`done == false`) is never suppressed: it
/// renders a live, animated `✶ Thinking…` line so the user sees the model is
/// actively reasoning (Gemini thought summaries, Anthropic thinking, OpenAI
/// reasoning all flow here). Once the block settles (`done == true`) it returns
/// to the quiet default — hidden unless the user expands it with Enter, EXCEPT a
/// long think (`elapsed >= COLLAPSED_DONE_MIN_ELAPSED`) keeps a one-line trace so
/// the user still sees how long the model reasoned. Unmeasured time (`None`) is
/// treated as short and stays hidden (the safe default — never claim a row the
/// renderer might not draw).
pub(crate) fn suppress_collapsed(
    _text: &str,
    done: bool,
    expanded: bool,
    elapsed: Option<Duration>,
) -> bool {
    !expanded && done && elapsed.is_none_or(|d| d < COLLAPSED_DONE_MIN_ELAPSED)
}

/// Render the reasoning widget.
///
/// The viewport is split into a fixed-width rail gutter and a wrapping
/// body column. The rail glyph is painted on every physical row of the
/// gutter so that long, soft-wrapped reasoning text keeps its rail on
/// every continuation row — matching native Claude Code.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    text: &str,
    done: bool,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    tick: u64,
    scroll_offset: u16,
    elapsed: Option<Duration>,
    seed: u64,
) {
    if suppress_collapsed(text, done, expanded, elapsed) {
        return;
    }

    if !expanded {
        frame.render_widget(
            Paragraph::new(body_lines(
                text, done, theme, focused, expanded, tick, elapsed, seed,
            ))
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0)),
            area,
        );
        return;
    }

    let [gutter_area, body_area] =
        Layout::horizontal([Constraint::Length(GUTTER_WIDTH), Constraint::Min(0)]).areas(area);

    // Rail gutter: one glyph per physical row of the block. Glyph flows through
    // `glyphs::REASONING_RAIL` (SSOT) and degrades to `:` under NO_COLOR.
    let rail_glyph = glyphs::pick(
        !theme.no_color,
        glyphs::REASONING_RAIL,
        glyphs::REASONING_RAIL_NC,
    );
    let rail_style = if theme.no_color {
        Style::new()
    } else {
        Style::new().fg(theme.palette.violet)
    };
    let rail_lines: Vec<Line<'_>> = (0..gutter_area.height.saturating_add(scroll_offset))
        .map(|_| Line::from(Span::styled(rail_glyph, rail_style)))
        .collect();
    frame.render_widget(
        Paragraph::new(rail_lines).scroll((scroll_offset, 0)),
        gutter_area,
    );

    // Body column: no rail spans here — let ratatui wrap freely.
    let lines = body_lines(text, done, theme, focused, expanded, tick, elapsed, seed);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0)),
        body_area,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_rows(
    text: &str,
    done: bool,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    width: u16,
    elapsed: Option<Duration>,
    seed: u64,
) -> u16 {
    if suppress_collapsed(text, done, expanded, elapsed) {
        return 0;
    }

    let body_width = if expanded {
        width.saturating_sub(GUTTER_WIDTH).max(1)
    } else {
        width.max(1)
    };
    wrapped_rows(
        &body_lines(text, done, theme, focused, expanded, 0, elapsed, seed),
        body_width,
    )
}

/// Body lines (without rail spans). The rail is painted separately in
/// the gutter column by [`draw`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) fn body_lines(
    text: &str,
    done: bool,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    tick: u64,
    elapsed: Option<Duration>,
    seed: u64,
) -> Vec<Line<'static>> {
    let body_style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::DIM)
        .add_modifier(Modifier::ITALIC);
    // 사고 시간 메타("· 2.7s") — 차분한 dim. 완료 후에도 유지된다.
    let meta_style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::DIM);

    // Collapsed by default (components.md §4). Expand only when
    // focused + explicitly expanded via Enter.
    if !expanded {
        // Streaming (done == false): show a live, animated reasoning cue so the
        // user sees the model is active. A neutral per-block verb is chosen by
        // `seed`; the dots cycle (`` → `.` → `..` → `...`) on the render tick,
        // and the elapsed `· N.Ns` rides alongside once timing is measured.
        if !done {
            let spark = glyphs::pick(!theme.no_color, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC);
            // The warning role distinguishes the live cue from quiet reasoning.
            // NO_COLOR degrades the hue to Reset and leans on BOLD + the `✦` spark.
            let label_style = Style::new()
                .fg(theme.palette.warn)
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::ITALIC);
            let verb = zo_reveal_verb(seed);
            // 4-phase cycle, width-stable: phase 0 reserves 3 trailing spaces so
            // the line never reflows as the dots grow/shrink. tick/6 → ~198ms
            // per phase, 792ms full cycle (~1.25Hz) — a calm, deliberate
            // "pondering" tempo rather than an anxious flicker.
            let dots = if reduce_motion_enabled() {
                "   " // settled: phase-0 placeholder, width-stable
            } else {
                match (tick / 6) % 4 {
                    0 => "   ",
                    1 => ".  ",
                    2 => ".. ",
                    _ => "...",
                }
            };
            let mut spans: Vec<Span<'_>> = Vec::new();
            if theme.no_color {
                spans.push(Span::raw("  "));
            }
            spans.push(Span::styled(format!("{spark} "), label_style));
            spans.push(Span::styled(format!("{verb}{dots}"), label_style));
            if let Some(d) = elapsed {
                spans.push(Span::styled(
                    format!(" · {}", format_reasoning_elapsed(d)),
                    meta_style,
                ));
            }
            return vec![Line::from(spans)];
        }

        let lead = if focused { "+ " } else { "  " };
        let summary = reasoning_summary_line(text);
        let summary = if summary.is_empty() {
            "worked".to_string()
        } else {
            summary
        };
        // One gradation below `dim` (v3 readability): the settled step line is
        // pure lead-in meta, and at `dim` it competed with the answer prose for
        // the eye — the bullet grammar has no header/rail left to separate
        // them, so the gradation gap must carry the hierarchy alone.
        let summary_style = Style::new()
            .fg(theme.palette.muted)
            .add_modifier(Modifier::DIM)
            .add_modifier(Modifier::ITALIC);
        let mut spans: Vec<Span<'_>> = Vec::new();
        spans.push(Span::styled(lead, Style::new().fg(theme.palette.dim)));
        spans.push(Span::styled(summary, summary_style));

        // Settled (done) collapsed step keeps its measured thinking time as a
        // permanent `· N.Ns` suffix. Streaming is handled by the animated
        // `✦ Thinking…` branch above, so no caret/dots fallback is needed here.
        if let Some(d) = elapsed {
            spans.push(Span::styled(
                format!(" · {}", format_reasoning_elapsed(d)),
                meta_style,
            ));
        }
        return vec![Line::from(spans)];
    }

    // Split into source lines only on the expanded path. The common collapsed
    // streaming path returns a single animated cue line, so doing this O(text)
    // split unconditionally cost two full-text scans per frame (this fn runs in
    // both `body_lines`→draw and `estimate_rows`→layout) while thinking streamed.
    let src_lines: Vec<&str> = text.split('\n').collect();
    let mut lines: Vec<Line<'_>> = Vec::with_capacity(src_lines.len().saturating_add(1));

    let header_style = Style::new()
        .fg(theme.palette.violet)
        .add_modifier(Modifier::BOLD)
        .add_modifier(Modifier::ITALIC);
    let mut header_spans = vec![Span::styled("work steps".to_string(), header_style)];
    if let Some(d) = elapsed {
        header_spans.push(Span::styled(
            format!(" · {}", format_reasoning_elapsed(d)),
            meta_style,
        ));
    } else if !done {
        // 폭이 흔들리지 않도록 tick%4==0 은 3-space 자리표시.
        let dots = if reduce_motion_enabled() {
            "   " // settled: phase-0 placeholder
        } else {
            match (tick / 5) % 4 {
                0 => "   ",
                1 => ".",
                2 => "..",
                _ => "...",
            }
        };
        header_spans.push(Span::styled(dots.to_string(), header_style));
    }
    lines.push(Line::from(header_spans));

    for (idx, body) in src_lines.iter().enumerate() {
        let mut spans: Vec<Span<'_>> = Vec::new();
        if theme.no_color && idx == 0 {
            spans.push(Span::raw(NO_COLOR_PREFIX));
        }
        spans.push(Span::styled(body.to_string(), body_style));
        lines.push(Line::from(spans));
    }
    lines
}

/// First non-empty reasoning line as a compact, CJK-safe, single-line title.
/// SSOT for the collapsed reasoning step header (done) and the activity/spinner
/// line (streaming), so a non-English — e.g. Korean — reasoning stream surfaces
/// the model's own topic sentence instead of collapsing to a hardcoded English
/// keyword bucket ("working"). Returns empty when no usable line has streamed
/// yet; the caller supplies any placeholder.
///
/// The first line grows monotonically as the model types, so echoing it reads as
/// the title typing out rather than flickering — no stateful debounce needed.
pub(crate) fn reasoning_summary_line(text: &str) -> String {
    const MAX_TITLE: usize = 72;
    let first = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .trim_matches(|ch: char| matches!(ch, '#' | '*' | '-' | '`' | '"' | '\''))
        .trim();
    // Drop control chars / stray backticks and collapse internal whitespace runs
    // so the one-line status stays clean regardless of the raw reasoning text.
    let mut title = String::with_capacity(first.len());
    let mut prev_space = false;
    for ch in first.chars() {
        // Whitespace first so a tab/other whitespace-control becomes a single
        // space (merging words on a dropped tab would be wrong); other control
        // chars and stray backticks are simply not appended.
        if ch.is_whitespace() {
            if !prev_space && !title.is_empty() {
                title.push(' ');
            }
            prev_space = true;
        } else if !ch.is_control() && ch != '`' {
            title.push(ch);
            prev_space = false;
        }
    }
    let title = title.trim_end();
    // CJK-safe: truncate by char, never by byte slice.
    if title.chars().count() > MAX_TITLE {
        let mut clipped: String = title.chars().take(MAX_TITLE.saturating_sub(1)).collect();
        clipped.push('…');
        return clipped;
    }
    title.to_string()
}


/// 사고 경과를 OpenCode 식으로 표기 — 1분 미만은 `2.7s` (소수 1자리),
/// 그 이상은 `1m 32s`. tool_call 의 `(26s)` 정수 괄호식과 달리 reasoning 은
/// 짧은 사고 버스트가 많아 소수 1자리가 더 또렷하다(components.md 디자인 의도).
fn format_reasoning_elapsed(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let total = d.as_secs();
        format!("{}m {}s", total / 60, total % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::Theme;

    /// 한 블록의 모든 span 텍스트를 한 줄로 이어붙인다.
    fn flat(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect()
    }

    #[test]
    fn long_settled_reasoning_keeps_one_collapsed_line_short_stays_hidden() {
        // Roadmap ③: a settled, collapsed reasoning block used to vanish to
        // height 0 unconditionally. Now a *long* think (>= COLLAPSED_DONE_MIN_ELAPSED)
        // keeps a one-line trace; short/unmeasured thinks stay hidden. Crucially
        // `estimate_rows` (measure) and `suppress_collapsed` (the draw gate) must
        // agree on the same elapsed threshold or scroll prefix-sums underflow.
        let theme = Theme::default_dark();
        // Streaming (done == false) is never suppressed — threshold irrelevant.
        assert!(!suppress_collapsed("x", false, false, None));
        // Collapsed + done + short (<3s) or unmeasured (None) → hidden (0 rows).
        assert!(suppress_collapsed(
            "짧은 생각",
            true,
            false,
            Some(Duration::from_millis(800))
        ));
        assert!(suppress_collapsed("x", true, false, None));
        assert_eq!(
            estimate_rows(
                "짧은 생각",
                true,
                &theme,
                false,
                false,
                40,
                Some(Duration::from_millis(800)),
                0
            ),
            0,
            "short settled thought stays hidden"
        );
        // Collapsed + done + long (>=3s) → kept as exactly one trace line, and the
        // measure path returns that same single row (no measure/draw divergence).
        assert!(!suppress_collapsed(
            "인증 흐름 분석",
            true,
            false,
            Some(Duration::from_secs(4))
        ));
        assert_eq!(
            estimate_rows(
                "인증 흐름 분석",
                true,
                &theme,
                false,
                false,
                40,
                Some(Duration::from_secs(4)),
                0
            ),
            1,
            "long settled thought keeps exactly one collapsed trace line"
        );
        // Expanded is always shown regardless of elapsed.
        assert!(!suppress_collapsed(
            "x",
            true,
            true,
            Some(Duration::from_millis(100))
        ));
    }

    #[test]
    fn reasoning_summary_echoes_korean_first_line_not_working_bucket() {
        // The regression: a Korean (non-English) reasoning stream used to collapse
        // to a hardcoded English "working" bucket. Now it echoes the model's own
        // first line verbatim.
        assert_eq!(
            reasoning_summary_line("인증 흐름을 먼저 살펴보자\n다음으로 토큰 갱신을 확인"),
            "인증 흐름을 먼저 살펴보자"
        );
        // English keeps echoing its real first line too (no bucket).
        assert_eq!(
            reasoning_summary_line("Let me trace the auth flow first\nthen tokens"),
            "Let me trace the auth flow first"
        );
        // Markdown lead chars and stray backticks/control chars are stripped.
        assert_eq!(
            reasoning_summary_line("## `Plan`: 모듈\t분리"),
            "Plan: 모듈 분리"
        );
        // Empty / whitespace-only → empty (caller supplies a placeholder).
        assert!(reasoning_summary_line("   \n  ").is_empty());
    }

    #[test]
    fn reasoning_summary_truncates_by_char_not_byte_for_cjk() {
        // A long CJK first line must truncate on a char boundary (never a byte
        // slice, which would panic or corrupt a multi-byte char) and end with '…'.
        let long = "가".repeat(100);
        let title = reasoning_summary_line(&long);
        assert!(title.ends_with('…'), "truncated title ends with ellipsis: {title:?}");
        assert_eq!(title.chars().count(), 72, "clamped to MAX_TITLE chars");
        // Round-trips as valid UTF-8 (no byte-slice corruption).
        assert!(title.chars().all(|c| c == '가' || c == '…'));
    }

    #[test]
    fn format_elapsed_sub_minute_uses_one_decimal() {
        assert_eq!(
            format_reasoning_elapsed(Duration::from_millis(2700)),
            "2.7s"
        );
        assert_eq!(format_reasoning_elapsed(Duration::from_millis(400)), "0.4s");
        assert_eq!(
            format_reasoning_elapsed(Duration::from_millis(6700)),
            "6.7s"
        );
        assert_eq!(format_reasoning_elapsed(Duration::from_secs(59)), "59.0s");
    }

    #[test]
    fn format_elapsed_over_minute_uses_m_s() {
        assert_eq!(format_reasoning_elapsed(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_reasoning_elapsed(Duration::from_secs(92)), "1m 32s");
    }

    #[test]
    fn collapsed_done_reasoning_shows_permanent_elapsed() {
        let theme = Theme::default_dark();
        // done == true, elapsed 동결값이 있으면 "· 2.7s" 가 영구 표기된다.
        let lines = body_lines(
            "Considering project analysis",
            true,
            &theme,
            false,
            false,
            0,
            Some(Duration::from_millis(2700)),
            0,
        );
        let text = flat(&lines);
        assert!(
            !text.contains("step ·"),
            "settled reasoning should read as quiet metadata: {text:?}"
        );
        assert!(
            text.contains("Considering project analysis"),
            "summary present: {text:?}"
        );
        assert!(
            text.contains("· 2.7s"),
            "permanent elapsed suffix: {text:?}"
        );
    }

    #[test]
    fn collapsed_streaming_shows_animated_reasoning_line() {
        let theme = Theme::default_dark();
        // !done → live animated reasoning cue, not the raw partial reasoning.
        // seed 0 → `ZO_REVEAL_VERBS[0]` == "Thinking". tick=6 → (6/6)%4 == 1
        // → one dot (".  ", width stable). No elapsed suffix when timing is
        // unmeasured.
        let text = flat(&body_lines(
            "thinking", false, &theme, false, false, 6, None, 0,
        ));
        assert_eq!(
            text, "\u{2726} Thinking.  ",
            "streaming shows the animated reasoning line"
        );
    }

    #[test]
    fn streaming_zo_verb_is_stable_per_seed_and_varies_across_blocks() {
        let theme = Theme::default_dark();
        // The verb is keyed by the block seed, not the tick — so it stays fixed
        // as the dots animate (no per-frame flicker) and different reasoning
        // blocks surface different words. Same seed, different ticks → same verb.
        let at = |tick, seed| {
            flat(&body_lines(
                "x", false, &theme, false, false, tick, None, seed,
            ))
        };
        let early = at(0, 3);
        let later = at(30, 3);
        assert_eq!(
            early.trim_end_matches([' ', '.']),
            later.trim_end_matches([' ', '.']),
            "the live reasoning verb is fixed across ticks for one block: {early:?} vs {later:?}"
        );
        // Every seed maps onto one of the SSOT verbs (drawn after the `✦ `).
        for seed in 0..(ZO_REVEAL_VERBS.len() as u64) {
            let line = at(0, seed);
            let verb = zo_reveal_verb(seed);
            assert!(
                line.contains(verb),
                "seed {seed} renders its live reasoning verb {verb:?}: {line:?}"
            );
        }
        // Distinct seeds across the set surface distinct verbs (variety reads).
        let verbs: std::collections::HashSet<_> = (0..ZO_REVEAL_VERBS.len() as u64)
            .map(zo_reveal_verb)
            .collect();
        assert_eq!(
            verbs.len(),
            ZO_REVEAL_VERBS.len(),
            "each slot is a distinct verb"
        );
    }

    #[test]
    fn collapsed_streaming_zo_carries_elapsed_when_timed() {
        let theme = Theme::default_dark();
        // done == false, tick = 0 → phase 0 dots are 3 spaces (width stable);
        // measured timing rides as `· 1.5s`. seed 0 → "Thinking".
        let lines = body_lines(
            "thinking",
            false,
            &theme,
            false,
            false,
            0,
            Some(Duration::from_millis(1500)),
            0,
        );
        let text = flat(&lines);
        assert!(
            text.starts_with("\u{2726} Thinking") && text.contains("· 1.5s"),
            "streaming reasoning line carries elapsed: {text:?}"
        );
        assert!(
            !text.contains("thinking") && !text.contains("step"),
            "raw reasoning text and the settled `step` label are hidden while streaming: {text:?}"
        );
    }

    #[test]
    fn collapsed_done_without_timing_is_quiet() {
        let theme = Theme::default_dark();
        // done && elapsed == None → 점·시간 모두 없음(기존 무소음 동작 보존).
        let text = flat(&body_lines(
            "done thought",
            true,
            &theme,
            false,
            false,
            0,
            None,
            0,
        ));
        assert_eq!(
            text, "  done thought",
            "no dots/elapsed suffix when done & untimed"
        );
    }
}
