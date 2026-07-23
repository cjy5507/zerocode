use std::path::PathBuf;
use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

use super::glyphs;
use super::theme::Theme;

/// Full launchpad height for the large ignition masthead.
const STARTUP_HEIGHT: u16 = 19;
/// Compact height for narrow terminals.
const STARTUP_HEIGHT_PLAIN: u16 = 12;
/// Short label for the startup summary suggestion shown in the launchpad.
pub const STARTUP_SUMMARIZE_REPO_LABEL: &str = "summarize this repo";
/// Prompt inserted by the startup summary shortcut.
pub const STARTUP_SUMMARIZE_REPO_PROMPT: &str =
    "Summarize this repository: architecture, main crates, and how to verify changes.";
/// Login command inserted by the startup Claude shortcut.
pub const STARTUP_LOGIN_CLAUDE_COMMAND: &str = "/login claude";
/// Login command inserted by the startup OpenAI shortcut.
pub const STARTUP_LOGIN_OPENAI_COMMAND: &str = "/login openai";
/// Permission command inserted by the startup permission shortcut.
pub const STARTUP_PERMISSIONS_COMMAND: &str = "/permissions";
/// Width below which the launchpad uses the compact information row set.
const RICH_MIN_WIDTH: u16 = 58;
/// Width below which the masthead falls back to the compact one-line wordmark.
const LARGE_MASTHEAD_MIN_WIDTH: u16 = 44;
/// Height below which the masthead falls back to the compact one-line wordmark.
const LARGE_MASTHEAD_MIN_HEIGHT: u16 = 18;
/// Width where the launchpad has enough room for a two-column command surface.
const DENSE_MIN_WIDTH: u16 = 72;
/// Wide launchpad starts filling more of the chat column before capping.
const WIDE_STARTUP_WIDTH: u16 = 132;
/// Absolute max width so the startup does not collide visually with side panels.
const MAX_STARTUP_WIDTH: u16 = 180;

/// Duration of the one-shot ignition intro. The existing Launchpad clock keeps
/// the 33 ms animation cadence alive only until this bounded sequence settles.
pub const INTRO_TOTAL_MS: u64 = 700;

const INTRO_BUCKET_MS: u64 = 33;
const INTRO_SPARK_END_MS: u64 = 250;
const INTRO_SWEEP_END_MS: u64 = 550;
const INTRO_FADE_STEP_MS: u64 = 625;
const LARGE_WORDMARK_ART: [&str; 3] = ["▰▰▰▰▰ ▰▰▰▰▰", "  ▰▰  ▰▰ ▰▰", "▰▰▰▰▰ ▰▰▰▰▰"];
const LARGE_WORDMARK_SHADOW: char = '░';
const LARGE_WORDMARK_RAIL: char = '▌';
const LARGE_WORDMARK_SPARK: char = '✦';

/// A recent session surfaced on the startup launchpad. `label` is the
/// pre-truncated display title (first user line or id prefix); `age` is a
/// short relative-time string (e.g. `3m-ago`). Both are computed by the
/// caller so `startup.rs` stays a pure renderer.
#[derive(Debug, Clone)]
pub struct RecentSession {
    pub label: String,
    pub age: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StartupAuthState {
    pub anthropic_oauth: bool,
    pub chatgpt_oauth: bool,
}

impl StartupAuthState {
    #[must_use]
    pub const fn needs_onboarding(self) -> bool {
        !self.anthropic_oauth || !self.chatgpt_oauth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupOnboardingStage {
    NeedsProvider,
    ReturningUser,
    ReadyWithSuggestions,
}

#[derive(Debug, Clone)]
pub struct StartupScreen {
    pub version: String,
    pub model: String,
    pub permissions: String,
    pub branch: String,
    pub workspace: String,
    pub directory: PathBuf,
    pub project_root: Option<PathBuf>,
    pub session_id: String,
    pub autosave_path: PathBuf,
    pub startup_ms: Option<u128>,
    pub memory_mb: Option<f64>,
    pub auth: StartupAuthState,
    /// Most-recent resumable sessions (newest first), already trimmed to a
    /// small N by the caller. Empty ⇒ the launchpad section is omitted.
    pub recent_sessions: Vec<RecentSession>,
}

/// Banner height for `width`. The rich launchpad is taller than the plain
/// fallback so the caller can reserve the right amount of sticky-banner
/// space once messages start scrolling underneath.
#[must_use]
pub fn preferred_height(width: u16) -> u16 {
    if width < LARGE_MASTHEAD_MIN_WIDTH {
        STARTUP_HEIGHT_PLAIN
    } else {
        STARTUP_HEIGHT
    }
}

/// Render the startup launchpad into `area`.
///
/// `intro` is quantized to the app's 33 ms animation cadence. `None` is the
/// reduce-motion and snapshot path and renders the settled masthead immediately.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    startup: &StartupScreen,
    theme: &Theme,
    intro: Option<Duration>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let content_area = startup_content_area(area);
    if content_area.width == 0 || content_area.height == 0 {
        return;
    }

    let compact = content_area.width < RICH_MIN_WIDTH
        || content_area.height < LARGE_MASTHEAD_MIN_HEIGHT;
    let compact_masthead = area.width < LARGE_MASTHEAD_MIN_WIDTH
        || area.height < LARGE_MASTHEAD_MIN_HEIGHT;
    let mut lines: Vec<Line<'_>> =
        Vec::with_capacity(usize::from(preferred_height(content_area.width)));

    lines.extend(masthead_lines(
        theme,
        compact_masthead,
        content_area.width,
        intro,
    ));
    lines.push(info_line(startup, theme));
    lines.push(workspace_line(startup, theme));
    if !compact {
        lines.push(path_line(startup, theme));
    }
    lines.push(Line::from(""));

    let stage = onboarding_stage(startup);
    render_stage_lines(
        &mut lines,
        content_area.height,
        content_area.width,
        compact,
        stage,
        startup,
        theme,
    );

    let paragraph = Paragraph::new(lines).alignment(Alignment::Left);
    frame.render_widget(paragraph, content_area);
}

fn startup_content_area(area: Rect) -> Rect {
    let mut out = area;
    let preferred = preferred_height(area.width).min(area.height);
    if area.height > preferred.saturating_add(4) {
        let pad = ((area.height - preferred) / 5).clamp(1, 3);
        out.y = out.y.saturating_add(pad);
        out.height = out.height.saturating_sub(pad).min(preferred);
    }
    let target_width = startup_content_width(out.width);
    if out.width > target_width {
        let horizontal_pad = (out.width - target_width) / 2;
        out.x = out.x.saturating_add(horizontal_pad);
        out.width = target_width;
    }
    out
}

fn startup_content_width(width: u16) -> u16 {
    if width <= WIDE_STARTUP_WIDTH {
        return width;
    }
    let responsive = width.saturating_mul(3) / 4;
    responsive.clamp(WIDE_STARTUP_WIDTH, MAX_STARTUP_WIDTH).min(width)
}

fn onboarding_stage(startup: &StartupScreen) -> StartupOnboardingStage {
    if startup.auth.needs_onboarding() {
        StartupOnboardingStage::NeedsProvider
    } else if startup.recent_sessions.is_empty() {
        StartupOnboardingStage::ReadyWithSuggestions
    } else {
        StartupOnboardingStage::ReturningUser
    }
}

fn render_stage_lines(
    lines: &mut Vec<Line<'_>>,
    area_height: u16,
    width: u16,
    compact: bool,
    stage: StartupOnboardingStage,
    startup: &StartupScreen,
    theme: &Theme,
) {
    let dense = !compact && width >= DENSE_MIN_WIDTH;
    match stage {
        StartupOnboardingStage::NeedsProvider => {
            for line in auth_onboarding_lines(startup.auth, theme) {
                lines.push(line);
            }
            lines.push(Line::from(""));
            lines.push(quickstart_line(theme));
        }
        StartupOnboardingStage::ReturningUser if dense => {
            render_returning_dense_lines(lines, startup, theme, width);
        }
        StartupOnboardingStage::ReturningUser => {
            lines.push(continue_header_line(theme));
            let max_sessions = returning_user_session_capacity(area_height, lines.len(), compact);
            for session in startup.recent_sessions.iter().take(max_sessions) {
                lines.push(recent_session_line(session, theme));
            }
            lines.push(resume_hint_line(theme));
            lines.push(Line::from(""));
            lines.push(quickstart_line(theme));
        }
        StartupOnboardingStage::ReadyWithSuggestions if dense => {
            render_ready_dense_lines(lines, startup, theme, width);
        }
        StartupOnboardingStage::ReadyWithSuggestions => {
            lines.push(ready_line(theme));
            lines.push(quickstart_line(theme));
        }
    }
    if dense && stage != StartupOnboardingStage::NeedsProvider {
        lines.push(Line::from(""));
        lines.push(session_context_line(startup, theme, width));
    }
    lines.push(hints_line(theme));
}

fn returning_user_session_capacity(
    area_height: u16,
    lines_after_header: usize,
    compact: bool,
) -> usize {
    // After the recent-session rows, returning users still need these rows to
    // keep the launchpad actionable: `/resume`, a spacer, the first-task
    // suggestion, and the global hints line. Fit sessions into whatever height
    // remains instead of letting narrow/plain banners clip the CTA footer, then
    // cap density so compact launchpads do not become a session list.
    const RETURNING_FOOTER_ROWS: usize = 4;
    let fits = usize::from(area_height)
        .saturating_sub(lines_after_header)
        .saturating_sub(RETURNING_FOOTER_ROWS);
    fits.min(if compact { 1 } else { 2 })
}

// ============================================================================
// Ignition masthead — cold steel at rest, hot core while entering the zone.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntroChrome {
    Hidden,
    Dim,
    Settled,
}

fn masthead_lines(
    theme: &Theme,
    compact: bool,
    width: u16,
    intro: Option<Duration>,
) -> Vec<Line<'static>> {
    if compact {
        return vec![brand_line(theme, true, width)];
    }
    let elapsed = quantized_intro_ms(intro);
    let chrome = intro_chrome(elapsed);
    let mut lines = large_wordmark_lines(theme, width, elapsed, chrome);
    lines.push(tagline_line(theme, chrome));
    lines.push(divider_line(theme, width, chrome));
    lines
}

fn quantized_intro_ms(intro: Option<Duration>) -> Option<u64> {
    let elapsed = intro?;
    let millis = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    if millis >= INTRO_TOTAL_MS {
        None
    } else {
        Some((millis / INTRO_BUCKET_MS) * INTRO_BUCKET_MS)
    }
}

fn intro_chrome(elapsed: Option<u64>) -> IntroChrome {
    match elapsed {
        Some(ms) if ms < INTRO_SWEEP_END_MS => IntroChrome::Hidden,
        Some(ms) if ms < INTRO_FADE_STEP_MS => IntroChrome::Dim,
        _ => IntroChrome::Settled,
    }
}

fn large_wordmark_lines(
    theme: &Theme,
    width: u16,
    elapsed: Option<u64>,
    chrome: IntroChrome,
) -> Vec<Line<'static>> {
    let art = LARGE_WORDMARK_ART
        .iter()
        .map(|row| row.chars().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let ncols = art.iter().map(Vec::len).max().unwrap_or_default();
    let gradient = theme.heat().wordmark_gradient(ncols);
    let rendered_width = ncols.saturating_add(1);
    let chip = "launchpad";
    let available = usize::from(width.saturating_sub(2));
    let brand_width = rendered_width.saturating_add(2);
    let gap = available
        .saturating_sub(brand_width.saturating_add(chip.len()))
        .max(2);
    let mut lines = Vec::with_capacity(4);

    for row in 0..4 {
        let mut spans = Vec::with_capacity(rendered_width.saturating_add(4));
        if row == 0 {
            spans.push(Span::styled(
                LARGE_WORDMARK_RAIL.to_string(),
                style(theme.heat().ember, true),
            ));
            spans.push(Span::raw(" "));
        } else {
            spans.push(Span::raw("  "));
        }
        for col in 0..rendered_width {
            if row == 0 && col == 0 && elapsed.is_some_and(|ms| ms < INTRO_SPARK_END_MS) {
                spans.push(Span::styled(
                    LARGE_WORDMARK_SPARK.to_string(),
                    style(spark_ramp_color(theme, elapsed.unwrap_or_default()), true),
                ));
                continue;
            }
            let foreground = art.get(row).and_then(|line| line.get(col)).copied();
            if let Some(glyph) = foreground.filter(|glyph| *glyph != ' ') {
                if let Some(glyph_style) = wordmark_column_style(theme, &gradient, col, ncols, elapsed)
                {
                    spans.push(Span::styled(glyph.to_string(), glyph_style));
                    continue;
                }
            }
            let cast_from_above = row
                .checked_sub(1)
                .and_then(|source_row| art.get(source_row))
                .and_then(|line| col.checked_sub(1).and_then(|source_col| line.get(source_col)))
                .is_some_and(|glyph| *glyph != ' ');
            // The extrusion only falls on the outer lower-right silhouette:
            // inside a counter or an inter-letter gap the row still has glyphs
            // to the right, and shade there reads as noise, not depth.
            let outside_glyph_run = art
                .get(row)
                .is_none_or(|line| line.iter().skip(col).all(|glyph| *glyph == ' '));
            if cast_from_above
                && outside_glyph_run
                && shadow_column_visible(col.saturating_sub(1), ncols, elapsed)
            {
                spans.push(Span::styled(
                    LARGE_WORDMARK_SHADOW.to_string(),
                    Style::new()
                        .fg(theme.heat().steel_dim)
                        .add_modifier(Modifier::DIM),
                ));
            } else {
                spans.push(Span::raw(" "));
            }
        }
        if row == 0 {
            spans.push(Span::raw(" ".repeat(gap)));
            let chip_style = match chrome {
                IntroChrome::Hidden => None,
                IntroChrome::Dim => Some(theme.typography.dim),
                IntroChrome::Settled => Some(style(theme.palette.accent, true)),
            };
            if let Some(chip_style) = chip_style {
                spans.push(Span::styled(chip.to_string(), chip_style));
            } else {
                spans.push(Span::raw(" ".repeat(chip.len())));
            }
        }
        lines.push(indented(spans));
    }
    lines
}

fn wordmark_column_style(
    theme: &Theme,
    gradient: &[Color],
    col: usize,
    ncols: usize,
    elapsed: Option<u64>,
) -> Option<Style> {
    let settled = || style(gradient.get(col).copied().unwrap_or(theme.heat().molten), true);
    match elapsed {
        None => Some(settled()),
        Some(ms) if ms < INTRO_SPARK_END_MS => None,
        Some(ms) if ms >= INTRO_SWEEP_END_MS => Some(settled()),
        Some(ms) => {
            let reveal_at = column_reveal_ms(col, ncols);
            if ms < reveal_at {
                None
            } else if ms < reveal_at.saturating_add(INTRO_BUCKET_MS) {
                Some(style(theme.heat().spark, true))
            } else {
                Some(settled())
            }
        }
    }
}

fn shadow_column_visible(col: usize, ncols: usize, elapsed: Option<u64>) -> bool {
    match elapsed {
        None => true,
        Some(ms) if ms >= INTRO_SWEEP_END_MS => true,
        Some(ms) if ms < INTRO_SPARK_END_MS => false,
        Some(ms) => ms
            >= column_reveal_ms(col, ncols)
                .saturating_add(INTRO_BUCKET_MS),
    }
}

fn column_reveal_ms(col: usize, ncols: usize) -> u64 {
    let columns = u64::try_from(ncols).unwrap_or(1).max(1);
    let col = u64::try_from(col).unwrap_or(u64::MAX);
    INTRO_SPARK_END_MS.saturating_add(col.saturating_mul(300 / columns))
}

fn spark_ramp_color(theme: &Theme, elapsed: u64) -> Color {
    let ramp = &theme.heat().ramp;
    let last = ramp.len().saturating_sub(1);
    let last_bucket = ((INTRO_SPARK_END_MS - 1) / INTRO_BUCKET_MS) * INTRO_BUCKET_MS;
    let index = usize::try_from(elapsed.saturating_mul(u64::try_from(last).unwrap_or(0))
        / last_bucket.max(1))
        .unwrap_or(last)
        .min(last);
    ramp[index]
}

fn tagline_line(theme: &Theme, chrome: IntroChrome) -> Line<'static> {
    const TAGLINE: &str = "in the zone — AI pair-programming for this repo";
    let tagline_style = match chrome {
        IntroChrome::Hidden => None,
        IntroChrome::Dim => Some(theme.typography.dim),
        IntroChrome::Settled => Some(theme.typography.body),
    };
    match tagline_style {
        Some(tagline_style) => indented(vec![Span::styled(TAGLINE.to_string(), tagline_style)]),
        None => Line::from(""),
    }
}

fn divider_line(theme: &Theme, width: u16, chrome: IntroChrome) -> Line<'static> {
    let rule_width = usize::from(width.saturating_sub(2));
    let divider_style = match chrome {
        IntroChrome::Hidden => None,
        IntroChrome::Dim => Some(theme.typography.dim),
        IntroChrome::Settled => Some(
            Style::new()
                .fg(theme.heat().steel_dim)
                .add_modifier(Modifier::DIM),
        ),
    };
    match divider_style {
        Some(divider_style) => indented(vec![Span::styled("─".repeat(rule_width), divider_style)]),
        None => Line::from(""),
    }
}

fn indented(spans: Vec<Span<'static>>) -> Line<'static> {
    let mut out = Vec::with_capacity(spans.len() + 1);
    out.push(Span::raw("  "));
    out.extend(spans);
    Line::from(out)
}

// ============================================================================
// Shared info / path / hint rows.
// ============================================================================

fn info_line(startup: &StartupScreen, theme: &Theme) -> Line<'static> {
    let nc = theme.no_color;
    let permission = permission_label_for_display(&startup.permissions);
    let model_label = compact_model_label(&startup.model).to_string();

    let git_icon = if nc {
        glyphs::GIT_BRANCH_NC
    } else {
        glyphs::GIT_BRANCH
    };
    let lock_icon = if nc {
        glyphs::PERMISSION_LOCK_NC
    } else {
        glyphs::PERMISSION_LOCK
    };
    // Primary separator: single-space middot keeps the meta row tight so
    // the logo stays the visual anchor. The startup-time tail uses an even
    // fainter separator + DIM text to drop a clear tier below the meta.
    let sep = style(theme.palette.muted, false);
    let sep_faint = Style::new()
        .fg(theme.palette.faint)
        .add_modifier(Modifier::DIM);

    // 메타 채도 축소(D1 단일 액센트 수렴): 종전 model=cyan·mode=warn·git=teal
    // 의 3색 경쟁을 모노크롬 + 앰버 1색으로 좁힌다. 모드 배지만 따뜻한 accent
    // 로 띄워 "대화 상단" 모드 표시 역할을 하고(입력 하단 HUD 와 짝), model 은
    // 중립 fg, git 은 차분한 dim 으로 물러난다.
    let mut spans = vec![
        Span::styled(model_label, style(theme.palette.fg, false)),
        Span::styled(" \u{00b7} ", sep),
        Span::styled(format!("{lock_icon} "), style(theme.palette.accent, false)),
        Span::styled(permission.to_string(), style(theme.palette.accent, true)),
        Span::styled(" \u{00b7} ", sep),
        Span::styled(format!("{git_icon} "), style(theme.palette.dim, false)),
        Span::styled(startup.branch.clone(), style(theme.palette.dim, false)),
    ];
    if let Some(ms) = startup.startup_ms {
        spans.push(Span::styled(" \u{00b7} ", sep_faint));
        spans.push(Span::styled(format!("{ms}ms"), sep_faint));
    }
    indented(spans)
}

fn brand_line(theme: &Theme, compact: bool, width: u16) -> Line<'static> {
    let rail = if theme.no_color { ">" } else { "▌" };
    let label = Style::new()
        .fg(theme.palette.accent)
        .add_modifier(Modifier::BOLD);
    let wordmark = "zo";
    let wordmark_colors = theme.heat().wordmark_gradient(wordmark.chars().count());
    let brand_width = 7; // rail + space + ZO
    let chip = "launchpad";
    let available = usize::from(width.saturating_sub(2));
    let gap = if compact {
        2
    } else {
        available
            .saturating_sub(brand_width + chip.len())
            .max(2)
    };
    let mut spans = vec![
        Span::styled(rail.to_string(), style(theme.heat().ember, true)),
        Span::raw(" "),
    ];
    spans.extend(
        wordmark
            .chars()
            .zip(wordmark_colors)
            .map(|(character, color)| Span::styled(character.to_string(), style(color, true))),
    );
    spans.push(Span::raw(" ".repeat(gap)));
    spans.push(Span::styled(chip.to_string(), label));
    indented(spans)
}

fn workspace_line(startup: &StartupScreen, theme: &Theme) -> Line<'static> {
    let label = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    let value = style(theme.heat().steel, false);
    let workspace = if startup.workspace.is_empty() {
        startup
            .project_root
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("workspace")
            .to_string()
    } else {
        startup.workspace.clone()
    };
    indented(vec![
        Span::styled("workspace ".to_string(), label),
        Span::styled(workspace, value),
    ])
}

fn quickstart_line(theme: &Theme) -> Line<'static> {
    let prompt = style(theme.palette.accent, true);
    let label = Style::new()
        .fg(theme.heat().steel)
        .add_modifier(Modifier::DIM);
    let divider = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    indented(vec![
        Span::styled("Alt+S ".to_string(), label),
        Span::styled(STARTUP_SUMMARIZE_REPO_LABEL.to_string(), prompt),
        Span::styled("  ·  ".to_string(), divider),
        Span::styled("review my diff".to_string(), prompt),
        Span::styled("  ·  ".to_string(), divider),
        Span::styled("find failing tests".to_string(), prompt),
    ])
}

fn path_line(startup: &StartupScreen, theme: &Theme) -> Line<'static> {
    indented(vec![Span::styled(
        startup.directory.display().to_string(),
        Style::new()
            .fg(theme.heat().steel)
            .add_modifier(Modifier::DIM),
    )])
}

fn auth_onboarding_lines(auth: StartupAuthState, theme: &Theme) -> Vec<Line<'static>> {
    let title = style(theme.palette.accent, true);
    let muted = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    let label = Style::new().fg(theme.heat().steel);
    vec![
        indented(vec![
            Span::styled("Get ready".to_string(), title),
            Span::styled(" in 60 seconds".to_string(), muted),
        ]),
        Line::from(auth_provider_step_spans(auth, theme)),
        indented(vec![
            Span::styled("2. Confirm access  ".to_string(), muted),
            Span::styled("Alt+P ".to_string(), muted),
            Span::styled(STARTUP_PERMISSIONS_COMMAND.to_string(), label),
            Span::styled(" explains the current mode".to_string(), muted),
        ]),
        indented(vec![
            Span::styled("3. Start a task    ".to_string(), muted),
            Span::styled("pick a suggestion below".to_string(), label),
        ]),
    ]
}

fn auth_provider_step_spans(auth: StartupAuthState, theme: &Theme) -> Vec<Span<'static>> {
    let muted = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    let divider = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    let mut spans = vec![Span::raw("  "), Span::styled("1. Connect provider  ".to_string(), muted)];
    spans.extend(provider_action_spans(
        "Claude",
        "Alt+C",
        STARTUP_LOGIN_CLAUDE_COMMAND,
        auth.anthropic_oauth,
        theme,
    ));
    spans.push(Span::styled("  ·  ".to_string(), divider));
    spans.extend(provider_action_spans(
        "OpenAI",
        "Alt+O",
        STARTUP_LOGIN_OPENAI_COMMAND,
        auth.chatgpt_oauth,
        theme,
    ));
    spans
}

fn provider_action_spans(
    label: &str,
    shortcut: &str,
    command: &str,
    connected: bool,
    theme: &Theme,
) -> Vec<Span<'static>> {
    if connected {
        vec![
            Span::styled(label.to_string(), style(theme.palette.fg, true)),
            Span::styled(" connected".to_string(), style(theme.palette.success, true)),
        ]
    } else {
        let hint = Style::new()
            .fg(theme.heat().steel)
            .add_modifier(Modifier::DIM);
        vec![
            Span::styled(format!("{shortcut} "), hint),
            Span::styled(command.to_string(), style(theme.palette.accent, true)),
        ]
    }
}

fn hints_line(theme: &Theme) -> Line<'static> {
    let nc = theme.no_color;
    let enter_glyph = if nc {
        glyphs::KEY_ENTER_NC
    } else {
        glyphs::KEY_ENTER
    };
    let key = Style::new()
        .fg(theme.heat().steel)
        .add_modifier(Modifier::DIM);
    let label = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    let divider = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    indented(vec![
        Span::styled(enter_glyph.to_string(), key),
        Span::styled(" submit".to_string(), label),
        Span::styled("  •  ".to_string(), divider),
        Span::styled("/help".to_string(), key),
        Span::styled(" commands".to_string(), label),
        Span::styled("  •  ".to_string(), divider),
        Span::styled("ctrl+p".to_string(), key),
        Span::styled(" model".to_string(), label),
        Span::styled("  •  ".to_string(), divider),
        Span::styled("ctrl+b".to_string(), key),
        Span::styled(" sidebar".to_string(), label),
        Span::styled("  •  ".to_string(), divider),
        Span::styled("esc".to_string(), key),
        Span::styled(" quit".to_string(), label),
    ])
}

fn ready_line(theme: &Theme) -> Line<'static> {
    let title = style(theme.palette.success, true);
    let label = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    indented(vec![
        Span::styled("Ready".to_string(), title),
        Span::styled(" — start with a task below".to_string(), label),
    ])
}

fn continue_header_line(theme: &Theme) -> Line<'static> {
    let title = style(theme.palette.accent, true);
    let label = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    indented(vec![
        Span::styled("Continue recent work".to_string(), title),
        Span::styled(" or start fresh".to_string(), label),
    ])
}

/// One recent-session row: a dim bullet, the (pre-truncated) title, then a
/// faint relative-age suffix — e.g. `· refactor the scroll handler  3m-ago`.
fn recent_session_line(session: &RecentSession, theme: &Theme) -> Line<'static> {
    let bullet = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    let title = style(theme.heat().steel, false);
    let age = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    indented(vec![
        Span::styled("\u{00b7} ".to_string(), bullet),
        Span::styled(session.label.clone(), title),
        Span::styled(format!("  {}", session.age), age),
    ])
}

/// Recommended next action shown beneath the recent-session list.
fn resume_hint_line(theme: &Theme) -> Line<'static> {
    let key = Style::new()
        .fg(theme.heat().steel)
        .add_modifier(Modifier::DIM);
    let label = Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM);
    indented(vec![
        Span::styled("/resume".to_string(), key),
        Span::styled(" continue".to_string(), label),
    ])
}

fn render_returning_dense_lines(
    lines: &mut Vec<Line<'_>>,
    startup: &StartupScreen,
    theme: &Theme,
    width: u16,
) {
    lines.push(two_column_line(
        width,
        vec![
            Span::styled("Continue recent work".to_string(), style(theme.palette.accent, true)),
            Span::styled(" or start fresh".to_string(), dim_label(theme)),
        ],
        vec![Span::styled("Start fast".to_string(), style(theme.palette.accent, true))],
    ));

    let mut sessions = startup.recent_sessions.iter();
    let first = sessions.next().map_or_else(
        || vec![Span::styled("No recent sessions".to_string(), dim_label(theme))],
        |session| recent_session_spans(session, theme),
    );
    let second = sessions.next().map_or_else(
        || resume_hint_spans(theme),
        |session| recent_session_spans(session, theme),
    );

    lines.push(two_column_line(width, first, summarize_action_spans(theme)));
    lines.push(two_column_line(width, second, action_spans("review my diff", theme)));
    lines.push(two_column_line(width, resume_hint_spans(theme), action_spans("find failing tests", theme)));
}

fn render_ready_dense_lines(
    lines: &mut Vec<Line<'_>>,
    startup: &StartupScreen,
    theme: &Theme,
    width: u16,
) {
    lines.push(two_column_line(
        width,
        vec![
            Span::styled("Ready".to_string(), style(theme.palette.success, true)),
            Span::styled(" — start with a task below".to_string(), dim_label(theme)),
        ],
        vec![Span::styled("Suggested prompts".to_string(), style(theme.palette.accent, true))],
    ));
    lines.push(two_column_line(
        width,
        workspace_summary_spans(startup, theme),
        summarize_action_spans(theme),
    ));
    lines.push(two_column_line(
        width,
        vec![Span::styled("No resume queue — fresh session".to_string(), dim_label(theme))],
        action_spans("review my diff", theme),
    ));
    lines.push(two_column_line(
        width,
        vec![Span::styled("Type a task or pick a prompt".to_string(), dim_label(theme))],
        action_spans("find failing tests", theme),
    ));
}

fn two_column_line(
    width: u16,
    left: Vec<Span<'static>>,
    right: Vec<Span<'static>>,
) -> Line<'static> {
    let available = usize::from(width.saturating_sub(2));
    let left_column = (available / 2).clamp(32, 72);
    let left_width = span_width(&left);
    let gap = left_column.saturating_sub(left_width).max(4);
    let mut spans = Vec::with_capacity(left.len() + right.len() + 2);
    spans.push(Span::raw("  "));
    spans.extend(left);
    spans.push(Span::raw(" ".repeat(gap)));
    spans.extend(right);
    Line::from(spans)
}

fn span_width(spans: &[Span<'_>]) -> usize {
    spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum()
}

fn recent_session_spans(session: &RecentSession, theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled("· ".to_string(), muted_label(theme)),
        Span::styled(session.label.clone(), style(theme.heat().steel, false)),
        Span::styled(format!("  {}", session.age), faint_label(theme)),
    ]
}

fn resume_hint_spans(theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled("/resume".to_string(), style(theme.heat().steel, false)),
        Span::styled(" continue".to_string(), faint_label(theme)),
    ]
}

fn summarize_action_spans(theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled("Alt+S ".to_string(), dim_label(theme)),
        Span::styled(
            STARTUP_SUMMARIZE_REPO_LABEL.to_string(),
            style(theme.palette.accent, true),
        ),
    ]
}

fn action_spans(label: &str, theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled("› ".to_string(), muted_label(theme)),
        Span::styled(label.to_string(), style(theme.palette.accent, true)),
    ]
}

fn workspace_summary_spans(startup: &StartupScreen, theme: &Theme) -> Vec<Span<'static>> {
    let workspace = if startup.workspace.is_empty() {
        "workspace".to_string()
    } else {
        startup.workspace.clone()
    };
    vec![
        Span::styled("workspace ".to_string(), faint_label(theme)),
        Span::styled(workspace, style(theme.heat().steel, false)),
    ]
}

fn session_context_line(startup: &StartupScreen, theme: &Theme, width: u16) -> Line<'static> {
    let short_session = startup.session_id.chars().take(12).collect::<String>();
    let mut right = vec![
        Span::styled(format!("v{}", startup.version), faint_label(theme)),
        Span::styled(" · ".to_string(), muted_label(theme)),
        Span::styled("autosave".to_string(), faint_label(theme)),
    ];
    if let Some(memory_mb) = startup.memory_mb {
        right.push(Span::styled(" · ".to_string(), muted_label(theme)));
        right.push(Span::styled(format!("{memory_mb:.0}MB"), faint_label(theme)));
    }
    two_column_line(
        width,
        vec![Span::styled(short_session, style(theme.heat().steel, false))],
        right,
    )
}

fn dim_label(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.heat().steel)
        .add_modifier(Modifier::DIM)
}

fn faint_label(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM)
}

fn muted_label(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.heat().steel_dim)
        .add_modifier(Modifier::DIM)
}

// ============================================================================
// Label helpers.
// ============================================================================

fn permission_label_for_display(mode: &str) -> &'static str {
    match mode {
        "read-only" => "read-only",
        "workspace-write" => "workspace-write",
        "danger-full-access" => "danger-full-access",
        other if other.eq_ignore_ascii_case("prompt") => "prompt",
        _ => "unknown-mode",
    }
}

fn compact_model_label(model: &str) -> &str {
    if model.contains("fable") {
        "Fable 5"
    } else if model.contains("claude-opus-4-8") || model == "opus" {
        "Opus 4.8"
    } else if model.contains("claude-opus-4-7") || model.contains("claude-opus-4-6") {
        "Opus 4.7"
    } else if model.contains("sonnet") {
        "Sonnet 4.6"
    } else if model.contains("haiku") {
        "Haiku 4.5"
    } else {
        model
    }
}

fn style(color: Color, bold: bool) -> Style {
    let mut style = Style::new().fg(color);
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    style
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn sample_startup_screen() -> StartupScreen {
        StartupScreen {
            version: "0.1.0".to_string(),
            model: "claude-opus-4-8".to_string(),
            permissions: "danger-full-access".to_string(),
            branch: "main".to_string(),
            workspace: "zo".to_string(),
            directory: PathBuf::from("/tmp/zo"),
            project_root: Some(PathBuf::from("/tmp/zo")),
            session_id: "session-1234567890".to_string(),
            autosave_path: PathBuf::from("/tmp/session.jsonl"),
            startup_ms: Some(1297),
            memory_mb: Some(42.0),
            auth: StartupAuthState::default(),
            recent_sessions: Vec::new(),
        }
    }

    fn dump_terminal(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn dump_cells_with_style(terminal: &Terminal<TestBackend>) -> Vec<String> {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| format!("{cell:?}"))
            .collect()
    }

    /// Row-by-row text dump (symbols only), trailing blanks trimmed.
    fn dump_rows(terminal: &Terminal<TestBackend>) -> Vec<String> {
        let buf = terminal.backend().buffer();
        let w = buf.area.width as usize;
        let cells = buf.content();
        (0..buf.area.height as usize)
            .map(|y| {
                let row: String = (0..w).map(|x| cells[y * w + x].symbol()).collect();
                row.trim_end().to_string()
            })
            .collect()
    }

    fn render(theme: &Theme, intro: Option<Duration>) -> Terminal<TestBackend> {
        let mut terminal = Terminal::new(TestBackend::new(72, STARTUP_HEIGHT)).expect("terminal");
        terminal
            .draw(|f| draw(f, f.area(), &sample_startup_screen(), theme, intro))
            .expect("draw");
        terminal
    }

    #[test]
    fn startup_onboarding_lists_oauth_providers_when_missing() {
        let dumped = dump_terminal(&render(&Theme::no_color(), None));
        for expected in [
            "Get ready",
            "Alt+C",
            "/login claude",
            "Alt+O",
            "/login openai",
            "Alt+P",
            "/permissions",
            "summarize this repo",
        ] {
            assert!(dumped.contains(expected), "missing {expected}: {dumped}");
        }
        assert!(!dumped.contains("Continue recent work"), "{dumped}");
    }

    #[test]
    fn startup_onboarding_summarizes_mixed_provider_state() {
        let theme = Theme::no_color();
        let mut screen = sample_startup_screen();
        screen.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: false,
        };
        let mut terminal = Terminal::new(TestBackend::new(72, STARTUP_HEIGHT)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &screen, &theme, None))
            .expect("draw");
        let dumped = dump_terminal(&terminal);
        assert!(dumped.contains("Claude connected"), "{dumped}");
        assert!(dumped.contains("/login openai"), "{dumped}");
        assert!(!dumped.contains("/login claude"), "{dumped}");
    }

    #[test]
    fn startup_ready_state_hides_provider_onboarding() {
        let theme = Theme::no_color();
        let mut screen = sample_startup_screen();
        screen.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: true,
        };
        let mut terminal = Terminal::new(TestBackend::new(72, STARTUP_HEIGHT)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &screen, &theme, None))
            .expect("draw");
        let dumped = dump_terminal(&terminal);
        assert!(dumped.contains("Ready"), "{dumped}");
        assert!(dumped.contains("summarize this repo"), "{dumped}");
        assert!(!dumped.contains("Get ready"), "{dumped}");
        assert!(!dumped.contains("/login claude"), "{dumped}");
    }

    #[test]
    fn narrow_plain_startup_keeps_onboarding_cta_visible() {
        let theme = Theme::no_color();
        let mut terminal = Terminal::new(TestBackend::new(42, STARTUP_HEIGHT_PLAIN))
            .expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &sample_startup_screen(), &theme, None))
            .expect("draw");
        let dumped = dump_terminal(&terminal);
        assert!(dumped.contains("zo"), "{dumped}");
        assert!(dumped.contains("Get ready"), "{dumped}");
        assert!(dumped.contains("/login claude"), "{dumped}");
    }

    #[test]
    fn startup_masthead_uses_large_ignition_lockup_without_card_chrome() {
        let terminal = render(&Theme::zo(), None);
        let dumped = dump_terminal(&terminal);
        let rows = dump_rows(&terminal);
        assert!(
            rows.iter().any(|row| {
                row.trim_start().starts_with("▌ ▰▰▰▰▰ ▰▰▰▰▰")
                    && row.contains("launchpad")
            }),
            "{rows:?}"
        );
        assert!(dumped.contains("▰▰ ▰▰░"), "right-edge extrusion missing: {rows:?}");
        assert!(dumped.contains("░░░░░ ░░░░░"), "baseline shade missing: {rows:?}");
        assert!(!dumped.contains("▰░▰"), "shade must not fill letter counters: {rows:?}");
        assert!(
            dumped.contains("in the zone — AI pair-programming for this repo"),
            "{dumped}"
        );
        assert!(dumped.contains("Opus 4.8"), "{dumped}");
        assert!(dumped.contains("danger-full-access"), "{dumped}");
        assert!(dumped.contains("Alt+S"), "wide launchpad should expose quick actions: {dumped}");
        assert!(dumped.contains("────────────────"), "wide divider should make the launchpad feel anchored: {dumped}");
        assert!(!dumped.contains("████"), "blocky FIGlet logo must stay gone: {dumped}");
        assert!(!dumped.contains('╭'), "card chrome must be gone: {dumped}");
        assert!(!dumped.contains('│'), "card chrome must be gone: {dumped}");
        assert!(!dumped.contains("*-["), "ASCII ornament must be gone: {dumped}");
    }

    #[test]
    fn wide_ready_launchpad_adds_density_without_card_chrome() {
        let mut startup = sample_startup_screen();
        startup.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: true,
        };
        let theme = Theme::zo();
        let mut terminal = Terminal::new(TestBackend::new(132, STARTUP_HEIGHT)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &startup, &theme, None))
            .expect("draw");
        let dumped = dump_terminal(&terminal);
        for expected in [
            "Ready",
            "Suggested prompts",
            "workspace zo",
            "Alt+S",
            "review my diff",
            "find failing tests",
            "session-123",
            "v0.1.0",
        ] {
            assert!(dumped.contains(expected), "missing {expected}: {dumped}");
        }
        assert!(!dumped.contains('╭'), "card chrome must stay gone: {dumped}");
        assert!(!dumped.contains('│'), "card chrome must stay gone: {dumped}");
    }

    #[test]
    fn wide_startup_content_area_is_centered() {
        let area = Rect::new(10, 0, 200, STARTUP_HEIGHT);
        let content = startup_content_area(area);
        let expected_width = startup_content_width(area.width);
        assert_eq!(content.width, expected_width);
        assert_eq!(content.x, 10 + (200 - expected_width) / 2);
    }

    #[test]
    fn startup_content_width_scales_before_capping() {
        assert_eq!(startup_content_width(96), 96);
        assert_eq!(startup_content_width(160), 132);
        assert_eq!(startup_content_width(220), 165);
        assert_eq!(startup_content_width(260), MAX_STARTUP_WIDTH);
    }

    #[test]
    fn no_color_masthead_keeps_the_large_structure_without_color() {
        let theme = Theme::no_color();
        let mut terminal = Terminal::new(TestBackend::new(60, STARTUP_HEIGHT)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &sample_startup_screen(), &theme, None))
            .expect("draw");

        let dumped = dump_terminal(&terminal);
        for expected in [
            "▌ ▰▰▰▰▰ ▰▰▰▰▰",
            "launchpad",
            "in the zone — AI pair-programming for this repo",
            "────────────────",
            "Opus 4.8",
            "danger-full-access",
            "@ main",
            "/tmp/zo",
            "1297ms",
        ] {
            assert!(dumped.contains(expected), "missing {expected}: {dumped}");
        }
        assert!(!dumped.contains("*-["), "{dumped}");
        assert!(!dumped.contains("*----+----*"), "{dumped}");
    }

    #[test]
    fn startup_intro_frames_are_bucketed_and_settle_deterministically() {
        let theme = Theme::zo();
        let first_in_bucket = dump_cells_with_style(&render(
            &theme,
            Some(Duration::from_millis(264)),
        ));
        let second_in_bucket = dump_cells_with_style(&render(
            &theme,
            Some(Duration::from_millis(296)),
        ));
        assert_eq!(first_in_bucket, second_in_bucket);

        let settled = dump_cells_with_style(&render(&theme, None));
        let at_boundary = dump_cells_with_style(&render(
            &theme,
            Some(Duration::from_millis(INTRO_TOTAL_MS)),
        ));
        let long_after = dump_cells_with_style(&render(
            &theme,
            Some(Duration::from_millis(5_000)),
        ));
        let reduce_motion = dump_cells_with_style(&render(&theme, None));
        assert_eq!(settled, at_boundary);
        assert_eq!(settled, long_after);
        assert_eq!(settled, reduce_motion);
    }

    #[test]
    fn large_masthead_glyphs_are_width_safe() {
        let structural = LARGE_WORDMARK_ART
            .iter()
            .flat_map(|row| row.chars())
            .chain([
                LARGE_WORDMARK_SHADOW,
                LARGE_WORDMARK_RAIL,
                LARGE_WORDMARK_SPARK,
            ]);
        let banned = ['█', '■', '·', '…', '↺'];
        for glyph in structural.filter(|glyph| *glyph != ' ') {
            let allowed = ('\u{2500}'..='\u{257f}').contains(&glyph)
                || matches!(glyph, '░' | '▰')
                || matches!(glyph, LARGE_WORDMARK_RAIL | LARGE_WORDMARK_SPARK);
            assert!(allowed, "unsafe masthead glyph {glyph:?}");
            assert!(!banned.contains(&glyph), "banned masthead glyph {glyph:?}");
        }
        // The mass letterforms and their shade live in width-sensitive rows
        // (row 0 right-aligns the chip), so each cell must hold one column
        // under a wide-ambiguous CJK locale — the same bar as the HUD gauges.
        for glyph in LARGE_WORDMARK_ART
            .iter()
            .flat_map(|row| row.chars())
            .chain([LARGE_WORDMARK_SHADOW])
            .filter(|glyph| *glyph != ' ')
        {
            assert_eq!(
                unicode_width::UnicodeWidthChar::width_cjk(glyph),
                Some(1),
                "masthead glyph {glyph:?} must stay Neutral-width"
            );
        }
        let rendered_width = LARGE_WORDMARK_ART
            .iter()
            .map(|row| row.chars().count())
            .max()
            .unwrap_or_default()
            .saturating_add(1);
        assert!(rendered_width <= 13, "wordmark + shadow width {rendered_width}");
    }

    #[test]
    fn small_terminal_masthead_is_compact_and_intro_invariant() {
        fn render_small(intro: Option<Duration>) -> Terminal<TestBackend> {
            let theme = Theme::zo();
            let mut terminal = Terminal::new(TestBackend::new(42, STARTUP_HEIGHT_PLAIN))
                .expect("terminal");
            terminal
                .draw(|frame| draw(frame, frame.area(), &sample_startup_screen(), &theme, intro))
                .expect("draw");
            terminal
        }

        let settled = dump_cells_with_style(&render_small(None));
        let early = dump_cells_with_style(&render_small(Some(Duration::from_millis(0))));
        let late = dump_cells_with_style(&render_small(Some(Duration::from_millis(699))));
        assert_eq!(settled, early);
        assert_eq!(settled, late);
        assert!(dump_terminal(&render_small(None)).contains("zo"));
    }

    #[test]
    fn preferred_height_tracks_compact_masthead() {
        assert_eq!(preferred_height(42), STARTUP_HEIGHT_PLAIN);
        assert_eq!(preferred_height(60), STARTUP_HEIGHT);
    }

    #[test]
    fn info_line_uses_single_space_middot_separators() {
        // The meta row separator tightened from "  ·  " (2 spaces each
        // side) to " · " (1 space) so the logo stays the visual anchor.
        let line = info_line(&sample_startup_screen(), &Theme::zo());
        let texts: Vec<&str> = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            texts.contains(&" \u{00b7} "),
            "expected single-space middot separator: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t.contains("  \u{00b7}  ")),
            "old double-space separator must be gone: {texts:?}"
        );
    }

    #[test]
    fn info_line_startup_ms_drops_to_a_subordinate_faint_tier() {
        // The startup-time tail sits one hierarchy tier below the meta:
        // its separator + value both render faint + DIM, distinct from the
        // brighter `muted` separator used between model/permission/branch.
        let theme = Theme::zo();
        let line = info_line(&sample_startup_screen(), &theme);
        let ms_span = line
            .spans
            .iter()
            .find(|s| s.content.ends_with("ms"))
            .expect("startup-ms span present (sample has Some(1297))");
        assert_eq!(ms_span.content.as_ref(), "1297ms");
        assert_eq!(ms_span.style.fg, Some(theme.palette.faint));
        assert!(
            ms_span.style.add_modifier.contains(Modifier::DIM),
            "startup-ms must be DIM: {:?}",
            ms_span.style
        );
        // The separator immediately before the ms tail is also faint+DIM,
        // a tier below the `muted` separators between the meta fields.
        let ms_pos = line
            .spans
            .iter()
            .position(|s| s.content.ends_with("ms"))
            .expect("ms span index");
        let tail_sep = &line.spans[ms_pos - 1];
        assert_eq!(tail_sep.content.as_ref(), " \u{00b7} ");
        assert_eq!(tail_sep.style.fg, Some(theme.palette.faint));
    }

    #[test]
    fn info_line_omits_startup_ms_when_absent() {
        let mut startup = sample_startup_screen();
        startup.startup_ms = None;
        let line = info_line(&startup, &Theme::zo());
        assert!(
            !line.spans.iter().any(|s| s.content.ends_with("ms")),
            "no startup-ms span expected when startup_ms is None"
        );
    }

    #[test]
    fn info_line_meta_converges_to_single_amber_accent() {
        // 메타 채도 축소(D1): 종전 model=cyan·git=teal 의 색 경쟁을 제거하고
        // 모드 배지만 단일 앰버 accent 로 띄운다.
        let theme = Theme::zo();
        let line = info_line(&sample_startup_screen(), &theme);
        for s in &line.spans {
            assert_ne!(
                s.style.fg,
                Some(theme.palette.cyan),
                "no cyan competition in startup meta: {:?}",
                s.content
            );
            assert_ne!(
                s.style.fg,
                Some(theme.palette.teal),
                "no teal competition in startup meta: {:?}",
                s.content
            );
        }
        assert!(
            line.spans
                .iter()
                .any(|s| s.style.fg == Some(theme.palette.accent)),
            "mode badge must carry the single amber accent"
        );
    }

    /// Render the launchpad tall enough to include the actionable section
    /// (the recent-session rows sit below the static masthead + hints).
    fn render_tall(startup: &StartupScreen, theme: &Theme) -> Terminal<TestBackend> {
        let mut terminal = Terminal::new(TestBackend::new(72, 24)).expect("terminal");
        terminal
            .draw(|f| draw(f, f.area(), startup, theme, None))
            .expect("draw");
        terminal
    }

    #[test]
    fn launchpad_lists_recent_sessions_when_provider_ready() {
        let mut startup = sample_startup_screen();
        startup.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: true,
        };
        startup.recent_sessions = vec![
            RecentSession {
                label: "refactor the scroll handler".to_string(),
                age: "3m-ago".to_string(),
            },
            RecentSession {
                label: "fix HUD adaptive sections".to_string(),
                age: "2h-ago".to_string(),
            },
        ];
        let dumped = dump_terminal(&render_tall(&startup, &Theme::zo()));
        for expected in [
            "Continue recent work",
            "refactor the scroll handler",
            "3m-ago",
            "fix HUD adaptive sections",
            "/resume",
            "summarize this repo",
        ] {
            assert!(dumped.contains(expected), "missing {expected}: {dumped}");
        }
        assert!(!dumped.contains("Get ready"), "{dumped}");
    }

    #[test]
    fn rich_preferred_height_returning_user_keeps_recent_and_cta_visible() {
        let mut startup = sample_startup_screen();
        startup.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: true,
        };
        startup.recent_sessions = vec![
            RecentSession {
                label: "first recent session".to_string(),
                age: "3m-ago".to_string(),
            },
            RecentSession {
                label: "second recent session".to_string(),
                age: "2h-ago".to_string(),
            },
        ];
        let theme = Theme::zo();
        let width = 60;
        let mut terminal = Terminal::new(TestBackend::new(width, preferred_height(width)))
            .expect("terminal");
        terminal
            .draw(|f| draw(f, f.area(), &startup, &theme, None))
            .expect("draw");
        let dumped = dump_terminal(&terminal);
        for expected in [
            "Continue recent work",
            "first recent session",
            "second recent session",
            "/resume",
            STARTUP_SUMMARIZE_REPO_LABEL,
            "/help",
        ] {
            assert!(dumped.contains(expected), "missing {expected}: {dumped}");
        }
    }

    #[test]
    fn narrow_returning_user_keeps_resume_and_suggestion_visible() {
        let mut startup = sample_startup_screen();
        startup.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: true,
        };
        startup.recent_sessions = vec![
            RecentSession {
                label: "first recent session".to_string(),
                age: "3m-ago".to_string(),
            },
            RecentSession {
                label: "second recent session".to_string(),
                age: "2h-ago".to_string(),
            },
        ];
        let theme = Theme::no_color();
        let mut terminal = Terminal::new(TestBackend::new(42, STARTUP_HEIGHT_PLAIN))
            .expect("terminal");
        terminal
            .draw(|f| draw(f, f.area(), &startup, &theme, None))
            .expect("draw");
        let dumped = dump_terminal(&terminal);
        for expected in [
            "Continue recent work",
            "first recent session",
            "/resume",
            "Alt+S",
            STARTUP_SUMMARIZE_REPO_LABEL,
            "/help",
        ] {
            assert!(dumped.contains(expected), "missing {expected}: {dumped}");
        }
        assert!(
            !dumped.contains("second recent session"),
            "narrow startup should preserve CTA rows before extra recents: {dumped}"
        );
    }

    #[test]
    fn recent_sessions_do_not_override_missing_provider_onboarding() {
        let mut startup = sample_startup_screen();
        startup.recent_sessions = vec![RecentSession {
            label: "resume me later".to_string(),
            age: "1m-ago".to_string(),
        }];
        let dumped = dump_terminal(&render_tall(&startup, &Theme::zo()));
        assert!(dumped.contains("Get ready"), "{dumped}");
        assert!(!dumped.contains("Continue recent work"), "{dumped}");
        assert!(!dumped.contains("resume me later"), "{dumped}");
    }

    /// Manual visual probe: `cargo test -p zo-cli --lib \
    /// startup::tests::dump_startup_masthead -- --ignored --nocapture`
    /// prints the settled launchpad for eyeballing the brand lockup layout.
    #[test]
    #[ignore = "manual visual probe"]
    fn dump_startup_masthead() {
        let mut startup = sample_startup_screen();
        startup.auth = StartupAuthState {
            anthropic_oauth: true,
            chatgpt_oauth: true,
        };
        startup.recent_sessions = vec![
            RecentSession {
                label: "배포도 된겨?".to_string(),
                age: "just-now".to_string(),
            },
            RecentSession {
                label: "tune startup launchpad".to_string(),
                age: "4m-ago".to_string(),
            },
        ];
        let mut terminal = Terminal::new(TestBackend::new(132, STARTUP_HEIGHT)).expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &startup, &Theme::zo(), None))
            .expect("draw");
        for row in dump_rows(&terminal) {
            println!("{row}");
        }
    }

    #[test]
    fn launchpad_section_hidden_when_no_recent_sessions() {
        // Empty `recent_sessions` (the default) must omit the section
        // entirely — no resume hint, no stray bullet rows.
        let startup = sample_startup_screen();
        assert!(startup.recent_sessions.is_empty());
        let dumped = dump_terminal(&render_tall(&startup, &Theme::zo()));
        assert!(
            !dumped.contains("/resume"),
            "resume hint should be hidden with no sessions: {dumped}"
        );
    }
}
