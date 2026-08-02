//! Right metadata sidebar panel (Zo ledger style).
//!
//! A toggleable panel that renders to the left of the transcript area,
//! showing live workspace metadata, changed files, Todo items, and LSP status.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Padding, Paragraph};

use core_types::RateLimitSnapshot;

use super::app::WakeSource;
use super::glyphs;
use super::hud::{
    AgentTaskSummary, HudState, McpHealth, McpHudStatus, McpHudStatusKind, McpSourcesSummary,
    PermissionMode, SecurityPosture, TodoChecklistStatus, scheduled_countdown,
};
use super::spinner::format_elapsed;
use super::text_metrics::{char_width, display_width};
use super::theme::Theme;

// 80행 터미널에서 스크롤 가능 범위 ~290개. 200이면 일반 터미널(40-60행)을
// 완전히 커버하면서 캐시 clone 비용 72µs, 메모리 17KB로 최소화.
pub(crate) const MAX_SIDEBAR_FILES: usize = 200;
/// Rows reserved at the bottom of the rail for the compact interaction legend.
/// Kept in sync with the line count produced by [`footer_lines`].
const FOOTER_ROWS: u16 = 2;
/// Preserve the existing short-terminal priority: metadata owns the rail until
/// there are at least eight inner rows available.
const FOOTER_MIN_HEIGHT: u16 = 8;
/// Rows a floating card spends on chrome: the closing `─` rule above the
/// content and the one below it.
const CARD_RULE_ROWS: u16 = 2;
/// Smallest card that still has a content row between its two closing rules.
const MIN_CARD_HEIGHT: u16 = CARD_RULE_ROWS + 1;

/// Status of a changed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStatus {
    /// File was modified.
    Modified,
    /// File was added (new).
    Added,
    /// File was deleted.
    Deleted,
}

/// A single changed file entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedFile {
    /// Relative path (or filename) of the changed file.
    pub path: String,
    /// Kind of change.
    pub status: FileStatus,
    /// Lines added vs HEAD (`git diff --numstat`). `0` for untracked or binary.
    pub adds: usize,
    /// Lines removed vs HEAD. `0` for untracked or binary.
    pub rems: usize,
}

/// Persistent state for the sidebar panel.
#[derive(Debug, Clone)]
pub struct SidebarState {
    /// Whether the sidebar is currently visible.
    pub visible: bool,
    /// Whether the running-agents tree is expanded (full per-agent list)
    /// or collapsed (single `✦ N agents [▶ expand]` line). Toggled by
    /// `Ctrl+A` so users can hide the per-agent breakdown on narrow
    /// terminals or focus the rest of the sidebar.
    pub agents_expanded: bool,
    /// Whether capped row lists (e.g. the MCP sources `+N more` tail) show
    /// every row. Toggled by clicking the panel — the row caps keep the
    /// resting card compact, and the click answers "what's behind +N more"
    /// without a keybinding.
    pub detail_expanded: bool,
    /// List of changed files to display (capped at [`MAX_SIDEBAR_FILES`]).
    pub changed_files: Vec<ChangedFile>,
    /// Total number of displayable changed files (may exceed
    /// `changed_files.len()` when capped).
    pub total_changed: usize,
    /// Vertical scroll offset (in rows).
    pub scroll: u16,
    /// Paths present in git status at session start. Changes matching
    /// these paths are hidden so only session-originated edits show.
    baseline_paths: std::collections::HashSet<String>,
    /// Total displayable changed-file count captured at session start. The
    /// baseline path list is capped for memory/render cost, but the header count
    /// must subtract the full baseline total or it can show e.g.
    /// `changes 5132 (showing 0)` when all visible changes are old dirt.
    baseline_total: usize,
}

impl Default for SidebarState {
    fn default() -> Self {
        Self {
            visible: true,
            agents_expanded: true,
            detail_expanded: false,
            changed_files: Vec::new(),
            total_changed: 0,
            scroll: 0,
            baseline_paths: std::collections::HashSet::new(),
            baseline_total: 0,
        }
    }
}

impl SidebarState {
    /// Create a new visible sidebar with no files.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle visibility on/off.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Toggle the running-agents tree expand state. Independent of
    /// [`Self::toggle`] (the sidebar can be visible with the agent tree
    /// collapsed and vice versa).
    pub fn toggle_agents(&mut self) {
        self.agents_expanded = !self.agents_expanded;
    }

    /// Toggle the capped-row detail view (mouse click on the panel).
    pub fn toggle_detail(&mut self) {
        self.detail_expanded = !self.detail_expanded;
    }

    /// Snapshot the current git status as the session baseline.
    /// Subsequent `set_changed_files` calls will hide paths present
    /// in this baseline, showing only session-originated changes.
    pub fn capture_baseline(&mut self, snapshot: &GitStatusSnapshot) {
        self.baseline_total = snapshot.total;
        self.baseline_paths = snapshot.files.iter().map(|f| f.path.clone()).collect();
    }

    /// Replace the file list, filtering out baseline paths so only
    /// session-originated changes are visible.
    pub fn set_changed_files(&mut self, files: Vec<ChangedFile>, total: usize) {
        let (new_files, new_total) = if self.baseline_paths.is_empty() {
            (files, total)
        } else {
            let filtered: Vec<ChangedFile> = files
                .into_iter()
                .filter(|f| !self.baseline_paths.contains(&f.path))
                .collect();
            let filtered_total = total.saturating_sub(self.baseline_total).max(filtered.len());
            (filtered, filtered_total)
        };
        // Only reset the scroll offset when the visible set actually changes.
        // A periodic mid-turn refresh that finds no new edits must not yank the
        // user's scroll position back to the top every tick.
        if new_files != self.changed_files {
            self.scroll = 0;
        }
        self.changed_files = new_files;
        self.total_changed = new_total;
    }

    /// Scroll down by `rows`, clamped to the content upper bound.
    ///
    /// The precise viewport clamp (`scroll.min(max_scroll)`) happens in
    /// `draw`, which knows the file-row height. Here we clamp the stored
    /// field to the number of changed files so repeated wheel events can't
    /// inflate `scroll` past any reachable offset (which would otherwise
    /// leave the panel unresponsive until an equal number of scroll-ups).
    pub fn scroll_down(&mut self, rows: u16) {
        let max = u16::try_from(self.changed_files.len()).unwrap_or(u16::MAX);
        self.scroll = self.scroll.saturating_add(rows).min(max);
    }

    /// Scroll up by `rows`.
    pub fn scroll_up(&mut self, rows: u16) {
        self.scroll = self.scroll.saturating_sub(rows);
    }
}

/// Snapshot of displayable `git status` results, capped at [`MAX_SIDEBAR_FILES`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitStatusSnapshot {
    pub files: Vec<ChangedFile>,
    pub total: usize,
}

impl GitStatusSnapshot {
    pub(crate) const EMPTY: Self = Self {
        files: Vec::new(),
        total: 0,
    };
}

#[derive(Clone, Copy)]
struct SidebarStyles {
    label: Style,
    value: Style,
    muted: Style,
    ok: Style,
    warn: Style,
    err: Style,
    cyan: Style,
    /// Brand accent — section titles, section counts, the live tool name.
    accent: Style,
}

impl SidebarStyles {
    fn new(theme: &Theme) -> Self {
        Self {
            label: Style::default().fg(theme.palette.muted),
            value: Style::default().fg(theme.palette.fg),
            muted: Style::default().fg(theme.palette.dim),
            ok: Style::default().fg(theme.palette.success),
            warn: Style::default().fg(theme.palette.warn),
            err: Style::default().fg(theme.palette.error),
            cyan: Style::default().fg(theme.palette.cyan),
            accent: Style::default().fg(theme.palette.accent),
        }
    }
}

/// Paths that are filtered from the sidebar by default to reduce
/// noise. Matches are prefix-based.
const FILTERED_PREFIXES: &[&str] = &[
    "target/",
    ".zo/agents/",
    ".zo/",
    "agent-",
    ".sandbox-",
];

/// Whether a repository-relative path is Zo/build noise rather than a user change.
#[must_use]
pub fn is_workspace_status_path_filtered(path: &str) -> bool {
    FILTERED_PREFIXES
        .iter()
        .any(|prefix| {
            path.starts_with(prefix)
                || prefix
                    .strip_suffix('/')
                    .is_some_and(|directory| path == directory)
        })
}

fn push_header(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let project = project_name(hud);
    let branch = hud.git_branch.as_deref().unwrap_or("detached");
    // The project is the panel's title line, so it carries the only bright BOLD
    // in the rail; the branch takes the teal every other git surface uses, and
    // the cwd below stays dim. Three weights, one glance.
    lines.push(aligned_sidebar_line(
        &project,
        Style::default()
            .fg(theme.palette.bright)
            .add_modifier(Modifier::BOLD),
        branch,
        Style::default().fg(theme.palette.teal),
        width,
    ));

    let cwd = compact_cwd(hud);
    let (status_label, status_style) = sidebar_header_status_badge(hud, theme, styles);
    lines.push(aligned_sidebar_line(
        &cwd,
        styles.muted,
        &status_label,
        status_style,
        width,
    ));
    lines.push(Line::default());
}

/// Compose one quiet two-column row. The left value truncates first while the
/// short status/branch anchor stays pinned to the right edge.
fn aligned_sidebar_line(
    left: &str,
    left_style: Style,
    right: &str,
    right_style: Style,
    width: u16,
) -> Line<'static> {
    const MIN_GAP: usize = 2;

    let width = usize::from(width);
    let right_budget = width.saturating_sub(MIN_GAP).min(16);
    let right = truncate_to_cells(right, right_budget);
    let right_width = display_width(&right);
    let left_budget = width.saturating_sub(right_width + MIN_GAP);
    let left = truncate_to_cells(left, left_budget);
    let left_width = display_width(&left);
    let gap = width.saturating_sub(left_width + right_width);

    Line::from(vec![
        Span::styled(left, left_style),
        Span::raw(" ".repeat(gap)),
        Span::styled(right, right_style),
    ])
}

fn sidebar_header_status_badge(
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) -> (String, Style) {
    if hud.security_posture == SecurityPosture::SandboxBlocked {
        return ("blocked".to_string(), styles.warn);
    }
    let workflow_running = hud.workflow.as_ref().is_some_and(|flow| {
        flow.status == "running"
            || flow.current_phase_status == "running"
            || flow.running_agents > 0
    });
    if workflow_running || hud.running_agents > 0 {
        return (
            "running".to_string(),
            Style::default().fg(theme.palette.info),
        );
    }
    // Idle: a calm "ready" activity lamp. The permission mode is deliberately
    // NOT echoed here — it owns the `mode` line in the session panel (its
    // single home), and mirroring it into the badge duplicated it on every
    // idle frame (the reported "권한 표시 중복"). So the badge stays a pure
    // activity indicator: ready (success) → running (info) → blocked (warn),
    // while the perm mode is always shown in exactly one place below.
    ("ready".to_string(), styles.ok)
}

/// Context-pressure row: `use  ▬▬▬▬░░░░░░  40%`. **Not currently rendered.**
///
/// Context pressure had grown three separate presentations — this gauge, a
/// `ctx <used> / <limit>` token row above it, and the HUD's own
/// `Context ▰▰▱▱▱▱ 34%` — so the same number was drawn twice in the rail and a
/// third time in the footer, each with its own glyph set and rounding. The HUD is
/// now the single owner: it is present in both inline and fullscreen mode,
/// whereas this rail is optional, so putting the one copy there means the
/// pressure signal never disappears with a toggle.
///
/// Kept (rather than deleted) because it is the shape the rail wants if context
/// ever moves back: the bar and BOLD percent share [`gauge_color`]'s ramp with
/// the rate-limit and quota meters, so all three pressure surfaces turn amber and
/// red at the same utilization — which a from-scratch rewrite would have to
/// rediscover.
#[allow(dead_code)] // documented single-owner decision above
fn push_context_use_line(
    lines: &mut Vec<Line<'_>>,
    theme: &Theme,
    styles: SidebarStyles,
    pct: u64,
) {
    let bar = token_gauge_bar(pct, CONTEXT_GAUGE_CELLS, theme);
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("use ", styles.muted),
        bar,
        Span::styled(format!(" {pct}%"), gauge_percent_style(pct, theme)),
    ]));
}

/// Below this inner width the rail is too tight to spend cells on indentation;
/// the same breakpoint the session and agent rows already switch layout at.
const NARROW_RAIL_COLS: u16 = 32;

/// Cells in the session context gauge. Ten matches the rate-limit and quota
/// bars so the three stack into one column.
const CONTEXT_GAUGE_CELLS: usize = 10;

/// The BOLD, ramp-colored style every gauge percent uses. Bold is what makes the
/// number readable *as* the headline of its row while the label stays dim.
fn gauge_percent_style(pct: u64, theme: &Theme) -> Style {
    let pct = u8::try_from(pct.min(100)).unwrap_or(100);
    Style::new()
        .fg(gauge_color(pct, theme))
        .add_modifier(Modifier::BOLD)
}

fn push_cache_split_lines(
    lines: &mut Vec<Line<'_>>,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
    stacked: bool,
) {
    if hud.ctx_cached == 0 {
        return;
    }

    if stacked {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("ctx   ", styles.muted),
            Span::styled(
                format!("{} new", format_tokens(hud.ctx_new_input)),
                styles.muted,
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("      ", styles.muted),
            Span::styled(
                format!("{} cached", format_tokens(hud.ctx_cached)),
                styles.muted,
            ),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(
                format!(
                    "ctx {} new · {} cached",
                    format_tokens(hud.ctx_new_input),
                    format_tokens(hud.ctx_cached)
                ),
                styles.muted,
            ),
        ]));
    }
}

/// `cost $0.37  mode workspace-write`, split onto two rows when the pair does
/// not fit. The fit is *measured*, not inferred from a width breakpoint: a long
/// mode label (`danger-full-access`) overflows a rail that a short one
/// (`plan`) fits comfortably, and a wrapped row loses its indentation.
fn push_cost_mode_line(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let cost_prefix = if hud.cost_approx { "~$" } else { "$" };
    let cost_text = format!("{cost_prefix}{:.2}", hud.cost_usd);
    let mode_label = permission_label(hud.perm_mode);
    let inline = display_width(indent_glyph(theme))
        + "cost ".len()
        + display_width(&cost_text)
        + "  mode ".len()
        + display_width(mode_label)
        <= usize::from(width);
    let mut spans = vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("cost ", styles.muted),
        Span::styled(cost_text, styles.value),
    ];
    if inline {
        spans.push(Span::styled("  mode ", styles.muted));
        spans.push(Span::styled(
            mode_label,
            permission_style(hud.perm_mode, theme),
        ));
        lines.push(Line::from(spans));
    } else {
        lines.push(Line::from(spans));
        // On the narrowest rails even `mode danger-full-access` overruns the
        // row. The mode value names itself, so the label is what gets dropped —
        // never the value, and never by wrapping.
        let indent = display_width(indent_glyph(theme));
        let labelled = indent + "mode ".len() + display_width(mode_label) <= usize::from(width);
        let mut mode_spans = vec![Span::styled(indent_glyph(theme), styles.muted)];
        if labelled {
            mode_spans.push(Span::styled("mode ", styles.muted));
        }
        mode_spans.push(Span::styled(
            mode_label,
            permission_style(hud.perm_mode, theme),
        ));
        lines.push(Line::from(mode_spans));
    }
}

fn push_session_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    lines.push(section_line("session", width, theme));
    if let Some(identity) = hud.session_identity.as_ref() {
        let badge_style = if theme.no_color {
            styles.value
        } else {
            Style::new().fg(identity.badge_color(theme))
        };
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("● ", badge_style),
            Span::styled(identity.name.clone(), styles.value),
        ]));
    }

    if width < 32 {
        // Context pressure is the HUD's alone (see `push_context_use_line`).
        push_cache_split_lines(lines, hud, theme, styles, true);
        push_compact_ceiling_line(lines, hud, theme, styles);
        if let Some(rl) = hud.rate_limit {
            lines.extend(rate_limit_gauges(rl, theme, styles.muted));
        }
        // Estimated rows render even without a measured Anthropic snapshot —
        // a non-Anthropic main model has no `rate_limit` but can be throttled.
        lines.extend(estimated_quota_gauges(&hud.provider_quotas, theme, styles.muted));
        push_auth_line(lines, hud, theme, styles);
        push_build_line(lines, width, theme, styles);
        push_cost_mode_line(lines, width, hud, theme, styles);
        return;
    }

    // Context pressure is the HUD's alone (see `push_context_use_line`).
    push_cache_split_lines(lines, hud, theme, styles, false);
    push_compact_ceiling_line(lines, hud, theme, styles);
    if let Some(rl) = hud.rate_limit {
        lines.extend(rate_limit_gauges(rl, theme, styles.muted));
    }
    // Estimated rows render even without a measured Anthropic snapshot — a
    // non-Anthropic main model has no `rate_limit` but can be throttled.
    lines.extend(estimated_quota_gauges(&hud.provider_quotas, theme, styles.muted));
    push_auth_line(lines, hud, theme, styles);
    push_build_line(lines, width, theme, styles);
    push_cost_mode_line(lines, width, hud, theme, styles);
}

/// One muted line naming the auto-compaction ceiling the `ctx` gauge measures
/// against (`⤷ compacts at 450.0k`), so the percent above reads as "distance
/// to compaction", not "share of the window".
fn push_compact_ceiling_line(
    lines: &mut Vec<Line<'_>>,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    if hud.compact_threshold == 0 {
        return;
    }
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled(
            format!("compacts at {}", format_tokens(hud.compact_threshold)),
            styles.muted,
        ),
    ]));
}

/// Claude credential rung in use. OAuth rungs render as plain values; the
/// env-key rung is metered billing on an OAuth-first tool, so it renders in
/// the warn style as a standing notice (no transition event needed — the row
/// itself is the warning).
fn push_auth_line(lines: &mut Vec<Line<'_>>, hud: &HudState, theme: &Theme, styles: SidebarStyles) {
    let Some(origin) = hud.auth_origin else {
        return;
    };
    let (label, metered) = match origin {
        api::ClaudeAuthOrigin::Keychain => ("oauth \u{00b7} keychain", false),
        api::ClaudeAuthOrigin::SavedOauth => ("oauth \u{00b7} zo login", false),
        api::ClaudeAuthOrigin::Env => ("env key \u{00b7} metered", true),
    };
    let value_style = if metered { styles.warn } else { styles.value };
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("auth ", styles.muted),
        Span::styled(label, value_style),
    ]));
}

/// The running build's identity, pinned next to `auth` because it answers the
/// same "what am I actually talking to" question.
///
/// The splash carries this too, but it scrolls away after the first prompt,
/// which left a working session with no way to tell a release apart from a local
/// build short of running `zo --version` — the ambiguity behind every "the fix
/// is in the source but not in my binary" round trip.
fn push_build_line(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    theme: &Theme,
    styles: SidebarStyles,
) {
    // `0.1.2 · a249da23-dirty` is long enough to wrap a 24-column rail, and a
    // wrapped row loses the indentation that puts it inside SESSION.
    let budget = usize::from(width)
        .saturating_sub(display_width(indent_glyph(theme)) + "build ".len())
        .max(6);
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("build ", styles.muted),
        Span::styled(
            truncate_to_cells(&super::build_identity::label(), budget),
            styles.value,
        ),
    ]));
}

/// Always-on `/restart` warning: the running binary has been replaced on disk
/// by a newer build, so the live session is executing stale code. Pinned near
/// the top of the sidebar in the warning tone with a `⚠` marker so it is
/// impossible to miss; absent entirely until [`HudState::stale_binary`] trips
/// (see [`super::stale_binary`]). The detection command itself (`/restart`) is a
/// separate concern — this row only names it.
fn push_stale_binary_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let Some(info) = hud.stale_binary.as_ref() else {
        return;
    };
    let marker = tree_glyph(theme, glyphs::WARN_TRIANGLE, glyphs::WARN_TRIANGLE_NC);
    let label = info.sidebar_label();
    // Reserve the marker + its trailing space out of the available cells.
    let max = usize::from(width).saturating_sub(6).max(8);
    lines.push(Line::from(vec![
        Span::styled(format!("{marker} "), styles.warn),
        Span::styled(truncate_to_cells(&label, max), styles.warn),
    ]));
}

fn push_automation_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    if hud.automation_lines.is_empty() {
        return;
    }
    lines.push(section_line("automation", width, theme));
    let max = usize::from(width).saturating_sub(4).max(8);
    for line in &hud.automation_lines {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(truncate_to_cells(line, max), styles.value),
        ]));
    }
}

fn push_live_activity_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    changed_files: usize,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let mcp_summary = McpSourcesSummary::from_encoded(&hud.mcp_servers);
    let mcp_health = mcp_summary.health();
    let show_mcp = matches!(mcp_health, McpHealth::Degraded | McpHealth::Connecting);
    if hud.last_tool.is_none()
        && hud.running_agents == 0
        && hud.background_tasks == 0
        && hud.scheduled_wake.is_none()
        && changed_files == 0
        && !show_mcp
        && hud.team_inbox_unread == 0
    {
        return;
    }

    lines.push(section_line("live", width, theme));
    if let Some(tool) = hud.last_tool.as_deref() {
        let label = truncate_to_cells(tool, usize::from(width).saturating_sub(9).max(8));
        // The running tool is the single most volatile fact in the panel, so it
        // takes the accent; everything else on the row stays quiet around it.
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(
                format!(
                    "{} ",
                    tree_glyph(theme, glyphs::CHEVRON_RIGHT, glyphs::CHEVRON_RIGHT_NC)
                ),
                styles.accent,
            ),
            Span::styled(label, styles.accent),
            Span::styled(" · tool", styles.muted),
        ]));
    }
    if hud.running_agents > 0 {
        let mut detail = format!("{} agents active", hud.running_agents);
        if let Some(agent) = hud.agents.iter().find(|agent| !agent.status.eq_ignore_ascii_case("completed")) {
            if let Some(activity) = agent.activity_label() {
                let _ = write!(detail, " · {activity}");
            } else if !agent.name.trim().is_empty() {
                let name = agent.name.as_str();
                let _ = write!(detail, " · {name}");
            }
        }
        let label = truncate_to_cells(&detail, usize::from(width).saturating_sub(6).max(8));
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(
                format!(
                    "{} ",
                    tree_glyph(theme, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC)
                ),
                styles.cyan,
            ),
            Span::styled(label, styles.value),
        ]));
    }
    if hud.background_tasks > 0 {
        let label = if hud.background_tasks == 1 {
            "1 background task active".to_string()
        } else {
            format!("{} background tasks active", hud.background_tasks)
        };
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(label, styles.cyan),
        ]));
    }
    push_scheduled_wake_row(lines, hud, theme, styles);
    if changed_files > 0 {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("edit ", styles.warn),
            Span::styled(format!("{changed_files} files changed"), styles.value),
        ]));
    }
    if show_mcp {
        // A degraded source is the one MCP state that costs the user a
        // capability, so it is the only one that spends BOLD; connecting is a
        // transient the amber alone covers.
        let (label, style) = match mcp_health {
            McpHealth::Degraded => (
                "sources degraded",
                styles.err.add_modifier(Modifier::BOLD),
            ),
            McpHealth::Connecting => ("sources connecting", styles.warn),
            McpHealth::Healthy | McpHealth::None => unreachable!(),
        };
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(label, style),
            Span::styled(
                format!(" · {}/{} ready", mcp_summary.ready, mcp_summary.total),
                styles.muted,
            ),
        ]));
    }
    if hud.team_inbox_unread > 0 {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("inbox ", styles.warn),
            Span::styled(team_inbox_unread_label(hud.team_inbox_unread), styles.value),
        ]));
    }
    lines.push(Line::default());
}

fn push_scheduled_wake_row(
    lines: &mut Vec<Line<'_>>,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let Some(wake) = hud.scheduled_wake.as_ref() else {
        return;
    };
    let fallback = match wake.source {
        WakeSource::Wakeup => "scheduled wakeup",
        WakeSource::Loop => "scheduled /loop run",
    };
    let reason = if wake.reason.trim().is_empty() {
        fallback
    } else {
        wake.reason.trim()
    };
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("⏱ ", styles.cyan),
        Span::styled(truncate_to_cells(reason, 28), styles.value),
        Span::styled(format!(" · {}", scheduled_countdown(wake)), styles.muted),
    ]));
}

/// Badge text for `N` unread `TeamInbox` updates — count only, never any
/// update summary/body text (the B4 scope boundary).
fn team_inbox_unread_label(unread: u64) -> String {
    if unread == 1 {
        "1 unread update".to_string()
    } else {
        format!("{unread} unread updates")
    }
}

fn push_todo_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    if hud.todo_items.is_empty() {
        return;
    }

    lines.push(section_rule(
        "todo",
        Some(&hud.todo_items.len().to_string()),
        styles.accent,
        width,
        theme,
    ));
    for todo in hud.todo_items.iter().take(6) {
        let (marker, marker_style) = todo_marker(todo.status, theme);
        let text = if todo.status == TodoChecklistStatus::InProgress
            && !todo.active_form.trim().is_empty()
        {
            todo.active_form.as_str()
        } else {
            todo.content.as_str()
        };
        let max_todo_len = usize::from(width).saturating_sub(7);
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(marker, marker_style),
            Span::raw(" "),
            Span::styled(truncate_to_cells(text, max_todo_len), styles.value),
        ]));
    }

    let total = hud.todo_items.len();
    let done = hud
        .todo_items
        .iter()
        .filter(|item| item.status == TodoChecklistStatus::Completed)
        .count();
    if done == total {
        // The ✓ markers above already carry the success green; a second green
        // row would double-count the same fact, so the tally reads muted.
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(format!("done · {done}/{total}"), styles.muted),
        ]));
    }
    lines.push(Line::default());
}

fn push_activity_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    changed_files: usize,
    detail_expanded: bool,
    theme: &Theme,
    styles: SidebarStyles,
) {
    lines.push(section_line("work", width, theme));
    let metrics = sorted_work_metrics(hud, changed_files);
    let mut work_line = vec![Span::styled(indent_glyph(theme), styles.muted)];
    if metrics.is_empty() {
        work_line.push(Span::styled("idle", styles.muted));
    } else {
        // Metrics are already sorted by magnitude, so a rail too narrow for all
        // four drops the *smallest* counters off the end instead of wrapping the
        // row (which would strand `files 2` in the panel's left margin).
        let mut used = display_width(indent_glyph(theme));
        for (idx, metric) in metrics.iter().enumerate() {
            let value = metric.value.to_string();
            let cost = usize::from(idx > 0) * 2 + metric.label.len() + 1 + value.len();
            if used + cost > usize::from(width) {
                break;
            }
            used += cost;
            if idx > 0 {
                work_line.push(Span::styled("  ", styles.muted));
            }
            work_line.push(Span::styled(format!("{} ", metric.label), styles.muted));
            work_line.push(Span::styled(value, styles.value));
        }
    }
    lines.push(Line::from(work_line));

    push_mcp_sources_section(lines, width, &hud.mcp_servers, detail_expanded, theme, styles);
    lines.push(Line::default());
}

/// Map an MCP lifecycle state to its sidebar headline color: a single failed
/// source turns the whole headline red, an in-flight one yellow, all-ready
/// green. Centralizes the "what color is MCP" decision so it lives in one place.
fn mcp_headline_style(health: McpHealth, styles: SidebarStyles) -> Style {
    match health {
        McpHealth::Degraded => styles.err,
        McpHealth::Connecting => styles.warn,
        McpHealth::Healthy | McpHealth::None => styles.ok,
    }
}

/// Display ordering key: the most actionable rows sort first so a `Failed`
/// source is never the one silently dropped past the row cap.
fn mcp_status_severity(kind: McpHudStatusKind) -> u8 {
    match kind {
        McpHudStatusKind::Failed => 0,
        McpHudStatusKind::AuthPending => 1,
        McpHudStatusKind::Discovering => 2,
        McpHudStatusKind::Ready => 3,
    }
}

/// Render the MCP "sources" headline and per-server rows.
///
/// Single responsibility: turn the encoded MCP source list into its sidebar
/// block. The headline count and color come from one [`McpSourcesSummary`]
/// folded over the *same* list the rows render, so the count can never disagree
/// with the rows (the old `mcp_count` drift), and a failing source recolors the
/// headline instead of staying green. Rows are capped at four — ordered worst-
/// first and topped with a `+N more` hint — so a failure past the cap is never
/// hidden.
fn push_mcp_sources_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    servers: &[String],
    expanded: bool,
    theme: &Theme,
    styles: SidebarStyles,
) {
    const MAX_ROWS: usize = 4;
    let max_rows = if expanded { usize::MAX } else { MAX_ROWS };

    let summary = McpSourcesSummary::from_encoded(servers);
    if summary.is_empty() {
        return;
    }

    // `N` when every source is ready, else `ready/total` so a degraded or
    // connecting set shows how many are actually up — not a flat green total.
    let count_text = if summary.ready == summary.total {
        summary.total.to_string()
    } else {
        format!("{}/{}", summary.ready, summary.total)
    };
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        section_heading_span("sources", theme),
        Span::raw(" "),
        Span::styled(count_text, mcp_headline_style(summary.health(), styles)),
    ]));

    let mut rows: Vec<McpHudStatus> = servers.iter().map(|s| McpHudStatus::decode(s)).collect();
    // Stable sort: equal severities keep the source list's alphabetical order.
    rows.sort_by_key(|status| mcp_status_severity(status.kind));

    let visible = rows.len().min(max_rows);
    let hidden = rows.len() - visible;
    let mcp_dot_style = Style::new().fg(theme.palette.violet);
    let ready_style = Style::new().fg(theme.palette.success);
    for (idx, status) in rows.into_iter().take(max_rows).enumerate() {
        let (label, label_style) = match status.kind {
            McpHudStatusKind::Discovering => ("discovering", styles.warn),
            McpHudStatusKind::Ready => ("ready", ready_style),
            // Waiting on the user's browser OAuth — warn (yellow), not err
            // (red): the server is not broken, it just needs authentication.
            McpHudStatusKind::AuthPending => ("auth pending", styles.warn),
            // The one row that means a capability is gone: BOLD red, matching
            // the `sources degraded` headline in LIVE.
            McpHudStatusKind::Failed => ("failed", styles.err.add_modifier(Modifier::BOLD)),
        };
        // The closing glyph belongs to the final printed line: the last row only
        // when nothing is hidden, otherwise the `+N more` line closes the tree.
        let is_last = idx + 1 == visible && hidden == 0;
        let mcp_branch = tree_glyph(
            theme,
            if is_last { "  └ " } else { "  ├ " },
            if is_last { "  - " } else { "  +- " },
        );
        let mut spans = vec![
            Span::styled(mcp_branch, styles.muted),
            Span::styled(status_dot(theme), mcp_dot_style),
            Span::styled(status.name, styles.value),
            Span::styled(" · ", styles.muted),
            Span::styled(label, label_style),
        ];
        if let Some(message) = status.message {
            spans.push(Span::styled(" · ", styles.muted));
            spans.push(Span::styled(
                truncate_to_cells(&message, usize::from(width).saturating_sub(18)),
                styles.muted,
            ));
        }
        lines.push(Line::from(spans));
    }
    if hidden > 0 {
        let more_branch = tree_glyph(theme, "  └ ", "  - ");
        lines.push(Line::from(vec![
            Span::styled(more_branch, styles.muted),
            Span::styled(format!("+{hidden} more"), styles.muted),
        ]));
    }
}

#[allow(clippy::too_many_lines)]
fn push_workflow_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let Some(flow) = hud.workflow.as_ref() else {
        return;
    };

    let state_style = workflow_status_style(&flow.status, theme);
    lines.push(section_line("workflow", width, theme));
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled(flow.status.clone(), state_style),
        Span::styled(" ", styles.muted),
        // Completion percent alone (the redundant "Y% left" half is dropped — it
        // is always 100−X and read as a broken "0%/100%" before any agent finished).
        // BOLD because it is the one number the whole section resolves to.
        Span::styled(
            format!("{}%", flow.progress_percent),
            styles.cyan.add_modifier(Modifier::BOLD),
        ),
    ]));

    if flow.phases.is_empty() {
        // No phase structure (a plain `SpawnMultiAgent` fan-out): the compact
        // aggregate current-phase / progress / next lines.
        let max_phase_len = usize::from(width).saturating_sub(17).max(8);
        let phase = truncate_to_cells(&flow.current_phase, max_phase_len);
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(
                format!("{}/{} ", flow.current_phase_index, flow.total_phases),
                styles.muted,
            ),
            Span::styled(phase, styles.value),
            Span::styled(" ", styles.muted),
            Span::styled(
                format!("[{}]", flow.current_phase_status),
                workflow_status_style(&flow.current_phase_status, theme),
            ),
        ]));

        let has_next = flow.next_phase.is_some();
        let progress_branch = tree_glyph(
            theme,
            if has_next { "  ├ " } else { "  └ " },
            if has_next { "  +- " } else { "  - " },
        );
        lines.push(Line::from(vec![
            Span::styled(progress_branch, styles.muted),
            Span::styled("progress ", styles.muted),
            Span::styled(
                format!(
                    "{}% · {}/{} phases",
                    flow.progress_percent, flow.completed_phases, flow.total_phases
                ),
                styles.value,
            ),
        ]));

        if let Some(next) = flow.next_phase.as_deref() {
            let max_next_len = usize::from(width).saturating_sub(12).max(8);
            lines.push(Line::from(vec![
                Span::styled(child_glyph(theme), styles.muted),
                Span::styled("next ", styles.muted),
                Span::styled(truncate_to_cells(next, max_next_len), styles.value),
            ]));
        }
    } else {
        // Multi-phase `Workflow`: the always-on Fleet — one progress bar per
        // phase, so the fan-out → reduce → synthesize pipeline is visible at a
        // glance instead of hidden behind Ctrl+O. The bar color alone encodes
        // phase status, keeping the phase labels typographically quiet.
        const BAR_CELLS: usize = 10;
        let max_id_len = usize::from(width)
            .saturating_sub(BAR_CELLS + 16)
            .max(6);
        for phase in &flow.phases {
            let status_style = workflow_status_style(&phase.status, theme);
            let bar = fleet_phase_bar(phase.terminal(), phase.total, BAR_CELLS, theme);
            let id_style = styles.value;
            let mut spans = vec![
                Span::styled(indent_glyph(theme), styles.muted),
                Span::styled(bar, status_style),
                Span::styled(" ", styles.muted),
                Span::styled(truncate_to_cells(&phase.id, max_id_len), id_style),
                Span::styled(
                    format!(" {}/{}", phase.terminal(), phase.total),
                    styles.muted,
                ),
            ];
            if phase.failed > 0 {
                // A space on BOTH sides of the separator: the tight "·1"
                // renders like "-1" at sidebar font sizes and reads as a
                // negative counter (live user report).
                spans.push(Span::styled(
                    format!(" · {} failed", phase.failed),
                    workflow_status_style("failed", theme),
                ));
            }
            lines.push(Line::from(spans));
        }
    }

    let terminal_agents = flow.completed_agents.saturating_add(flow.failed_agents);
    let mut meta = if flow.total_agents > 0 {
        format!("{terminal_agents}/{} agents", flow.total_agents)
    } else {
        "0 agents".to_string()
    };
    if flow.running_agents > 0 {
        let _ = write!(meta, " · {} running", flow.running_agents);
    }
    if flow.failed_agents > 0 {
        let _ = write!(meta, " · {} failed", flow.failed_agents);
    }
    if !flow.mode.is_empty() {
        let _ = write!(meta, " · {}", flow.mode);
    }
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled(meta, styles.muted),
    ]));
    lines.push(Line::default());
}

#[derive(Debug, Clone, Copy)]
struct WorkMetric {
    label: &'static str,
    value: u32,
    order: u8,
}

fn sorted_work_metrics(hud: &HudState, changed_files: usize) -> Vec<WorkMetric> {
    let mut metrics = [
        WorkMetric {
            label: "read",
            value: hud.read_count,
            order: 0,
        },
        WorkMetric {
            label: "edit",
            value: hud.edit_count,
            order: 1,
        },
        WorkMetric {
            label: "run",
            value: hud.bash_count,
            order: 2,
        },
        WorkMetric {
            label: "files",
            value: u32::try_from(changed_files).unwrap_or(u32::MAX),
            order: 3,
        },
    ]
    .into_iter()
    .filter(|metric| metric.value > 0)
    .collect::<Vec<_>>();
    metrics.sort_by(|left, right| {
        right
            .value
            .cmp(&left.value)
            .then_with(|| left.order.cmp(&right.order))
    });
    metrics
}

fn push_lsp_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    if hud.lsp_servers.is_empty() {
        // Every other section disappears when it has nothing to report; this one
        // used to spend two rows saying "disabled", which is the panel's default
        // state and therefore not news.
        return;
    }
    lines.push(section_rule(
        "lsp",
        Some(&hud.lsp_servers.len().to_string()),
        styles.accent,
        width,
        theme,
    ));
    let lsp_visible_count = hud.lsp_servers.len().min(4);
    for (idx, server) in hud.lsp_servers.iter().take(4).enumerate() {
        let style = lsp_status_style(&server.status, theme);
        let is_last = idx + 1 == lsp_visible_count;
        let lsp_branch = tree_glyph(
            theme,
            if is_last { "  └ " } else { "  ├ " },
            if is_last { "  - " } else { "  +- " },
        );
        // Reserve the branch + dot + status so a long language name truncates
        // instead of wrapping the status onto its own unindented row.
        let reserved = display_width(lsp_branch)
            + display_width(status_dot(theme))
            + 1
            + display_width(&server.status);
        let language = truncate_to_cells(
            &server.language,
            usize::from(width).saturating_sub(reserved).max(4),
        );
        lines.push(Line::from(vec![
            Span::styled(lsp_branch, styles.muted),
            Span::styled(status_dot(theme), style),
            Span::styled(language, styles.value),
            Span::styled(" ", styles.muted),
            Span::styled(server.status.clone(), style),
        ]));
    }
    lines.push(Line::default());
}

fn push_changes_section(
    lines: &mut Vec<Line<'_>>,
    body_height: u16,
    width: u16,
    state: &SidebarState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    if state.total_changed == 0 {
        return;
    }

    let changes_label = if state.total_changed > state.changed_files.len() {
        format!(
            "{} (showing {})",
            state.total_changed,
            state.changed_files.len()
        )
    } else {
        state.total_changed.to_string()
    };
    lines.push(section_rule(
        "changes",
        Some(&changes_label),
        styles.accent,
        width,
        theme,
    ));
    // Session diff magnitude, in the same green/red the per-file tallies use, so
    // the header answers "how big is this change" before any path is read.
    let (adds, rems) = changed_totals(&state.changed_files);
    if adds > 0 || rems > 0 {
        let mut spans = vec![Span::styled(indent_glyph(theme), styles.muted)];
        if adds > 0 {
            spans.push(Span::styled(format!("+{adds}"), styles.ok));
        }
        if rems > 0 {
            if adds > 0 {
                spans.push(Span::styled(" ", styles.muted));
            }
            spans.push(Span::styled(format!("-{rems}"), styles.err));
        }
        spans.push(Span::styled(" lines", styles.muted));
        lines.push(Line::from(spans));
    }

    let file_rows = body_height.saturating_sub(u16::try_from(lines.len()).unwrap_or(u16::MAX));
    let max_scroll = u16::try_from(state.changed_files.len())
        .unwrap_or(u16::MAX)
        .saturating_sub(file_rows);
    let skip = usize::from(state.scroll.min(max_scroll));
    let take = usize::from(file_rows);
    // Number of rows that will actually render (used only for `is_last`).
    // Arithmetic avoids a second pass over the window each frame.
    let visible_files = visible_window_len(state.changed_files.len(), skip, take);
    for (idx, file) in state.changed_files.iter().skip(skip).take(take).enumerate() {
        let (indicator, indicator_style) = match file.status {
            FileStatus::Modified => ("~", styles.warn),
            FileStatus::Added => ("+", styles.ok),
            FileStatus::Deleted => ("-", styles.err),
        };

        // Reserve room for a trailing ` +N -M` tally so the path truncates
        // before it rather than colliding with it.
        let tally = change_tally_label(file);
        let reserved = 6 + tally.as_ref().map_or(0, |t| t.chars().count() + 1);
        let max_path_len = usize::from(width).saturating_sub(reserved);
        let display_path = truncate_path(&file.path, max_path_len);
        let is_last = idx + 1 == visible_files;
        let file_branch = tree_glyph(
            theme,
            if is_last { "  └ " } else { "  ├ " },
            if is_last { "  - " } else { "  +- " },
        );
        let mut spans = vec![
            Span::styled(file_branch, styles.muted),
            Span::styled(indicator, indicator_style),
            Span::raw(" "),
            Span::styled(display_path, styles.value),
        ];
        if file.adds > 0 {
            spans.push(Span::styled(format!(" +{}", file.adds), styles.ok));
        }
        if file.rems > 0 {
            spans.push(Span::styled(format!(" -{}", file.rems), styles.err));
        }
        lines.push(Line::from(spans));
    }
}

/// Rows a `skip(skip).take(take)` window yields over a `total`-element slice —
/// i.e. `iter().skip(skip).take(take).count()` without the extra pass. Used to
/// pick the last-rendered file row's terminal-branch glyph each frame.
const fn visible_window_len(total: usize, skip: usize, take: usize) -> usize {
    let remaining = total.saturating_sub(skip);
    if remaining < take { remaining } else { take }
}

/// Session-wide `(adds, rems)` across the visible changed files.
fn changed_totals(files: &[ChangedFile]) -> (usize, usize) {
    files.iter().fold((0, 0), |(adds, rems), file| {
        (adds + file.adds, rems + file.rems)
    })
}

/// `Some("+N -M")` line-magnitude label for a file with a non-zero tally, used
/// only to size the path column (the spans are styled per-sign when rendered).
fn change_tally_label(file: &ChangedFile) -> Option<String> {
    match (file.adds, file.rems) {
        (0, 0) => None,
        (a, 0) => Some(format!("+{a}")),
        (0, r) => Some(format!("-{r}")),
        (a, r) => Some(format!("+{a} -{r}")),
    }
}

/// Padding of the sidebar panel — shared by [`draw`] and
/// [`workflow_section_on_screen`] so the probe's geometry can never drift
/// from the real render.
const PANEL_PADDING: Padding = Padding::new(2, 1, 0, 0);

/// The padded inner rect and the body row budget (after the bottom footer
/// reservation) for a sidebar drawn into `area`.
fn panel_body_metrics(area: Rect) -> (Rect, u16) {
    let inner = Block::default().padding(PANEL_PADDING).inner(area);
    let body_height = if inner.height >= FOOTER_MIN_HEIGHT {
        inner.height - FOOTER_ROWS
    } else {
        inner.height
    };
    (inner, body_height)
}

/// `true` when a sidebar drawn into `area` for this state actually gets the
/// workflow phase line on screen. The body is top-anchored and unscrollable:
/// the header + session section above can push the workflow section past the
/// visible budget on short terminals — and further whenever the session
/// section grows (rate-limit rows, auth origin) — in which case the HUD must
/// keep its dedicated workflow row instead of trusting the sidebar to carry
/// the phase. Replays the real section builders above the workflow section
/// and counts *wrapped rows* with the same [`wrapped_row_count`] the clamp
/// uses — line count alone under-counts whenever a session line (rate-limit
/// gauge, long branch name) soft-wraps in the narrow panel. Note
/// [`wrapped_row_count`]'s `div_ceil` is a word-wrap approximation: measured
/// safe for today's short-word session lines, but re-check this envelope
/// before adding session rows made of many medium-length words.
pub(crate) fn workflow_section_on_screen(area: Rect, hud: &HudState, theme: &Theme) -> bool {
    if hud.workflow.is_none() || area.width == 0 || area.height == 0 {
        return false;
    }
    let (inner, body_height) = panel_body_metrics(area);
    if inner.width == 0 || body_height == 0 {
        return false;
    }
    let styles = SidebarStyles::new(theme);
    let mut lines: Vec<Line<'_>> = Vec::new();
    push_header(&mut lines, inner.width, hud, theme, styles);
    push_session_section(&mut lines, inner.width, hud, theme, styles);
    let header_rows: usize = lines
        .iter()
        .map(|line| wrapped_row_count(line, inner.width))
        .sum();
    // The section header plus the phase line right below it must both land
    // inside the budget, keeping one row of slack for the clamp's "+N more"
    // marker (the workflow rows themselves are truncate_to_cells-bounded, so
    // they never wrap). Under-estimating is safe: the HUD grants its
    // dedicated row and the phase shows on both surfaces for one boundary
    // row, never on none. Four rows: the section rule, the status/percent
    // row under it, the phase line itself, and one row of clamp slack.
    header_rows + 4 <= usize::from(body_height)
}

/// How the panel meets the surface behind it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PanelFrame {
    /// Layout column (fullscreen): the panel owns its strip down to the
    /// terminal's bottom edge, so the left rail alone carries the boundary.
    Column,
    /// Floating card (inline): the panel is painted over the transcript, so it
    /// hugs its content and closes with a rule above and below.
    Card,
}

/// Where each piece of a drawn panel lands, once the content has been measured.
struct PanelGeometry {
    /// The surface actually painted — cleared, washed, and framed.
    panel: Rect,
    /// Content rows inside [`Self::panel`].
    body: Rect,
    /// Legend rows, when the height budget keeps them.
    footer: Option<Rect>,
}

/// Draw the sidebar into `area` using the given theme.
///
/// The sidebar renders an inspector surface: a two-column identity header,
/// colored pressure gauges, adaptive live sections, and compact interaction
/// hints — anchored to the transcript by a single left rule, with the section
/// boundaries carried by inlaid `─ TITLE ───` heading rules instead of boxes.
pub fn draw(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    state: &SidebarState,
    hud: &HudState,
    theme: &Theme,
) -> Option<Rect> {
    draw_panel(frame, area, state, hud, theme, PanelFrame::Column)
}

/// Draw the sidebar as a floating card over the transcript (inline mode).
///
/// Unlike [`draw`], the card owns its whole surface: it measures its content
/// first, shrinks `area` to what it will actually paint, and only then clears.
/// Painted at full viewport height it wiped the transcript's right columns
/// behind rows the panel had no content for, and trailed a lone left rail down
/// to the bottom edge — a boundary around nothing.
pub fn draw_floating(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    state: &SidebarState,
    hud: &HudState,
    theme: &Theme,
) -> Option<Rect> {
    draw_panel(frame, area, state, hud, theme, PanelFrame::Card)
}

fn draw_panel(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    state: &SidebarState,
    hud: &HudState,
    theme: &Theme,
    kind: PanelFrame,
) -> Option<Rect> {
    if area.width == 0 || area.height == 0 {
        return None;
    }

    // A card spends its first and last row closing itself, so its content is
    // measured against what those two rules leave behind.
    let card = kind == PanelFrame::Card && area.height >= MIN_CARD_HEIGHT;
    let content_area = if card {
        Rect::new(
            area.x,
            area.y.saturating_add(1),
            area.width,
            area.height - CARD_RULE_ROWS,
        )
    } else {
        area
    };
    let (inner, body_height) = panel_body_metrics(content_area);
    if inner.width == 0 || inner.height == 0 {
        return None;
    }

    // Bottom-anchored interaction legend. It uses two content rows without a
    // separator rule and disappears on short terminals so metadata always wins.
    let show_footer = inner.height >= FOOTER_MIN_HEIGHT;
    let body = Rect::new(inner.x, inner.y, inner.width, body_height);

    let styles = SidebarStyles::new(theme);
    let mut lines = panel_lines(inner.width, body.height, state, hud, theme, styles);

    // The body is top-anchored and unscrollable, so tall live state (lsp /
    // changes) would otherwise clip off the bottom edge with no signal. Trim to
    // the visible budget and drop in a dim `+N more` row so the clip is visible.
    let overflowed = clamp_lines_with_overflow(&mut lines, body, styles.muted);

    // Measure before painting. Content that overflows already fills the panel,
    // so it keeps today's geometry — full height, `+N more` marker, legend
    // pinned to the bottom edge.
    let geometry = if card && !overflowed {
        hug_to_content(area, inner, &lines, show_footer)
    } else {
        PanelGeometry {
            panel: area,
            body,
            footer: show_footer.then(|| {
                Rect::new(
                    inner.x,
                    inner.y + inner.height - FOOTER_ROWS,
                    inner.width,
                    FOOTER_ROWS,
                )
            }),
        }
    };

    // The panel keeps its code_bg wash — no box, no glass. The wash alone was
    // enough while the sidebar split the layout, but inline mode floats it over
    // the transcript, and on terminals whose background already matches
    // `code_bg` (and under NO_COLOR, where the wash is `Reset`) the floating
    // edge had no boundary at all: text simply changed subject mid-row. One
    // 1-column rule in the resting-border blue anchors that edge. It lives
    // inside the panel's existing 2-column left padding, so no geometry moves.
    let surface = Style::default().bg(theme.palette.code_bg);
    if kind == PanelFrame::Card {
        // A floating panel must blank what it covers: `Block` only sets a
        // style, so without this the transcript's text shows through the wash.
        // Clearing `geometry.panel` — the hugged rect, not the caller's
        // full-height one — is what keeps the reading column beside a short
        // panel intact.
        frame.render_widget(Clear, geometry.panel);
    }
    frame.render_widget(
        Block::default().style(surface).padding(PANEL_PADDING),
        geometry.panel,
    );
    if card {
        draw_card_rules(frame, geometry.panel, theme);
    }
    draw_left_anchor_rule(frame, rail_rect(geometry.panel, card), theme);

    // Indentation is the panel's hierarchy: `  ctx …` sits *inside* SESSION and
    // `   ⤷ …` hangs off its agent row. `Wrap { trim: true }` strips exactly
    // that leading whitespace, which flattened every section to one column and
    // left a last-agent continuation row (whose prefix is all spaces) dangling
    // under the `└─` with no indent at all. So the rail keeps its indentation —
    // except on the narrowest rails, where every row's budget was tuned against
    // the trimmed width and honoring the indent costs two cells the content
    // cannot spare. Below the 32-column breakpoint the panel falls back to the
    // flat, flush-left layout that fits.
    if inner.width < NARROW_RAIL_COLS {
        // A narrow rail shows one fact per row instead of wrapping. Wrapping here
        // produced orphans — `2/4 read-code [running]` became `2/4 read-co…` and a
        // bare `[running]`, `progress 25% · 1/4 phases` became `… 1/4` and a bare
        // `phases` — and because `trim` stripped the leading whitespace those
        // continuations also lost the indent that places them inside their
        // section, so the column stopped reading as a list at all. Flattening and
        // truncating keeps every row anchored to its section and each row a
        // complete-enough phrase, which is what `push_build_line` already chose to
        // do for the one row that always overran.
        let budget = usize::from(inner.width);
        let flattened: Vec<Line<'_>> = lines
            .into_iter()
            .map(|line| crate::tui::text_metrics::truncate_line_to_cells(flatten_indent(line), budget))
            .collect();
        frame.render_widget(
            Paragraph::new(flattened).style(surface),
            geometry.body,
        );
    } else {
        // A roomy rail keeps its indentation and lets a long row wrap *inside* its
        // section: `trim: false` preserves the leading whitespace that expresses
        // the hierarchy, so a continuation stays in its own column rather than
        // dangling flush-left under a `└─`.
        frame.render_widget(
            Paragraph::new(crate::tui::blocks::wrap_rows(&lines, geometry.body.width))
                .style(surface),
            geometry.body,
        );
    }

    if let Some(footer_rect) = geometry.footer {
        let footer = Paragraph::new(footer_lines(theme, inner.width)).style(surface);
        frame.render_widget(footer, footer_rect);
    }
    Some(geometry.panel)
}

/// Build the ledger body for a panel `inner_width` wide with `body_height` rows
/// of budget. Split out of [`draw_panel`] so the geometry decision above it can
/// measure the real lines instead of guessing at them.
fn panel_lines(
    inner_width: u16,
    body_height: u16,
    state: &SidebarState,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(usize::from(body_height));
    push_header(&mut lines, inner_width, hud, theme, styles);
    push_stale_binary_section(&mut lines, inner_width, hud, theme, styles);
    push_session_section(&mut lines, inner_width, hud, theme, styles);
    push_workflow_section(&mut lines, inner_width, hud, theme, styles);
    push_automation_section(&mut lines, inner_width, hud, theme, styles);
    push_live_activity_section(
        &mut lines,
        inner_width,
        hud,
        state.total_changed,
        theme,
        styles,
    );
    push_agents_section(&mut lines, inner_width, body_height, state, hud, theme, styles);
    push_todo_section(&mut lines, inner_width, hud, theme, styles);
    push_activity_section(
        &mut lines,
        inner_width,
        hud,
        state.total_changed,
        state.detail_expanded,
        theme,
        styles,
    );
    push_lsp_section(&mut lines, inner_width, hud, theme, styles);
    push_changes_section(&mut lines, body_height, inner_width, state, theme, styles);
    lines
}

/// Shrink a floating card to the rows it actually fills: the wrapped content,
/// the legend when it survives, and the two closing rules. [`PANEL_PADDING`] is
/// horizontal-only, so nothing else is owed vertical space.
fn hug_to_content(
    area: Rect,
    inner: Rect,
    lines: &[Line<'_>],
    show_footer: bool,
) -> PanelGeometry {
    let measured: usize = lines
        .iter()
        .map(|line| wrapped_row_count(line, inner.width))
        .sum();
    // `max(1)` so a card is never framed around nothing; the clamp above
    // already guarantees the content fits `inner.height`.
    let used = u16::try_from(measured)
        .unwrap_or(u16::MAX)
        .clamp(1, inner.height);
    // Keep the short-terminal rule, now judged against the *hugged* height: a
    // card only a few rows tall gives those rows to metadata, not to hints.
    let keep_footer = show_footer && used.saturating_add(FOOTER_ROWS) >= FOOTER_MIN_HEIGHT;
    let footer_rows = if keep_footer { FOOTER_ROWS } else { 0 };
    let panel_height = used
        .saturating_add(footer_rows)
        .saturating_add(CARD_RULE_ROWS)
        .min(area.height);
    PanelGeometry {
        panel: Rect::new(area.x, area.y, area.width, panel_height),
        body: Rect::new(inner.x, inner.y, inner.width, used),
        footer: keep_footer
            .then(|| Rect::new(inner.x, inner.y + used, inner.width, FOOTER_ROWS)),
    }
}

/// The rows the left anchor rail spans. On a card the closing rules own the
/// first and last row across the full width, so the rail fills only what lies
/// between them — light rules meeting a light rail, no corner glyphs.
fn rail_rect(panel: Rect, card: bool) -> Rect {
    if card {
        Rect::new(
            panel.x,
            panel.y.saturating_add(1),
            1,
            panel.height.saturating_sub(CARD_RULE_ROWS),
        )
    } else {
        Rect::new(panel.x, panel.y, 1, panel.height)
    }
}

/// Close a floating card with a light `─` rule on its top and bottom rows,
/// each starting on the corner that joins it to the left anchor rail (`-` and
/// `+` under `NO_COLOR`, like every other rule in the panel).
fn draw_card_rules(frame: &mut ratatui::Frame<'_>, panel: Rect, theme: &Theme) {
    if panel.height < MIN_CARD_HEIGHT || panel.width == 0 {
        return;
    }
    let dash = tree_glyph(theme, glyphs::HORIZONTAL_RULE, glyphs::HORIZONTAL_RULE_NC);
    let span = dash.repeat(usize::from(panel.width).saturating_sub(1));
    let style = Style::default()
        .fg(theme.palette.border)
        .bg(theme.palette.code_bg);
    // Column 0 is the joint, not more rule. The rail deliberately stops a row
    // short of each rule (see `rail_rect`), so painting a dash here overwrote
    // the one cell that closes the bracket — leaving the frame as three orphan
    // strokes, broken at exactly the two corners the eye checks first.
    let top = tree_glyph(theme, glyphs::CARD_CORNER_TOP, glyphs::CARD_CORNER_NC);
    let bottom = tree_glyph(theme, glyphs::CARD_CORNER_BOTTOM, glyphs::CARD_CORNER_NC);
    for (y, corner) in [(panel.y, top), (panel.bottom() - 1, bottom)] {
        frame.render_widget(
            Paragraph::new(Line::from(format!("{corner}{span}"))).style(style),
            Rect::new(panel.x, y, panel.width, 1),
        );
    }
}

/// Paint the panel's 1-column left anchor (`│` per row, `|` under `NO_COLOR`)
/// in the resting-border blue. Occupies column 0 of the panel's own left
/// padding, so it costs no content cells.
fn draw_left_anchor_rule(frame: &mut ratatui::Frame<'_>, rail: Rect, theme: &Theme) {
    let glyph = tree_glyph(theme, glyphs::VERTICAL_SEP, glyphs::VERTICAL_SEP_NC);
    let rows: Vec<Line<'static>> = (0..rail.height).map(|_| Line::from(glyph)).collect();
    frame.render_widget(
        Paragraph::new(rows).style(
            Style::default()
                .fg(theme.palette.border)
                .bg(theme.palette.code_bg),
        ),
        rail,
    );
}

/// Number of terminal rows `line` occupies once word-wrapped into `width`
/// cells. A blank line still costs one row. Indentation counts either way: the
/// span widths are summed as authored, which matches the untrimmed rail exactly
/// and over-counts by at most the two-cell indent on the trimmed narrow one —
/// erring toward reserving a row, never toward clipping one.
fn wrapped_row_count(line: &Line<'_>, width: u16) -> usize {
    let width = usize::from(width).max(1);
    let cells: usize = line
        .spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum();
    cells.div_ceil(width).max(1)
}

/// Trim `lines` so the (word-wrapped) content fits inside `body`, replacing the
/// last visible row with a dim `+N more` indicator when rows are dropped.
///
/// The sidebar body is top-anchored and has no whole-panel scroll, so without
/// this a tall stack of live sections (lsp + changes) clips silently off the
/// bottom. The indicator makes the hidden state legible.
///
/// Returns `true` when rows were dropped — a floating card that overflows keeps
/// the full height it was given instead of hugging.
fn clamp_lines_with_overflow(lines: &mut Vec<Line<'_>>, body: Rect, muted: Style) -> bool {
    let budget = usize::from(body.height);
    if budget == 0 {
        let had_lines = !lines.is_empty();
        lines.clear();
        return had_lines;
    }

    let original_len = lines.len();

    // Walk lines, accumulating wrapped rows until the budget is exhausted.
    let mut used = 0usize;
    let mut keep = 0usize;
    for line in lines.iter() {
        let rows = wrapped_row_count(line, body.width);
        if used + rows > budget {
            break;
        }
        used += rows;
        keep += 1;
    }
    if keep == original_len {
        return false; // everything fits — no indicator needed.
    }

    // Reserve the final visible row for the indicator. `keep` counts whole
    // lines, so peel back until that one extra row fits the budget.
    lines.truncate(keep);
    while !lines.is_empty() && used + 1 > budget {
        if let Some(last) = lines.pop() {
            used -= wrapped_row_count(&last, body.width);
        }
    }
    let hidden = original_len - lines.len();
    lines.push(Line::from(Span::styled(
        format!("  +{hidden} more"),
        muted.add_modifier(Modifier::ITALIC),
    )));
    true
}

/// Running-agent tree section: a collapsible header (`✦ N agents · meta`) and,
/// when expanded, a per-agent breakdown (name/model, status/elapsed/activity).
/// Extracted from [`draw`] to keep the rail's section flow readable.
#[allow(clippy::too_many_lines)]
/// Which competing field a meta-row segment is, so the allocator's per-index
/// grant can be rendered back in visual order with the right style.
enum MetaKind {
    Model,
    Activity,
}

#[allow(
    clippy::too_many_lines,
    reason = "one cohesive fleet renderer: header + per-agent title/meta/narrow \
              rows assembled in order; splitting would scatter shared row state"
)]
fn push_agents_section(
    lines: &mut Vec<Line<'_>>,
    inner_width: u16,
    body_height: u16,
    state: &SidebarState,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    if hud.running_agents == 0 {
        return;
    }
    let SidebarStyles {
        muted,
        ok,
        warn,
        err,
        cyan,
        ..
    } = styles;
    let agent_label = Style::new().fg(theme.palette.info);
    // Chevron flows through `glyphs::CHEVRON_*` so it matches the prototype
    // (▾/▸) and degrades to `v`/`>` under NO_COLOR.
    let chevron = if state.agents_expanded {
        tree_glyph(theme, glyphs::CHEVRON_DOWN, glyphs::CHEVRON_DOWN_NC)
    } else {
        tree_glyph(theme, glyphs::CHEVRON_RIGHT, glyphs::CHEVRON_RIGHT_NC)
    };
    // Key hints (chevron, `Ctrl+A`) stay dim; the state/model summary reads as
    // secondary *content*, so it takes muted — one step up from the hints.
    let hint_style = Style::new().fg(theme.palette.dim);
    let meta_style = styles.label;
    let display_agent_count = if hud.agents.is_empty() {
        usize::from(hud.running_agents)
    } else {
        hud.agents.len()
    };
    let count_label = format!("{display_agent_count} agents");
    let meta = agent_header_meta(&hud.agents, !state.agents_expanded);
    let fixed_width = display_width(indent_glyph(theme))
        + display_width(chevron)
        + 1
        + display_width(tree_glyph(
            theme,
            glyphs::ZO_SPARK,
            glyphs::ZO_SPARK_NC,
        ))
        + 1
        + display_width(&count_label)
        + display_width("  Ctrl+A");
    let meta_label = meta.and_then(|meta| {
        let available = usize::from(inner_width).saturating_sub(fixed_width + 3);
        (available >= 10).then(|| truncate_to_cells(&meta, available))
    });
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), muted),
        Span::styled(format!("{chevron} "), hint_style),
        Span::styled(
            format!(
                "{} ",
                tree_glyph(theme, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC)
            ),
            agent_label,
        ),
        Span::styled(count_label, agent_label),
        Span::styled(
            meta_label
                .as_deref()
                .map_or_else(String::new, |meta| format!(" · {meta}")),
            meta_style,
        ),
        Span::styled("  Ctrl+A", hint_style),
    ]));
    // Tree body — rendered only when expanded. Per-agent breakdown is
    // zo 의 차별화 포인트지만 좁은 터미널 / 다른 sidebar 정보에
    // 집중하고 싶을 때 collapse 가능. agents_expanded 는 SidebarState
    // 가 보관하므로 sidebar 자체를 닫았다 열어도 상태 유지.
    if state.agents_expanded {
        // Scale the agent list to the available vertical space instead of a flat
        // cap: each agent is ~2 rows, and the fleet shares the body with the other
        // sections, so allow roughly a fifth of the body height in agents, clamped
        // to a sensible 3..=12. `clamp_lines_with_overflow` is the hard backstop
        // that trims to the real remaining height on a small monitor, so this only
        // needs to pick a fair share, not an exact fit.
        let max_visible = (usize::from(body_height) / 5).clamp(3, 12);
        let total = hud.agents.len();
        let visible = total.min(max_visible);
        for (idx, agent) in hud.agents.iter().take(visible).enumerate() {
            let is_last = idx + 1 == visible && visible == total;
            // Pi's tree vocabulary: the connector carries its horizontal arm
            // (`├─`/`└─`) so the name reads as hanging off the trunk rather
            // than floating one column right of a bare corner.
            let branch = tree_glyph(theme, if is_last { "└─" } else { "├─" }, "|-");
            let vertical = tree_glyph(
                theme,
                if is_last { " " } else { "│" },
                if is_last { " " } else { "|" },
            );
            let status_style = match agent.status.as_str() {
                "completed" => ok,
                "failed" => err,
                "stopped" => warn,
                _ => cyan,
            };

            // Line 1: name, with the model id in the title when it fits. The
            // allocator arbitrates name vs model so a tight title drops the
            // static model whole and gives the name the full width, instead of
            // always reserving `model+3` and chopping the name to
            // `agent-workflow-too…`. `model_cells` is the granted model width
            // (Some) or None when it was dropped (then the meta line below shows
            // the model as a fallback).
            let model_label = crate::tui::workflow_progress::short_model(agent.model.as_str());
            // `"  " + "├─" + " "` = five cells of prefix before the name.
            let title_budget = usize::from(inner_width).saturating_sub(5);
            let title_grant = if !model_label.is_empty() && inner_width >= 32 {
                let segs = [
                    segments::Seg::flex(display_cells(&agent.name), 10, 3),
                    segments::Seg::flex(display_cells(&model_label) + 3, 8, 1),
                ];
                segments::allocate(title_budget, &segs, 0)
            } else {
                vec![Some(title_budget.max(10)), None]
            };
            let name_cells = title_grant[0].unwrap_or(title_budget).max(10);
            let model_cells = title_grant.get(1).copied().flatten();
            let name = truncate_to_cells(&agent.name, name_cells);
            // Per-agent identity hue (stable hash of the agent id, falling back
            // to its name for legacy manifests that carry no id) + BOLD: in a
            // fan-out the name is how you tell rows apart, and one flat `fg` for
            // every agent made a twelve-agent tree unreadable. The static model
            // beside it is reference data, so it drops to muted.
            let name_style = Style::new()
                .fg(theme.agent_color(agent_hue_key(agent)))
                .add_modifier(Modifier::BOLD);
            let mut agent_title_spans = vec![
                Span::styled(format!("  {branch} "), muted),
                Span::styled(name, name_style),
            ];
            if let Some(mc) = model_cells {
                let model_shown = truncate_to_cells(&model_label, mc.saturating_sub(3).max(1));
                agent_title_spans.push(Span::styled(format!(" ({model_shown})"), styles.label));
            }
            lines.push(Line::from(agent_title_spans));

            // Line 2: Status, elapsed, and active tool
            let elapsed_str = format_elapsed(agent.elapsed_secs);
            let elapsed_len = elapsed_str.len() + 1;
            let bracketed_status = format!("[{}]", agent.status);
            let status_len = bracketed_status.len();

            let mut agent_meta_spans = vec![
                Span::styled(format!("  {vertical}  ⤷ "), muted),
                Span::styled(format!("{elapsed_str} "), muted),
                Span::styled(bracketed_status, status_style),
            ];

            if inner_width >= 32 {
                // Meta line: a fixed prefix (`⤷ {elapsed} [status]`) then the
                // fields that compete for the remainder. Live activity outranks
                // the static model (shown here only if it was dropped from the
                // title) and the token sparkline, so a tight row keeps *what the
                // agent is doing* and drops the rest whole — no `waiting for ap…`.
                let used = 7 + elapsed_len + status_len;
                let budget = usize::from(inner_width).saturating_sub(used);
                let activity = agent.activity_label();
                // The text fields share via the allocator (live activity outranks
                // the static model, which appears here only if it was dropped from
                // the title). The token sparkline is decorative, so it is appended
                // only from the *leftover* after the text is placed — it never
                // crowds the activity into `grep_s…`.
                let mut metas: Vec<(MetaKind, segments::Seg)> = Vec::new();
                if model_cells.is_none() && !model_label.is_empty() {
                    metas.push((
                        MetaKind::Model,
                        segments::Seg::flex(display_cells(&model_label) + 3, 8, 1),
                    ));
                }
                if let Some(activity) = activity {
                    metas.push((
                        MetaKind::Activity,
                        segments::Seg::flex(display_cells(activity) + 3, 8, 3),
                    ));
                }
                let segs: Vec<segments::Seg> = metas.iter().map(|(_, s)| *s).collect();
                let grant = segments::allocate(budget, &segs, 0);
                let mut text_used = 0usize;
                for ((kind, _), cells) in metas.iter().zip(grant.iter()) {
                    let Some(cells) = *cells else { continue };
                    text_used += cells;
                    match kind {
                        MetaKind::Model => {
                            let m = truncate_to_cells(&model_label, cells.saturating_sub(3).max(1));
                            agent_meta_spans.push(Span::styled(format!(" · {m}"), styles.label));
                        }
                        MetaKind::Activity => {
                            let a = truncate_to_cells(
                                activity.unwrap_or(""),
                                cells.saturating_sub(3).max(1),
                            );
                            // Wait phases render in the warning tone so a parked
                            // agent is visibly distinct from a working one.
                            let style = if agent.activity_is_wait() {
                                Style::new().fg(theme.palette.warn)
                            } else {
                                cyan
                            };
                            agent_meta_spans.push(Span::styled(format!(" · {a}"), style));
                        }
                    }
                }
                // Sparkline only if the text left a clear ~7 cells (" " + 6 bars).
                if !agent.token_history.is_empty() && budget.saturating_sub(text_used) >= 7 {
                    let spark = inline_sparkline(&agent.token_history, 6, theme);
                    agent_meta_spans.push(Span::styled(format!(" {spark}"), styles.accent));
                }
                lines.push(Line::from(agent_meta_spans));
            } else {
                lines.push(Line::from(agent_meta_spans));
                // Under narrow layout, stack activity/model on Line 3 to
                // prevent wrapping. Live activity (tool or wait phase)
                // leads so truncation eats the static model name, not the
                // signal that tells the user what the agent is doing.
                if !model_label.is_empty() || agent.activity_label().is_some() {
                    let mut detail = agent.activity_label().unwrap_or("").to_string();
                    if !model_label.is_empty() {
                        if !detail.is_empty() {
                            detail.push_str(" · ");
                        }
                        detail.push_str(&model_label);
                    }
                    let max_detail_len = usize::from(inner_width).saturating_sub(8).max(5);
                    let detail_truncated = truncate_to_cells(&detail, max_detail_len);
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {vertical}    ⤷ "), muted),
                        Span::styled(detail_truncated, cyan),
                    ]));
                }
            }
        }
        if total > visible {
            lines.push(Line::from(vec![
                Span::styled("  └ ".to_string(), muted),
                Span::styled(format!("+{} more", total - visible), muted),
            ]));
        }
    }
}

/// Bottom-anchored interaction legend for the ledger rail.
///
/// Returns exactly [`FOOTER_ROWS`] content rows with no decorative rule. The
/// panel surface and whitespace already establish the boundary; another line
/// would compete with the metadata above it.
fn footer_lines(theme: &Theme, width: u16) -> Vec<Line<'static>> {
    // Pi's key-hint idiom: `dim key` + `muted description`, joined by a muted
    // ` · `. The keys used to render in full `fg`, which made the quietest row
    // in the panel the brightest thing on it.
    let key_style = Style::default().fg(theme.palette.dim);
    let label_style = Style::default().fg(theme.palette.muted);
    let ctrl = if theme.no_color { "^" } else { "\u{2303}" };
    let rows = [
        vec![("drag".to_string(), "copy"), ("click".to_string(), "expand")],
        vec![
            (format!("{ctrl}F"), "find"),
            (format!("{ctrl}P"), "cmds"),
            ("?".to_string(), "help"),
        ],
    ];

    // Hints are dropped whole, never clipped. These rows are laid out from fixed
    // strings, so at the rail's minimum width (24 cells) they overran it and the
    // renderer cut them mid-word — the footer read `drag copy · click exp` and
    // `⌃F find · ⌃P cmds · ?`, which turns a legend into debris. Measuring each
    // `key label` pair with its separator and stopping before the edge keeps
    // whatever fits fully readable, and a truncated *hint* is worse than a
    // missing one: the reader cannot act on half a key name.
    rows.into_iter()
        .map(|row| {
            let budget = usize::from(width);
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(8);
            let mut used = 0usize;
            for (key, label) in row {
                let separator = if spans.is_empty() { "" } else { " \u{00b7} " };
                let hint = format!("{separator}{key} {label}");
                let hint_width = display_width(&hint);
                if used + hint_width > budget {
                    break;
                }
                used += hint_width;
                if !separator.is_empty() {
                    spans.push(Span::styled(separator, label_style));
                }
                spans.push(Span::styled(key, key_style));
                spans.push(Span::styled(format!(" {label}"), label_style));
            }
            Line::from(spans)
        })
        .collect()
}

/// The session context bar: a `width`-cell `▬`/`░` meter tinted by
/// [`gauge_color`]. Renders at every percent including zero — an all-empty bar
/// is the honest picture of a fresh session, and hiding it made the row jump a
/// column the first time usage landed.
fn token_gauge_bar(pct: u64, width: usize, theme: &Theme) -> Span<'static> {
    let pct_u8 = u8::try_from(pct.min(100)).unwrap_or(100);
    let pct_usize = usize::from(pct_u8);
    let filled = ((pct_usize * width) + 50) / 100;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);

    // Fill/empty glyphs are East-Asian **Neutral** (`▬`/`░` → `width_cjk()==1`)
    // so the fixed-`width` bar renders one column per cell even under a `ko_KR`
    // wide-ambiguous tmux; the old `■` (Ambiguous) doubled the filled run and
    // overflowed the sidebar. `NO_COLOR` degrades to `#`/`.`, which is the only
    // way the meter still reads once the ramp color is gone (R10).
    let fill_glyph = glyphs::pick(!theme.no_color, glyphs::GAUGE_FILL, glyphs::GAUGE_FILL_NC);
    let empty_glyph = glyphs::pick(!theme.no_color, glyphs::GAUGE_EMPTY, glyphs::GAUGE_EMPTY_NC);

    let gauge_str = format!("{}{}", fill_glyph.repeat(filled), empty_glyph.repeat(empty));

    // Share the single gauge-color ramp with the rate-limit bars so both gauges
    // flip to amber/red at the same utilization (they previously disagreed:
    // context erred at >=85%, rate-limit at >=80%).
    let color = if theme.no_color {
        theme.palette.fg
    } else {
        gauge_color(pct_u8, theme)
    };

    Span::styled(gauge_str, Style::new().fg(color))
}

fn project_name(hud: &HudState) -> String {
    hud.cwd
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("zo")
        .to_string()
}

fn compact_cwd(hud: &HudState) -> String {
    let Some(parent) = hud.cwd.parent().and_then(|path| path.file_name()) else {
        return "~".to_string();
    };
    let parent = parent.to_string_lossy();
    format!("~/{parent}/{}", project_name(hud))
}

fn format_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 1_000_000 {
        #[allow(clippy::cast_precision_loss)]
        let v = tokens as f64 / 1_000.0;
        format!("{v:.1}k")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let v = tokens as f64 / 1_000_000.0;
        if tokens.is_multiple_of(1_000_000) {
            format!("{v:.1}M")
        } else {
            format!("{v:.2}M")
        }
    }
}

fn status_dot(theme: &Theme) -> &'static str {
    if theme.no_color { "* " } else { "\u{25cf} " }
}

fn tree_glyph(theme: &Theme, unicode: &'static str, ascii: &'static str) -> &'static str {
    if theme.no_color { ascii } else { unicode }
}

/// Key the per-agent identity hue is drawn from: the manifest `agentId` when
/// present, else the agent name. Legacy manifests carry an empty id, and hashing
/// `""` would paint every one of them the same color — the exact failure the
/// per-agent hue exists to prevent.
fn agent_hue_key(agent: &AgentTaskSummary) -> &str {
    if agent.id.is_empty() {
        agent.name.as_str()
    } else {
        agent.id.as_str()
    }
}

fn agent_header_meta(agents: &[AgentTaskSummary], include_models: bool) -> Option<String> {
    if agents.is_empty() {
        return None;
    }

    let mut running = 0usize;
    let mut queued = 0usize;
    let mut failed = 0usize;
    let mut stopped = 0usize;
    let mut other_active = 0usize;
    let mut models = BTreeMap::<String, usize>::new();

    for agent in agents {
        match agent.status.as_str() {
            "running" => running += 1,
            "pending" | "queued" => queued += 1,
            "failed" => failed += 1,
            "stopped" => stopped += 1,
            "completed" => {}
            _ => other_active += 1,
        }

        if include_models {
            let model = crate::tui::workflow_progress::short_model(agent.model.as_str());
            if !model.is_empty() {
                *models.entry(model).or_default() += 1;
            }
        }
    }

    let mut parts = Vec::new();
    push_count_part(&mut parts, running, "running");
    push_count_part(&mut parts, queued, "queued");
    push_count_part(&mut parts, failed, "failed");
    push_count_part(&mut parts, stopped, "stopped");
    push_count_part(&mut parts, other_active, "active");

    if !models.is_empty() {
        let mut model_parts = Vec::new();
        for (model, count) in models.iter().take(3) {
            if *count == 1 {
                model_parts.push(model.clone());
            } else {
                model_parts.push(format!("{model} x{count}"));
            }
        }
        if models.len() > 3 {
            model_parts.push(format!("+{} models", models.len() - 3));
        }
        parts.push(model_parts.join(", "));
    }

    (!parts.is_empty()).then(|| parts.join(" · "))
}

fn push_count_part(parts: &mut Vec<String>, count: usize, label: &str) {
    if count > 0 {
        parts.push(format!("{count} {label}"));
    }
}

/// 누적 셀폭이 `max_cells` 를 넘지 않도록 truncate. 넘으면 `…` 추가.
/// CJK (셀폭 2) 인식 — sidebar agent name 한국어 혼용 대응.
/// Drop a row's leading indent, the way `Wrap { trim: true }` used to.
///
/// The narrow rail's per-row budgets were tuned against the trimmed width, so
/// honouring the two-cell indent there costs content the column cannot spare.
/// Doing it here rather than via `trim` means the *rest* of the row is left alone
/// — `trim` also collapsed the whitespace inside a wrapped continuation, which is
/// what flattened the tree alignment.
fn flatten_indent(line: Line<'_>) -> Line<'_> {
    let mut spans = line.spans;
    while let Some(first) = spans.first() {
        let trimmed = first.content.trim_start();
        if trimmed.is_empty() {
            // An all-whitespace span is pure indent: drop it and look at the next.
            spans.remove(0);
            continue;
        }
        if trimmed.len() != first.content.len() {
            let style = first.style;
            let owned = trimmed.to_string();
            spans[0] = Span::styled(owned, style);
        }
        break;
    }
    Line::from(spans)
}

fn truncate_to_cells(s: &str, max_cells: usize) -> String {
    if max_cells == 0 {
        return String::new();
    }
    let mut acc: usize = 0;
    let mut end: usize = 0;
    let ellipsis = '…';
    let ellipsis_width = 1usize;
    for (i, ch) in s.char_indices() {
        let w = char_width(ch);
        if acc + w > max_cells {
            // ellipsis 가 들어갈 자리 확보를 위해 한 char 더 뒤로
            while end > 0 && acc + ellipsis_width > max_cells {
                let removed_byte = end;
                if let Some((idx, ch)) = s[..removed_byte].char_indices().next_back() {
                    end = idx;
                    acc = acc.saturating_sub(char_width(ch));
                } else {
                    break;
                }
            }
            let mut out = String::with_capacity(end + ellipsis.len_utf8());
            out.push_str(&s[..end]);
            out.push(ellipsis);
            return out;
        }
        acc += w;
        end = i + ch.len_utf8();
    }
    s.to_string()
}

/// Render the last `width` samples of `series` as an inline sparkline using
/// the 8-step block glyph progression `▁▂▃▄▅▆▇█` (a flat `#` run under
/// `NO_COLOR`). Returns an empty string when the series itself is empty so
/// callers can skip the span altogether.
///
/// Why inline glyphs over `ratatui::widgets::Sparkline`: the widget needs a
/// dedicated `Rect` and per-frame `render_widget` call, but the sidebar
/// agent row composes a single `Line` of styled `Span`s — embedding a glyph
/// string preserves the row model and keeps the widget tree shallow.
fn inline_sparkline(series: &[u32], width: usize, theme: &Theme) -> String {
    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if series.is_empty() || width == 0 {
        return String::new();
    }
    let recent = if series.len() > width {
        &series[series.len() - width..]
    } else {
        series
    };
    // NO_COLOR can't render the 8-step block ramp meaningfully (every glyph is
    // the terminal default), so degrade to a flat `#` run like the prototype —
    // the sparkline still reads as "N samples present" without color (R10).
    if theme.no_color {
        return "#".repeat(recent.len());
    }
    let max = recent.iter().copied().max().unwrap_or(1).max(1);
    recent
        .iter()
        .map(|v| {
            let scaled = (u64::from(*v) * (GLYPHS.len() as u64 - 1)) / u64::from(max);
            #[allow(clippy::cast_possible_truncation)]
            let idx = (scaled as usize).min(GLYPHS.len() - 1);
            GLYPHS[idx]
        })
        .collect()
}

fn indent_glyph(theme: &Theme) -> &'static str {
    tree_glyph(theme, "  ", "  ")
}

fn child_glyph(theme: &Theme) -> &'static str {
    tree_glyph(theme, "  └ ", "  - ")
}

/// A section heading in Pi's dialog idiom: the title inlaid into a single
/// full-width rule, `─ SESSION ───────────────`.
///
/// One line does the work the old bare heading needed whitespace for: the rule
/// draws the section boundary in the resting-border blue while the BOLD accent
/// title stays the only thing the eye lands on. `count` rides between the title
/// and the rule (`─ CHANGES 3 ────`) so a section's magnitude is legible from
/// the heading alone.
///
/// Under `NO_COLOR` the rule glyphs are dropped entirely and the heading
/// degrades to plain `SESSION` / `CHANGES 3` text: without color the rule is
/// indistinguishable from content and only spends cells (R10).
fn section_rule(
    title: &str,
    count: Option<&str>,
    count_style: Style,
    width: u16,
    theme: &Theme,
) -> Line<'static> {
    let heading = section_heading_span(title, theme);
    let mut spans = Vec::with_capacity(5);
    if theme.no_color {
        spans.push(heading);
        if let Some(count) = count {
            spans.push(Span::styled(format!(" {count}"), count_style));
        }
        return Line::from(spans);
    }

    let rule_style = Style::default().fg(theme.palette.border);
    // `─ ` lead-in + title + optional ` N` + the single space before the rule.
    let mut used = 2 + display_width(heading.content.as_ref()) + 1;
    spans.push(Span::styled(glyphs::HORIZONTAL_RULE.to_string() + " ", rule_style));
    spans.push(heading);
    if let Some(count) = count {
        used += 1 + display_width(count);
        spans.push(Span::styled(format!(" {count}"), count_style));
    }
    let fill = usize::from(width).saturating_sub(used);
    if fill > 0 {
        spans.push(Span::styled(
            format!(" {}", glyphs::HORIZONTAL_RULE.repeat(fill)),
            rule_style,
        ));
    }
    Line::from(spans)
}

/// [`section_rule`] for a section with no count.
fn section_line(title: &str, width: u16, theme: &Theme) -> Line<'static> {
    section_rule(title, None, Style::default(), width, theme)
}

/// A sidebar section heading.
///
/// Headings are the panel's only navigation, so they carry the accent and
/// uppercase weight instead of the muted tone of the rows they introduce. Before
/// this, every line in the sidebar rendered at the same weight and the sections
/// were only findable by counting blank rows.
fn section_heading_span(title: &str, theme: &Theme) -> Span<'static> {
    Span::styled(
        title.to_uppercase(),
        Style::new()
            .fg(theme.palette.accent)
            .add_modifier(Modifier::BOLD),
    )
}

/// Build the 5h/7d rate-limit gauge lines — one per present window.
fn rate_limit_gauges(rl: RateLimitSnapshot, theme: &Theme, muted: Style) -> Vec<Line<'static>> {
    let now = now_unix();
    let mut lines = Vec::new();
    for (label, window) in [("5h", rl.five_hour), ("7d", rl.seven_day)] {
        let Some(w) = window else {
            continue;
        };
        let pct = w.used_percent();
        let bar_style = Style::new().fg(gauge_color(pct, theme));
        let mut spans = vec![
            Span::styled(indent_glyph(theme), muted),
            Span::styled(format!("{label}  "), muted),
            Span::styled(gauge_bar(w.utilization, CONTEXT_GAUGE_CELLS, theme), bar_style),
            Span::styled(format!(" {pct}%"), gauge_percent_style(u64::from(pct), theme)),
        ];
        if let Some(reset) = w.resets_at_unix {
            spans.push(Span::styled(
                format!("  ↺ {}", format_reset(now, reset)),
                muted,
            ));
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// Build the 429-estimated quota rows for throttled providers — the
/// cross-provider stack under the measured 5h/7d gauge. One line per estimated
/// view row: the provider's rate-limit key, a used-style bar, `~NN%` (the `~`
/// plus a trailing `est` marker say "inferred from 429s, not measured"), and
/// the cool-down countdown when known. Measured (non-estimated) rows are
/// skipped here: the Anthropic windows already render from the streamed
/// snapshot above, and duplicating them would show the same window twice.
/// Rows without a remaining figure are omitted — never a fabricated bar.
fn estimated_quota_gauges(
    views: &[api::quota::ProviderQuotaView],
    theme: &Theme,
    muted: Style,
) -> Vec<Line<'static>> {
    let now = now_unix();
    let mut lines = Vec::new();
    for view in views {
        if !view.estimated {
            continue;
        }
        let Some(remaining) = view.remaining_percent else {
            continue;
        };
        let used = 100u8.saturating_sub(remaining);
        let bar_style = Style::new().fg(gauge_color(used, theme));
        let mut spans = vec![
            Span::styled(indent_glyph(theme), muted),
            Span::styled(format!("{} ", view.provider.rate_limit_key()), muted),
            Span::styled(
                gauge_bar(f64::from(used) / 100.0, CONTEXT_GAUGE_CELLS, theme),
                bar_style,
            ),
            Span::styled(format!(" ~{used}%"), gauge_percent_style(u64::from(used), theme)),
            Span::styled(" est", muted),
        ];
        if let Some(reset) = view.resets_at_unix {
            spans.push(Span::styled(
                format!("  ↺ {}", format_reset(now, reset)),
                muted,
            ));
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// A 10-cell `▬`/`░` utilization bar — degrades to `#`/`.` under `NO_COLOR`.
///
/// The fill glyph is East-Asian **Neutral** (`▬`, `width_cjk()==1`) so the bar
/// keeps its width under a `ko_KR` wide-ambiguous tmux; the old `█` (Ambiguous)
/// painted two columns per filled cell there. Quota and Fleet meters share this
/// compact visual vocabulary; session context remains text-first.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn gauge_bar(utilization: f64, width: usize, theme: &Theme) -> String {
    let filled = (utilization.clamp(0.0, 1.0) * width as f64).round() as usize;
    let filled = filled.min(width);
    let fill_ch = glyphs::pick(!theme.no_color, glyphs::GAUGE_FILL, glyphs::GAUGE_FILL_NC);
    let empty_ch = glyphs::pick(!theme.no_color, glyphs::GAUGE_EMPTY, glyphs::GAUGE_EMPTY_NC);
    let mut bar = String::with_capacity(width * 3);
    for _ in 0..filled {
        bar.push_str(fill_ch);
    }
    for _ in filled..width {
        bar.push_str(empty_ch);
    }
    bar
}

/// Single gauge-color ramp shared by quota utilization bars: calm green under
/// 50%, warn amber under 80%, and error red at/above 80%.
fn gauge_color(pct: u8, theme: &Theme) -> ratatui::style::Color {
    if pct >= 80 {
        theme.palette.error
    } else if pct >= 50 {
        theme.palette.warn
    } else {
        theme.palette.success
    }
}

/// Compact "time until reset" — `2h11m`, `3d`, `45m`, or `now`.
fn format_reset(now: u64, resets_at: u64) -> String {
    if resets_at <= now {
        return "now".to_string();
    }
    let mins = (resets_at - now) / 60;
    if mins >= 1440 {
        let days = mins / 1440;
        let hours = (mins % 1440) / 60;
        if hours > 0 {
            format!("{days}d{hours}h")
        } else {
            format!("{days}d")
        }
    } else if mins >= 60 {
        format!("{}h{:02}m", mins / 60, mins % 60)
    } else {
        format!("{mins}m")
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub(crate) fn permission_style(mode: PermissionMode, theme: &Theme) -> Style {
    let color = match mode {
        PermissionMode::ReadOnly => theme.palette.info,
        // Plan is a read-only planning gate; the reasoning violet sets it
        // apart from plain read-only ("model is drafting a plan") without
        // spending the brand accent on a status badge.
        PermissionMode::Plan => theme.palette.violet,
        // Workspace-write is the everyday default — quiet, not a warning.
        PermissionMode::Workspace => theme.palette.fg,
        // Full access is the dangerous end of the scale: it must read as
        // danger. (The old success-green here inverted the semantic lamp —
        // "danger-full-access" rendered as a reassuring all-clear.)
        PermissionMode::All => theme.palette.error,
    };
    let style = Style::default().fg(color);
    if mode == PermissionMode::All {
        style.add_modifier(Modifier::BOLD)
    } else {
        style
    }
}

fn lsp_status_style(status: &str, theme: &Theme) -> Style {
    match status {
        "connected" => Style::default().fg(theme.palette.success),
        "starting" => Style::default().fg(theme.palette.warn),
        "error" => Style::default().fg(theme.palette.error),
        "disconnected" => Style::default().fg(theme.palette.dim),
        _ => Style::default().fg(theme.palette.fg),
    }
}

fn workflow_status_style(status: &str, theme: &Theme) -> Style {
    match status {
        "running" => Style::default().fg(theme.palette.accent),
        "completed" | "done" | "resumed" => Style::default().fg(theme.palette.success),
        "failed" | "cancelled" | "budget_exhausted" => Style::default().fg(theme.palette.error),
        "pending" => Style::default().fg(theme.palette.dim),
        _ => Style::default().fg(theme.palette.fg),
    }
}

/// A fixed-width progress bar for one Fleet phase: `filled` = terminal agents
/// (completed + failed), the rest empty. Mirrors [`token_gauge_bar`]'s glyphs
/// (`▬`/`░`, or `#`/`.` under `no_color`) so the sidebar reads consistently. A
/// phase with no agents yet renders all-empty rather than a misleading full bar.
fn fleet_phase_bar(terminal: usize, total: usize, cells: usize, theme: &Theme) -> String {
    let filled = if total == 0 {
        0
    } else {
        ((terminal * cells) + total / 2) / total
    }
    .min(cells);
    let fill_char = glyphs::pick(!theme.no_color, glyphs::GAUGE_FILL, glyphs::GAUGE_FILL_NC);
    let empty_char = glyphs::pick(!theme.no_color, glyphs::GAUGE_EMPTY, glyphs::GAUGE_EMPTY_NC);
    let mut bar = String::with_capacity(cells * 3);
    for _ in 0..filled {
        bar.push_str(fill_char);
    }
    for _ in filled..cells {
        bar.push_str(empty_char);
    }
    bar
}

/// Checklist marker + style. The in-progress marker is the only one that takes
/// BOLD: exactly one item is active at a time, and it is what the user is
/// looking for when they glance at the list.
fn todo_marker(status: TodoChecklistStatus, theme: &Theme) -> (&'static str, Style) {
    let active = Style::default()
        .fg(theme.palette.warn)
        .add_modifier(Modifier::BOLD);
    if theme.no_color {
        match status {
            TodoChecklistStatus::Pending => ("[ ]", Style::default().fg(theme.palette.dim)),
            TodoChecklistStatus::InProgress => ("[-]", active),
            TodoChecklistStatus::Completed => ("[x]", Style::default().fg(theme.palette.success)),
        }
    } else {
        match status {
            TodoChecklistStatus::Pending => ("○", Style::default().fg(theme.palette.dim)),
            TodoChecklistStatus::InProgress => ("●", active),
            TodoChecklistStatus::Completed => ("✓", Style::default().fg(theme.palette.success)),
        }
    }
}

pub(crate) const fn permission_label(mode: PermissionMode) -> &'static str {
    mode.label()
}

/// Truncate a file path to fit within `max_len` terminal cells.
///
/// If the full path fits, it is returned as-is. Otherwise the path is
/// shortened to show just the filename (last component), or the
/// filename itself is truncated with a `…` ellipsis if even that is too
/// long. Width is measured in display cells via [`sidebar_char_width`]
/// (CJK = 2) — consistent with [`truncate_to_cells`] — so paths with
/// Hangul / CJK characters don't overflow the panel.
fn truncate_path(path: &str, max_len: usize) -> String {
    if display_cells(path) <= max_len {
        return path.to_string();
    }

    // Try just the filename.
    let filename = path.rsplit('/').next().unwrap_or(path);
    if display_cells(filename) <= max_len {
        return filename.to_string();
    }

    // Truncate filename to fit, appending the ellipsis (1 cell).
    truncate_to_cells(filename, max_len)
}

/// Total display width of `s` in terminal cells (CJK = 2).
fn display_cells(s: &str) -> usize {
    display_width(s)
}

mod segments;

#[cfg(test)]
mod tests;
