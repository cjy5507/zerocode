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
use ratatui::widgets::{Block, Padding, Paragraph, Wrap};

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
    lines.push(aligned_sidebar_line(
        &project,
        styles.value,
        branch,
        styles.cyan,
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

fn push_context_use_line(
    lines: &mut Vec<Line<'_>>,
    theme: &Theme,
    styles: SidebarStyles,
    pct: u64,
) {
    lines.push(Line::from(vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("use  ", styles.muted),
        Span::styled(format!("{pct}%"), styles.value),
    ]));
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

fn push_cost_mode_line(
    lines: &mut Vec<Line<'_>>,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
    inline: bool,
) {
    let cost_prefix = if hud.cost_approx { "~$" } else { "$" };
    let mut spans = vec![
        Span::styled(indent_glyph(theme), styles.muted),
        Span::styled("cost ", styles.muted),
        Span::styled(
            format!("{cost_prefix}{:.2}", hud.cost_usd),
            styles.value,
        ),
    ];
    if inline {
        spans.push(Span::styled("  mode ", styles.muted));
        spans.push(Span::styled(
            permission_label(hud.perm_mode),
            permission_style(hud.perm_mode, theme),
        ));
        lines.push(Line::from(spans));
    } else {
        lines.push(Line::from(spans));
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("mode ", styles.muted),
            Span::styled(
                permission_label(hud.perm_mode),
                permission_style(hud.perm_mode, theme),
            ),
        ]));
    }
}

fn push_session_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    let pct = context_percent(hud);
    lines.push(section_line("session", theme, styles.label));
    if let Some(identity) = hud.session_identity.as_ref() {
        let badge_style = if theme.no_color {
            styles.value
        } else {
            Style::new().fg(identity.color)
        };
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("● ", badge_style),
            Span::styled(identity.name.clone(), styles.value),
        ]));
    }

    if width < 32 {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("ctx  ", styles.muted),
            Span::styled(format_context_used_tokens(hud), styles.value),
            Span::styled(" / ", styles.muted),
            Span::styled(format_tokens(hud.ctx_limit), styles.value),
        ]));
        push_context_use_line(lines, theme, styles, pct);
        push_cache_split_lines(lines, hud, theme, styles, true);
        push_compact_ceiling_line(lines, hud, theme, styles);
        if let Some(rl) = hud.rate_limit {
            lines.extend(rate_limit_gauges(rl, theme, styles.muted, styles.value));
        }
        // Estimated rows render even without a measured Anthropic snapshot —
        // a non-Anthropic main model has no `rate_limit` but can be throttled.
        lines.extend(estimated_quota_gauges(&hud.provider_quotas, theme, styles.muted, styles.value));
        push_auth_line(lines, hud, theme, styles);
        push_cost_mode_line(lines, hud, theme, styles, false);
        return;
    }

    if width >= 38 {
        let used_str = format_context_used_tokens(hud);
        let limit_str = format_tokens(hud.ctx_limit);
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("ctx ", styles.muted),
            Span::styled(format!("{used_str} / {limit_str}"), styles.value),
            Span::styled(format!("  {pct}%"), styles.muted),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("ctx ", styles.muted),
            Span::styled(
                format!(
                    "{} / {}",
                    format_context_used_tokens(hud),
                    format_tokens(hud.ctx_limit)
                ),
                styles.value,
            ),
        ]));
        push_context_use_line(lines, theme, styles, pct);
    }
    push_cache_split_lines(lines, hud, theme, styles, false);
    push_compact_ceiling_line(lines, hud, theme, styles);
    if let Some(rl) = hud.rate_limit {
        lines.extend(rate_limit_gauges(rl, theme, styles.muted, styles.value));
    }
    // Estimated rows render even without a measured Anthropic snapshot — a
    // non-Anthropic main model has no `rate_limit` but can be throttled.
    lines.extend(estimated_quota_gauges(&hud.provider_quotas, theme, styles.muted, styles.value));
    push_auth_line(lines, hud, theme, styles);
    push_cost_mode_line(lines, hud, theme, styles, true);
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
    lines.push(section_line("automation", theme, styles.label));
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

    lines.push(section_line("live", theme, styles.label));
    if let Some(tool) = hud.last_tool.as_deref() {
        let label = truncate_to_cells(tool, usize::from(width).saturating_sub(9).max(8));
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled(
                format!(
                    "{} ",
                    tree_glyph(theme, glyphs::CHEVRON_RIGHT, glyphs::CHEVRON_RIGHT_NC)
                ),
                styles.cyan,
            ),
            Span::styled(label, styles.cyan),
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
        let (label, style) = match mcp_health {
            McpHealth::Degraded => ("sources degraded", styles.err),
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

    lines.push(Line::from(vec![
        Span::styled(section_prefix(theme), styles.muted),
        Span::styled("todo", styles.label),
        Span::styled(format!(" {}", hud.todo_items.len()), styles.muted),
    ]));
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
        lines.push(Line::from(vec![
            Span::styled(indent_glyph(theme), styles.muted),
            Span::styled("done", styles.ok),
            Span::styled(format!(" · {done}/{total}"), styles.muted),
        ]));
    }
    lines.push(Line::default());
}

fn push_activity_section(
    lines: &mut Vec<Line<'_>>,
    width: u16,
    hud: &HudState,
    changed_files: usize,
    theme: &Theme,
    styles: SidebarStyles,
) {
    lines.push(section_line("work", theme, styles.label));
    let metrics = sorted_work_metrics(hud, changed_files);
    let mut work_line = vec![Span::styled(indent_glyph(theme), styles.muted)];
    if metrics.is_empty() {
        work_line.push(Span::styled("idle", styles.muted));
    } else {
        for (idx, metric) in metrics.iter().enumerate() {
            if idx > 0 {
                work_line.push(Span::styled("  ", styles.muted));
            }
            work_line.push(Span::styled(format!("{} ", metric.label), styles.muted));
            work_line.push(Span::styled(metric.value.to_string(), styles.value));
        }
    }
    lines.push(Line::from(work_line));

    push_mcp_sources_section(lines, width, &hud.mcp_servers, theme, styles);
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
    theme: &Theme,
    styles: SidebarStyles,
) {
    const MAX_ROWS: usize = 4;

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
        Span::styled("sources ", styles.muted),
        Span::styled(count_text, mcp_headline_style(summary.health(), styles)),
    ]));

    let mut rows: Vec<McpHudStatus> = servers.iter().map(|s| McpHudStatus::decode(s)).collect();
    // Stable sort: equal severities keep the source list's alphabetical order.
    rows.sort_by_key(|status| mcp_status_severity(status.kind));

    let visible = rows.len().min(MAX_ROWS);
    let hidden = rows.len() - visible;
    let mcp_dot_style = Style::new().fg(theme.palette.violet);
    let ready_style = Style::new().fg(theme.palette.success);
    for (idx, status) in rows.into_iter().take(MAX_ROWS).enumerate() {
        let (label, label_style) = match status.kind {
            McpHudStatusKind::Discovering => ("discovering", styles.warn),
            McpHudStatusKind::Ready => ("ready", ready_style),
            // Waiting on the user's browser OAuth — warn (yellow), not err
            // (red): the server is not broken, it just needs authentication.
            McpHudStatusKind::AuthPending => ("auth pending", styles.warn),
            McpHudStatusKind::Failed => ("failed", styles.err),
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
    lines.push(Line::from(vec![
        Span::styled(section_prefix(theme), styles.muted),
        Span::styled("workflow", styles.label),
        Span::styled(" ", styles.muted),
        Span::styled(flow.status.clone(), state_style),
        Span::styled(" ", styles.muted),
        // Completion percent alone (the redundant "Y% left" half is dropped — it
        // is always 100−X and read as a broken "0%/100%" before any agent finished).
        Span::styled(format!("{}%", flow.progress_percent), styles.cyan),
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
    hud: &HudState,
    theme: &Theme,
    styles: SidebarStyles,
) {
    lines.push(Line::from(vec![
        Span::styled(section_prefix(theme), styles.muted),
        Span::styled("lsp", styles.label),
        if hud.lsp_servers.is_empty() {
            Span::styled("  disabled", styles.muted)
        } else {
            Span::styled(format!(" {}", hud.lsp_servers.len()), styles.muted)
        },
    ]));
    let lsp_visible_count = hud.lsp_servers.len().min(4);
    for (idx, server) in hud.lsp_servers.iter().take(4).enumerate() {
        let style = lsp_status_style(&server.status, theme);
        let is_last = idx + 1 == lsp_visible_count;
        let lsp_branch = tree_glyph(
            theme,
            if is_last { "  └ " } else { "  ├ " },
            if is_last { "  - " } else { "  +- " },
        );
        lines.push(Line::from(vec![
            Span::styled(lsp_branch, styles.muted),
            Span::styled(status_dot(theme), style),
            Span::styled(server.language.clone(), styles.value),
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
            " {} (showing {})",
            state.total_changed,
            state.changed_files.len()
        )
    } else {
        format!(" {}", state.total_changed)
    };
    lines.push(Line::from(vec![
        Span::styled(section_prefix(theme), styles.muted),
        Span::styled("changes", styles.label),
        Span::styled(changes_label, styles.muted),
    ]));

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
    // row, never on none.
    header_rows + 3 <= usize::from(body_height)
}

/// Draw the sidebar into `area` using the given theme.
///
/// The sidebar renders a quiet inspector surface: a two-column identity header,
/// text-first session metrics, adaptive live sections, and compact interaction
/// hints without decorative border or rule chrome.
#[allow(clippy::too_many_lines)] // cohesive sidebar frame render
pub fn draw(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    state: &SidebarState,
    hud: &HudState,
    theme: &Theme,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    // Match the bottom HUD's neutral surface. Whitespace and padding separate
    // the inspector from the transcript; no border or glass effect competes
    // with the content hierarchy.
    let panel = Block::default()
        .style(Style::default().bg(theme.palette.code_bg))
        .padding(PANEL_PADDING);
    frame.render_widget(panel, area);
    let (inner, body_height) = panel_body_metrics(area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    // Bottom-anchored interaction legend. It uses two content rows without a
    // separator rule and disappears on short terminals so metadata always wins.
    let show_footer = inner.height >= FOOTER_MIN_HEIGHT;
    let body = Rect::new(inner.x, inner.y, inner.width, body_height);

    let mut lines: Vec<Line<'_>> = Vec::with_capacity(usize::from(body.height));
    let styles = SidebarStyles::new(theme);
    let SidebarStyles { muted, .. } = styles;

    push_header(&mut lines, inner.width, hud, theme, styles);
    push_stale_binary_section(&mut lines, inner.width, hud, theme, styles);
    push_session_section(&mut lines, inner.width, hud, theme, styles);
    push_workflow_section(&mut lines, inner.width, hud, theme, styles);
    push_automation_section(&mut lines, inner.width, hud, theme, styles);
    push_live_activity_section(
        &mut lines,
        inner.width,
        hud,
        state.total_changed,
        theme,
        styles,
    );
    push_agents_section(&mut lines, inner.width, body.height, state, hud, theme, styles);

    push_todo_section(&mut lines, inner.width, hud, theme, styles);
    push_activity_section(&mut lines, inner.width, hud, state.total_changed, theme, styles);
    push_lsp_section(&mut lines, hud, theme, styles);
    push_changes_section(&mut lines, body.height, inner.width, state, theme, styles);

    // The body is top-anchored and unscrollable, so tall live state (lsp /
    // changes) would otherwise clip off the bottom edge with no signal. Trim to
    // the visible budget and drop in a dim `+N more` row so the clip is visible.
    clamp_lines_with_overflow(&mut lines, body, muted);

    let paragraph = Paragraph::new(lines)
        .style(Style::default().bg(theme.palette.code_bg))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, body);

    if show_footer {
        let footer_rect = Rect::new(
            inner.x,
            inner.y + inner.height - FOOTER_ROWS,
            inner.width,
            FOOTER_ROWS,
        );
        let footer = Paragraph::new(footer_lines(theme, inner.width))
            .style(Style::default().bg(theme.palette.code_bg));
        frame.render_widget(footer, footer_rect);
    }
}

/// Number of terminal rows `line` occupies once word-wrapped into `width`
/// cells (`Wrap { trim: true }`). A blank line still costs one row.
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
fn clamp_lines_with_overflow(lines: &mut Vec<Line<'_>>, body: Rect, muted: Style) {
    let budget = usize::from(body.height);
    if budget == 0 {
        lines.clear();
        return;
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
        return; // everything fits — no indicator needed.
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
        value,
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
    let hint_style = Style::new().fg(theme.palette.dim);
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
            hint_style,
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
            let branch = tree_glyph(theme, if is_last { "└" } else { "├" }, "|-");
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
            let title_budget = usize::from(inner_width).saturating_sub(4);
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
            let mut agent_title_spans = vec![
                Span::styled(format!("  {branch} "), muted),
                Span::styled(name, value),
            ];
            if let Some(mc) = model_cells {
                let model_shown = truncate_to_cells(&model_label, mc.saturating_sub(3).max(1));
                agent_title_spans.push(Span::styled(format!(" ({model_shown})"), cyan));
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
                            agent_meta_spans.push(Span::styled(format!(" · {m}"), cyan));
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
                    agent_meta_spans.push(Span::styled(format!(" {spark}"), cyan));
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
fn footer_lines(theme: &Theme, _width: u16) -> Vec<Line<'static>> {
    let key_style = Style::default().fg(theme.palette.fg);
    let label_style = Style::default().fg(theme.palette.dim);
    let ctrl = if theme.no_color { "^" } else { "\u{2303}" };
    let rows = [
        vec![("drag".to_string(), "copy"), ("click".to_string(), "expand")],
        vec![
            (format!("{ctrl}F"), "find"),
            (format!("{ctrl}P"), "cmds"),
            ("?".to_string(), "help"),
        ],
    ];

    rows.into_iter()
        .map(|row| {
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(6);
            for (idx, (key, label)) in row.into_iter().enumerate() {
                if idx > 0 {
                    spans.push(Span::styled("   ", label_style));
                }
                spans.push(Span::styled(key, key_style));
                spans.push(Span::styled(format!(" {label}"), label_style));
            }
            Line::from(spans)
        })
        .collect()
}

#[cfg(test)]
fn token_gauge_bar(pct: u64, width: usize, theme: &Theme) -> Span<'static> {
    if theme.no_color || pct == 0 {
        return Span::raw("");
    }
    let pct_u8 = u8::try_from(pct.min(100)).unwrap_or(100);
    let pct_usize = usize::from(pct_u8);
    let filled = ((pct_usize * width) + 50) / 100;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);

    // Fill/empty glyphs are East-Asian **Neutral** (`▬`/`░` → `width_cjk()==1`)
    // so the fixed-`width` bar renders one column per cell even under a `ko_KR`
    // wide-ambiguous tmux; the old `■` (Ambiguous) doubled the filled run and
    // overflowed the sidebar. (The no-color arm is dead behind the early return
    // above but stays correct via `pick` should that guard ever move.)
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

/// Context-pressure percent: occupancy of the *auto-compaction ceiling* when
/// known (80% of the window for Claude, 85% otherwise, or the user override),
/// else of the nominal window — measured against the nominal window the gauge
/// understated how close the silent compact was, so amber/red fired late.
fn context_percent(hud: &HudState) -> u64 {
    crate::tui::hud::context_pressure_percent(hud).unwrap_or(0)
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

fn format_context_used_tokens(hud: &HudState) -> String {
    if hud.ctx_used == 0 {
        "pending".to_string()
    } else if hud.ctx_limit > 0 && hud.ctx_used > hud.ctx_limit {
        format!("{}+", format_tokens(hud.ctx_limit))
    } else {
        format_tokens(hud.ctx_used)
    }
}

fn status_dot(theme: &Theme) -> &'static str {
    if theme.no_color { "* " } else { "\u{25cf} " }
}

fn tree_glyph(theme: &Theme, unicode: &'static str, ascii: &'static str) -> &'static str {
    if theme.no_color { ascii } else { unicode }
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

fn section_prefix(_theme: &Theme) -> &'static str {
    ""
}

fn indent_glyph(theme: &Theme) -> &'static str {
    tree_glyph(theme, "  ", "  ")
}

fn child_glyph(theme: &Theme) -> &'static str {
    tree_glyph(theme, "  └ ", "  - ")
}

fn section_line(title: &'static str, _theme: &Theme, style: Style) -> Line<'static> {
    Line::from(Span::styled(title, style))
}

/// Build the 5h/7d rate-limit gauge lines — one per present window.
fn rate_limit_gauges(
    rl: RateLimitSnapshot,
    theme: &Theme,
    muted: Style,
    value: Style,
) -> Vec<Line<'static>> {
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
            Span::styled(gauge_bar(w.utilization, 10, theme), bar_style),
            Span::styled(format!(" {pct}%"), value),
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
    value: Style,
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
            Span::styled(gauge_bar(f64::from(used) / 100.0, 10, theme), bar_style),
            Span::styled(format!(" ~{used}%"), value),
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

fn todo_marker(status: TodoChecklistStatus, theme: &Theme) -> (&'static str, Style) {
    if theme.no_color {
        match status {
            TodoChecklistStatus::Pending => ("[ ]", Style::default().fg(theme.palette.dim)),
            TodoChecklistStatus::InProgress => ("[-]", Style::default().fg(theme.palette.warn)),
            TodoChecklistStatus::Completed => ("[x]", Style::default().fg(theme.palette.success)),
        }
    } else {
        match status {
            TodoChecklistStatus::Pending => ("○", Style::default().fg(theme.palette.dim)),
            TodoChecklistStatus::InProgress => ("●", Style::default().fg(theme.palette.warn)),
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
