use super::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use runtime::message_stream::ActiveModel;

use crate::tui::app::{ScheduledWakeHud, WakeSource};
use crate::tui::hud::{
    AgentTaskSummary, LspStatusItem, McpHudStatus, SecurityPosture, SessionIdentity,
    TodoChecklistItem, TodoChecklistStatus,
};
use crate::tui::theme::Theme;
use crate::tui::workspace_status::{GitCliStatus, WorkspaceStatusSource};
use crate::tui::workflow_progress::{FleetPhase, WorkflowSummary};

fn load_changed_files_from_git(repo: &Path) -> GitStatusSnapshot {
    GitCliStatus
        .snapshot(repo, Arc::new(AtomicBool::new(false)))
        .expect("git status fixture should load")
}

#[test]
fn gauge_bar_fills_proportionally() {
    let color = Theme::default_dark();
    // Fill glyph is `▬` (EAW=Neutral, `width_cjk()==1`) — the old `█` (Ambiguous)
    // doubled under a wide-ambiguous tmux; empty stays `░` (also Neutral).
    assert_eq!(gauge_bar(0.0, 10, &color), "░░░░░░░░░░");
    assert_eq!(gauge_bar(1.0, 10, &color), "▬▬▬▬▬▬▬▬▬▬");
    assert_eq!(
        gauge_bar(0.68, 10, &color)
            .chars()
            .filter(|&c| c == '▬')
            .count(),
        7
    );
    // Over-range utilization clamps to a full bar.
    assert_eq!(gauge_bar(1.5, 10, &color), "▬▬▬▬▬▬▬▬▬▬");
    // NO_COLOR degrades the solid/empty blocks to `#`/`.` (R10).
    let mono = Theme::no_color();
    assert_eq!(gauge_bar(0.0, 10, &mono), "..........");
    assert_eq!(gauge_bar(1.0, 10, &mono), "##########");
    assert_eq!(
        gauge_bar(0.68, 10, &mono)
            .chars()
            .filter(|&c| c == '#')
            .count(),
        7
    );
}

/// Every sidebar gauge must keep its budgeted `width` even under a `ko_KR`
/// wide-ambiguous tmux, where East-Asian-Ambiguous glyphs paint two columns.
/// `width_cjk()` models that host: the bar's rendered width must equal its cell
/// count, which fails the moment a fill glyph regresses to `■`/`█` (cjk == 2).
#[test]
fn sidebar_gauges_stay_one_cell_under_wide_ambiguous() {
    use unicode_width::UnicodeWidthStr;
    let color = Theme::default_dark();
    let mono = Theme::no_color();
    for theme in [&color, &mono] {
        for pct in [0u64, 33, 50, 80, 100] {
            let gauge = token_gauge_bar(pct, 10, theme);
            let w = UnicodeWidthStr::width_cjk(gauge.content.as_ref());
            assert!(
                gauge.content.is_empty() || w == 10,
                "ctx gauge at {pct}% renders {w} cols under wide-ambiguous, want 10"
            );
        }
        assert_eq!(UnicodeWidthStr::width_cjk(gauge_bar(0.6, 10, theme).as_str()), 10);
        assert_eq!(
            UnicodeWidthStr::width_cjk(fleet_phase_bar(3, 8, 10, theme).as_str()),
            10
        );
    }
}

/// The estimated cross-provider stack renders ONLY `estimated` rows with a
/// known remaining figure: measured Anthropic windows stay on the 5h/7d gauge
/// (no duplicate), and an unknown remaining renders nothing rather than a
/// fabricated bar. The row carries the provider key, the `~NN%` used figure,
/// the `est` marker, and the cool-down countdown.
#[test]
fn estimated_quota_gauges_render_only_estimated_rows() {
    let theme = Theme::default_dark();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let views = vec![
        api::quota::ProviderQuotaView {
            provider: api::ProviderKind::Anthropic,
            window_label: "5h".to_string(),
            remaining_percent: Some(40),
            resets_at_unix: Some(now + 3_600),
            estimated: false,
        },
        api::quota::ProviderQuotaView {
            provider: api::ProviderKind::OpenAi,
            window_label: "429".to_string(),
            remaining_percent: Some(10),
            resets_at_unix: Some(now + 120),
            estimated: true,
        },
        api::quota::ProviderQuotaView {
            provider: api::ProviderKind::Google,
            window_label: "429".to_string(),
            remaining_percent: None,
            resets_at_unix: None,
            estimated: true,
        },
    ];
    let lines = estimated_quota_gauges(
        &views,
        &theme,
        Style::default(),
        Style::default(),
    );
    assert_eq!(lines.len(), 1, "only the estimated row with a known figure renders");
    let text: String = lines[0]
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    assert!(text.contains("openai"), "{text}");
    assert!(text.contains("~90%"), "{text}");
    assert!(text.contains("est"), "{text}");
    assert!(text.contains("↺"), "{text}");
    assert!(!text.contains("anthropic") && !text.contains("google"), "{text}");

    // No views at all → no lines (the stack simply doesn't appear).
    assert!(estimated_quota_gauges(&[], &theme, Style::default(), Style::default()).is_empty());
}

#[test]
fn format_reset_compact_forms() {
    assert_eq!(format_reset(100, 100), "now");
    assert_eq!(format_reset(0, 45 * 60), "45m");
    assert_eq!(format_reset(0, 2 * 3600 + 11 * 60), "2h11m");
    assert_eq!(format_reset(0, 3 * 86_400), "3d");
    assert_eq!(format_reset(0, 3 * 86_400 + 5 * 3600), "3d5h");
}

fn sample_hud() -> HudState {
    HudState {
        session_identity: None,
        model: ActiveModel {
            provider: "openai",
            alias: "gpt-5.5".to_string(),
            display_name: "GPT-5.5 Fast".to_string(),
            context_limit: 400_000,
        },
        turn_fallback_model: None,
        quota_fallback_model: None,
        ctx_used: 139_112,
        ctx_limit: 400_000,
        ctx_new_input: 0,
        ctx_cached: 0,
        compact_threshold: 340_000,
        cost_usd: 0.37,
        cost_approx: false,
        cwd: PathBuf::from("/Users/joe/2026/zo"),
        git_branch: Some("main".to_string()),
        perm_mode: PermissionMode::All,
        security_posture: SecurityPosture::SandboxActive,
        effort: None,
        architect_impl: None,
        mcp_servers: vec!["almanac".to_string(), "context7".to_string()],
        bash_count: 2,
        read_count: 8,
        edit_count: 5,
        changed_files: 0,
        agents: Vec::new(),
        todo_summary: Some("4 active".to_string()),
        todo_items: vec![
            TodoChecklistItem {
                step_id: None,
                content: "Lock sidebar data contracts".to_string(),
                status: TodoChecklistStatus::Completed,
                active_form: "Locking sidebar data contracts".to_string(),
            },
            TodoChecklistItem {
                step_id: None,
                content: "Wire live git diff".to_string(),
                status: TodoChecklistStatus::InProgress,
                active_form: "Wiring live git diff".to_string(),
            },
            TodoChecklistItem {
                step_id: None,
                content: "Render LSP status".to_string(),
                status: TodoChecklistStatus::Pending,
                active_form: "Rendering LSP status".to_string(),
            },
        ],
        automation_lines: Vec::new(),
        lsp_servers: vec![
            LspStatusItem {
                language: "rust".to_string(),
                status: "connected".to_string(),
            },
            LspStatusItem {
                language: "typescript".to_string(),
                status: "starting".to_string(),
            },
        ],
        running_agents: 0,
        workflow: None,
        last_tool: None,
        rate_limit: None,
        provider_quotas: Vec::new(),
        auth_origin: None,
        status_line: None,
        team_inbox_unread: 0,
        stale_binary: None,
        background_tasks: 0,
        scheduled_wake: None,
    }
}

#[test]
fn sidebar_live_section_surfaces_current_activity_and_edits() {
    let theme = Theme::no_color();
    let mut state = SidebarState::new();
    let mut hud = sample_hud();
    hud.last_tool = Some("Checking docs: Clean code".to_string());
    hud.running_agents = 2;
    hud.agents = vec![AgentTaskSummary {
        id: "agent-1".to_string(),
        tool_call_id: None,
        name: "Inspect admin flow".to_string(),
        status: "running".to_string(),
        model: "claude-opus-4-8".to_string(),
        elapsed_secs: 12,
        token_history: Vec::new(),
        current_tool: Some("grep admin_query.go".to_string()),
        current_phase: None,
        last_activity_at: None,
        subagent_type: Some("Explore".to_string()),
        tool_calls: Some(3),
        tokens: 1200,
        created_at: Some(10),
        output_tail: Some("checking handlers".to_string()),
        route_reason: None,
    }];
    state.set_changed_files(
        vec![
            ChangedFile {
                path: "src/main.rs".to_string(),
                status: FileStatus::Modified,
                adds: 3,
                rems: 1,
            },
            ChangedFile {
                path: "src/lib.rs".to_string(),
                status: FileStatus::Modified,
                adds: 2,
                rems: 0,
            },
        ],
        3,
    );
    hud.mcp_servers = vec![
        McpHudStatus::ready("context7").encode(),
        McpHudStatus::failed("chrome-devtools", "connection refused").encode(),
    ];

    let backend = TestBackend::new(74, 38);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(dumped.contains("live"), "live section should be visible: {dumped}");
    assert!(
        dumped.contains("Checking docs: Clean code"),
        "current tool activity should be surfaced: {dumped}"
    );
    assert!(
        dumped.contains("2 agents active") && dumped.contains("grep admin_query.go"),
        "agent fleet activity should show count and current work: {dumped}"
    );
    assert!(
        dumped.contains("edit 3 files changed"),
        "edit/diff activity should be explicit near the top: {dumped}"
    );
    assert!(
        dumped.contains("sources degraded") && dumped.contains("1/2 ready"),
        "MCP health should be summarized in live activity: {dumped}"
    );
    assert!(
        dumped.contains("files 3"),
        "work metrics should include changed-file count even when changes list is clipped: {dumped}"
    );
}

#[test]
fn sidebar_live_section_surfaces_background_tasks() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.last_tool = None;
    hud.running_agents = 0;
    hud.agents.clear();
    hud.background_tasks = 2;

    let backend = TestBackend::new(74, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(dumped.contains("live"), "live section should be visible: {dumped}");
    assert!(
        dumped.contains("2 background tasks"),
        "active background task count should be visible: {dumped}"
    );
}

#[test]
fn sidebar_session_section_renders_named_session_badge() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.session_identity = SessionIdentity::named("session-123", Some("deploy watch"));

    let backend = TestBackend::new(74, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("● deploy watch"),
        "session badge missing: {dumped}"
    );
}

#[test]
fn sidebar_surfaces_scheduled_wake_row_only_when_armed() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.last_tool = None;
    hud.running_agents = 0;
    hud.agents.clear();

    let backend = TestBackend::new(74, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let unarmed = dump(&terminal);
    assert!(!unarmed.contains('⏱'), "unarmed wake row leaked: {unarmed}");

    hud.scheduled_wake = Some(ScheduledWakeHud {
        due_at_epoch: 0,
        reason: "next CI status check after deployment".to_string(),
        source: WakeSource::Wakeup,
    });
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let armed = dump(&terminal);
    assert!(
        armed.contains("⏱ next CI status check") && armed.contains(" · now"),
        "scheduled wake row missing: {armed}"
    );
}

/// B4 `TeamInbox` badge: an unread count > 0 surfaces the `live` section with an
/// `inbox N unread updates` row (count only — no update text has any path into
/// the HUD), a count of 1 uses the singular label, and a count of 0 leaves the
/// sidebar exactly as before (no `inbox` row, and no `live` section when
/// nothing else is live).
#[test]
fn sidebar_team_inbox_badge_shows_unread_count_only() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = state_hud();
    hud.last_tool = None;
    hud.rate_limit = None;

    // Zero unread → no badge, and no live section without other live signal.
    let backend = TestBackend::new(74, 38);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);
    assert!(
        !dumped.contains("inbox"),
        "no inbox badge at zero unread: {dumped}"
    );

    // Unread present → the live section appears with the count-only badge.
    hud.team_inbox_unread = 3;
    let backend = TestBackend::new(74, 38);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);
    assert!(
        dumped.contains("live"),
        "unread inbox should surface the live section: {dumped}"
    );
    assert!(
        dumped.contains("inbox 3 unread updates"),
        "inbox badge should show the unread count: {dumped}"
    );

    // Singular form for one unread update.
    hud.team_inbox_unread = 1;
    let backend = TestBackend::new(74, 38);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);
    assert!(
        dumped.contains("inbox 1 unread update") && !dumped.contains("1 unread updates"),
        "singular label for one unread update: {dumped}"
    );
}

/// The stale-binary `/restart` warning is absent until `HudState::stale_binary`
/// trips, then appears as an always-on top-of-sidebar row naming the disk build
/// date and the `/restart` action.
#[test]
fn sidebar_stale_binary_warning_appears_only_when_stale() {
    use crate::tui::stale_binary::StaleBinaryInfo;

    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = state_hud();
    hud.last_tool = None;
    hud.rate_limit = None;

    // No replacement detected → no warning row at all.
    let backend = TestBackend::new(74, 38);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let clean_dump = dump(&terminal);
    assert!(
        !clean_dump.contains("new build on disk") && !clean_dump.contains("/restart"),
        "no stale-binary warning while the running binary matches disk: {clean_dump}"
    );

    // A newer build on disk (2026-07-10) → the always-on warning row surfaces,
    // leading with the /restart action and naming the disk date.
    hud.stale_binary = Some(StaleBinaryInfo {
        disk_mtime: 1_783_641_600,
    });
    let backend = TestBackend::new(74, 38);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let stale_dump = dump(&terminal);
    assert!(
        stale_dump.contains("new build on disk"),
        "stale binary must surface the restart warning: {stale_dump}"
    );
    assert!(
        stale_dump.contains("2026-07-10"),
        "warning names the disk build date: {stale_dump}"
    );
    assert!(
        stale_dump.contains("/restart"),
        "warning names the /restart action: {stale_dump}"
    );
}

#[test]
fn sidebar_header_status_badge_tracks_live_state() {
    let theme = Theme::no_color();
    let styles = SidebarStyles::new(&theme);
    let mut hud = sample_hud();

    // Idle is a pure activity lamp ("ready"), NOT the permission mode — the
    // perm mode has its own `mode` line in the session panel, so echoing it
    // here duplicated it. Changing the perm mode must not change the idle
    // badge.
    assert_eq!(
        sidebar_header_status_badge(&hud, &theme, styles).0,
        "ready"
    );

    hud.perm_mode = PermissionMode::ReadOnly;
    assert_eq!(
        sidebar_header_status_badge(&hud, &theme, styles).0,
        "ready",
        "the idle badge tracks activity, not the permission mode"
    );

    hud.workflow = Some(WorkflowSummary {
        name: "code-health".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "inspect".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 2,
        next_phase: None,
        total_agents: 2,
        progress_percent: 50,
        completed_phases: 0,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 2,
        phases: Vec::new(),
    });
    assert_eq!(
        sidebar_header_status_badge(&hud, &theme, styles).0,
        "running"
    );

    hud.security_posture = SecurityPosture::SandboxBlocked;
    assert_eq!(
        sidebar_header_status_badge(&hud, &theme, styles).0,
        "blocked"
    );
}

#[test]
fn modern_sidebar_header_uses_aligned_quiet_rows() {
    let theme = Theme::no_color();
    let styles = SidebarStyles::new(&theme);
    let mut lines = Vec::new();
    push_header(&mut lines, 36, &sample_hud(), &theme, styles);

    assert_eq!(lines.len(), 3, "two header rows plus breathing room");
    for line in &lines[..2] {
        let text: String = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(display_width(&text), 36, "header row stays aligned: {text:?}");
        assert!(
            line.spans
                .iter()
                .all(|span| !span.style.add_modifier.contains(Modifier::BOLD)),
            "ordinary header metadata should not spend the bold emphasis budget"
        );
    }
    let first: String = lines[0]
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    let second: String = lines[1]
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    assert!(first.contains("zo") && first.ends_with("main"), "{first:?}");
    assert!(second.ends_with("ready"), "{second:?}");
    assert!(!first.contains(" · "), "header no longer chains fields with separators");
}

#[test]
fn modern_sidebar_sections_and_footer_avoid_decorative_chrome() {
    let theme = Theme::no_color();
    let styles = SidebarStyles::new(&theme);
    let section = section_line("session", &theme, styles.label);
    let section_text: String = section
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    assert_eq!(section_text, "session");
    assert!(
        section
            .spans
            .iter()
            .all(|span| !span.style.add_modifier.contains(Modifier::BOLD))
    );

    let footer = footer_lines(&theme, 36);
    assert_eq!(footer.len(), usize::from(FOOTER_ROWS));
    let footer_text: String = footer
        .iter()
        .flat_map(|line| line.spans.iter())
        .map(|span| span.content.as_ref())
        .collect();
    for expected in ["drag", "copy", "click", "expand", "^F", "find", "?", "help"] {
        assert!(footer_text.contains(expected), "missing {expected}: {footer_text}");
    }
    assert!(!footer_text.contains('─') && !footer_text.contains("--------"));
}

#[test]
fn color_sidebar_context_remains_text_first() {
    let theme = Theme::default_dark();
    let state = SidebarState::new();
    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &sample_hud(), &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(dumped.contains("ctx") && dumped.contains("40%"), "{dumped}");
    assert!(
        !dumped.contains(glyphs::GAUGE_FILL) && !dumped.contains(glyphs::GAUGE_EMPTY),
        "session context should not render a color gauge: {dumped}"
    );
}

fn temp_dir(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("zo-sidebar-{label}-{unique}"))
}

fn run_git(repo: &std::path::Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .expect("git should run");
    assert!(
        output.status.success(),
        "git {:?} failed\nstdout: {}\nstderr: {}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn dump(terminal: &Terminal<TestBackend>) -> String {
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect()
}

#[test]
fn mcp_sources_render_lifecycle_statuses() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.mcp_servers = vec![
        McpHudStatus::discovering("atlassian").encode(),
        McpHudStatus::ready("context7").encode(),
        McpHudStatus::failed("chrome-devtools", "tools/list timed out").encode(),
    ];

    let backend = TestBackend::new(72, 28);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    // One ready of three configured: the headline reports `ready/total`, not a
    // flat green total, so a failing source can't masquerade as all-healthy.
    assert!(
        dumped.contains("sources 1/3"),
        "ready/total count shown: {dumped}"
    );
    assert!(dumped.contains("atlassian"), "server name shown: {dumped}");
    assert!(
        dumped.contains("discovering"),
        "pending status shown: {dumped}"
    );
    assert!(dumped.contains("context7"), "ready server shown: {dumped}");
    assert!(dumped.contains("ready"), "ready status shown: {dumped}");
    assert!(
        dumped.contains("chrome-devtools"),
        "failed server shown: {dumped}"
    );
    assert!(dumped.contains("failed"), "failed status shown: {dumped}");
}

/// Foreground color of the MCP "sources" count cell (the cell right after the
/// `sources ` label). Lets a test assert the headline recolors on failure
/// instead of only checking its text.
fn mcp_headline_count_fg(
    hud: &HudState,
    theme: &Theme,
) -> Option<ratatui::style::Color> {
    let state = SidebarState::new();
    let backend = TestBackend::new(72, 28);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, hud, theme))
        .expect("draw sidebar");
    let buffer = terminal.backend().buffer();
    let area = *buffer.area();
    let label = "sources ";
    for y in 0..area.height {
        let row: String = (0..area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect();
        if let Some(byte_idx) = row.find(label) {
            // `find` returns a byte offset; the row is ASCII up to the label, so
            // the count cell column is the char count of the label's end.
            let col = row[..byte_idx].chars().count() + label.chars().count();
            return buffer[(u16::try_from(col).unwrap(), y)].fg.into();
        }
    }
    None
}

#[test]
fn mcp_headline_recolors_red_when_a_source_failed() {
    // A failed source must turn the whole "sources" headline red — the old
    // denormalized count rendered an unconditional green total, so a failure was
    // invisible in the headline while only a per-server row went red.
    let theme = Theme::default_dark();
    let mut hud = sample_hud();
    hud.mcp_servers = vec![
        McpHudStatus::ready("context7").encode(),
        McpHudStatus::failed("atlassian", "tools/list timed out").encode(),
    ];
    assert_eq!(
        mcp_headline_count_fg(&hud, &theme),
        Some(theme.palette.error),
        "headline count must be red when any source failed"
    );

    // All-ready stays green.
    hud.mcp_servers = vec![
        McpHudStatus::ready("context7").encode(),
        McpHudStatus::ready("atlassian").encode(),
    ];
    assert_eq!(
        mcp_headline_count_fg(&hud, &theme),
        Some(theme.palette.success),
        "all-ready headline stays green"
    );

    // Still discovering (none failed) is amber, not green or red.
    hud.mcp_servers = vec![
        McpHudStatus::ready("context7").encode(),
        McpHudStatus::discovering("atlassian").encode(),
    ];
    assert_eq!(
        mcp_headline_count_fg(&hud, &theme),
        Some(theme.palette.warn),
        "a discovering source makes the headline amber"
    );
}

#[test]
fn mcp_rows_cap_at_four_with_failures_surfaced_first() {
    // Six configured sources with the failure sorting alphabetically last: it
    // must still appear in the capped rows (worst-first ordering) and the cap
    // must be flagged with a `+N more` hint — never silently dropped.
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.mcp_servers = vec![
        McpHudStatus::ready("aaa").encode(),
        McpHudStatus::ready("bbb").encode(),
        McpHudStatus::ready("ccc").encode(),
        McpHudStatus::ready("ddd").encode(),
        McpHudStatus::ready("eee").encode(),
        McpHudStatus::failed("zzz", "spawn failed").encode(),
    ];

    let backend = TestBackend::new(72, 28);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("sources 5/6"),
        "ready/total reflects the failed source: {dumped}"
    );
    assert!(
        dumped.contains("zzz") && dumped.contains("failed"),
        "the failed source must be surfaced despite sorting last: {dumped}"
    );
    assert!(
        dumped.contains("+2 more"),
        "the row cap must be flagged with a +N more hint: {dumped}"
    );
}

/// The running-agents tree degrades every decorative glyph under
/// `NO_COLOR`: the expand chevron (▾→`v`/`>`), the spark (✦→`+`), and the
/// per-agent token sparkline (▁▂▃→`#` run). §8 sidebar parity (R10).
#[test]
fn agents_tree_degrades_glyphs_under_no_color() {
    let theme = Theme::no_color();
    let state = SidebarState::new(); // agents_expanded = true by default
    let mut hud = sample_hud();
    hud.rate_limit = None; // keep `#` attributable to the sparkline alone
    hud.running_agents = 2;
    hud.agents = vec![
        AgentTaskSummary {
            name: "explorer".to_string(),
            status: "running".to_string(),
            model: "sonnet".to_string(),
            elapsed_secs: 12,
            token_history: vec![1, 4, 2, 8, 5],
            current_tool: Some("Grep".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "reviewer".to_string(),
            status: "completed".to_string(),
            model: "opus".to_string(),
            elapsed_secs: 40,
            token_history: Vec::new(),
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
    ];

    let backend = TestBackend::new(60, 34);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("2 agents"),
        "agents count line present: {dumped}"
    );
    // The spark and the expand chevron are ASCII under NO_COLOR — no rich
    // glyphs survive (the bug was both hardcoded outside the glyph table).
    assert!(!dumped.contains('✦'), "spark degrades to ASCII: {dumped}");
    assert!(!dumped.contains('▾'), "chevron degrades to ASCII: {dumped}");
    assert!(!dumped.contains('▼'), "no rich down-triangle: {dumped}");
    // The token sparkline degrades to a `#` run — no braille blocks remain.
    for braille in ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'] {
        assert!(
            !dumped.contains(braille),
            "sparkline braille {braille:?} must degrade under NO_COLOR: {dumped}"
        );
    }
    assert!(
        dumped.contains('#'),
        "sparkline degrades to a # run: {dumped}"
    );
}

#[test]
fn agents_tree_shows_resolved_model_once() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.rate_limit = None;
    hud.running_agents = 1;
    hud.agents = vec![AgentTaskSummary {
        name: "runtime-streaming-audit".to_string(),
        status: "running".to_string(),
        model: "openai:gpt-5.5-fast".to_string(),
        elapsed_secs: 37,
        token_history: Vec::new(),
        current_tool: Some("Read".to_string()),
        current_phase: None,
        last_activity_at: None,
        ..Default::default()
    }];

    let backend = TestBackend::new(72, 26);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("gpt-5.5-fast"),
        "resolved model id should be visible: {dumped}"
    );
    assert!(
        !dumped.contains("openai:gpt-5.5-fast"),
        "provider prefix should not crowd the sidebar: {dumped}"
    );
    assert_eq!(
        dumped.matches("gpt-5.5-fast").count(),
        1,
        "agent row should not repeat the model on both title and metadata lines: {dumped}"
    );
}

#[test]
fn agent_row_shows_wait_phase_when_no_tool_is_active() {
    // A quota-parked agent (rate-limit cool-down / governor queue) has no
    // current tool; its `currentPhase` must surface in the row so the
    // agent reads as alive instead of a frozen `[running]`.
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.rate_limit = None;
    hud.running_agents = 1;
    hud.agents = vec![AgentTaskSummary {
        name: "memory-optimization".to_string(),
        status: "running".to_string(),
        model: "claude-opus-4-8".to_string(),
        elapsed_secs: 410,
        token_history: Vec::new(),
        current_tool: None,
        current_phase: Some("rate-limited \u{00b7} resumes in ~45s".to_string()),
        last_activity_at: None,
        ..Default::default()
    }];

    let backend = TestBackend::new(72, 26);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("rate-limited"),
        "wait phase must surface on the agent row: {dumped}"
    );
}

#[test]
fn agents_header_summarizes_state_and_model_mix_when_collapsed() {
    let theme = Theme::no_color();
    let mut state = SidebarState::new();
    state.toggle_agents();
    let mut hud = sample_hud();
    hud.rate_limit = None;
    hud.running_agents = 3;
    hud.agents = vec![
        AgentTaskSummary {
            name: "runtime-streaming-audit".to_string(),
            status: "running".to_string(),
            model: "OpenAI GPT-5.5 Fast".to_string(),
            elapsed_secs: 37,
            token_history: Vec::new(),
            current_tool: Some("Read".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "cli-health".to_string(),
            status: "pending".to_string(),
            model: "openai:gpt-5.5-fast".to_string(),
            elapsed_secs: 2,
            token_history: Vec::new(),
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "provider-audit".to_string(),
            status: "failed".to_string(),
            model: "anthropic/claude-opus-4-8".to_string(),
            elapsed_secs: 61,
            token_history: Vec::new(),
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
    ];

    let backend = TestBackend::new(118, 26);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(dumped.contains("3 agents"), "count missing: {dumped}");
    assert!(
        dumped.contains("1 running") && dumped.contains("1 queued") && dumped.contains("1 failed"),
        "state summary missing: {dumped}"
    );
    assert!(
        dumped.contains("gpt-5.5-fast x2") && dumped.contains("claude-opus-4-8"),
        "resolved model mix should stay visible when the agent tree is collapsed: {dumped}"
    );
    assert!(
        !dumped.contains("runtime-streaming-audit"),
        "collapsed tree should not render individual agent rows: {dumped}"
    );
}

#[test]
fn agents_tree_keeps_resolved_model_visible_when_narrow() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.rate_limit = None;
    hud.running_agents = 1;
    hud.agents = vec![AgentTaskSummary {
        name: "runtime-streaming-audit".to_string(),
        status: "running".to_string(),
        model: "openai:gpt-5.5-fast".to_string(),
        elapsed_secs: 37,
        token_history: Vec::new(),
        current_tool: Some("Bash".to_string()),
        current_phase: None,
        last_activity_at: None,
        ..Default::default()
    }];

    let backend = TestBackend::new(34, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("gpt-5.5-fast"),
        "narrow agent row must still show the resolved model id: {dumped}"
    );
    assert!(
        dumped.contains("Bash"),
        "narrow agent row should keep the active tool beside the model: {dumped}"
    );
    assert_eq!(
        dumped.matches("gpt-5.5-fast").count(),
        1,
        "narrow agent row should not duplicate the model id: {dumped}"
    );
}

#[test]
fn draw_shows_live_workflow_phase_summary() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.workflow = Some(WorkflowSummary {
        name: "code-health".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("synthesize".to_string()),
        total_agents: 12,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 3,
        failed_agents: 1,
        running_agents: 8,
        phases: Vec::new(),
    });

    let backend = TestBackend::new(60, 34);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(
        dumped.contains("workflow"),
        "workflow section shown: {dumped}"
    );
    assert!(dumped.contains("2/4"), "phase index shown: {dumped}");
    assert!(
        dumped.contains("read-code"),
        "current phase shown: {dumped}"
    );
    assert!(dumped.contains("next"), "next phase label shown: {dumped}");
    assert!(
        dumped.contains("synthesize"),
        "next phase name shown: {dumped}"
    );
    assert!(
        dumped.contains("25%"),
        "workflow progress percentage shown: {dumped}"
    );
    assert!(
        !dumped.contains("% left"),
        "the redundant '% left' half must no longer be shown: {dumped}"
    );
    assert!(
        dumped.contains("1/4 phases"),
        "workflow phase completion shown: {dumped}"
    );
    assert!(
        dumped.contains("4/12 agents") && dumped.contains("8 running"),
        "workflow agent tally shown: {dumped}"
    );
    assert!(
        dumped.contains("1 failed"),
        "workflow failure tally shown: {dumped}"
    );
}

#[test]
fn draw_shows_fleet_phase_bars_when_phases_present() {
    // A multi-phase Workflow surfaces the always-on Fleet: one progress bar per
    // phase (terminal/total), instead of the single aggregate current-phase line
    // used for a plain fan-out. This is the signature multi-agent view.
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.workflow = Some(WorkflowSummary {
        name: "review-changes".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "Review".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 2,
        next_phase: Some("Verify".to_string()),
        total_agents: 11,
        progress_percent: 55,
        completed_phases: 0,
        completed_agents: 5,
        failed_agents: 1,
        running_agents: 2,
        phases: vec![
            FleetPhase {
                id: "Review".to_string(),
                step_id: None,
                agent_ids: Vec::new(),
                status: "running".to_string(),
                total: 8,
                completed: 5,
                failed: 1,
                running: 2,
            },
            FleetPhase {
                id: "Verify".to_string(),
                step_id: None,
                agent_ids: Vec::new(),
                status: "pending".to_string(),
                total: 3,
                completed: 0,
                failed: 0,
                running: 0,
            },
        ],
    });

    let backend = TestBackend::new(60, 34);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);

    assert!(dumped.contains("workflow"), "workflow header shown: {dumped}");
    // Both phase rows render with their per-phase terminal/total tally …
    assert!(
        dumped.contains("Review") && dumped.contains("6/8"),
        "running phase row + tally shown (5 done + 1 failed = 6 of 8): {dumped}"
    );
    assert!(
        dumped.contains("Verify") && dumped.contains("0/3"),
        "pending phase row + tally shown: {dumped}"
    );
    // … a filled bar segment (no-color glyph is `#`) …
    assert!(dumped.contains('#'), "fleet progress bar rendered: {dumped}");
    // … and the per-phase failure annotation.
    assert!(
        dumped.contains("1 failed"),
        "per-phase failure tally shown: {dumped}"
    );
    // The Fleet path replaces the single aggregate "next <phase>" line.
    assert!(
        !dumped.contains("next Verify"),
        "fleet tree should not also print the aggregate next-phase line: {dumped}"
    );
}

#[test]
fn fleet_phase_bar_fills_proportionally_and_guards_empty() {
    let theme = Theme::no_color();
    // 6 of 8 terminal → 8 of 10 cells filled (rounded); no-color glyphs `#`/`.`.
    assert_eq!(fleet_phase_bar(6, 8, 10, &theme), "########..");
    // No agents yet → all-empty, never a misleading full bar.
    assert_eq!(fleet_phase_bar(0, 0, 10, &theme), "..........");
    // Fully terminal → full bar.
    assert_eq!(fleet_phase_bar(3, 3, 6, &theme), "######");
    // Round-half-up over `total` (NOT `cells`): a small terminal count still
    // shows ≥1 filled cell rather than rounding to empty. 1/3·10 = 3.33 → 3;
    // 2/7·10 = 2.86 → 3; 3/4·10 = 7.5 → 8 (half rounds up); 1/100·10 = 0.1 → 0.
    assert_eq!(fleet_phase_bar(1, 3, 10, &theme), "###.......");
    assert_eq!(fleet_phase_bar(2, 7, 10, &theme), "###.......");
    assert_eq!(fleet_phase_bar(3, 4, 10, &theme), "########..");
    assert_eq!(fleet_phase_bar(1, 100, 10, &theme), "..........");
}

#[test]
fn truncate_path_short_path_unchanged() {
    assert_eq!(truncate_path("src/main.rs", 30), "src/main.rs");
}

#[test]
fn truncate_path_long_path_uses_filename() {
    let long = "very/deep/nested/directory/structure/file.rs";
    assert_eq!(truncate_path(long, 10), "file.rs");
}

#[test]
fn truncate_path_long_filename_gets_ellipsis() {
    let long = "very_long_filename_that_wont_fit.rs";
    let result = truncate_path(long, 10);
    // Cell-width based: ascii filename truncated to 10 cells with a
    // single `…` ellipsis (truncate_to_cells), not the old "..." run.
    assert_eq!(display_cells(&result), 10);
    assert!(result.ends_with('…'));
}

#[test]
fn truncate_path_cjk_respects_cell_width_not_bytes() {
    // 한국어 파일명: 각 글자 3바이트, 셀폭 2. 6글자 = 12 cells, 18 bytes.
    let cjk = "한글파일이름.rs";
    // max_len 8 cells: full string (12 + 3 = 15 cells) overflows, falls
    // back to the filename (== itself, still overflows), then truncates
    // to cells. Byte-based logic would mis-measure and overflow the panel.
    let result = truncate_path(cjk, 8);
    assert!(
        display_cells(&result) <= 8,
        "CJK truncation must fit cell budget, got {} cells: {result}",
        display_cells(&result)
    );
    assert!(
        result.ends_with('…'),
        "truncated CJK keeps ellipsis: {result}"
    );

    // A CJK path that fits exactly in cells is returned untouched even
    // though its byte length far exceeds max_len.
    let fits = "한글.rs"; // 2*2 + 3 = 7 cells, 9 bytes
    assert_eq!(truncate_path(fits, 7), fits);
}

#[test]
fn truncate_path_cjk_directory_falls_back_to_filename() {
    // Deep CJK directory path; filename alone fits the cell budget.
    let path = "아주/깊은/디렉토리/구조/file.rs"; // filename "file.rs" = 7 cells
    assert_eq!(truncate_path(path, 10), "file.rs");
}

#[test]
fn scroll_down_clamps_to_changed_file_count() {
    let mut state = SidebarState::new();
    state.set_changed_files(
        (0..5)
            .map(|i| ChangedFile {
                path: format!("file{i}.rs"),
                status: FileStatus::Modified,
                adds: 0,
                rems: 0,
            })
            .collect(),
        5,
    );
    assert_eq!(state.scroll, 0);
    // Wheel spam far beyond content must not inflate scroll past the
    // file count — otherwise draw's scroll.min(max_scroll) leaves the
    // panel unresponsive until an equal number of scroll-ups.
    for _ in 0..50 {
        state.scroll_down(3);
    }
    assert_eq!(
        state.scroll, 5,
        "scroll must clamp to changed_files.len(), got {}",
        state.scroll
    );
    // Scrolling back up still works from the clamped value.
    state.scroll_up(2);
    assert_eq!(state.scroll, 3);
    state.scroll_up(100);
    assert_eq!(state.scroll, 0, "scroll_up saturates at 0");
}

#[test]
fn scroll_down_no_files_stays_at_zero() {
    let mut state = SidebarState::new();
    state.scroll_down(10);
    assert_eq!(state.scroll, 0, "no files means no scrollable rows");
}

#[test]
fn baseline_filter_subtracts_full_baseline_total_not_capped_path_count() {
    let mut state = SidebarState::new();
    let baseline_files: Vec<ChangedFile> = (0..MAX_SIDEBAR_FILES)
        .map(|i| ChangedFile {
            path: format!("generated-{i:04}.rs"),
            status: FileStatus::Modified,
            adds: 0,
            rems: 0,
        })
        .collect();
    state.baseline_paths = baseline_files.iter().map(|file| file.path.clone()).collect();
    state.baseline_total = 5_132;

    state.set_changed_files(baseline_files, 5_132);

    assert!(
        state.changed_files.is_empty(),
        "all capped paths are baseline dirt and should be hidden"
    );
    assert_eq!(
        state.total_changed, 0,
        "the header count must subtract the full baseline total, not only the capped path count"
    );
}

#[test]
fn set_changed_files_preserves_scroll_when_unchanged() {
    let files: Vec<ChangedFile> = (0..5)
        .map(|i| ChangedFile {
            path: format!("file{i}.rs"),
            status: FileStatus::Modified,
            adds: 0,
            rems: 0,
        })
        .collect();
    let mut state = SidebarState::new();
    state.set_changed_files(files.clone(), 5);
    state.scroll_down(3);
    assert_eq!(state.scroll, 3);

    // A mid-turn re-scan that finds the identical set must leave the user's
    // scroll position alone (the periodic refresh would otherwise reset it
    // to the top every ~1s).
    state.set_changed_files(files, 5);
    assert_eq!(state.scroll, 3, "unchanged re-scan must preserve scroll");

    // But a genuine change (new edit appears) resets to the top so the
    // freshest state is in view.
    state.set_changed_files(
        vec![ChangedFile {
            path: "new.rs".to_string(),
            status: FileStatus::Added,
            adds: 0,
            rems: 0,
        }],
        1,
    );
    assert_eq!(state.scroll, 0, "changed set resets scroll to top");
}

#[test]
fn draw_hides_context_bar_at_zero_percent() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    // Fresh session: nothing consumed yet.
    hud.ctx_used = 0;

    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    // SESSION header + "0%" still render, but no empty gauge bar is painted.
    // NO_COLOR would draw an empty gauge cell as '.', so a 10-dot run would
    // reveal a regression to the old meter.
    assert!(
        dumped.contains("session"),
        "session header still shown: {dumped}"
    );
    assert!(
        dumped.contains("ctx pending"),
        "fresh usage should read as pending instead of a broken zero: {dumped}"
    );
    assert!(
        !dumped.contains("ctx 0 /"),
        "fresh usage must not look like an authoritative zero: {dumped}"
    );
    assert!(dumped.contains("0%"), "0% label still shown: {dumped}");
    assert!(
        !dumped.contains(".........."),
        "empty gauge bar must be hidden at 0%: {dumped}"
    );
}

#[test]
fn draw_shows_context_as_text_not_gauge() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let hud = sample_hud(); // ctx_used = 139_112 (34%)

    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    // OpenCode 식 텍스트 우선: 게이지(no_color 채움 '#') 대신 % 와 ctx 토큰을
    // 텍스트로만 표기한다. '#' 채움 글리프는 더 이상 나오지 않는다.
    // 40% = 139.1k / 340k(자동 압축 임계값) — 명목 윈도우(400k) 기준이 아니라
    // 압축 천장 기준으로 압력을 표시한다.
    assert!(
        dumped.contains("40%"),
        "context percent shown as text: {dumped}"
    );
    assert!(
        dumped.contains("compacts at 340.0k"),
        "compaction ceiling named: {dumped}"
    );
    assert!(dumped.contains("ctx"), "ctx label shown: {dumped}");
    assert!(
        !dumped.contains('#'),
        "no fill gauge glyph remains: {dumped}"
    );
}

#[test]
fn draw_states_lsp_disabled_when_no_servers() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.lsp_servers.clear(); // 서버 미부착 — 비활성 상태.

    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    // 숨기지 않고 OpenCode 처럼 한 문장으로 비활성을 명시한다.
    assert!(dumped.contains("lsp"), "LSP header present: {dumped}");
    assert!(
        dumped.contains("disabled"),
        "LSP disabled stated, not hidden: {dumped}"
    );
}

#[test]
fn session_renders_cache_split_when_cached() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.ctx_new_input = 4_000;
    hud.ctx_cached = 135_000;

    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(dumped.contains("new"), "cache split shows new: {dumped}");
    assert!(
        dumped.contains("cached"),
        "cache split shows cached: {dumped}"
    );
    assert!(
        !dumped.contains("cache hit") && !dumped.contains("still ctx"),
        "sidebar should show cache numbers, not explanatory copy: {dumped}"
    );
}

#[test]
fn work_section_sorts_nonzero_activity_and_names_reads() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.bash_count = 0;
    hud.read_count = 99;
    hud.edit_count = 3;
    hud.mcp_servers.clear();

    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(
        dumped.contains("read 99  edit 3"),
        "largest visible work should lead and read/open work should be named: {dumped}"
    );
    assert!(
        !dumped.contains("run 0") && !dumped.contains("explore"),
        "zero-value and vague work labels should stay hidden: {dumped}"
    );
}

#[test]
fn sidebar_state_toggle() {
    let mut state = SidebarState::new();
    assert!(state.visible);
    state.toggle();
    assert!(!state.visible);
    state.toggle();
    assert!(state.visible);
}

#[test]
fn draw_surfaces_readable_opencode_plus_metadata() {
    let theme = Theme::no_color();
    let mut state = SidebarState::new();
    state.set_changed_files(
        vec![
            ChangedFile {
                path: "crates/runtime/src/conversation.rs".to_string(),
                status: FileStatus::Modified,
                adds: 12,
                rems: 3,
            },
            ChangedFile {
                path: "crates/runtime/src/new_provider.rs".to_string(),
                status: FileStatus::Added,
                adds: 0,
                rems: 0,
            },
            ChangedFile {
                path: "crates/runtime/src/old_snapshot.rs".to_string(),
                status: FileStatus::Deleted,
                adds: 0,
                rems: 7,
            },
        ],
        3,
    );

    let backend = TestBackend::new(52, 34);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &sample_hud(), &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    for expected in [
        "danger-full-access",
        "zo",
        "main",
        "session",
        "139.1k / 400.0k",
        // Percent measures against the auto-compaction ceiling (340k), not
        // the nominal window.
        "40%",
        "$0.37",
        "sources 2",
        "almanac",
        "context7",
        "lsp 2",
        "rust",
        "connected",
        "typescript",
        "starting",
        "[x]",
        "Lock sidebar data",
        "[-]",
        "Wiring live git diff",
        "[ ]",
        "Render LSP status",
        "changes 3",
        "conversation.rs",
        "+12",
        "-3",
        "new_provider.rs",
        "old_snapshot.rs",
        "-7",
        "danger-full-access",
    ] {
        assert!(dumped.contains(expected), "missing {expected}: {dumped}");
    }
}

#[test]
fn draw_hides_empty_todo_section_adaptively() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.todo_summary = None;
    hud.todo_items.clear();

    let backend = TestBackend::new(56, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    // Adaptive ledger: an empty TODO section is hidden entirely — no
    // header, no "empty" placeholder, and no fake sample items.
    assert!(
        !dumped.contains("todo"),
        "empty TODO section must be hidden, not shown: {dumped}"
    );
    assert!(
        !dumped.contains(".zo-todos.json"),
        "no empty-state placeholder when section is hidden: {dumped}"
    );
    assert!(
        !dumped.contains("Lock sidebar data") && !dumped.contains("Wire live git diff"),
        "must not render sample todo content: {dumped}"
    );
}

#[test]
fn draw_clamps_context_percent_when_usage_exceeds_limit() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    hud.ctx_used = 1_370_000;
    hud.ctx_limit = 1_000_000;

    let backend = TestBackend::new(56, 18);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(dumped.contains("100%"), "percent should clamp: {dumped}");
    assert!(
        dumped.contains("1.0M+ / 1.0M"),
        "over-limit context should render capped plus notation: {dumped}"
    );
    assert!(
        !dumped.contains("137%"),
        "context percent must not exceed 100: {dumped}"
    );
    assert!(
        !dumped.contains("1.4M"),
        "context display must not show cumulative-looking over-limit count: {dumped}"
    );
}

#[test]
fn sidebar_token_formatter_keeps_precise_million_limits() {
    assert_eq!(format_tokens(1_050_000), "1.05M");
    assert_eq!(format_tokens(1_000_000), "1.0M");
}

#[test]
fn load_changed_files_from_git_reports_worktree_statuses() {
    let repo = temp_dir("git-status");
    std::fs::create_dir_all(&repo).expect("temp repo");
    run_git(&repo, &["init"]);
    run_git(&repo, &["config", "user.email", "zo@example.com"]);
    run_git(&repo, &["config", "user.name", "Zo Test"]);

    std::fs::write(repo.join("tracked.txt"), "before\n").expect("write tracked");
    std::fs::write(repo.join("gone.txt"), "bye\n").expect("write deleted");
    run_git(&repo, &["add", "tracked.txt", "gone.txt"]);
    run_git(&repo, &["commit", "-m", "seed"]);

    std::fs::write(repo.join("tracked.txt"), "after\n").expect("modify tracked");
    std::fs::write(repo.join("new.txt"), "new\n").expect("write new");
    std::fs::remove_file(repo.join("gone.txt")).expect("delete tracked");

    let snapshot = load_changed_files_from_git(&repo);
    assert_eq!(snapshot.total, 3);

    let tracked = snapshot
        .files
        .iter()
        .find(|file| file.path == "tracked.txt")
        .expect("tracked.txt in snapshot");
    assert!(matches!(tracked.status, FileStatus::Modified));
    // "before\n" -> "after\n" is a 1-line replace: +1 -1 vs HEAD.
    assert_eq!(
        (tracked.adds, tracked.rems),
        (1, 1),
        "modified file must carry its git numstat line tally"
    );
    assert!(
        snapshot
            .files
            .iter()
            .any(|file| file.path == "new.txt" && matches!(file.status, FileStatus::Added))
    );
    assert!(
        snapshot
            .files
            .iter()
            .any(|file| file.path == "gone.txt" && matches!(file.status, FileStatus::Deleted))
    );

    std::fs::remove_dir_all(repo).ok();
}

#[test]
fn load_changed_files_from_git_excludes_generated_noise_from_total() {
    let repo = temp_dir("git-status-filtered");
    std::fs::create_dir_all(repo.join(".zo/cost")).expect("zo cost dir");
    std::fs::create_dir_all(repo.join(".zo/sessions")).expect("zo sessions dir");
    run_git(&repo, &["init"]);

    std::fs::write(repo.join("src.rs"), "fn main() {}\n").expect("write source");
    std::fs::write(repo.join("agent-123.json"), "{}\n").expect("write agent log");
    std::fs::write(repo.join(".zo/cost/sample.jsonl"), "{}\n").expect("write cost");
    std::fs::write(repo.join(".zo/runtime.json"), "{}\n").expect("write runtime");
    std::fs::write(repo.join(".zo/sessions/session.json"), "{}\n").expect("write session");

    let snapshot = load_changed_files_from_git(&repo);
    assert_eq!(snapshot.total, 1);
    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "src.rs");

    std::fs::remove_dir_all(repo).ok();
}

#[test]
fn draw_pins_keybinding_legend_to_bottom_when_tall_enough() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let hud = sample_hud();

    // Tall panel: the footer must appear and the live metadata still renders.
    let backend = TestBackend::new(40, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    // Mouse gestures are always available; keyboard accelerators use `^` under
    // NO_COLOR.
    for expected in ["drag copy", "click expand", "^F find", "^P cmds"] {
        assert!(
            dumped.contains(expected),
            "footer legend missing {expected}: {dumped}"
        );
    }
    assert!(!dumped.contains("^T"), "removed mouse toggle leaked into footer");
    // The rail head still renders — footer reserves space, doesn't replace.
    assert!(
        dumped.contains("danger-full-access"),
        "ledger head still shown: {dumped}"
    );
    // The footer rule degrades to ASCII under NO_COLOR (no box-drawing ─).
    assert!(
        !dumped.contains('\u{2500}'),
        "NO_COLOR footer rule must be ASCII, not ─: {dumped}"
    );
}

#[test]
fn draw_hides_legend_on_short_terminals() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let hud = sample_hud();

    // Short panel: the footer is suppressed so it never crowds the metadata.
    let backend = TestBackend::new(40, 7);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(
        !dumped.contains("panel") && !dumped.contains("cmds"),
        "footer legend must be hidden on short terminals: {dumped}"
    );
}

#[test]
fn draw_marks_todo_list_done_when_all_completed() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let mut hud = sample_hud();
    // Drive every checklist item to Completed.
    for item in &mut hud.todo_items {
        item.status = TodoChecklistStatus::Completed;
    }
    let total = hud.todo_items.len();

    let backend = TestBackend::new(44, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(
        dumped.contains(&format!("done · {total}/{total}")),
        "completed checklist must show done · N/N marker: {dumped}"
    );
}

#[test]
fn draw_omits_done_marker_while_todo_in_progress() {
    let theme = Theme::no_color();
    let state = SidebarState::new();
    let hud = sample_hud(); // sample has one InProgress + one Pending item.

    let backend = TestBackend::new(44, 30);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(
        !dumped.contains("done ·"),
        "an unfinished checklist must not show the done marker: {dumped}"
    );
}

#[test]
fn char_width_matches_unicode_width() {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

    // The sidebar width helper must agree cell-for-cell with the crate the HUD
    // and spinner use (unicode-width), not the old hand-rolled CJK table.
    for ch in [
        'a',
        ' ',
        '\u{2726}', // ✦ ZO_SPARK — narrow (1).
        '한',       // Hangul — wide (2).
        '中',       // CJK ideograph — wide (2).
        'ｱ',        // half-width katakana — narrow (1).
        '✓',        // check mark — narrow (1).
        '\u{0301}', // combining acute accent — zero width.
        '\u{200b}', // zero-width space.
    ] {
        assert_eq!(
            char_width(ch),
            UnicodeWidthChar::width(ch).unwrap_or(0),
            "char_width disagrees with unicode-width on {ch:?} (U+{:04X})",
            ch as u32,
        );
    }

    // The aggregate helpers must also match `UnicodeWidthStr` so truncation and
    // column math line up with the rest of the TUI.
    let spark_row = "\u{2726} 3 agents";
    assert_eq!(display_width(spark_row), UnicodeWidthStr::width(spark_row));
    assert_eq!(display_cells(spark_row), UnicodeWidthStr::width(spark_row));
    assert_eq!(display_width(spark_row), display_width(" 3 agents") + 1);
    assert_eq!(char_width('\u{2726}'), 1, "ZO_SPARK must measure 1 cell");
}

#[test]
fn gauge_color_thresholds_are_shared_across_gauges() {
    // Distinct success/warn/error so the ramp boundaries are observable
    // (no_color collapses them to Reset).
    let theme = Theme::default_dark();
    let p = theme.palette;

    // Single ramp: green < 50, amber < 80, red >= 80.
    assert_eq!(gauge_color(0, &theme), p.success);
    assert_eq!(gauge_color(49, &theme), p.success);
    assert_eq!(gauge_color(50, &theme), p.warn);
    assert_eq!(gauge_color(79, &theme), p.warn);
    assert_eq!(gauge_color(80, &theme), p.error);
    assert_eq!(gauge_color(100, &theme), p.error);

    // The context token bar must use the same ramp it shares with the rate-limit
    // bars — previously it only erred at >=85% while gauge_color erred at >=80%.
    // At 80% the context bar must already be red, matching gauge_color.
    let bar_80 = token_gauge_bar(80, 10, &theme);
    assert_eq!(
        bar_80.style.fg,
        Some(p.error),
        "context bar at 80% must be red like the rate-limit gauge"
    );
    let bar_55 = token_gauge_bar(55, 10, &theme);
    assert_eq!(
        bar_55.style.fg,
        Some(p.warn),
        "context bar at 55% must be amber"
    );
}

#[test]
fn visible_window_len_matches_iterator_count() {
    // Item C regression: the visible-row count is now computed by arithmetic
    // instead of a second `iter().skip().take().count()` pass. It must yield
    // exactly the same value for every (total, skip, take) combination, since
    // the result drives the last-rendered row's terminal-branch glyph.
    for total in 0..12 {
        for skip in 0..14 {
            for take in 0..14 {
                let expected = (0..total).skip(skip).take(take).count();
                assert_eq!(
                    visible_window_len(total, skip, take),
                    expected,
                    "visible_window_len({total}, {skip}, {take}) diverged from \
                     the iterator it replaced"
                );
            }
        }
    }
}

#[test]
fn tall_sidebar_shows_overflow_indicator_when_clipped() {
    // Item D regression: the body is top-anchored and unscrollable, so a tall
    // stack of live sections must surface a `+N more` indicator instead of
    // silently clipping off the bottom.
    let theme = Theme::no_color();
    let mut state = SidebarState::new();
    state.set_changed_files(
        (0..60)
            .map(|i| ChangedFile {
                path: format!("changed-file-number-{i:02}.rs"),
                status: FileStatus::Modified,
                adds: 1,
                rems: 1,
            })
            .collect(),
        60,
    );

    // A very short body forces the assembled lines to overflow.
    let backend = TestBackend::new(40, 12);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &state, &state_hud(), &theme))
        .expect("draw sidebar");

    let dumped = dump(&terminal);
    assert!(
        dumped.contains("more"),
        "clipped body must show a `+N more` overflow indicator: {dumped}"
    );

    // A roomy body must NOT show the indicator (nothing is clipped).
    let mut roomy = SidebarState::new();
    roomy.set_changed_files(
        vec![ChangedFile {
            path: "only.rs".to_string(),
            status: FileStatus::Modified,
            adds: 1,
            rems: 0,
        }],
        1,
    );
    let backend = TestBackend::new(44, 60);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| draw(frame, frame.area(), &roomy, &state_hud(), &theme))
        .expect("draw sidebar");
    let dumped = dump(&terminal);
    assert!(
        !dumped.contains("more"),
        "a body with room to spare must not show an overflow indicator: {dumped}"
    );
}

/// A HUD with no running agents / workflow so the changes + overflow tests
/// exercise the file window rather than the agent tree.
fn state_hud() -> HudState {
    let mut hud = sample_hud();
    hud.running_agents = 0;
    hud.agents = Vec::new();
    hud.workflow = None;
    hud
}

/// Width-sweep responsive guard: across a range of sidebar widths the segment
/// allocator must (a) never let a rendered row overflow its width, and (b) keep
/// the high-priority live signal — the agent's current activity — visible, even
/// when the static model id has to be dropped. The pre-allocator layout chopped
/// the activity to `waiting for ap…` on a narrow rail; this proves it survives.
#[test]
fn agent_fleet_is_responsive_across_widths() {
    let theme = Theme::no_color();
    let state = SidebarState::new(); // agents_expanded = true
    let mut hud = sample_hud();
    hud.rate_limit = None;
    hud.running_agents = 1;
    hud.agents = vec![AgentTaskSummary {
        name: "agentworkflowtools".to_string(),
        status: "running".to_string(),
        model: "gpt-5.5-fast".to_string(),
        elapsed_secs: 14,
        token_history: vec![1, 4, 2, 8, 5],
        current_tool: Some("grep_search".to_string()),
        current_phase: None,
        last_activity_at: None,
        ..Default::default()
    }];

    for w in [32u16, 40, 56, 80, 120] {
        let backend = TestBackend::new(w, 24);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, frame.area(), &state, &hud, &theme))
            .expect("draw sidebar");

        // (a) No rendered row may exceed the terminal width. Each buffer row is
        // exactly `w` cells, so re-measure the trimmed symbols to catch any
        // double-width-cell miscount.
        let buf = terminal.backend().buffer();
        let rows: Vec<String> = (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect();
        for row in &rows {
            let cells: usize = row.chars().map(char_width).sum();
            assert!(
                cells <= usize::from(w),
                "row overflows width {w} ({cells} cells): {row:?}"
            );
        }

        // (b) The title must NOT wrap. The sidebar renders with Wrap{trim:true},
        // so a 1-cell-too-wide title would spill the model's closing ")" onto the
        // next buffer row instead of clipping. Assert that any row showing the
        // model open-paren also shows its close-paren on the SAME row — the
        // direct guard against the prefix-width / budget off-by-one (a blind
        // per-row cell check cannot see a wrap, since each wrapped row still fits
        // the width).
        let title_row = rows
            .iter()
            .find(|r| r.contains("agentworkflow"))
            .unwrap_or_else(|| panic!("agent title row missing at width {w}: {rows:?}"));
        if title_row.contains('(') {
            assert!(
                title_row.contains(')'),
                "title model wrapped to the next row at width {w}: {title_row:?}"
            );
        }

        // (c) The live activity must survive at every width — it is the signal
        // that tells the user what the agent is doing.
        let dumped = rows.concat();
        assert!(
            dumped.contains("grep_search"),
            "live activity must survive at width {w}: {dumped}"
        );
    }
}
