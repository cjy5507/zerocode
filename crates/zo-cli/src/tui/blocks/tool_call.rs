//! `RenderBlock::ToolCall` widget — compact Codex-style event row.
//!
//! Tool invocations render as transcript events such as `Called`, `Ran`,
//! `Explored`, or `Updated Plan`, with raw JSON hidden behind a short summary.
//!
//! See `code-rules.md` R2 (no ANSI), R9 (`&Theme` styling).

use std::borrow::Cow;
use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use runtime::message_stream::{ToolCallStatus, ToolPreview};
use unicode_width::UnicodeWidthChar;

use crate::tui::glyphs;
use crate::tui::hud::AgentTaskSummary;
use crate::tui::theme::{Palette, Theme};

use super::tool_provenance::{self, Origin};
use super::{compact_path_label, sanitize_inline, wrapped_rows};

/// Braille spinner frames, in order.
pub const SPINNER_FRAMES: [&str; 10] = [
    "\u{280b}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283c}", "\u{2834}", "\u{2826}", "\u{2827}",
    "\u{2807}", "\u{280f}",
];

/// Max display cells for a settled `Ran <command>` label. Long enough to keep a
/// command identifiable, short enough that a sprawling one-liner no longer dumps
/// its whole raw text into the transcript.
const BASH_LABEL_CAP: usize = 64;

const LIVE_TAIL_MAX_BYTES: usize = 8 * 1024;
const LIVE_TAIL_MAX_LINES: usize = 12;

// Status markers and agent-tree box glyphs, each paired with a 1-cell ASCII
// sibling so the transcript stays legible under `NO_COLOR` / `TERM=dumb` /
// non-Nerd-Font terminals (code-rules R10). Routed through [`glyphs::pick`]
// keyed on `!theme.no_color`, mirroring the rest of the TUI's glyph plumbing.
// These live here rather than in `glyphs.rs` because the markers (`○●✓×⊘`) and
// the `⎿` result hook are local to this block's vocabulary.

/// `○` pending / `●` running / `✓` ok / `×` errored / `⊘` cancelled markers.
const MARKER_PENDING: (&str, &str) = ("\u{25cb}", "o");
const MARKER_RUNNING: (&str, &str) = ("\u{25cf}", "*");
const MARKER_OK: (&str, &str) = ("\u{2713}", "v");
const MARKER_ERRORED: (&str, &str) = ("\u{00d7}", "x");
const MARKER_CANCELLED: (&str, &str) = ("\u{2298}", "/");

/// Seconds an in-flight call may run before its row grows an explicit
/// "still waiting … esc to interrupt" hint. Long enough that ordinary
/// builds/fetches never nag; short enough that a hung MCP request stops
/// masquerading as progress within the first minute.
const LONG_WAIT_HINT_SECS: u64 = 30;

/// Animated marker frame for an in-flight tool row — the unmistakable
/// "this is actually running" signal. A static `●` on a call that streams
/// no output (an MCP request returns nothing until it completes) read as
/// frozen, and users cancelled healthy turns; a moving glyph proves both
/// the app and the call are alive. All frames are width-1 so measure and
/// draw agree; NO_COLOR degrades to the classic ASCII spinner.
fn running_marker(tick: u64, color: bool, reduced: bool) -> &'static str {
    if color {
        // ~1Hz spark heartbeat on the 33ms tick clock, matching the activity
        // line's ✦/✧ pulse so every "alive" signal shares one brand beat.
        crate::tui::spinner::spark_glyph_for_elapsed(tick.saturating_mul(33), reduced, true)
    } else {
        // NO_COLOR keeps the classic ASCII rotation: a static glyph here read
        // as frozen and users cancelled healthy turns.
        const FRAMES_NC: [&str; 4] = ["-", "\\", "|", "/"];
        let idx = if reduced {
            0
        } else {
            usize::try_from(tick / 4).unwrap_or(0)
        };
        FRAMES_NC[idx % FRAMES_NC.len()]
    }
}

/// Agent-tree branch (`├`), last-child elbow (`└`), stem (`│`), and the `⎿`
/// completion hook, with ASCII siblings (`+`/`` ` ``/`|`/`+`).
const TREE_BRANCH: (&str, &str) = ("\u{251c}", "+");
const TREE_ELBOW: (&str, &str) = ("\u{2514}", "`");
const TREE_STEM: (&str, &str) = ("\u{2502}", "|");
const TREE_HOOK: (&str, &str) = ("\u{23bf}", "+");
/// Sub-line lead for a running agent's streamed `outputTail` (`⤷` / ASCII `>`).
const TREE_TAIL: (&str, &str) = ("\u{2937}", ">");

/// The brightness step for a tool-call verb, so the transcript reads as a quiet
/// hierarchy (consequential mutate stands out, bookkeeping recedes) rather than
/// a rainbow of category hues. This deliberately drops the old 7-hue band
/// (cyan/info/warn/violet/teal/…): chromatic ink is now reserved for *semantic*
/// state — error / cancelled / the running pulse, resolved by the caller — so a
/// color in the verb column always means "status", never just "category". Verbs
/// come from [`tool_event_summary`]; the three steps are `bright > fg > dim`,
/// staying within the "카드당 최대 3계조" budget. `NO_COLOR` degrades each step
/// to `Reset`, and the marker glyph shape still carries status independently.
fn verb_color(verb: &str, palette: &Palette) -> Color {
    match verb {
        // The consequential band (create + in-place edit) pops via brightness.
        "Wrote" | "Edited" => palette.bright,
        // Plan / state bookkeeping recedes below the body text.
        "Updated" => palette.dim,
        // Read / run / search / delegate / unknown all sit at the body fg —
        // the action's identity is carried by its verb word, not a hue.
        _ => palette.fg,
    }
}

/// One sub-agent row retained for the pinned panel, sidebar, and Ctrl+G viewer.
/// The transcript itself renders the batch as a compact aggregate so the same
/// per-agent detail is not duplicated in the primary conversation flow.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentTreeRow {
    pub agent_id: String,
    pub name: String,
    pub model: String,
    /// `pending` | `running` | `completed` | `failed` | `stopped` | `still_running`.
    pub status: String,
    pub subagent_type: Option<String>,
    pub tool_calls: Option<usize>,
    pub tokens: u64,
    pub elapsed_secs: u64,
    /// Live activity label while running (current tool / wait phase).
    pub activity: Option<String>,
    /// Last streamed chars (manifest `outputTail`), available in the dedicated
    /// agent detail surfaces so the user can inspect what an agent is producing.
    pub output_tail: Option<String>,
    /// Completion sequence (1-based) — rows flip to `⎿ Done` in the order the
    /// agents actually finished, not spawn order.
    pub done_order: Option<u32>,
    /// Manifest `createdAt` — spawn-order sort key (not rendered).
    pub created_at: Option<u64>,
    /// Why the Smart router picked this agent's model (manifest `routeReason`).
    /// The pinned/sidebar/Ctrl+G detail surfaces keep this explainability while
    /// the transcript shows only the batch aggregate. `None` for explicit
    /// models, routing off, or legacy manifests.
    pub route_reason: Option<String>,
}

impl AgentTreeRow {
    fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "completed" | "failed" | "stopped")
    }
}

/// Live tree state for one Spawn-family tool call. Owned by the transcript as
/// a side table keyed by `tool_call_id`, so the boundary `RenderBlock` type
/// stays untouched (code-rules R1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentTree {
    /// Spawn-order rows (stable); per-row `done_order` carries completion order.
    pub rows: Vec<AgentTreeRow>,
    /// Optional provenance label shown in batch headers, e.g. `Smart` for host
    /// prelude batches. Plain model-invoked batches leave this unset.
    pub batch_label: Option<String>,
    /// True once the owning tool call returned its result.
    pub finished: bool,
}

impl AgentTree {
    fn terminal_count(&self) -> usize {
        self.rows.iter().filter(|row| row.is_terminal()).count()
    }

    /// Rows present but not yet all terminal and the call has not returned — the
    /// fan-out is still running. Matches the live-vs-finished split in
    /// [`rendered_lines_with_tree`] and the pinned-panel visibility gate.
    pub(crate) fn is_live(&self) -> bool {
        !self.rows.is_empty() && !self.finished && self.terminal_count() != self.rows.len()
    }
}

/// Whether a tool name is one of the sub-agent spawn family — these calls get
/// the live batch aggregate under their transcript row.
#[must_use]
pub fn is_spawn_family(name: &str) -> bool {
    let display = tool_provenance::display_name(name);
    display.eq_ignore_ascii_case("SpawnMultiAgent")
        || display.eq_ignore_ascii_case("Task")
        || display.eq_ignore_ascii_case("Agent")
}

#[must_use]
pub(crate) fn is_bash(name: &str) -> bool {
    tool_provenance::display_name(name).eq_ignore_ascii_case("bash")
}

/// Whether an in-flight call to this tool should host a live agent-batch summary
/// under its transcript row. The spawn family always does; the `Workflow` tool
/// drives its agents through the *same* manifest machinery (they already feed
/// the live HUD scan, the sidebar, and the Ctrl+O viewer) but is not part of
/// the spawn family — so without this its agents showed up everywhere *except*
/// the conversation transcript. A batch that never gains a row renders nothing
/// (see the `!tree.rows.is_empty()` guards), so opening one for a workflow that
/// spawns no sub-agents is a safe no-op.
#[must_use]
pub fn opens_agent_batch(name: &str) -> bool {
    is_spawn_family(name) || tool_provenance::display_name(name).eq_ignore_ascii_case("Workflow")
}

/// `812` / `4.1k` / `95.1k` / `1.2M` — Claude Code's compact token figure.
#[allow(clippy::cast_precision_loss)] // display-only rounding to one decimal
fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        let m = n as f64 / 1_000_000.0;
        format!("{m:.1}M")
    } else if n >= 1_000 {
        let k = n as f64 / 1_000.0;
        format!("{k:.1}k")
    } else {
        n.to_string()
    }
}

/// Plural-aware `N tool uses` segment; `None` count renders nothing.
fn tool_uses_segment(count: Option<usize>) -> Option<String> {
    let count = count?;
    Some(if count == 1 {
        "1 tool use".to_string()
    } else {
        format!("{count} tool uses")
    })
}

/// The agent-tree details affordance, shared by the live and finished batch
/// headers. `Ctrl+G` (`open_agents_viewer`) opens the per-agent viewer modal
/// and works for both live and finished batches.
const AGENT_TREE_DETAIL_HINT: &str = "ctrl+g for details";

/// `"Explore "` when every row shares one `subagent_type`, else `""`. Both batch
/// headers use this so the agent type is labelled identically (`3 Explore agents`).
fn uniform_agent_type_label(tree: &AgentTree) -> String {
    tree.rows
        .first()
        .and_then(|first| {
            let t = first.subagent_type.as_deref()?;
            tree.rows
                .iter()
                .all(|row| row.subagent_type.as_deref() == Some(t))
                .then(|| format!("{t} "))
        })
        .unwrap_or_default()
}

fn agent_type_count_label(tree: &AgentTree) -> String {
    let n = tree.rows.len();
    let noun = if n == 1 { "agent" } else { "agents" };
    format!("{n} {}{noun}", uniform_agent_type_label(tree))
}

fn batch_label_with_glyph(label: &str, color: bool) -> Cow<'_, str> {
    if label == "Smart" {
        Cow::Owned(format!(
            "{} {label}",
            glyphs::pick(color, glyphs::SMART_AUTO, glyphs::SMART_AUTO_NC)
        ))
    } else {
        Cow::Borrowed(label)
    }
}

/// The header for a live batch: `Running 3 Explore agents… (ctrl+g for details)`.
/// Mirrors [`finished_header`]; shown while the batch is still collecting rows.
fn running_header(tree: &AgentTree, color: bool) -> String {
    let batch = agent_type_count_label(tree);
    match tree.batch_label.as_deref() {
        Some(label) if !label.is_empty() => {
            let label = batch_label_with_glyph(label, color);
            format!("{label} running {batch}\u{2026} ({AGENT_TREE_DETAIL_HINT})")
        }
        _ => format!("Running {batch}\u{2026} ({AGENT_TREE_DETAIL_HINT})"),
    }
}

/// The header for a finished batch: `3 Explore agents finished (ctrl+g for details)`.
fn finished_header(tree: &AgentTree, color: bool) -> String {
    let batch = agent_type_count_label(tree);
    match tree.batch_label.as_deref() {
        Some(label) if !label.is_empty() => {
            let label = batch_label_with_glyph(label, color);
            format!("{label} {batch} finished ({AGENT_TREE_DETAIL_HINT})")
        }
        _ => format!("{batch} finished ({AGENT_TREE_DETAIL_HINT})"),
    }
}

/// One quiet child line for an agent batch. Per-agent names, output tails,
/// routing reasons, and completion rows remain available in the pinned panel,
/// sidebar, and Ctrl+G viewer; the transcript keeps only the batch outcome and
/// aggregate work so it stays readable after a large fan-out settles.
fn compact_agent_batch_line(tree: &AgentTree, theme: &Theme) -> Line<'static> {
    let completed = tree
        .rows
        .iter()
        .filter(|row| row.status == "completed")
        .count();
    let failed = tree
        .rows
        .iter()
        .filter(|row| row.status == "failed")
        .count();
    let stopped = tree
        .rows
        .iter()
        .filter(|row| row.status == "stopped")
        .count();
    let active = tree
        .rows
        .len()
        .saturating_sub(completed + failed + stopped);

    let dim = Style::new().fg(theme.palette.dim);
    let mut segments: Vec<(String, Style)> = Vec::with_capacity(6);
    if completed > 0 {
        segments.push((format!("{completed} done"), dim));
    }
    if failed > 0 {
        segments.push((
            format!("{failed} failed"),
            Style::new()
                .fg(theme.palette.error)
                .add_modifier(Modifier::BOLD),
        ));
    }
    if stopped > 0 {
        segments.push((
            format!("{stopped} stopped"),
            Style::new().fg(theme.palette.warn),
        ));
    }
    if active > 0 {
        let label = if tree.finished { "unfinished" } else { "running" };
        segments.push((
            format!("{active} {label}"),
            Style::new().fg(theme.palette.teal),
        ));
    }

    let known_tool_calls = tree.rows.iter().filter_map(|row| row.tool_calls).sum::<usize>();
    if known_tool_calls > 0 {
        if let Some(label) = tool_uses_segment(Some(known_tool_calls)) {
            segments.push((label, dim));
        }
    }
    let tokens = tree.rows.iter().map(|row| row.tokens).sum::<u64>();
    if tokens > 0 {
        segments.push((format!("{} tokens", format_tokens(tokens)), dim));
    }

    let mut spans = vec![Span::styled(
        format!(
            "  {} ",
            glyphs::pick(!theme.no_color, TREE_ELBOW.0, TREE_ELBOW.1)
        ),
        dim,
    )];
    for (index, (label, style)) in segments.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        spans.push(Span::styled(label, style));
    }
    Line::from(spans)
}

/// Render the per-agent tree lines under a Spawn-family tool call row.
///
/// Live (unfinished) rows show name · tools · elapsed · current activity; a row
/// that reached a terminal state gains a `⎿ Done` / `⎿ Failed` line the moment
/// it lands — in completion order — exactly like Claude Code's agent batches.
#[cfg(test)]
fn agent_tree_lines(tree: &AgentTree, theme: &Theme) -> Vec<Line<'static>> {
    agent_tree_lines_with_spans(tree, theme, None).0
}

/// Spans-returning form of [`agent_tree_lines`]: identical lines, plus per agent
/// row `(agent_id, line_count)` so the pinned live panel can map a screen row
/// back to the agent it belongs to for click-to-inspect. `hovered` underlines
/// that agent's title row (a NO_COLOR-safe hover affordance). Inline transcript
/// callers use the `None` wrapper above and get byte- and style-identical output
/// to before.
#[allow(clippy::too_many_lines)] // one row's worth of spans assembled in documented order
fn agent_tree_lines_with_spans(
    tree: &AgentTree,
    theme: &Theme,
    hovered: Option<&str>,
) -> (Vec<Line<'static>>, Vec<(String, u16)>) {
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(tree.rows.len() * 2);
    let mut row_spans: Vec<(String, u16)> = Vec::with_capacity(tree.rows.len());
    let last = tree.rows.len().saturating_sub(1);
    let color = !theme.no_color;
    for (i, row) in tree.rows.iter().enumerate() {
        let line_start = lines.len();
        let branch = if i == last {
            glyphs::pick(color, TREE_ELBOW.0, TREE_ELBOW.1)
        } else {
            glyphs::pick(color, TREE_BRANCH.0, TREE_BRANCH.1)
        };
        let stem = if i == last {
            " "
        } else {
            glyphs::pick(color, TREE_STEM.0, TREE_STEM.1)
        };
        // Per-row status marker + color band. Every agent row — the running
        // ones too, not just the terminal `⎿ Done` line — now leads with a
        // colored status dot and a status-tinted tree connector, so the eye
        // reads a live vertical color band down the batch (teal running / green
        // done / red failed / amber stopped) the whole time. Before, a running
        // tree was a flat grey wall that only gained any color once an agent
        // finished, so an in-flight fan-out (the state you actually watch) had
        // no at-a-glance status. Markers reuse the row markers' glyph pairs, so
        // they degrade to 1-cell ASCII siblings under NO_COLOR (R10) and status
        // is still carried by the glyph shape, never color alone.
        let (marker_pair, status_color) = match row.status.as_str() {
            "completed" => (MARKER_OK, theme.palette.success),
            "failed" => (MARKER_ERRORED, theme.palette.error),
            "stopped" => (MARKER_CANCELLED, theme.palette.warn),
            "pending" => (MARKER_PENDING, theme.palette.dim),
            // running / still_running / any other live state.
            _ => (MARKER_RUNNING, theme.palette.teal),
        };
        let marker = glyphs::pick(color, marker_pair.0, marker_pair.1);
        let marker_style = Style::new().fg(status_color);
        // The agent name is the row's primary identifier, so it carries weight
        // (BOLD) plus a per-agent identity hue keyed on a stable hash of the
        // agent id (`agent_color`), so sibling agents in a fan-out are told apart
        // at a glance and an agent keeps its color as others finish — the status
        // is carried separately by the marker/connector, so the name hue reads as
        // *which* agent, not its state. Bold survives NO_COLOR (a text attribute,
        // not a hue), and `agent_color` drops to `Reset` there, so the emphasis
        // holds and rows stay distinct by position on a dumb terminal. A hovered
        // row also gets UNDERLINE — another text attribute, so the mouse-hover
        // affordance is legible under NO_COLOR too.
        let mut name_style = Style::new()
            .fg(theme.agent_color(&row.agent_id))
            .add_modifier(Modifier::BOLD);
        if hovered.is_some_and(|id| id == row.agent_id) {
            name_style = name_style.add_modifier(Modifier::UNDERLINED);
        }
        let meta_style = Style::new().fg(theme.palette.dim);

        let mut meta: Vec<String> = Vec::with_capacity(4);
        if let Some(tools) = tool_uses_segment(row.tool_calls) {
            meta.push(tools);
        }
        if row.tokens > 0 {
            meta.push(format!("{} tokens", format_tokens(row.tokens)));
        }
        if !row.is_terminal() {
            if row.elapsed_secs >= 1 {
                meta.push(format_elapsed_secs(row.elapsed_secs));
            }
            if let Some(activity) = row.activity.as_deref() {
                meta.push(truncate_activity(activity, 48));
            }
        }
        // Smart routing explainability: a truncated `routed: …` segment when
        // the manifest carries a routeReason (exploration/learned-shadow/quota
        // suffixes ride inside it already). Absent for explicit models /
        // routing off / legacy manifests — the common case today — so this is
        // additive noise only when Smart AUTO actually made a decision.
        if let Some(reason) = row.route_reason.as_deref() {
            meta.push(format!("routed: {}", truncate_activity(reason, 40)));
        }
        let mut spans: Vec<Span<'static>> = vec![
            Span::raw("   "),
            Span::styled(format!("{branch} "), meta_style),
            Span::styled(format!("{marker} "), marker_style),
            Span::styled(sanitize_inline(&row.name), name_style),
        ];
        if !meta.is_empty() {
            spans.push(Span::styled(format!(" · {}", meta.join(" · ")), meta_style));
        }
        lines.push(Line::from(spans));

        // Live output tail: the last line the agent streamed, shown dim/italic
        // under its row (CC-style) so the user reads what each agent is producing
        // — only while running, since a terminal row's `⎿ Done` carries the
        // outcome instead.
        if !row.is_terminal() {
            if let Some(tail) = row.output_tail.as_deref().and_then(agent_tail_preview) {
                lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::styled(format!("{stem} "), Style::new().fg(theme.palette.dim)),
                    Span::styled(
                        format!("{}  ", glyphs::pick(color, TREE_TAIL.0, TREE_TAIL.1)),
                        Style::new().fg(theme.palette.faint),
                    ),
                    Span::styled(
                        tail,
                        Style::new()
                            .fg(theme.palette.dim)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
        }

        if row.is_terminal() {
            // The outcome label and its `⎿` hook share one color band so a
            // glance down the tree's left edge reads completed (green) vs failed
            // (red) vs stopped (amber) without reading the words. The label is
            // BOLD so the verdict pops out of the dim meta around it, and the
            // hook is tinted to the same hue instead of a flat `dim` — the hook
            // was the one part of the outcome row that stayed grey. `stopped`
            // uses `warn` (amber) to agree with the sidebar's agent tree, which
            // already maps stopped→warn; the old `muted` grey was both lower
            // contrast and inconsistent across the two surfaces.
            let (label, label_color) = match row.status.as_str() {
                "completed" => ("Done".to_string(), theme.palette.success),
                "stopped" => ("Stopped".to_string(), theme.palette.warn),
                _ => ("Failed".to_string(), theme.palette.error),
            };
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(format!("{stem} "), Style::new().fg(theme.palette.dim)),
                Span::styled(
                    format!("{}  ", glyphs::pick(color, TREE_HOOK.0, TREE_HOOK.1)),
                    Style::new().fg(label_color),
                ),
                Span::styled(
                    label,
                    Style::new()
                        .fg(label_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        }

        // Record this agent's on-screen span (1 title line, plus an optional
        // output-tail or terminal line) so a click on any of its rows maps back
        // to this agent id. `line_start` was captured before the row's lines
        // were pushed, so the count is exact regardless of how many the row
        // produced.
        let line_count = u16::try_from(lines.len().saturating_sub(line_start)).unwrap_or(u16::MAX);
        row_spans.push((row.agent_id.clone(), line_count));
    }
    (lines, row_spans)
}

/// A single dim `⎿ spawning…` line shown while a spawn batch has no agent rows
/// yet — fills the gap between the tool call and the first agent manifest so the
/// row never stalls as a bare verb line (the silent `agents 0/N` gap).
fn spawning_placeholder_line(theme: &Theme) -> Line<'static> {
    let color = !theme.no_color;
    Line::from(vec![
        Span::raw("   "),
        Span::styled(
            format!("{}  ", glyphs::pick(color, TREE_HOOK.0, TREE_HOOK.1)),
            Style::new().fg(theme.palette.dim),
        ),
        Span::styled(
            "spawning\u{2026}",
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::ITALIC),
        ),
    ])
}


/// The last non-blank line of an agent's rolling `outputTail` — what it most
/// recently streamed — for the dim `⤷ …` activity sub-line. `None` when the
/// tail is empty/whitespace.
fn last_output_line(tail: &str) -> Option<&str> {
    tail.lines().map(str::trim).rev().find(|line| !line.is_empty())
}

const AGENT_TAIL_PREVIEW_CELLS: usize = 64;

/// Stable one-line preview for manifest `outputTail` snapshots.
///
/// Sub-agent output arrives as polled snapshots rather than token deltas, so a
/// true typewriter pacer would need timestamped per-agent state. At render time
/// we can still smooth the bursty shape: show only the latest non-blank line,
/// sanitize control characters, collapse whitespace, and cap display width so a
/// large manifest flush cannot widen or wrap the agent tree unexpectedly.
fn agent_tail_preview(tail: &str) -> Option<String> {
    let line = last_output_line(tail)?;
    let sanitized = sanitize_inline(line);
    let compact = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    let compact = compact.trim();
    if compact.is_empty() {
        None
    } else {
        Some(truncate_activity(compact, AGENT_TAIL_PREVIEW_CELLS))
    }
}

/// Max agent rows shown in the pinned live panel above the input. A larger
/// fan-out is summarised with a `… +N more` line rather than eating the
/// transcript; the full set is always in the sidebar / `ctrl+g` viewer.
pub(crate) const LIVE_PANEL_MAX_AGENTS: usize = 6;

/// Build the lines for the pinned **live-agent panel** drawn just above the
/// input while a fan-out runs (Claude-Code parity: a `Running N agents…` header
/// + the per-agent tree with live tool / tokens / elapsed / output-tail).
///
/// Sourced from the HUD's [`AgentTaskSummary`] scan — the *same* data the
/// sidebar reads — so the panel renders for every fan-out path (host pre-spawn,
/// model-invoked `SpawnMultiAgent`/`Workflow`, or a scrolled-away host row),
/// not only when a host `ToolCall` transcript row is on screen. Reuses
/// [`running_header`] + [`agent_tree_lines_with_spans`] for the full detail tree;
/// the transcript intentionally uses only the compact aggregate. Returns empty
/// when there are no agents to show.
/// Absolute placement of one agent's rows inside the pinned live panel: the
/// agent id, the 0-based line offset of its first row within the panel's line
/// vec, and how many lines it spans. `draw_agent_panel` turns these into screen
/// rects for click-to-inspect routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentRowSpan {
    pub id: String,
    pub start: u16,
    pub len: u16,
}

/// Build live-agent panel lines plus per-agent [`AgentRowSpan`]s. The header
/// occupies line 0, so the first agent row starts at absolute offset 1.
/// `hovered` underlines that agent's title row. Returns empty vecs when there
/// are no agents to show.
pub(crate) fn live_agent_panel_lines_with_spans(
    agents: &[AgentTaskSummary],
    theme: &Theme,
    batch_label: Option<&str>,
    hovered: Option<&str>,
) -> (Vec<Line<'static>>, Vec<AgentRowSpan>) {
    if agents.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted: Vec<&AgentTaskSummary> = agents.iter().collect();
    // Spawn order (stable), shared with the detailed agent tree.
    sorted.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.name.cmp(&b.name)));
    let overflow = sorted.len().saturating_sub(LIVE_PANEL_MAX_AGENTS);
    let rows: Vec<AgentTreeRow> = sorted
        .into_iter()
        .take(LIVE_PANEL_MAX_AGENTS)
        .map(summary_to_tree_row)
        .collect();
    let tree = AgentTree {
        rows,
        batch_label: batch_label.map(str::to_string),
        finished: false,
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!(
                "{} ",
                glyphs::pick(!theme.no_color, MARKER_RUNNING.0, MARKER_RUNNING.1)
            ),
            Style::new().fg(theme.palette.teal),
        ),
        Span::styled(
            running_header(&tree, !theme.no_color),
            Style::new()
                .fg(theme.palette.fg)
                .add_modifier(Modifier::BOLD),
        ),
    ])];
    // The header occupies line 0, so agent rows begin at absolute offset 1.
    let header_offset = u16::try_from(lines.len()).unwrap_or(1);
    let (tree_lines, row_spans) = agent_tree_lines_with_spans(&tree, theme, hovered);
    lines.extend(tree_lines);
    let mut spans: Vec<AgentRowSpan> = Vec::with_capacity(row_spans.len());
    let mut cursor = header_offset;
    for (id, len) in row_spans {
        spans.push(AgentRowSpan {
            id,
            start: cursor,
            len,
        });
        cursor = cursor.saturating_add(len);
    }
    if overflow > 0 {
        lines.push(Line::from(Span::styled(
            format!("   \u{2026} +{overflow} more"),
            Style::new().fg(theme.palette.dim),
        )));
    }
    (lines, spans)
}

/// One HUD agent summary → a tree row for the live panel. Carries the live
/// tool/phase (`activity_label`) and the streamed `output_tail`; `done_order`
/// is `None` because the live panel only ever holds non-terminal agents (the
/// HUD scan filters terminals out before this point).
fn summary_to_tree_row(summary: &AgentTaskSummary) -> AgentTreeRow {
    AgentTreeRow {
        agent_id: summary.id.clone(),
        name: summary.name.clone(),
        model: summary.model.clone(),
        status: summary.status.clone(),
        subagent_type: summary.subagent_type.clone(),
        tool_calls: summary.tool_calls,
        tokens: summary.tokens,
        elapsed_secs: summary.elapsed_secs,
        activity: summary.activity_label().map(str::to_string),
        output_tail: summary.output_tail.clone(),
        done_order: None,
        created_at: summary.created_at,
        route_reason: summary.route_reason.clone(),
    }
}

/// `50s` / `1m 32s` — bare (no parens) elapsed for tree meta segments.
fn format_elapsed_secs(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}m {s}s")
        }
    }
}

/// Ellipsis appended when an over-long single-line tool row is clipped to width.
const CLIP_ELLIPSIS: char = '\u{2026}';

/// Clip each line's spans to fit `width` display cells, appending an ellipsis
/// (`…`) to mark right-side truncation while preserving span styles.
///
/// Every line a ToolCall block produces is single-line **by design** — the verb
/// row, the agent-tree rows (`├ ● name · … · routed: …`), the `⤷` output-tail
/// and `⎿` terminal sub-lines, the batch headers, the `spawning…` placeholder —
/// each is meant to occupy exactly one screen row. Fed through `Paragraph::wrap`
/// (which the renderer and [`wrapped_rows`] both use), an over-long row wrapped
/// to column 0 instead, scattering orphan meta fragments (`selector wi…`) with
/// no hanging indent and shattering the tree's left structure (the live
/// screenshot bug). Clipping from the RIGHT keeps the left tree glyphs
/// (`├`/`└`/`⎿`/`⤷` + indent) intact and drops only the trailing meta, and keeps
/// line-count == row-count so the `wrapped_rows` height and the paint agree —
/// which in turn lets the per-row card / selection wash cover every rendered row
/// (a wrapped continuation row previously washed only its col-0 fragment).
///
/// Fitting lines are returned untouched (no span rebuild), so the common wide
/// case stays byte-identical to the pre-clip output. Widths use the crate's
/// `text_metrics` SSOT (unicode-width default tables — the same tables ratatui's
/// wrap consults), so a clipped line measures `<= width` for ratatui too and
/// never re-wraps.
fn clip_lines_to_width(lines: Vec<Line<'static>>, width: u16) -> Vec<Line<'static>> {
    let width = usize::from(width);
    lines
        .into_iter()
        .map(|line| clip_line_to_width(line, width))
        .collect()
}

/// Right-clip one line to `width` display cells (see [`clip_lines_to_width`]).
fn clip_line_to_width(line: Line<'static>, width: usize) -> Line<'static> {
    if width == 0 {
        return Line {
            style: line.style,
            alignment: line.alignment,
            spans: Vec::new(),
        };
    }
    let total: usize = line
        .spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum();
    if total <= width {
        // Fits — return untouched so the wide common case is byte-identical.
        return line;
    }
    // Reserve exactly one cell for the ellipsis; the clipped body fills the rest.
    let body_limit = width - 1;
    let mut used = 0usize;
    let mut clipped: Vec<Span<'static>> = Vec::with_capacity(line.spans.len() + 1);
    // The ellipsis inherits the style of the span it truncates (typically the
    // dim meta), so the `…` reads as part of that trailing segment.
    let mut ellipsis_style = Style::default();
    for span in &line.spans {
        ellipsis_style = span.style;
        let span_w = display_width(span.content.as_ref());
        if used + span_w <= body_limit {
            used += span_w;
            clipped.push(span.clone());
            continue;
        }
        // This span straddles the limit — keep the char prefix that still fits.
        let mut prefix = String::new();
        for ch in span.content.chars() {
            let w = char_width(ch);
            if used + w > body_limit {
                break;
            }
            used += w;
            prefix.push(ch);
        }
        if !prefix.is_empty() {
            clipped.push(Span::styled(prefix, span.style));
        }
        break;
    }
    clipped.push(Span::styled(CLIP_ELLIPSIS.to_string(), ellipsis_style));
    Line {
        style: line.style,
        alignment: line.alignment,
        spans: clipped,
    }
}

/// Render a `ToolCall` row in Claude Code style:
/// `● Name(summary)  status` — borderless, single line.
///
/// `is_tail_active` is `true` when this tool call is the most recent
/// in-flight one in the transcript (status Pending or Running and no
/// later tool call exists). The caller keeps this in the signature so
/// transcript layout can evolve without changing the block API.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    tool_call_id: &str,
    name: &str,
    summary: &str,
    preview: &ToolPreview,
    status: ToolCallStatus,
    theme: &Theme,
    tick: u64,
    scroll_offset: u16,
    is_tail_active: bool,
    elapsed: Option<Duration>,
    agent_tree: Option<&AgentTree>,
    live_tail_expanded: bool,
) {
    // Clip every design-line to the draw width so no single-line row wraps to
    // column 0 (see `clip_lines_to_width`). The measurement seam (`estimate_rows`)
    // clips to the same width, keeping cached height and paint in lockstep.
    let lines = clip_lines_to_width(
        rendered_lines_with_state(
            Some(tool_call_id),
            name,
            summary,
            preview,
            status,
            theme,
            tick,
            is_tail_active,
            elapsed,
            agent_tree,
            live_tail_expanded,
        ),
        area.width,
    );
    let para = Paragraph::new(lines)
        .style(Style::new().fg(theme.palette.fg))
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));
    frame.render_widget(para, area);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_rows(
    tool_call_id: Option<&str>,
    name: &str,
    summary: &str,
    preview: &ToolPreview,
    status: ToolCallStatus,
    theme: &Theme,
    width: u16,
    agent_tree: Option<&AgentTree>,
    live_tail_expanded: bool,
) -> u16 {
    // Height must be measured over the SAME clipped lines the draw paints, or a
    // long row the draw clips to one line would be measured (pre-clip) as
    // multiple wrapped rows, reserving a phantom row. Clip, then count.
    wrapped_rows(
        &clip_lines_to_width(
            rendered_lines_with_state(
                tool_call_id,
                name,
                summary,
                preview,
                status,
                theme,
                0,
                false,
                None,
                agent_tree,
                live_tail_expanded,
            ),
            width,
        ),
        width,
    )
}

/// Tree-less convenience for the block's unit tests.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rendered_lines(
    name: &str,
    summary: &str,
    preview: &ToolPreview,
    status: ToolCallStatus,
    theme: &Theme,
    tick: u64,
    is_tail_active: bool,
    elapsed: Option<Duration>,
) -> Vec<Line<'static>> {
    rendered_lines_with_tree(
        name,
        summary,
        preview,
        status,
        theme,
        tick,
        is_tail_active,
        elapsed,
        None,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[cfg(test)]
pub(crate) fn rendered_lines_with_tree(
    name: &str,
    summary: &str,
    preview: &ToolPreview,
    status: ToolCallStatus,
    theme: &Theme,
    tick: u64,
    is_tail_active: bool,
    elapsed: Option<Duration>,
    agent_tree: Option<&AgentTree>,
) -> Vec<Line<'static>> {
    rendered_lines_with_state(
        None,
        name,
        summary,
        preview,
        status,
        theme,
        tick,
        is_tail_active,
        elapsed,
        agent_tree,
        false,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn rendered_lines_with_state(
    tool_call_id: Option<&str>,
    name: &str,
    summary: &str,
    preview: &ToolPreview,
    status: ToolCallStatus,
    theme: &Theme,
    tick: u64,
    is_tail_active: bool,
    elapsed: Option<Duration>,
    agent_tree: Option<&AgentTree>,
    live_tail_expanded: bool,
) -> Vec<Line<'static>> {
    // A `TodoWrite` / `TaskList` call is rendered entirely by its ToolResult
    // checklist block (`• Updated Plan · N/M done` + rows). Rendering a head
    // here too produced a duplicate `Updated Plan` line above that block, so the
    // call row renders nothing — the result block owns the plan display. (A
    // plan-update never carries an agent tree, so nothing else is lost.)
    if is_plan_update_tool(name) {
        return Vec::new();
    }
    // A finished agent batch keeps one outcome header and one aggregate child
    // line. The full tree already lives in the sidebar and Ctrl+G viewer.
    if let Some(tree) = agent_tree {
        if !tree.rows.is_empty() && (tree.finished || tree.terminal_count() == tree.rows.len()) {
            let failed = tree.rows.iter().any(|row| row.status == "failed");
            let stopped = tree.rows.iter().any(|row| row.status == "stopped");
            let (marker_pair, marker_color) = if failed {
                (MARKER_ERRORED, theme.palette.error)
            } else if stopped {
                (MARKER_CANCELLED, theme.palette.warn)
            } else {
                (MARKER_OK, theme.palette.success)
            };
            let lines = vec![Line::from(vec![
                Span::styled(
                    format!(
                        "{} ",
                        glyphs::pick(!theme.no_color, marker_pair.0, marker_pair.1)
                    ),
                    Style::new().fg(marker_color),
                ),
                Span::styled(
                    finished_header(tree, !theme.no_color),
                    Style::new()
                        .fg(theme.palette.fg)
                        .add_modifier(Modifier::BOLD),
                ),
            ]), compact_agent_batch_line(tree, theme)];
            return lines;
        }
    }
    // A live batch shows the batch header plus aggregate progress. Per-agent
    // activity stays in the pinned panel instead of being duplicated inline.
    if let Some(tree) = agent_tree {
        if !tree.rows.is_empty()
            && matches!(status, ToolCallStatus::Pending | ToolCallStatus::Running)
        {
            let lines = vec![Line::from(vec![
                Span::styled(
                    format!(
                        "{} ",
                        running_marker(
                            tick,
                            !theme.no_color,
                            crate::tui::term::reduce_motion_enabled()
                        )
                    ),
                    Style::new().fg(theme.palette.teal),
                ),
                Span::styled(
                    running_header(tree, !theme.no_color),
                    Style::new()
                        .fg(theme.palette.fg)
                        .add_modifier(Modifier::BOLD),
                ),
            ]), compact_agent_batch_line(tree, theme)];
            return lines;
        }
    }
    let (verb, detail) = if matches!(status, ToolCallStatus::Pending | ToolCallStatus::Running) {
        pending_tool_event_summary(name, summary, preview)
            .unwrap_or_else(|| tool_event_summary(name, summary, preview))
    } else {
        tool_event_summary(name, summary, preview)
    };
    // Snapshot the reduce-motion preference once for this row's marker and verb:
    // when set, the running row settles to its non-pulsing hue and first frame.
    let reduced = crate::tui::term::reduce_motion_enabled();
    let marker_style = match status {
        ToolCallStatus::Errored => Style::new()
            .fg(theme.palette.error)
            .add_modifier(Modifier::BOLD),
        ToolCallStatus::Pending => Style::new().fg(theme.palette.dim),
        ToolCallStatus::Running if is_tail_active && !reduced => {
            if (tick / 8).is_multiple_of(2) {
                Style::new().fg(theme.palette.success)
            } else {
                Style::new().fg(theme.palette.muted)
            }
        }
        ToolCallStatus::Running | ToolCallStatus::Ok => Style::new().fg(theme.palette.success),
        ToolCallStatus::Cancelled => Style::new().fg(theme.palette.muted),
    };
    let color = !theme.no_color;
    let marker = match status {
        ToolCallStatus::Pending => glyphs::pick(color, MARKER_PENDING.0, MARKER_PENDING.1),
        // In-flight rows spin instead of holding a static dot, so a silent
        // long call (no streamed output until it returns) still visibly runs.
        ToolCallStatus::Running => running_marker(tick, color, reduced),
        ToolCallStatus::Ok => glyphs::pick(color, MARKER_OK.0, MARKER_OK.1),
        ToolCallStatus::Errored => glyphs::pick(color, MARKER_ERRORED.0, MARKER_ERRORED.1),
        ToolCallStatus::Cancelled => glyphs::pick(color, MARKER_CANCELLED.0, MARKER_CANCELLED.1),
    };
    // Role color: each verb category carries a distinct hue so the eye scans the
    // transcript by color band first, text second (the flat one-color wall was the
    // chrome>content problem). While a tool is actively running its verb pulses to
    // bright — settling to the category hue on completion — so the live row is
    // visibly alive. NO_COLOR degrades every palette hue to `Reset`, and status is
    // still carried by the marker glyph shape, so color is never the only signal.
    let verb_base = verb_color(verb, &theme.palette);
    let verb_style = match status {
        ToolCallStatus::Running if is_tail_active && !reduced => {
            let hue = if (tick / 8).is_multiple_of(2) {
                theme.palette.bright
            } else {
                verb_base
            };
            Style::new().fg(hue).add_modifier(Modifier::BOLD)
        }
        ToolCallStatus::Errored => Style::new()
            .fg(theme.palette.error)
            .add_modifier(Modifier::BOLD),
        ToolCallStatus::Cancelled => Style::new()
            .fg(theme.palette.muted)
            .add_modifier(Modifier::BOLD),
        _ => Style::new().fg(verb_base).add_modifier(Modifier::BOLD),
    };
    let detail_style = if matches!(tool_provenance::classify(name), Origin::Mcp(_)) {
        Style::new()
            .fg(theme.palette.info)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(theme.palette.fg)
    };

    let mut spans: Vec<Span<'_>> = Vec::with_capacity(8);
    spans.push(Span::styled(format!("{marker} "), marker_style));
    spans.push(Span::styled(verb.to_string(), verb_style));
    if !detail.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(detail, detail_style));
    }

    if let Some(d) = elapsed {
        if matches!(status, ToolCallStatus::Running | ToolCallStatus::Pending) && d.as_secs() >= 1 {
            let label = format_elapsed(d);
            spans.push(Span::styled(" · ", Style::new().fg(theme.palette.dim)));
            spans.push(Span::styled(
                label,
                Style::new()
                    .fg(theme.palette.dim)
                    .add_modifier(Modifier::ITALIC),
            ));
        }
        // A long-silent in-flight call says explicitly that the host is still
        // waiting — and on which MCP server — plus the interrupt affordance.
        // Elapsed alone can't separate "working" from "hung" (the 52-minute
        // stuck MCP request), so users cancelled healthy turns and sat on dead
        // ones; naming the wait makes that call an informed one. A running
        // spawn batch with rows never reaches this path (its live-header
        // branch returned above), so a healthy delegation can't read as hung.
        if matches!(status, ToolCallStatus::Running) && d.as_secs() >= LONG_WAIT_HINT_SECS {
            let wait = match tool_provenance::classify(name) {
                Origin::Mcp(server) => format!("waiting on {server} \u{2014} esc to interrupt"),
                Origin::Local => "still waiting \u{2014} esc to interrupt".to_string(),
            };
            spans.push(Span::styled(" · ", Style::new().fg(theme.palette.dim)));
            spans.push(Span::styled(
                wait,
                Style::new()
                    .fg(theme.palette.warn)
                    .add_modifier(Modifier::ITALIC),
            ));
        }
    }
    if matches!(status, ToolCallStatus::Errored | ToolCallStatus::Cancelled) {
        let (label, color) = if matches!(status, ToolCallStatus::Errored) {
            ("error", theme.palette.error)
        } else {
            ("cancelled", theme.palette.muted)
        };
        spans.push(Span::styled(" · ", Style::new().fg(theme.palette.dim)));
        spans.push(Span::styled(
            label.to_string(),
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        ));
    }

    let mut lines = vec![Line::from(spans)];
    // Agent batch under a Spawn-family / Workflow call. With rows, keep one
    // compact progress line; the full tree remains in the dedicated surfaces.
    // While the batch is live but has no rows yet (first manifest/completion not
    // in), show a `⎿ spawning…` placeholder so the row never stalls as a bare
    // verb line — the silent `agents 0/N` gap was the complaint. The placeholder
    // is strictly for the row-less spawn window: a batch with rows always shows
    // its real aggregate, never a stand-in.
    match agent_tree {
        Some(tree) if !tree.rows.is_empty() => lines.push(compact_agent_batch_line(tree, theme)),
        _ if opens_agent_batch(name)
            && matches!(status, ToolCallStatus::Pending | ToolCallStatus::Running) =>
        {
            lines.push(spawning_placeholder_line(theme));
        }
        _ => {}
    }
    if live_tail_expanded && status == ToolCallStatus::Running && is_bash(name) {
        lines.extend(live_tail_lines(tool_call_id, elapsed, theme));
    }
    lines
}

fn live_tail_lines(
    tool_call_id: Option<&str>,
    row_elapsed: Option<Duration>,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let snapshot = live_tail_snapshot(tool_call_id);
    let elapsed = snapshot
        .as_ref()
        .map_or_else(|| row_elapsed.unwrap_or_default(), |snapshot| snapshot.elapsed);
    let dim = Style::new().fg(theme.palette.dim);
    let header_style = dim.add_modifier(Modifier::BOLD);

    let Some(snapshot) = snapshot else {
        return vec![Line::styled(
            format!("no output yet · running {}", format_elapsed(elapsed)),
            header_style,
        )];
    };
    let stdout = tools::live_output::sanitize_live_output(&snapshot.stdout_tail);
    let stderr = tools::live_output::sanitize_live_output(&snapshot.stderr_tail);
    let (content, is_stderr) = if stdout.trim().is_empty() {
        (stderr, true)
    } else {
        (stdout, false)
    };
    if content.trim().is_empty() {
        return vec![Line::styled(
            format!("no output yet · running {}", format_elapsed(elapsed)),
            header_style,
        )];
    }

    let age = snapshot.last_output_age.unwrap_or_default();
    let mut lines = vec![Line::styled(
        format!("live output · {} ago", format_elapsed(age)),
        header_style,
    )];
    let content_lines: Vec<&str> = content.lines().collect();
    let start = content_lines.len().saturating_sub(LIVE_TAIL_MAX_LINES);
    for (index, line) in content_lines[start..].iter().enumerate() {
        if is_stderr && index == 0 {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("stderr ", dim.add_modifier(Modifier::BOLD)),
                Span::styled((*line).to_string(), Style::new().fg(theme.palette.fg)),
            ]));
        } else {
            let prefix = if is_stderr { "         " } else { "  " };
            lines.push(Line::styled(
                format!("{prefix}{line}"),
                Style::new().fg(theme.palette.fg),
            ));
        }
    }
    lines
}

#[must_use]
pub(crate) fn live_tail_row_count(tool_call_id: &str) -> u16 {
    let Some(snapshot) = live_tail_snapshot(Some(tool_call_id)) else {
        return 1;
    };
    let stdout = tools::live_output::sanitize_live_output(&snapshot.stdout_tail);
    let stderr = tools::live_output::sanitize_live_output(&snapshot.stderr_tail);
    let content = if stdout.trim().is_empty() {
        stderr
    } else {
        stdout
    };
    if content.trim().is_empty() {
        1
    } else {
        1 + u16::try_from(content.lines().count().min(LIVE_TAIL_MAX_LINES)).unwrap_or(0)
    }
}

fn live_tail_snapshot(tool_call_id: Option<&str>) -> Option<tools::live_output::LiveTailSnapshot> {
    tool_call_id
        .and_then(|key| tools::live_output::snapshot(key, LIVE_TAIL_MAX_BYTES))
        .or_else(|| tools::live_output::current(LIVE_TAIL_MAX_BYTES))
}

fn pending_tool_event_summary(
    name: &str,
    summary: &str,
    preview: &ToolPreview,
) -> Option<(&'static str, String)> {
    let ToolPreview::Generic { input_summary, .. } = preview else {
        return None;
    };
    if !summary.trim().is_empty() || !input_summary.trim().is_empty() {
        return None;
    }
    let display = canonical_tool_basename(name);
    let phrase = pending_action(display)?.trim_end_matches('\u{2026}');
    Some(("Starting", phrase.to_string()))
}

/// `1s` / `26s` / `1m 32s` / `3m` 형식 — TurnActivity 의 elapsed 표기와 일관.
fn format_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}m {s}s")
        }
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        format!("{h}h {m}m")
    }
}

fn canonical_tool_basename(name: &str) -> &str {
    let display = tool_provenance::display_name(name);
    display
        .rsplit(['.', '/', ':'])
        .find(|part| !part.is_empty())
        .unwrap_or(display)
}

fn is_plan_update_tool(name: &str) -> bool {
    let display = canonical_tool_basename(name);
    display.eq_ignore_ascii_case("TodoWrite") || display.eq_ignore_ascii_case("TaskList")
}

fn display_tool_name(name: &str) -> String {
    let name = sanitize_inline(canonical_tool_basename(name));
    match name.as_str() {
        "SpawnMultiAgent" => "agents".to_string(),
        "TaskList" => "tasks".to_string(),
        "Sleep" => "wait".to_string(),
        other => other.to_string(),
    }
}

/// Present-progressive action phrase shown for an in-flight tool call
/// whose concrete arguments have not arrived yet.
///
/// `(canonical_name, phrase)` — matched case-insensitively by
/// [`pending_action`]. Inspired by opencode's pending labels
/// ("Searching content…", "Writing command…") so a streaming row reads
/// as *what* is starting, not just *that* something is. Tools absent
/// here fall back to no phrase (the bare name + spinner already convey
/// activity).
const PENDING_ACTIONS: &[(&str, &str)] = &[
    ("bash", "Writing command\u{2026}"),
    ("read", "Reading file\u{2026}"),
    ("write", "Preparing write\u{2026}"),
    ("edit", "Preparing edit\u{2026}"),
    ("multiedit", "Preparing edits\u{2026}"),
    ("notebookedit", "Preparing edit\u{2026}"),
    ("glob", "Finding files\u{2026}"),
    ("grep", "Searching content\u{2026}"),
    ("webfetch", "Fetching from the web\u{2026}"),
    ("websearch", "Searching web\u{2026}"),
    ("todowrite", "Updating todos\u{2026}"),
    ("skill", "Loading skill\u{2026}"),
    ("spawnmultiagent", "Delegating\u{2026}"),
    ("task", "Delegating\u{2026}"),
    ("agent", "Delegating\u{2026}"),
    ("askuserquestion", "Asking questions\u{2026}"),
];

/// Look up the in-flight action phrase for `name`, case-insensitively.
///
/// Returns `None` for tools without an entry so the caller can fall
/// back to an empty summary.
fn pending_action(name: &str) -> Option<&'static str> {
    PENDING_ACTIONS
        .iter()
        .find(|(tool, _)| name.eq_ignore_ascii_case(tool))
        .map(|(_, phrase)| *phrase)
}

/// Human-readable activity line for live status surfaces.
///
/// This is intentionally provider-neutral: OpenAI and Anthropic adapters both
/// emit the same `RenderBlock::ToolCall`, so the TUI can explain "what is
/// happening now" without backend-specific branches.
pub(crate) fn activity_summary(name: &str, summary: &str) -> String {
    let origin = tool_provenance::classify(name);
    let display = canonical_tool_basename(name);
    if let Origin::Mcp(server) = origin {
        return mcp_activity_summary(server, display, summary);
    }

    let mut action = activity_verb(display);
    let raw_detail = display_tool_summary(name, summary);
    let mut detail = raw_detail.trim().trim_end_matches('\u{2026}');
    let bash_detail;
    if display.eq_ignore_ascii_case("bash") {
        if let Some(path) = detail.strip_prefix("read ").map(str::trim) {
            action = "Reading file";
            bash_detail = path;
            detail = bash_detail;
        } else if let Some(query) = detail.strip_prefix("search ").map(str::trim) {
            action = "Searching";
            bash_detail = query;
            detail = bash_detail;
        }
    }

    if !detail.is_empty()
        && !looks_like_raw_payload(detail)
        && !detail_repeats_action(action, detail)
    {
        return format!("{action}: {}", truncate_activity(detail, 96));
    }

    pending_action(display).map_or_else(
        || format!("Running {display}"),
        |phrase| phrase.trim_end_matches('\u{2026}').to_string(),
    )
}

fn activity_verb(name: &str) -> &'static str {
    if name.eq_ignore_ascii_case("bash") {
        "Running command"
    } else if is_file_tool(name, &["read_file", "Read", "read"]) {
        "Reading file"
    } else if is_file_tool(name, &["write_file", "Write", "write"]) {
        "Writing file"
    } else if is_file_tool(name, &["edit_file", "Edit", "edit", "MultiEdit"]) {
        "Editing file"
    } else if name.eq_ignore_ascii_case("Glob") || name.eq_ignore_ascii_case("glob_search") {
        "Finding files"
    } else if name.eq_ignore_ascii_case("Grep") || name.eq_ignore_ascii_case("grep_search") {
        "Searching"
    } else if name.eq_ignore_ascii_case("SpawnMultiAgent")
        || name.eq_ignore_ascii_case("Task")
        || name.eq_ignore_ascii_case("Agent")
    {
        "Delegating"
    } else {
        "Running"
    }
}

fn looks_like_raw_payload(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with('{') || trimmed.starts_with('[')
}

fn detail_repeats_action(action: &str, detail: &str) -> bool {
    normalize_activity_phrase(action) == normalize_activity_phrase(detail)
}

fn normalize_activity_phrase(text: &str) -> String {
    text.trim()
        .trim_end_matches('\u{2026}')
        .trim_end_matches('.')
        .trim()
        .to_ascii_lowercase()
}

use crate::tui::text_metrics::{char_width, display_width};

fn truncate_activity(text: &str, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    let total_width = display_width(text);
    if total_width <= limit {
        return text.to_string();
    }

    let body_limit = limit.saturating_sub(1);
    let mut used = 0;
    let mut truncated = String::new();
    for ch in text.chars() {
        let width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + width > body_limit {
            break;
        }
        used += width;
        truncated.push(ch);
    }
    truncated.push('\u{2026}');
    truncated
}

#[allow(clippy::too_many_lines)] // flat verb-classification table; one alias block per tool family
fn tool_event_summary(name: &str, summary: &str, preview: &ToolPreview) -> (&'static str, String) {
    let origin = tool_provenance::classify(name);
    let display = tool_provenance::display_name(name);
    if let Origin::Mcp(server) = origin {
        return mcp_event_summary(server, display, summary);
    }

    match preview {
        ToolPreview::Bash { command } => {
            let command = collapse_command(command);
            if is_exploratory_command(&command) {
                ("Explored", command_event_detail(&command))
            } else {
                // Keep the command identifiable but stop dumping the whole raw
                // line: a long `cd … INTENDED=( … )` is noise, not signal.
                ("Ran", smart_truncate_command(&command, BASH_LABEL_CAP))
            }
        }
        ToolPreview::Read { path, range } => {
            let range = range
                .map(|(start, end)| format!(":{start}-{end}"))
                .unwrap_or_default();
            (
                "Explored",
                format!("read {}{range}", compact_path_label(path)),
            )
        }
        ToolPreview::Write { path, byte_count } => (
            "Wrote",
            format!("{} · {byte_count} bytes", compact_path_label(path)),
        ),
        ToolPreview::Edit { path, hunk_count } => (
            "Edited",
            format!("{} · {hunk_count} hunks", compact_path_label(path)),
        ),
        ToolPreview::Glob { pattern } => ("Explored", format!("glob {pattern}")),
        ToolPreview::Grep { pattern, path } => {
            ("Explored", format_grep_detail(pattern, path.as_deref()))
        }
        ToolPreview::Search { query } => ("Searched", query.clone()),
        ToolPreview::Generic {
            name: preview_name,
            input_summary,
        } => {
            let display = canonical_tool_basename(display);
            if display.eq_ignore_ascii_case("TodoWrite") || display.eq_ignore_ascii_case("TaskList") {
                return ("Updated", "Plan".to_string());
            }
            if display.eq_ignore_ascii_case("bash") {
                let detail = display_tool_summary(name, summary);
                if !detail.is_empty() {
                    let verb = if detail.starts_with("read ") || detail.starts_with("search ") {
                        "Explored"
                    } else {
                        "Ran"
                    };
                    return (verb, smart_truncate_command(&detail, BASH_LABEL_CAP));
                }
            }
            if is_file_tool(display, &["read_file", "Read", "read"])
                || display.eq_ignore_ascii_case("Glob")
                || display.eq_ignore_ascii_case("glob_search")
                || display.eq_ignore_ascii_case("Grep")
                || display.eq_ignore_ascii_case("grep_search")
            {
                let detail = display_tool_summary(name, summary);
                if !detail.is_empty() {
                    return ("Explored", truncate_activity(&detail, 120));
                }
                if display.eq_ignore_ascii_case("Grep")
                    || display.eq_ignore_ascii_case("grep_search")
                    || preview_name.eq_ignore_ascii_case("Grep")
                    || preview_name.eq_ignore_ascii_case("grep_search")
                {
                    return ("Explored", "grep".to_string());
                }
            }
            if is_file_tool(display, &["write_file", "Write", "write"]) {
                let detail = display_tool_summary(name, summary);
                if !detail.is_empty() {
                    return ("Wrote", truncate_activity(&detail, 120));
                }
            }
            if is_file_tool(display, &["edit_file", "Edit", "edit", "MultiEdit"]) {
                let detail = display_tool_summary(name, summary);
                if !detail.is_empty() {
                    return ("Edited", truncate_activity(&detail, 120));
                }
            }
            if display.eq_ignore_ascii_case("SpawnMultiAgent")
                || display.eq_ignore_ascii_case("Task")
                || display.eq_ignore_ascii_case("Agent")
            {
                let detail = display_tool_summary(name, summary);
                let detail = if detail.is_empty() {
                    input_summary.clone()
                } else {
                    detail
                };
                return ("Spawned", truncate_activity(&detail, 120));
            }
            let detail = if input_summary.trim().is_empty() {
                format!(
                    "{}({})",
                    display_tool_name(preview_name),
                    call_args(summary)
                )
            } else {
                format!(
                    "{}({})",
                    display_tool_name(preview_name),
                    truncate_activity(input_summary, 120)
                )
            };
            ("Called", detail)
        }
    }
}

fn normalize_mcp_server(server: &str) -> String {
    server.replace('_', "-")
}

fn mcp_event_summary(server: &str, tool: &str, summary: &str) -> (&'static str, String) {
    let verb = mcp_event_verb(server, tool);
    let detail = mcp_summary_subject(summary).map_or_else(
        || mcp_tool_name(server, tool),
        |subject| {
            format!(
                "{} · {}",
                truncate_activity(&subject, 96),
                normalize_mcp_server(server)
            )
        },
    );
    (verb, detail)
}

fn mcp_activity_summary(server: &str, tool: &str, summary: &str) -> String {
    let action = mcp_activity_verb(server, tool);
    let detail = mcp_summary_subject(summary).unwrap_or_else(|| mcp_tool_name(server, tool));
    format!("{action}: {}", truncate_activity(&detail, 96))
}

fn mcp_event_verb(server: &str, tool: &str) -> &'static str {
    let tool = tool.to_ascii_lowercase();
    if server.eq_ignore_ascii_case("context7") {
        if tool.contains("resolve") {
            "Found docs"
        } else {
            "Checked docs"
        }
    } else if tool.contains("search") {
        "Searched"
    } else if tool.contains("read") || tool.contains("open") || tool.contains("fetch") {
        "Read source"
    } else if server.eq_ignore_ascii_case("browser")
        || server.eq_ignore_ascii_case("chrome")
        || server.eq_ignore_ascii_case("chrome-devtools")
    {
        "Checked browser"
    } else {
        "Consulted"
    }
}

fn mcp_activity_verb(server: &str, tool: &str) -> &'static str {
    let tool = tool.to_ascii_lowercase();
    if server.eq_ignore_ascii_case("context7") {
        if tool.contains("resolve") {
            "Finding docs"
        } else {
            "Checking docs"
        }
    } else if tool.contains("search") {
        "Searching"
    } else if tool.contains("read") || tool.contains("open") || tool.contains("fetch") {
        "Reading source"
    } else if server.eq_ignore_ascii_case("browser")
        || server.eq_ignore_ascii_case("chrome")
        || server.eq_ignore_ascii_case("chrome-devtools")
    {
        "Checking browser"
    } else {
        "Consulting source"
    }
}

fn mcp_tool_name(server: &str, tool: &str) -> String {
    format!(
        "{}.{}",
        normalize_mcp_server(server),
        display_tool_name(tool)
    )
}

fn mcp_summary_subject(summary: &str) -> Option<String> {
    extract_call_json(summary).and_then(|value| subject_from_json(&value))
}

fn subject_from_json(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Object(map) => {
            for key in [
                "query",
                "q",
                "question",
                "url",
                "href",
                "libraryName",
                "libraryId",
                "path",
                "pattern",
                "location",
                "ticker",
            ] {
                if let Some(subject) = map
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|subject| !subject.is_empty())
                {
                    return Some(sanitize_inline(subject));
                }
            }
            for key in ["queries", "search_query"] {
                if let Some(subject) = map.get(key).and_then(subject_from_json) {
                    return Some(subject);
                }
            }
            None
        }
        serde_json::Value::Array(items) => items.iter().find_map(subject_from_json),
        serde_json::Value::String(subject) => {
            let subject = subject.trim();
            (!subject.is_empty()).then(|| sanitize_inline(subject))
        }
        _ => None,
    }
}

fn format_grep_detail(pattern: &str, path: Option<&str>) -> String {
    let pattern = compact_grep_pattern(&clean_grep_pattern(pattern));
    let scope = path
        .filter(|path| !path.trim().is_empty())
        .map_or(String::new(), |path| {
            format!(" in {}", truncate_activity(&compact_path_label(path), 40))
        });
    format!("grep {}{}", truncate_activity(&pattern, 48), scope)
}

fn clean_grep_pattern(pattern: &str) -> String {
    let mut pattern = sanitize_inline(pattern).trim().to_string();
    for marker in [
        " io error:",
        " regex parse error:",
        " error ",
        " Wait tool JSON",
    ] {
        if let Some(idx) = pattern.find(marker) {
            pattern.truncate(idx);
        }
    }
    pattern.trim_matches('/').trim().to_string()
}

fn compact_grep_pattern(pattern: &str) -> String {
    let parts = pattern
        .split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() <= 1 {
        return pattern.to_string();
    }
    format!("{} +{} more", parts[0], parts.len() - 1)
}

fn call_args(summary: &str) -> String {
    extract_call_json(summary)
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| {
            let raw = sanitize_inline(summary);
            if raw.trim().is_empty() {
                "{}".to_string()
            } else {
                truncate_activity(raw.trim(), 120)
            }
        })
}

fn is_exploratory_command(command: &str) -> bool {
    let trimmed = command.trim_start();
    bash_read_target(trimmed).is_some()
        || trimmed.starts_with("ls")
        || trimmed.starts_with("find ")
        || trimmed.starts_with("rg --files")
        || trimmed.starts_with("fd ")
}

fn command_event_detail(command: &str) -> String {
    if let Some(path) = bash_read_target(command) {
        return format!("read {}", compact_path_label(&path));
    }
    smart_truncate_command(command, 120)
}

fn describe_bash_command(command: &str) -> String {
    let command = collapse_command(command);
    if let Some(path) = bash_read_target(&command) {
        return format!("read {}", compact_path_label(&path));
    }
    if let Some(query) = bash_search_target(&command) {
        return format!("search {query}");
    }
    smart_truncate_command(&command, 96)
}

fn collapse_command(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn bash_read_target(command: &str) -> Option<String> {
    let before_pipe = command.split('|').next().unwrap_or(command).trim();
    let candidates = [
        path_after_prefix(before_pipe, "nl -ba "),
        path_after_prefix(before_pipe, "cat "),
        path_after_last_arg(before_pipe, "sed -n "),
        path_after_last_arg(before_pipe, "head "),
        path_after_last_arg(before_pipe, "tail "),
    ];
    candidates
        .into_iter()
        .flatten()
        .map(|path| clean_shell_path(&path))
        .find(|path| looks_like_source_path(path))
}

fn bash_search_target(command: &str) -> Option<String> {
    let trimmed = command.trim_start();
    if !trimmed.starts_with("rg ") && trimmed != "rg" {
        return None;
    }
    let mut args = trimmed.split_whitespace().skip(1);
    while let Some(arg) = args.next() {
        if arg.starts_with('-') {
            if matches!(arg, "-e" | "--regexp" | "-g" | "--glob") {
                let _ = args.next();
            }
            continue;
        }
        return Some(clean_shell_path(arg));
    }
    Some("content".to_string())
}

fn path_after_prefix(command: &str, prefix: &str) -> Option<String> {
    command
        .strip_prefix(prefix)
        .and_then(|rest| rest.split_whitespace().next())
        .map(str::to_string)
}

fn path_after_last_arg(command: &str, prefix: &str) -> Option<String> {
    command.strip_prefix(prefix)?;
    command.split_whitespace().last().map(str::to_string)
}

fn clean_shell_path(path: &str) -> String {
    path.trim()
        .trim_matches(|ch| matches!(ch, '\'' | '"' | '`'))
        .to_string()
}

const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "go", "ts", "tsx", "js", "jsx", "py", "java", "md", "json", "toml", "yml", "yaml",
];

fn looks_like_source_path(path: &str) -> bool {
    // Case-insensitive extension match (`README.MD`, `Main.JAVA`) — a
    // case-sensitive `ends_with` silently misclassified upper-cased files.
    let has_source_extension = || {
        std::path::Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                SOURCE_EXTENSIONS
                    .iter()
                    .any(|known| ext.eq_ignore_ascii_case(known))
            })
    };
    !path.is_empty() && path != "-" && (path.contains('/') || has_source_extension())
}

pub(crate) fn display_tool_summary(name: &str, summary: &str) -> String {
    let raw = sanitize_inline(summary);
    if raw.trim().is_empty() || raw.trim() == "{}" {
        // Concrete args haven't arrived; let the in-flight branch in
        // `rendered_lines` decide whether to show an action phrase. A
        // settled call with no args simply has no summary.
        return String::new();
    }

    let parsed = extract_call_json(&raw);

    if name.eq_ignore_ascii_case("bash") {
        if let Some(value) = parsed.as_ref() {
            if let Some(command) = value.get("command").and_then(serde_json::Value::as_str) {
                return describe_bash_command(command);
            }
        }
    }

    if name == "Sleep" {
        if let Some(value) = parsed.as_ref() {
            if let Some(ms) = value.get("duration_ms").and_then(serde_json::Value::as_u64) {
                return format!("{}s", ms / 1000);
            }
        }
    }

    if name == "SpawnMultiAgent" {
        if let Some(value) = parsed.as_ref() {
            if let Some(agents) = value.get("agents").and_then(serde_json::Value::as_array) {
                return format!("{} agents", agents.len());
            }
        }
        return "delegating".to_string();
    }

    if is_file_tool(name, &["read_file", "Read", "read"]) {
        if let Some(value) = parsed.as_ref() {
            return extract_json_path(value);
        }
    }

    if is_file_tool(name, &["write_file", "Write", "write"]) {
        if let Some(value) = parsed.as_ref() {
            let path = extract_json_path(value);
            let byte_count = value
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map_or(0, str::len);
            return format!("{path} · {byte_count} bytes");
        }
    }

    if is_file_tool(name, &["edit_file", "Edit", "edit"]) {
        if let Some(value) = parsed.as_ref() {
            let path = extract_json_path(value);
            let mode = if value
                .get("replace_all")
                .or_else(|| value.get("replaceAll"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                "replace all"
            } else {
                "edit"
            };
            return format!("{path} · {mode}");
        }
    }

    if name.eq_ignore_ascii_case("grep_search") || name.eq_ignore_ascii_case("Grep") {
        if let Some(value) = parsed.as_ref() {
            if let Some(pattern) = value.get("pattern").and_then(serde_json::Value::as_str) {
                let path = value
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string);
                return format_grep_detail(pattern, path.as_deref());
            }
        }
        if let Some(detail) = summarize_raw_grep(&raw, name) {
            return detail;
        }
        return String::new();
    }

    if let Some(inner) = raw
        .strip_prefix(&format!("{name}("))
        .and_then(|rest| rest.strip_suffix(')'))
    {
        sanitize_inline(inner)
    } else if looks_like_raw_payload(&raw) || raw.contains("\"command\"") {
        String::new()
    } else {
        raw
    }
}

fn summarize_raw_grep(raw: &str, name: &str) -> Option<String> {
    let mut body = raw.trim();
    if let Some(rest) = body.strip_prefix(&format!("{name}(")) {
        body = rest.trim_end_matches(')').trim();
    }
    if body.is_empty() || looks_like_raw_payload(body) {
        return None;
    }

    let mut pattern = body;
    let mut path = None;
    if let Some((left, right)) = body.rsplit_once(" in ") {
        pattern = left;
        path = Some(right.trim().trim_end_matches(')').to_string());
    }
    let pattern = clean_grep_pattern(pattern);
    if pattern.is_empty() {
        return None;
    }
    Some(format_grep_detail(&pattern, path.as_deref()))
}

fn is_file_tool(name: &str, aliases: &[&str]) -> bool {
    aliases.iter().any(|alias| name.eq_ignore_ascii_case(alias))
}

fn extract_json_path(value: &serde_json::Value) -> String {
    let path = value
        .get("file_path")
        .or_else(|| value.get("filePath"))
        .or_else(|| value.get("path"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    compact_path_label(path)
}

fn extract_call_json(summary: &str) -> Option<serde_json::Value> {
    let body = summary
        .split_once('(')
        .and_then(|(_, rest)| rest.strip_suffix(')'))
        .unwrap_or(summary)
        .trim();
    parse_json_payload(body).or_else(|| {
        body.find('{')
            .and_then(|start| parse_json_payload(&body[start..]))
    })
}

fn parse_json_payload(body: &str) -> Option<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(inner) = value.as_str() {
        serde_json::from_str(inner).ok()
    } else {
        Some(value)
    }
}

fn middle_truncate_word(word: &str, edge_chars: usize) -> String {
    let char_count = word.chars().count();
    if char_count <= edge_chars * 2 {
        return word.to_string();
    }

    let start: String = word.chars().take(edge_chars).collect();
    let end: String = word
        .chars()
        .rev()
        .take(edge_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{start}...{end}")
}

/// Smart truncation for bash commands that preserves important flags (-p, --test, etc.)
/// and file paths, while middle-truncating excessively long arguments.
fn smart_truncate_command(command: &str, limit: usize) -> String {
    let collapsed = command.split_whitespace().collect::<Vec<_>>().join(" ");
    if display_width(&collapsed) <= limit {
        return collapsed;
    }

    let words: Vec<&str> = collapsed.split_whitespace().collect();
    let mut shortened_words = Vec::new();

    for word in words {
        if word.starts_with('-') {
            // Keep flags intact
            shortened_words.push(word.to_string());
        } else if word.contains('/') || word.contains('\\') {
            // Shorten paths
            let path = std::path::Path::new(word);
            if let Some(file_name) = path.file_name().and_then(|f| f.to_str()) {
                if let Some(parent) = path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|p| p.to_str())
                {
                    shortened_words.push(format!("{parent}/.../{file_name}"));
                } else {
                    shortened_words.push(file_name.to_string());
                }
            } else {
                shortened_words.push(word.to_string());
            }
        } else if display_width(word) > 24 && !word.starts_with("http") {
            // Middle-truncate long arguments
            shortened_words.push(middle_truncate_word(word, 10));
        } else {
            shortened_words.push(word.to_string());
        }
    }

    let combined = shortened_words.join(" ");
    if display_width(&combined) <= limit {
        combined
    } else {
        truncate_activity(&combined, limit)
    }
}

#[cfg(test)]
mod tests;
