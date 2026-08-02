//! Minimal in-tree Mermaid `graph`/`flowchart` renderer for the TUI.
//!
//! Mermaid diagrams arrive as fenced ```` ```mermaid ```` code blocks. Rather
//! than show the raw source, [`render`] parses a useful subset of the grammar
//! and lays it out as a box-and-arrow diagram drawn with Unicode line glyphs,
//! returning rows in the same `Vec<Vec<Span>>` shape the markdown
//! [`crate::tui::markdown::Renderer`] accumulates.
//!
//! Supported subset (Phase 1):
//! * `graph`/`flowchart` with direction `TD`/`TB` (vertical) or `LR`/`RL`
//!   (horizontal).
//! * Node shapes `id[label]`, `id(label)`, `id{label}`, `id([label])`; bare
//!   `id` is its own label. `<br/>` inside a label becomes a line break.
//! * Edges `-->`, `---`, `-.->`, `==>`, each optionally `|label|`; both
//!   endpoints may carry an inline definition (`A[Foo] --> B[Bar]`).
//! * `subgraph Title … end` groups — drawn as a labelled border around the
//!   bounding box of its members when that box is clean.
//! * `style`/`classDef`/`class`/`click`/`linkStyle`/`%%` lines are ignored.
//!
//! Anything outside the subset (or a graph too wide for the viewport) makes
//! [`render`] return `None` so the caller falls back to the raw code block —
//! content is never lost.

use ratatui::style::{Modifier, Style};
use ratatui::text::Span;

use super::theme::Theme;

/// Hard caps that keep a pathological diagram from blowing up the layout.
const MAX_NODES: usize = 64;
const MAX_LABEL_W: usize = 24;
const RANK_GAP: usize = 2;
const NODE_GAP: usize = 3;

/// Render `source` (the mermaid fence body) into diagram rows, or `None` when
/// the diagram is unsupported / empty / too wide to fit `width`.
#[must_use]
pub fn render(source: &str, width: u16, theme: &Theme) -> Option<Vec<Vec<Span<'static>>>> {
    let graph = parse(source)?;
    if graph.nodes.is_empty() {
        return None;
    }
    let grid = layout(&graph, width)?;
    Some(grid.into_spans(theme))
}

// ============================================================================
// Parsing
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Vertical,
    Horizontal,
}

struct Node {
    id: String,
    label: Vec<String>,
}

struct Subgraph {
    title: String,
    members: Vec<usize>,
}

struct Graph {
    direction: Direction,
    nodes: Vec<Node>,
    edges: Vec<(usize, usize)>,
    subgraphs: Vec<Subgraph>,
}

impl Graph {
    /// Find or insert a node by id, returning its index. `label`, when present,
    /// updates a placeholder/bare label.
    fn intern(&mut self, id: &str, label: Option<Vec<String>>) -> usize {
        if let Some(idx) = self.nodes.iter().position(|n| n.id == id) {
            if let Some(label) = label {
                if !label.is_empty() {
                    self.nodes[idx].label = label;
                }
            }
            return idx;
        }
        let label = label.unwrap_or_else(|| vec![id.to_string()]);
        self.nodes.push(Node {
            id: id.to_string(),
            label,
        });
        self.nodes.len() - 1
    }
}

fn parse(source: &str) -> Option<Graph> {
    let mut graph = Graph {
        direction: Direction::Vertical,
        nodes: Vec::new(),
        edges: Vec::new(),
        subgraphs: Vec::new(),
    };
    let mut saw_header = false;
    let mut subgraph_stack: Vec<Subgraph> = Vec::new();

    for raw in source.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") {
            continue;
        }
        let lower = line.to_ascii_lowercase();

        if !saw_header && (lower.starts_with("graph") || lower.starts_with("flowchart")) {
            graph.direction = parse_direction(&lower);
            saw_header = true;
            continue;
        }
        // Tolerate a missing header (treat first statement line as content).
        saw_header = true;

        if lower.starts_with("subgraph") {
            let title = line["subgraph".len()..].trim();
            let title = parse_subgraph_title(title);
            subgraph_stack.push(Subgraph {
                title,
                members: Vec::new(),
            });
            continue;
        }
        if lower == "end" {
            if let Some(sg) = subgraph_stack.pop() {
                graph.subgraphs.push(sg);
            }
            continue;
        }
        if is_ignored_statement(&lower) {
            continue;
        }

        let touched = parse_statement(line, &mut graph)?;
        for sg in &mut subgraph_stack {
            for &idx in &touched {
                if !sg.members.contains(&idx) {
                    sg.members.push(idx);
                }
            }
        }
        if graph.nodes.len() > MAX_NODES {
            return None;
        }
    }
    while let Some(sg) = subgraph_stack.pop() {
        graph.subgraphs.push(sg);
    }
    Some(graph)
}

fn parse_direction(header_lower: &str) -> Direction {
    if header_lower.contains("lr") || header_lower.contains("rl") {
        Direction::Horizontal
    } else {
        Direction::Vertical
    }
}

fn parse_subgraph_title(raw: &str) -> String {
    parse_node_token(raw)
        .and_then(|(_, label)| label)
        .and_then(|label| label.into_iter().next())
        .filter(|label| !label.trim().is_empty())
        .unwrap_or_else(|| strip_brackets(raw).unwrap_or_else(|| raw.trim().to_string()))
}

fn is_ignored_statement(lower: &str) -> bool {
    const PREFIXES: [&str; 6] = [
        "style",
        "classdef",
        "class ",
        "click",
        "linkstyle",
        "direction",
    ];
    PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Parse one statement (an edge chain or a lone node), interning nodes into the
/// graph. Returns the node indices the statement referenced (for subgraph
/// membership). Returns `None` only on hard malformation.
fn parse_statement(line: &str, graph: &mut Graph) -> Option<Vec<usize>> {
    // Split the chain on edge operators, keeping endpoint tokens. Edge labels
    // (`-->|text|` or `-- text -->`) are stripped — Phase 1 omits edge labels.
    let segments = split_on_edges(line);
    if segments.is_empty() {
        return Some(Vec::new());
    }
    let mut touched = Vec::new();
    let mut prev: Option<usize> = None;
    for seg in segments {
        let token = seg.trim();
        if token.is_empty() {
            continue;
        }
        let (id, label) = parse_node_token(token)?;
        let idx = graph.intern(&id, label);
        if !touched.contains(&idx) {
            touched.push(idx);
        }
        if let Some(from) = prev {
            if from != idx && !graph.edges.contains(&(from, idx)) {
                graph.edges.push((from, idx));
            }
        }
        prev = Some(idx);
    }
    Some(touched)
}

/// Split a statement into endpoint tokens on any edge operator, discarding the
/// operators and any inline edge labels.
fn split_on_edges(line: &str) -> Vec<String> {
    // Normalise the operator variants to a single sentinel, then split.
    let mut s = line.to_string();
    for op in ["-.->", "-.-", "==>", "===", "-->", "---", "==", "--"] {
        s = s.replace(op, "\u{1}");
    }
    s.split('\u{1}')
        .map(|seg| strip_edge_label(seg.trim()).to_string())
        .filter(|seg| !seg.is_empty())
        .collect()
}

/// Drop a leading/trailing `|edge label|` left attached to an endpoint token
/// after splitting on the operator.
fn strip_edge_label(seg: &str) -> &str {
    let seg = seg.trim();
    if let Some(rest) = seg.strip_prefix('|') {
        if let Some(end) = rest.find('|') {
            return rest[end + 1..].trim();
        }
    }
    if let Some(open) = seg.find('|') {
        return seg[..open].trim();
    }
    seg
}

/// Parse `id[label]` / `id(label)` / `id{label}` / `id([label])` / bare `id`
/// into `(id, optional multi-line label)`.
fn parse_node_token(token: &str) -> Option<(String, Option<Vec<String>>)> {
    let open = token.find(['[', '(', '{']);
    let Some(open) = open else {
        // Bare id — reject obviously non-identifier junk.
        let id = token.trim();
        if id.is_empty() {
            return None;
        }
        return Some((id.to_string(), None));
    };
    let id = token[..open].trim().to_string();
    if id.is_empty() {
        return None;
    }
    let label = strip_brackets(token[open..].trim()).map(|raw| split_label(&raw));
    Some((id, label))
}

/// Strip one layer of `[]`/`()`/`{}`/`([])` brackets and surrounding quotes.
fn strip_brackets(s: &str) -> Option<String> {
    let s = s.trim();
    let inner = s
        .strip_prefix("([")
        .and_then(|r| r.strip_suffix("])"))
        .or_else(|| s.strip_prefix('[').and_then(|r| r.strip_suffix(']')))
        .or_else(|| s.strip_prefix('(').and_then(|r| r.strip_suffix(')')))
        .or_else(|| s.strip_prefix('{').and_then(|r| r.strip_suffix('}')))?;
    let inner = inner.trim();
    let inner = inner
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or(inner);
    Some(inner.to_string())
}

/// Split a label on `<br/>` (and `<br>`) into display lines, trimmed.
fn split_label(raw: &str) -> Vec<String> {
    let normalised = raw.replace("<br/>", "\n").replace("<br>", "\n");
    let lines: Vec<String> = normalised
        .split('\n')
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        vec![String::new()]
    } else {
        lines
    }
}

// ============================================================================
// Layout
// ============================================================================

/// Cell style classes, mapped to theme colors when converting to spans.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ink {
    Empty,
    Line,
    Label,
    Title,
}

struct Grid {
    w: usize,
    h: usize,
    ch: Vec<char>,
    ink: Vec<Ink>,
}

impl Grid {
    fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            ch: vec![' '; w * h],
            ink: vec![Ink::Empty; w * h],
        }
    }

    fn put(&mut self, x: usize, y: usize, c: char, ink: Ink) {
        if x < self.w && y < self.h {
            self.ch[y * self.w + x] = c;
            self.ink[y * self.w + x] = ink;
        }
    }

    fn get(&self, x: usize, y: usize) -> char {
        if x < self.w && y < self.h {
            self.ch[y * self.w + x]
        } else {
            ' '
        }
    }

    /// Draw a connector cell, merging with any existing connector glyph by
    /// unioning their N/S/E/W directions so corners become T-junctions and
    /// crossings as routes overlap. Node-box glyphs (drawn later) are not
    /// touched here, so box interiors/borders always win.
    fn line(&mut self, x: usize, y: usize, c: char) {
        let existing = self.get(x, y);
        let merged = match mask_of(existing) {
            Some(prev) => glyph_of(prev | mask_of(c).unwrap_or(0)),
            None => c,
        };
        self.put(x, y, merged, Ink::Line);
    }

    fn into_spans(self, theme: &Theme) -> Vec<Vec<Span<'static>>> {
        let line_style = Style::new().fg(theme.palette.muted);
        let label_style = Style::new().fg(theme.palette.fg);
        let title_style = Style::new()
            .fg(theme.palette.bright)
            .add_modifier(Modifier::BOLD);
        let mut rows = Vec::with_capacity(self.h);
        for y in 0..self.h {
            let mut spans: Vec<Span<'static>> = Vec::new();
            let mut buf = String::new();
            let mut cur = Ink::Empty;
            for x in 0..self.w {
                let c = self.ch[y * self.w + x];
                let ink = self.ink[y * self.w + x];
                if !buf.is_empty() && ink != cur {
                    spans.push(styled(&buf, cur, line_style, label_style, title_style));
                    buf.clear();
                }
                cur = ink;
                buf.push(c);
            }
            // Trim trailing spaces to avoid painting empty cells.
            let trimmed = buf.trim_end();
            if !trimmed.is_empty() {
                spans.push(styled(trimmed, cur, line_style, label_style, title_style));
            }
            if spans.is_empty() {
                spans.push(Span::raw(String::new()));
            }
            rows.push(spans);
        }
        rows
    }
}

fn styled(text: &str, ink: Ink, line: Style, label: Style, title: Style) -> Span<'static> {
    let style = match ink {
        Ink::Empty => Style::default(),
        Ink::Line => line,
        Ink::Label => label,
        Ink::Title => title,
    };
    Span::styled(text.to_string(), style)
}

// Box-drawing direction bits for merging overlapping connector glyphs.
const DIR_N: u8 = 1;
const DIR_S: u8 = 2;
const DIR_E: u8 = 4;
const DIR_W: u8 = 8;

/// Direction bitmask of a connector glyph, or `None` if `c` is not one (space,
/// arrow, label, or a node-box corner) and so should not be merged.
fn mask_of(c: char) -> Option<u8> {
    Some(match c {
        '│' | '┆' => DIR_N | DIR_S,
        '─' | '┄' => DIR_E | DIR_W,
        '╭' | '┌' => DIR_S | DIR_E,
        '╮' | '┐' => DIR_S | DIR_W,
        '╰' | '└' => DIR_N | DIR_E,
        '╯' | '┘' => DIR_N | DIR_W,
        '├' => DIR_N | DIR_S | DIR_E,
        '┤' => DIR_N | DIR_S | DIR_W,
        '┬' => DIR_S | DIR_E | DIR_W,
        '┴' => DIR_N | DIR_E | DIR_W,
        '┼' => DIR_N | DIR_S | DIR_E | DIR_W,
        _ => return None,
    })
}

/// Rounded-style connector glyph for a direction bitmask.
fn glyph_of(mask: u8) -> char {
    match mask {
        m if m == DIR_S | DIR_E => '╭',
        m if m == DIR_S | DIR_W => '╮',
        m if m == DIR_N | DIR_E => '╰',
        m if m == DIR_N | DIR_W => '╯',
        m if m == DIR_N | DIR_S | DIR_E => '├',
        m if m == DIR_N | DIR_S | DIR_W => '┤',
        m if m == DIR_S | DIR_E | DIR_W => '┬',
        m if m == DIR_N | DIR_E | DIR_W => '┴',
        m if m == DIR_N | DIR_S | DIR_E | DIR_W => '┼',
        m if m & (DIR_E | DIR_W) != 0 && m & (DIR_N | DIR_S) == 0 => '─',
        _ => '│',
    }
}

/// A placed node: its rect in grid coordinates plus center of each axis.
struct Placed {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Placed {
    fn cx(&self) -> usize {
        self.x + self.w / 2
    }
    fn cy(&self) -> usize {
        self.y + self.h / 2
    }
}

fn layout(graph: &Graph, width: u16) -> Option<Grid> {
    let avail = usize::from(width).max(20);
    let ranks = assign_ranks(graph);
    let order = group_by_rank(&ranks, graph.nodes.len());

    // Box dimensions per node.
    let boxes: Vec<(usize, usize)> = graph.nodes.iter().map(|n| box_size(&n.label)).collect();

    match graph.direction {
        Direction::Vertical => layout_vertical(graph, &order, &boxes, avail),
        Direction::Horizontal => layout_horizontal(graph, &order, &boxes, avail),
    }
}

/// Longest-path rank assignment over a DAG; back-edges (cycles) are ignored for
/// ranking so the function always terminates.
fn assign_ranks(graph: &Graph) -> Vec<usize> {
    let n = graph.nodes.len();
    let mut rank = vec![0usize; n];
    // Relax up to n times (Bellman-Ford-style longest path on a DAG-ish graph).
    for _ in 0..n {
        let mut changed = false;
        for &(a, b) in &graph.edges {
            if rank[b] < rank[a] + 1 {
                rank[b] = rank[a] + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    rank
}

fn group_by_rank(ranks: &[usize], n: usize) -> Vec<Vec<usize>> {
    let max_rank = ranks.iter().copied().max().unwrap_or(0);
    let mut out = vec![Vec::new(); max_rank + 1];
    for idx in 0..n {
        out[ranks[idx]].push(idx);
    }
    out
}

/// Box outer size `(w, h)` for a multi-line label: `│ text │` + borders.
fn box_size(label: &[String]) -> (usize, usize) {
    let text_w = label
        .iter()
        .map(|l| display_w(l).min(MAX_LABEL_W))
        .max()
        .unwrap_or(1)
        .max(1);
    (text_w + 4, label.len().max(1) + 2)
}

fn layout_vertical(
    graph: &Graph,
    order: &[Vec<usize>],
    boxes: &[(usize, usize)],
    avail: usize,
) -> Option<Grid> {
    let n = graph.nodes.len();
    let mut placed: Vec<Placed> = (0..n)
        .map(|_| Placed {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        })
        .collect();

    let max_row_w = avail.saturating_sub(2).max(1);
    let visual_rows = wrap_vertical_rows(order, boxes, max_row_w);

    // Row width = widest visual row.
    let mut total_w = 0usize;
    let mut y = 0usize;
    for rank_nodes in &visual_rows {
        let row_w: usize = rank_nodes.iter().map(|&i| boxes[i].0).sum::<usize>()
            + NODE_GAP * rank_nodes.len().saturating_sub(1);
        total_w = total_w.max(row_w);
        let row_h = rank_nodes.iter().map(|&i| boxes[i].1).max().unwrap_or(1);
        // Center the rank within the eventual total width later; place x now
        // relative to row start, offset applied in a second pass.
        let mut x = 0usize;
        for &i in rank_nodes {
            placed[i] = Placed {
                x,
                y,
                w: boxes[i].0,
                h: boxes[i].1,
            };
            x += boxes[i].0 + NODE_GAP;
        }
        y += row_h + RANK_GAP;
    }
    let needs_subgraph_padding = !graph.subgraphs.is_empty();
    let padded_w = total_w + usize::from(needs_subgraph_padding) * 2;
    if padded_w + 2 > avail {
        return None; // too wide — fall back to raw source
    }
    // Second pass: center each rank row horizontally within total_w.
    for rank_nodes in &visual_rows {
        let Some(&last) = rank_nodes.last() else {
            continue;
        };
        let row_w = placed[last].x + placed[last].w;
        let offset = (total_w - row_w) / 2;
        for &i in rank_nodes {
            placed[i].x += offset;
        }
    }

    let mut grid_h = y.saturating_sub(RANK_GAP);
    if needs_subgraph_padding {
        for p in &mut placed {
            p.x += 1;
            p.y += 1;
        }
        total_w = padded_w;
        grid_h += 2;
    }
    let mut grid = Grid::new(total_w, grid_h);
    route_edges_vertical(&mut grid, graph, &placed);
    draw_subgraphs(&mut grid, graph, &placed);
    for (i, p) in placed.iter().enumerate() {
        draw_box(&mut grid, p, &graph.nodes[i].label);
    }
    Some(grid)
}

fn wrap_vertical_rows(
    order: &[Vec<usize>],
    boxes: &[(usize, usize)],
    max_row_w: usize,
) -> Vec<Vec<usize>> {
    let mut rows = Vec::new();
    for rank_nodes in order {
        let mut current = Vec::new();
        let mut current_w = 0usize;
        for &idx in rank_nodes {
            let node_w = boxes[idx].0;
            let next_w = if current.is_empty() {
                node_w
            } else {
                current_w + NODE_GAP + node_w
            };
            if !current.is_empty() && next_w > max_row_w {
                rows.push(std::mem::take(&mut current));
                current_w = 0;
            }
            if current.is_empty() {
                current_w = node_w;
            } else {
                current_w += NODE_GAP + node_w;
            }
            current.push(idx);
        }
        if current.is_empty() {
            rows.push(Vec::new());
        } else {
            rows.push(current);
        }
    }
    rows
}

fn layout_horizontal(
    graph: &Graph,
    order: &[Vec<usize>],
    boxes: &[(usize, usize)],
    avail: usize,
) -> Option<Grid> {
    let n = graph.nodes.len();
    let mut placed: Vec<Placed> = (0..n)
        .map(|_| Placed {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        })
        .collect();

    // Columns are ranks; nodes stack vertically within a column.
    let mut x = 0usize;
    let mut total_h = 0usize;
    let mut col_widths = Vec::new();
    for rank_nodes in order {
        let col_w = rank_nodes.iter().map(|&i| boxes[i].0).max().unwrap_or(1);
        col_widths.push(col_w);
        let mut yy = 0usize;
        for &i in rank_nodes {
            placed[i] = Placed {
                x,
                y: yy,
                w: boxes[i].0,
                h: boxes[i].1,
            };
            yy += boxes[i].1 + 1;
        }
        total_h = total_h.max(yy.saturating_sub(1));
        x += col_w + NODE_GAP + RANK_GAP;
    }
    let needs_subgraph_padding = !graph.subgraphs.is_empty();
    let mut total_w = x.saturating_sub(NODE_GAP + RANK_GAP);
    let padded_w = total_w + usize::from(needs_subgraph_padding) * 2;
    if padded_w + 2 > avail {
        return None;
    }
    // Center each column vertically.
    for rank_nodes in order {
        let Some(&last) = rank_nodes.last() else {
            continue;
        };
        let col_h = placed[last].y + placed[last].h;
        let offset = (total_h - col_h) / 2;
        for &i in rank_nodes {
            placed[i].y += offset;
        }
    }

    let mut grid_h = total_h.max(1);
    if needs_subgraph_padding {
        for p in &mut placed {
            p.x += 1;
            p.y += 1;
        }
        total_w = padded_w;
        grid_h += 2;
    }

    let mut grid = Grid::new(total_w, grid_h);
    route_edges_horizontal(&mut grid, graph, &placed);
    draw_subgraphs(&mut grid, graph, &placed);
    for (i, p) in placed.iter().enumerate() {
        draw_box(&mut grid, p, &graph.nodes[i].label);
    }
    Some(grid)
}

/// Route every edge top→down: a vertical drop from the parent's bottom-center,
/// a horizontal jog at the gap row, then a vertical rise with a `▼` arrow into
/// the child's top-center. Boxes are drawn last, so any line that grazes an
/// intermediate box is overwritten cleanly.
fn route_edges_vertical(grid: &mut Grid, graph: &Graph, placed: &[Placed]) {
    for &(a, b) in &graph.edges {
        let (pa, pb) = (&placed[a], &placed[b]);
        let (top, bot) = if pa.cy() <= pb.cy() {
            (pa, pb)
        } else {
            (pb, pa)
        };
        let sx = top.cx();
        let ex = bot.cx();
        let y_start = top.y + top.h; // first gap row below the parent box
        let y_end = bot.y; // top border row of the child box
        if y_start > y_end || y_end == 0 {
            continue;
        }
        if sx == ex {
            for y in y_start..y_end {
                grid.line(sx, y, '│');
            }
        } else {
            // Corner out of the parent column, jog horizontally, corner into
            // the child column, then drop to the child.
            grid.line(sx, y_start, corner_from_parent(sx, ex));
            let (lo, hi) = (sx.min(ex), sx.max(ex));
            for x in (lo + 1)..hi {
                grid.line(x, y_start, '─');
            }
            grid.line(ex, y_start, corner_to_child(sx, ex));
            for y in (y_start + 1)..y_end {
                grid.line(ex, y, '│');
            }
        }
        grid.put(ex, y_end.saturating_sub(1), '▼', Ink::Line);
    }
}

fn route_edges_horizontal(grid: &mut Grid, graph: &Graph, placed: &[Placed]) {
    for &(a, b) in &graph.edges {
        let (pa, pb) = (&placed[a], &placed[b]);
        let (left, right) = if pa.cx() <= pb.cx() {
            (pa, pb)
        } else {
            (pb, pa)
        };
        let start_x = left.x + left.w;
        let start_y = left.cy();
        let end_x = right.x;
        let end_y = right.cy();
        if end_x == 0 || start_x > end_x {
            continue;
        }
        let jog = start_x;
        let (lo, hi) = (start_y.min(end_y), start_y.max(end_y));
        for y in lo..=hi {
            grid.line(jog, y, '│');
        }
        for x in (jog + 1)..end_x {
            grid.line(x, end_y, '─');
        }
        if end_x > 0 {
            grid.put(end_x.saturating_sub(1), end_y, '▶', Ink::Line);
        }
    }
}

fn corner_from_parent(start_x: usize, end_x: usize) -> char {
    if end_x > start_x { '╰' } else { '╯' }
}

fn corner_to_child(start_x: usize, end_x: usize) -> char {
    if end_x > start_x { '╮' } else { '╭' }
}

/// Draw a rounded box with a centered multi-line label.
fn draw_box(grid: &mut Grid, p: &Placed, label: &[String]) {
    if p.w < 2 || p.h < 2 {
        return;
    }
    let right = p.x + p.w - 1;
    let bottom = p.y + p.h - 1;
    grid.put(p.x, p.y, '╭', Ink::Line);
    grid.put(right, p.y, '╮', Ink::Line);
    grid.put(p.x, bottom, '╰', Ink::Line);
    grid.put(right, bottom, '╯', Ink::Line);
    for x in (p.x + 1)..right {
        grid.put(x, p.y, '─', Ink::Line);
        grid.put(x, bottom, '─', Ink::Line);
    }
    for y in (p.y + 1)..bottom {
        grid.put(p.x, y, '│', Ink::Line);
        grid.put(right, y, '│', Ink::Line);
        for x in (p.x + 1)..right {
            grid.put(x, y, ' ', Ink::Label);
        }
    }
    let inner_w = p.w.saturating_sub(4);
    for (li, text) in label.iter().enumerate() {
        let y = p.y + 1 + li;
        if y >= bottom {
            break;
        }
        let truncated = truncate_w(text, inner_w);
        let tw = display_w(&truncated);
        let pad = inner_w.saturating_sub(tw) / 2;
        let mut x = p.x + 2 + pad;
        for c in truncated.chars() {
            grid.put(x, y, c, Ink::Label);
            x += char_w(c);
        }
    }
}

/// Draw a labelled light border around the bounding box of a subgraph's
/// members, when that box does not collide with non-member boxes.
fn draw_subgraphs(grid: &mut Grid, graph: &Graph, placed: &[Placed]) {
    for sg in &graph.subgraphs {
        if sg.members.is_empty() {
            continue;
        }
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        for &m in &sg.members {
            let p = &placed[m];
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x + p.w);
            max_y = max_y.max(p.y + p.h);
        }
        let (x0, y0) = (min_x.saturating_sub(1), min_y.saturating_sub(1));
        let (x1, y1) = (
            (max_x).min(grid.w.saturating_sub(1)),
            (max_y).min(grid.h.saturating_sub(1)),
        );
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        for x in x0..=x1 {
            soft_border(grid, x, y0, '┄');
            soft_border(grid, x, y1, '┄');
        }
        for y in y0..=y1 {
            soft_border(grid, x0, y, '┆');
            soft_border(grid, x1, y, '┆');
        }
        // Title sits on the top border, inset by 1.
        let title = truncate_w(&sg.title, x1.saturating_sub(x0).saturating_sub(2));
        let title_y = if title_area_clear(grid, x0 + 1, y0, &title) {
            y0
        } else if title_area_clear(grid, x0 + 1, y1, &title) {
            y1
        } else {
            y0
        };
        let mut x = x0 + 1;
        for c in title.chars() {
            grid.put(x, title_y, c, Ink::Title);
            x += char_w(c);
        }
    }
}

fn title_area_clear(grid: &Grid, x: usize, y: usize, title: &str) -> bool {
    let width = display_w(title);
    (x..x.saturating_add(width)).all(|col| matches!(grid.get(col, y), ' ' | '┄'))
}

/// Place a subgraph border glyph only over empty cells so it never paints over
/// node boxes or routed edges.
fn soft_border(grid: &mut Grid, x: usize, y: usize, c: char) {
    if grid.get(x, y) == ' ' {
        grid.put(x, y, c, Ink::Line);
    }
}

// ============================================================================
// Width helpers (CJK-aware, mirroring the markdown table renderer)
// ============================================================================

fn char_w(c: char) -> usize {
    unicode_width::UnicodeWidthChar::width(c).unwrap_or(0)
}

fn display_w(s: &str) -> usize {
    s.chars().map(char_w).sum()
}

/// Truncate `s` to at most `max` display columns, appending `…` if cut.
fn truncate_w(s: &str, max: usize) -> String {
    if display_w(s) <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for c in s.chars() {
        let cw = char_w(c);
        if w + cw > max.saturating_sub(1) {
            break;
        }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::Theme;

    fn flatten(rows: &[Vec<Span<'static>>]) -> String {
        rows.iter()
            .map(|r| r.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parses_direction_and_nodes() {
        let g = parse("graph LR\n  A[Start] --> B[End]").expect("parse");
        assert_eq!(g.direction as u8, Direction::Horizontal as u8);
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges, vec![(0, 1)]);
        assert_eq!(g.nodes[0].label, vec!["Start".to_string()]);
    }

    #[test]
    fn splits_br_labels_into_lines() {
        let g = parse("graph TD\n  CLI[\"zo<br/>(bin)\"] --> RT[runtime]").expect("parse");
        assert_eq!(
            g.nodes[0].label,
            vec!["zo".to_string(), "(bin)".to_string()]
        );
    }

    #[test]
    fn ignores_style_lines() {
        let g = parse("graph TD\nA-->B\nstyle A fill:#fff\nclassDef x fill:#000").expect("parse");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn renders_boxes_and_arrows_for_simple_td() {
        let theme = Theme::default_dark();
        let rows = render("graph TD\n A[Top] --> B[Bottom]", 60, &theme).expect("render");
        let flat = flatten(&rows);
        assert!(flat.contains("Top"), "node label present:\n{flat}");
        assert!(flat.contains("Bottom"), "child label present:\n{flat}");
        assert!(
            flat.contains('╭') && flat.contains('╯'),
            "boxes drawn:\n{flat}"
        );
        assert!(flat.contains('▼'), "downward arrow drawn:\n{flat}");
    }

    #[test]
    fn subgraph_title_is_rendered() {
        let theme = Theme::default_dark();
        let src = "graph TD\n subgraph L1[Base]\n  CT[core] \n end\n CT --> API[api]";
        let rows = render(src, 80, &theme).expect("render");
        let flat = flatten(&rows);
        assert!(flat.contains("core") && flat.contains("api"));
    }

    #[test]
    fn single_box_too_wide_falls_back_to_none() {
        let theme = Theme::default_dark();
        // A single capped-width Mermaid box cannot fit a 20-col viewport.
        let src = "graph TD\n A[abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz]";
        assert!(render(src, 20, &theme).is_none());
    }

    #[test]
    fn wide_vertical_rank_wraps_instead_of_falling_back() {
        let theme = Theme::default_dark();
        let src = "flowchart TB\n\
            ROOT[Root]\n\
            ROOT --> A[Service A]\n\
            ROOT --> B[Service B]\n\
            ROOT --> C[Service C]\n\
            ROOT --> D[Service D]";
        let rows = render(src, 44, &theme).expect("wide rank should wrap");
        let flat = flatten(&rows);
        for label in ["Root", "Service A", "Service B", "Service C", "Service D"] {
            assert!(flat.contains(label), "missing {label}:\n{flat}");
        }
        assert!(!flat.contains("flowchart"), "raw Mermaid leaked:\n{flat}");
    }

    #[test]
    fn renders_nested_subgraphs_and_labeled_edges() {
        let theme = Theme::default_dark();
        let src = "flowchart TB\n\
            %% comment\n\
            subgraph KBANK[\"K-bank AWS 환경\"]\n\
              subgraph ORG[\"AWS Organizations\"]\n\
                MGMT[\"Management Account\"]\n\
                AUDIT[\"Security / Audit Account\"]\n\
                MEMBER1[\"Member Account A\"]\n\
                MEMBER2[\"Member Account B\"]\n\
                MGMT -->|ListAccounts| AUDIT\n\
                MGMT --> MEMBER1\n\
                MGMT --> MEMBER2\n\
              end\n\
              subgraph IAM[\"AWS IAM / StackSet\"]\n\
                HUBROLE[\"Hub Role<br/>Management / Audit\"]\n\
                SPOKE1[\"ReadRole<br/>Member A\"]\n\
                SPOKE2[\"ReadRole<br/>Member B\"]\n\
              end\n\
              AUDIT --> HUBROLE\n\
              HUBROLE -->|sts:AssumeRole| SPOKE1\n\
              HUBROLE -->|sts:AssumeRole| SPOKE2\n\
              SPOKE1 --> MEMBER1\n\
              SPOKE2 --> MEMBER2\n\
            end";
        let rows = render(src, 90, &theme).expect("nested Mermaid should render");
        let flat = flatten(&rows);
        for label in [
            "K-bank AWS",
            "Organizations",
            "AWS IAM / StackSet",
            "Hub Role",
            "Management / Audit",
            "ReadRole",
            "Member A",
        ] {
            assert!(flat.contains(label), "missing {label}:\n{flat}");
        }
        assert!(!flat.contains("-->"), "raw edge op leaked:\n{flat}");
    }

    #[test]
    fn empty_source_is_none() {
        let theme = Theme::default_dark();
        assert!(render("graph TD", 80, &theme).is_none());
    }

    #[test]
    fn multi_rank_graph_with_fan_out_is_clean() {
        let theme = Theme::default_dark();
        let src = "graph TD\n\
            CLI[\"zo<br/>(bin)\"]\n\
            CLI --> RT[runtime]\n\
            CLI --> TOOLS[tools]\n\
            RT --> API[api]\n\
            RT --> PLUG[plugins]\n\
            API --> CT[core-types]\n\
            PLUG --> CT";
        let rows = render(src, 80, &theme).expect("render");
        let flat = flatten(&rows);
        // All nodes present, fan-out/merge junctions drawn, no raw mermaid ops.
        for label in ["zo", "runtime", "tools", "api", "plugins", "core-types"] {
            assert!(flat.contains(label), "missing {label}:\n{flat}");
        }
        assert!(
            flat.contains('┴') || flat.contains('┬'),
            "junctions:\n{flat}"
        );
        assert!(!flat.contains("-->"), "raw edge op leaked:\n{flat}");
    }
}
