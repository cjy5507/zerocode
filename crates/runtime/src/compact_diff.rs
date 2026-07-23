const DEFAULT_CONTEXT_LINES: usize = 3;
const EXACT_DIFF_MAX_CELLS: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactDiffHunk {
    pub old_start: usize,
    pub old_lines: usize,
    pub new_start: usize,
    pub new_lines: usize,
    pub lines: Vec<CompactDiffLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactDiffLine {
    pub kind: CompactDiffLineKind,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactDiffLineKind {
    Context,
    Removed,
    Added,
}

#[derive(Clone, Copy, Debug)]
struct LineDiffOp<'a> {
    kind: CompactDiffLineKind,
    old_index: usize,
    new_index: usize,
    text: &'a str,
}

#[must_use]
pub fn compact_line_diff(original: &str, updated: &str) -> Vec<CompactDiffHunk> {
    if original == updated {
        return Vec::new();
    }

    let original_lines: Vec<&str> = original.lines().collect();
    let updated_lines: Vec<&str> = updated.lines().collect();
    if let Some(ops) = line_diff_ops(&original_lines, &updated_lines) {
        return hunks_from_ops(&ops);
    }

    vec![single_hunk_from_lines(&original_lines, &updated_lines)]
}

fn line_diff_ops<'a>(old_lines: &[&'a str], new_lines: &[&'a str]) -> Option<Vec<LineDiffOp<'a>>> {
    let cells = old_lines.len().saturating_mul(new_lines.len());
    if cells > EXACT_DIFF_MAX_CELLS {
        return None;
    }

    let width = new_lines.len() + 1;
    let index = |old_index: usize, new_index: usize| old_index * width + new_index;
    let mut lengths = vec![0usize; (old_lines.len() + 1) * width];
    for old_index in (0..old_lines.len()).rev() {
        for new_index in (0..new_lines.len()).rev() {
            lengths[index(old_index, new_index)] = if old_lines[old_index] == new_lines[new_index] {
                lengths[index(old_index + 1, new_index + 1)] + 1
            } else {
                lengths[index(old_index + 1, new_index)]
                    .max(lengths[index(old_index, new_index + 1)])
            };
        }
    }

    let mut ops = Vec::new();
    let mut old_index = 0usize;
    let mut new_index = 0usize;
    while old_index < old_lines.len() && new_index < new_lines.len() {
        if old_lines[old_index] == new_lines[new_index] {
            ops.push(LineDiffOp {
                kind: CompactDiffLineKind::Context,
                old_index,
                new_index,
                text: old_lines[old_index],
            });
            old_index += 1;
            new_index += 1;
        } else if lengths[index(old_index + 1, new_index)]
            >= lengths[index(old_index, new_index + 1)]
        {
            ops.push(LineDiffOp {
                kind: CompactDiffLineKind::Removed,
                old_index,
                new_index,
                text: old_lines[old_index],
            });
            old_index += 1;
        } else {
            ops.push(LineDiffOp {
                kind: CompactDiffLineKind::Added,
                old_index,
                new_index,
                text: new_lines[new_index],
            });
            new_index += 1;
        }
    }

    while old_index < old_lines.len() {
        ops.push(LineDiffOp {
            kind: CompactDiffLineKind::Removed,
            old_index,
            new_index,
            text: old_lines[old_index],
        });
        old_index += 1;
    }
    while new_index < new_lines.len() {
        ops.push(LineDiffOp {
            kind: CompactDiffLineKind::Added,
            old_index,
            new_index,
            text: new_lines[new_index],
        });
        new_index += 1;
    }

    Some(ops)
}

fn hunks_from_ops(ops: &[LineDiffOp<'_>]) -> Vec<CompactDiffHunk> {
    let mut hunks = Vec::new();
    let mut current = Vec::new();
    let mut pending_context = Vec::new();
    let mut in_hunk = false;

    for &op in ops {
        if op.kind == CompactDiffLineKind::Context {
            if in_hunk {
                pending_context.push(op);
            } else {
                pending_context.push(op);
                if pending_context.len() > DEFAULT_CONTEXT_LINES {
                    pending_context.remove(0);
                }
            }
            continue;
        }

        if !in_hunk {
            current.extend_from_slice(&pending_context);
            pending_context.clear();
            in_hunk = true;
        } else if pending_context.len() > DEFAULT_CONTEXT_LINES * 2 {
            current.extend_from_slice(&pending_context[..DEFAULT_CONTEXT_LINES]);
            if let Some(hunk) = hunk_from_ops(&current) {
                hunks.push(hunk);
            }
            current.clear();
            current.extend_from_slice(
                &pending_context[pending_context.len() - DEFAULT_CONTEXT_LINES..],
            );
            pending_context.clear();
        } else {
            current.extend_from_slice(&pending_context);
            pending_context.clear();
        }

        current.push(op);
    }

    if in_hunk {
        current.extend_from_slice(
            &pending_context[..pending_context.len().min(DEFAULT_CONTEXT_LINES)],
        );
        if let Some(hunk) = hunk_from_ops(&current) {
            hunks.push(hunk);
        }
    }

    hunks
}

fn hunk_from_ops(ops: &[LineDiffOp<'_>]) -> Option<CompactDiffHunk> {
    let first = ops.first()?;
    Some(CompactDiffHunk {
        old_start: first.old_index.saturating_add(1),
        old_lines: ops
            .iter()
            .filter(|op| op.kind != CompactDiffLineKind::Added)
            .count(),
        new_start: first.new_index.saturating_add(1),
        new_lines: ops
            .iter()
            .filter(|op| op.kind != CompactDiffLineKind::Removed)
            .count(),
        lines: ops
            .iter()
            .map(|op| CompactDiffLine {
                kind: op.kind,
                text: op.text.to_string(),
            })
            .collect(),
    })
}

fn single_hunk_from_lines(original_lines: &[&str], updated_lines: &[&str]) -> CompactDiffHunk {
    if original_lines.is_empty() {
        return CompactDiffHunk {
            old_start: 1,
            old_lines: 0,
            new_start: 1,
            new_lines: updated_lines.len(),
            lines: updated_lines
                .iter()
                .map(|line| CompactDiffLine {
                    kind: CompactDiffLineKind::Added,
                    text: (*line).to_string(),
                })
                .collect(),
        };
    }
    if updated_lines.is_empty() {
        return CompactDiffHunk {
            old_start: 1,
            old_lines: original_lines.len(),
            new_start: 1,
            new_lines: 0,
            lines: original_lines
                .iter()
                .map(|line| CompactDiffLine {
                    kind: CompactDiffLineKind::Removed,
                    text: (*line).to_string(),
                })
                .collect(),
        };
    }

    let mut prefix = 0usize;
    while prefix < original_lines.len()
        && prefix < updated_lines.len()
        && original_lines[prefix] == updated_lines[prefix]
    {
        prefix += 1;
    }

    let mut suffix = 0usize;
    while suffix < original_lines.len().saturating_sub(prefix)
        && suffix < updated_lines.len().saturating_sub(prefix)
        && original_lines[original_lines.len() - 1 - suffix]
            == updated_lines[updated_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let old_change_end = original_lines.len().saturating_sub(suffix);
    let new_change_end = updated_lines.len().saturating_sub(suffix);
    let context_start = prefix.saturating_sub(DEFAULT_CONTEXT_LINES);
    let old_context_end = old_change_end
        .saturating_add(DEFAULT_CONTEXT_LINES)
        .min(original_lines.len());
    let new_context_end = new_change_end
        .saturating_add(DEFAULT_CONTEXT_LINES)
        .min(updated_lines.len());

    let mut lines = Vec::new();
    for line in &original_lines[context_start..prefix] {
        lines.push(CompactDiffLine {
            kind: CompactDiffLineKind::Context,
            text: (*line).to_string(),
        });
    }
    for line in &original_lines[prefix..old_change_end] {
        lines.push(CompactDiffLine {
            kind: CompactDiffLineKind::Removed,
            text: (*line).to_string(),
        });
    }
    for line in &updated_lines[prefix..new_change_end] {
        lines.push(CompactDiffLine {
            kind: CompactDiffLineKind::Added,
            text: (*line).to_string(),
        });
    }
    for line in &updated_lines[new_change_end..new_context_end] {
        lines.push(CompactDiffLine {
            kind: CompactDiffLineKind::Context,
            text: (*line).to_string(),
        });
    }

    CompactDiffHunk {
        old_start: context_start.saturating_add(1),
        old_lines: old_context_end.saturating_sub(context_start),
        new_start: context_start.saturating_add(1),
        new_lines: new_context_end.saturating_sub(context_start),
        lines,
    }
}
