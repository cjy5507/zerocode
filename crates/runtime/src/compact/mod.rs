use crate::session::{AnchorSummary, ContentBlock, ConversationMessage, MessageRole, Session};

const COMPACT_CONTINUATION_PREAMBLE: &str = "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n";
const COMPACT_RECENT_MESSAGES_NOTE: &str = "Recent messages are preserved verbatim.";
const COMPACT_DIRECT_RESUME_INSTRUCTION: &str = "Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, and do not preface with continuation text.";

/// System prompt sent to the API when requesting an LLM-generated compaction summary.
///
/// The model receives the conversation messages to be compacted along with this prompt,
/// and should respond with `<analysis>…</analysis>` and `<summary>…</summary>` blocks.
pub const COMPACTION_SYSTEM_PROMPT: &str = "\
You are summarizing a coding conversation that is about to run out of context window. Produce a detailed but information-dense summary that lets the assistant resume the work seamlessly, losing no critical technical detail.

Rules:
1. Read the entire conversation and capture every fact needed to continue.
2. Preserve exact identifiers verbatim: file paths, function/type names, error messages, command lines, and key tool outputs.
3. Be specific, not vague — prefer concrete names and values over descriptions.
4. Do NOT include pleasantries, meta-commentary, or filler.

Respond in this exact format:

<analysis>
Briefly walk through the conversation chronologically and note what matters for each of the summary sections below. This is your scratchpad.
</analysis>

<summary>
1. Primary Request and Intent: [the user's explicit asks and overall goal, in detail]
2. Key Technical Concepts: [technologies, libraries, patterns, and conventions in play]
3. Files and Code Sections: [each file viewed or changed, with why it matters and the specific functions/edits — include important code snippets where they aid resumption]
4. Errors and Fixes: [errors encountered, how they were resolved, and any user feedback]
5. Problem Solving: [problems solved and ongoing troubleshooting, with reasoning]
6. All User Messages: [list the user's non-tool messages so intent and corrections are not lost]
7. Pending Tasks: [explicitly requested work that is not yet done]
8. Current Work: [precisely what was being done right before this summary, with file names and code so it can be picked up immediately] and Next Step: [the single most logical next action, only if it directly continues the current work]
</summary>";

/// Build the compaction-summary system prompt, optionally augmented with the
/// user's `/compact <focus>` directive (Claude Code "Compact Instructions"
/// parity).
///
/// Bare compaction (`focus == None`, or an all-whitespace focus) returns
/// [`COMPACTION_SYSTEM_PROMPT`] byte-for-byte, so the no-focus path — and the
/// prompt-cache blocks that ride the system prompt — are unchanged. When a
/// focus is present it is appended as an explicit, highest-priority
/// preservation instruction so the model steers the *retained* detail toward
/// what the user asked to keep. This is what makes `/compact <focus>` raise
/// summary quality on the focus area, instead of the old behavior where routing
/// to the deterministic extractor made the summary *worse* the more specific
/// the request got.
#[must_use]
pub fn compaction_system_prompt(focus: Option<&str>) -> String {
    match focus.map(str::trim).filter(|focus| !focus.is_empty()) {
        Some(focus) => format!(
            "{COMPACTION_SYSTEM_PROMPT}\n\nCompact Instructions: the user asked to focus this summary on \"{focus}\". Preserve every detail relevant to {focus} above all else — when you must condense, condense unrelated material first and keep information about {focus} verbatim. Still produce all eight numbered sections in the exact format above."
        ),
        None => COMPACTION_SYSTEM_PROMPT.to_string(),
    }
}

/// Thresholds controlling when and how a session is compacted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionConfig {
    pub preserve_recent_messages: usize,
    pub max_estimated_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            preserve_recent_messages: 4,
            max_estimated_tokens: 10_000,
        }
    }
}

/// Result of compacting a session into a summary plus preserved tail messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    pub summary: String,
    pub formatted_summary: String,
    pub compacted_session: Session,
    pub removed_message_count: usize,
}

/// Prepared compaction boundaries — the messages to remove, the tail to preserve,
/// and any existing summary from a prior compaction round.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// Messages that will be summarized and removed.
    pub messages_to_compact: Vec<ConversationMessage>,
    /// Messages preserved verbatim at the end of the session.
    pub preserved_tail: Vec<ConversationMessage>,
    /// Typed anchor from a prior compaction round, if one exists — recovered
    /// from `session.compaction.anchor` (or parsed from a legacy continuation
    /// message on the first post-upgrade round). The next summary folds into it.
    pub existing_anchor: Option<AnchorSummary>,
    /// Lightweight clone of the original session with messages cleared,
    /// carrying only the metadata (id, timestamps, compaction state, fork).
    /// Avoids cloning the entire message history (P0 L2).
    session_shell: Session,
}

/// Trait for generating a summary of messages being compacted.
///
/// Implementations can produce summaries locally (deterministic extraction)
/// or by calling an LLM API (higher quality, matches upstream TS behavior).
pub trait CompactionSummarizer {
    /// Generate a summary of the given messages.
    ///
    /// The returned string should ideally contain `<summary>…</summary>` tags.
    /// An `<analysis>…</analysis>` block is optional and will be stripped during formatting.
    fn summarize(&self, messages: &[ConversationMessage]) -> String;
}

/// Local deterministic summarizer — extracts structure from messages without API calls.
///
/// This is the default summarizer and matches the behavior prior to the trait introduction.
#[derive(Debug, Default)]
pub struct LocalSummarizer;

impl CompactionSummarizer for LocalSummarizer {
    fn summarize(&self, messages: &[ConversationMessage]) -> String {
        summarize_messages(messages)
    }
}

/// Summarizer that injects a user focus directive (from `/compact <focus>`) as
/// the first line inside the `<summary>` block of the deterministic summary, so
/// the resumed assistant prioritizes it. Extraction stays deterministic; the
/// focus is carried as an explicit header rather than steering the extraction.
#[derive(Debug)]
pub struct FocusSummarizer {
    /// The user's focus directive (already trimmed, non-empty).
    pub focus: String,
}

impl CompactionSummarizer for FocusSummarizer {
    fn summarize(&self, messages: &[ConversationMessage]) -> String {
        let base = LocalSummarizer.summarize(messages);
        let focus_line = format!("Focus (user /compact directive): {}", self.focus);
        match base.find("<summary>") {
            Some(idx) => {
                let cut = idx + "<summary>".len();
                format!("{}\n{focus_line}{}", &base[..cut], &base[cut..])
            }
            None => format!("{focus_line}\n{base}"),
        }
    }
}

/// Roughly estimates the token footprint of the current session transcript.
#[must_use]
pub fn estimate_session_tokens(session: &Session) -> usize {
    session.messages.iter().map(estimate_message_tokens).sum()
}

/// At least this many messages must remain summarizable after the preserved
/// tail is carved off, or a token-budgeted tail could swallow the whole
/// transcript and turn the round into a no-op that immediately re-triggers.
const MIN_COMPACTABLE_MESSAGES: usize = 8;

/// Number of trailing messages whose estimated tokens fit `budget_tokens` —
/// the CC-parity token-budgeted preserved tail. Capped so at least
/// [`MIN_COMPACTABLE_MESSAGES`] remain to summarize, then floored at the
/// legacy 4-message default ([`CompactionConfig::default`]): the floor wins
/// on a session too small for both bounds, which keeps small-session
/// behavior byte-identical to the fixed-tail era (the budget only ever
/// GROWS the tail on sessions big enough to afford it).
#[must_use]
pub fn preserved_tail_len_for_budget(
    messages: &[ConversationMessage],
    budget_tokens: u64,
) -> usize {
    let mut total: u64 = 0;
    let mut count = 0usize;
    for message in messages.iter().rev() {
        total = total.saturating_add(estimate_message_tokens(message) as u64);
        if total > budget_tokens {
            break;
        }
        count += 1;
    }
    let cap = messages.len().saturating_sub(MIN_COMPACTABLE_MESSAGES);
    count
        .min(cap)
        .max(CompactionConfig::default().preserve_recent_messages)
}

/// Per-tool-result cap (chars) on the copy of the transcript sent to the
/// summarizer. The 8-section summary needs the structural facts a result
/// carries, not a 40k-char build log verbatim; oversized bodies keep head and
/// tail (`elide_middle`) so identifiers at either end survive.
const SUMMARY_INPUT_TOOL_RESULT_MAX_CHARS: usize = 6_000;

/// P3: pre-trim the SUMMARY REQUEST copy of the messages (never the session):
/// oversized tool-result bodies are middle-elided and image blocks are
/// replaced with a placeholder — both pure input-cost reduction for the
/// summary round-trip. Everything else passes through byte-identical.
#[must_use]
pub fn pretrim_messages_for_summary(messages: &[ConversationMessage]) -> Vec<ConversationMessage> {
    messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            let needs_trim = message.blocks.iter().any(|block| match block {
                ContentBlock::ToolResult { output, .. } => {
                    output.chars().count() > SUMMARY_INPUT_TOOL_RESULT_MAX_CHARS
                }
                ContentBlock::Image { .. } => true,
                _ => false,
            });
            if !needs_trim {
                return message;
            }
            message.blocks = message
                .blocks
                .iter()
                .map(|block| match block {
                    ContentBlock::ToolResult { output, .. }
                        if output.chars().count() > SUMMARY_INPUT_TOOL_RESULT_MAX_CHARS =>
                    {
                        let mut trimmed = block.clone();
                        if let ContentBlock::ToolResult { output, .. } = &mut trimmed {
                            *output = core_types::text::elide_middle(
                                output,
                                SUMMARY_INPUT_TOOL_RESULT_MAX_CHARS,
                            );
                        }
                        trimmed
                    }
                    ContentBlock::Image { .. } => ContentBlock::Text {
                        text: "[image omitted from summary input]".to_string(),
                    },
                    other => other.clone(),
                })
                .collect();
            message
        })
        .collect()
}

/// Returns `true` when the session exceeds the configured compaction budget.
#[must_use]
pub fn should_compact(session: &Session, config: CompactionConfig) -> bool {
    let start = compacted_summary_prefix_len(session);
    let compactable = &session.messages[start..];

    compactable.len() > config.preserve_recent_messages
        && compactable
            .iter()
            .map(estimate_message_tokens)
            .sum::<usize>()
            >= config.max_estimated_tokens
}

/// Normalizes a compaction summary into user-facing continuation text.
#[must_use]
pub fn format_compact_summary(summary: &str) -> String {
    let without_analysis = strip_tag_block(summary, "analysis");
    let formatted = if let Some(content) = extract_tag_block(&without_analysis, "summary") {
        without_analysis.replace(
            &format!("<summary>{content}</summary>"),
            &format!("Summary:\n{}", content.trim()),
        )
    } else {
        without_analysis
    };

    collapse_blank_lines(&formatted).trim().to_string()
}

/// Builds the synthetic system message used after session compaction.
///
/// `vault_ranges` are the Raw Vault seq spans this session has sealed (LAVA P1).
/// When non-empty, a single recall-affordance line is appended AFTER the summary
/// body and the resume/notes so the model knows the exact pre-compaction
/// originals are recoverable and how to address them — see
/// [`format_vault_recall_affordance`]. Pass `&[]` for an unsealed session (or a
/// caller that does not surface recovery).
#[must_use]
pub fn get_compact_continuation_message(
    summary: &str,
    suppress_follow_up_questions: bool,
    recent_messages_preserved: bool,
    vault_ranges: &[(u32, u32)],
) -> String {
    let mut base = format!(
        "{COMPACT_CONTINUATION_PREAMBLE}{}",
        format_compact_summary(summary)
    );

    if recent_messages_preserved {
        base.push_str("\n\n");
        base.push_str(COMPACT_RECENT_MESSAGES_NOTE);
    }

    if suppress_follow_up_questions {
        base.push('\n');
        base.push_str(COMPACT_DIRECT_RESUME_INSTRUCTION);
    }

    // The recall affordance rides OUTSIDE the summary body (after the
    // resume/notes) so the legacy prose reparse in `parse_anchor_from_summary`,
    // which only inspects the extracted `<summary>`/continuation body, never
    // folds it into the anchor.
    if let Some(affordance) = format_vault_recall_affordance(vault_ranges) {
        base.push_str("\n\n");
        base.push_str(&affordance);
    }

    base
}

/// Render the model-facing Raw Vault recall affordance appended to the
/// continuation message: one line naming the exact `session_recall` seq spans
/// the raw pre-compaction originals live at, so the model can pull back an exact
/// detail instead of trusting the lossy summary. Returns `None` when nothing was
/// sealed (no ranges), so the line appears only when there is a vault to point
/// at. The parameter names match the `session_recall` tool spec (`seq_from` /
/// `seq_to` / `include_tool_results`) so the model reads a directly-usable call.
fn format_vault_recall_affordance(vault_ranges: &[(u32, u32)]) -> Option<String> {
    if vault_ranges.is_empty() {
        return None;
    }
    let spans = vault_ranges
        .iter()
        .map(|(lo, hi)| format!("{lo}-{hi}"))
        .collect::<Vec<_>>()
        .join(", ");
    let seq_from = vault_ranges.iter().map(|(lo, _)| *lo).min().unwrap_or(0);
    let seq_to = vault_ranges.iter().map(|(_, hi)| *hi).max().unwrap_or(0);
    Some(format!(
        "Raw originals of every compacted message are preserved in this session's vault (seq {spans}). \
         Retrieve exact pre-compaction originals with the session_recall tool: \
         {{\"session_ref\": \"current\", \"seq_from\": {seq_from}, \"seq_to\": {seq_to}}} \
         (optional: query, role, include_tool_results)."
    ))
}

/// Prepares a compaction plan without generating the summary yet.
///
/// Returns `None` if the session does not need compaction.
#[must_use]
pub fn prepare_compaction(session: &Session, config: CompactionConfig) -> Option<CompactionPlan> {
    if !should_compact(session, config) {
        return None;
    }

    // Detect (structurally) whether index 0 is a prior continuation message,
    // and recover the prior round's state as a TYPED anchor — from the stored
    // `compaction.anchor` (P1 path), or, for a session compacted before P1,
    // by parsing the legacy continuation prose (upgrades it to typed on this
    // round). The structural detection (not the anchor) drives the cut point,
    // so the boundary logic is unchanged from before P1.
    let existing_continuation = session
        .messages
        .first()
        .and_then(extract_existing_compacted_summary);
    let compacted_prefix_len = usize::from(existing_continuation.is_some());
    let existing_anchor = session
        .compaction
        .as_ref()
        .and_then(|compaction| compaction.anchor.clone())
        // A `Some` but entirely-empty anchor (e.g. a `"anchor": null` from
        // corruption/migration) is indistinguishable from real state here, so
        // treat it as absent and fall back to recovering from the rendered
        // continuation rather than silently dropping prior context.
        .filter(|anchor| !anchor.is_empty())
        .or_else(|| {
            existing_continuation
                .as_deref()
                .map(parse_anchor_from_summary)
        });
    let mut keep_from = session
        .messages
        .len()
        .saturating_sub(config.preserve_recent_messages);

    // Never let the preserved tail begin with an orphan `tool_result`: the Anthropic
    // API rejects such payloads with `unexpected tool_use_id in tool_result blocks`
    // because the matching assistant `tool_use` would be lost in the summary. Walk
    // the cut point backward past any user message carrying a tool_result so the
    // preceding assistant `tool_use` travels with it into the tail.
    while keep_from > compacted_prefix_len
        && session
            .messages
            .get(keep_from)
            .is_some_and(message_has_tool_result)
    {
        keep_from -= 1;
    }

    let mut session_shell = session.clone();
    session_shell.messages = std::sync::Arc::new(Vec::new());

    Some(CompactionPlan {
        messages_to_compact: session.messages[compacted_prefix_len..keep_from].to_vec(),
        preserved_tail: session.messages[keep_from..].to_vec(),
        existing_anchor,
        session_shell,
    })
}

/// Returns `true` if `message` contains any `ToolResult` content block.
///
/// Used by [`prepare_compaction`] to keep `tool_use`/`tool_result` pairs together
/// across the compaction cut point — otherwise the preserved tail can begin
/// with an orphaned `tool_result` that the Anthropic API rejects.
fn message_has_tool_result(message: &ConversationMessage) -> bool {
    message
        .blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
}

/// Applies a compaction plan using the given raw summary text.
///
/// The `raw_summary` is the output from a [`CompactionSummarizer`] — either a local
/// extraction or an LLM-generated summary with `<analysis>` and `<summary>` tags.
/// Applies a compaction plan by value, consuming the plan to avoid
/// redundant clones of `preserved_tail` and `original_session`.
#[must_use]
pub fn apply_compaction(plan: CompactionPlan, raw_summary: &str) -> CompactionResult {
    let CompactionPlan {
        mut messages_to_compact,
        preserved_tail,
        existing_anchor,
        session_shell,
    } = plan;
    let removed_message_count = messages_to_compact.len();
    // Fold this round's delta into the prior typed anchor (verbatim carry, no
    // re-truncation), then RENDER the model-facing summary from the folded
    // anchor — the single source of truth. This replaces the prose
    // merge-of-summaries that eroded identifiers over many rounds.
    let delta_anchor = parse_anchor_from_summary(raw_summary);
    let mut folded_anchor = fold_anchor(existing_anchor.as_ref(), delta_anchor);

    // The session shell carries the persistence path and the pre-compaction
    // `first_message_index`, so it is both the seal target and the source the
    // microcompact-restore reads its raw originals from.
    let mut compacted_session = session_shell;

    // LAVA lossless: before sealing, swap any microcompact-cleared tool-result
    // body back to its raw original from the append-only transcript on disk —
    // which `record_compaction` has not overwritten yet — so the vault seals the
    // pre-clear original, not the `[Old tool result content cleared]`
    // placeholder. Best-effort: a missing/corrupt/unmatched source degrades to
    // sealing the in-memory (placeholder) state, never failing the compaction.
    restore_microcompacted_bodies_from_disk(&mut messages_to_compact, &compacted_session);

    // Seal the raw evicted messages to the append-only vault BEFORE
    // `record_compaction` overwrites the transcript with the lossy summary, so
    // the originals survive this and every later compaction round losslessly.
    // The returned span records which vault seqs this round owns; append it to
    // the anchor (the fold carries prior rounds' spans forward verbatim, so the
    // new span is added exactly once, here) so the continuation can advertise
    // the exact recall range.
    if let Some(span) = compacted_session.seal_evicted_to_vault(&messages_to_compact) {
        folded_anchor.vault_ranges.push(span);
    }

    // Keep the running summary from growing to dominate the window: the fold's
    // unbounded union is trimmed to its most recent content per section now that
    // the raw originals are sealed to the vault (so any trimmed detail remains
    // recoverable via `session_recall`).
    bound_anchor(&mut folded_anchor);

    let summary = render_anchor_to_summary_text(&folded_anchor);
    let formatted_summary = format_compact_summary(&summary);
    let continuation = get_compact_continuation_message(
        &summary,
        true,
        !preserved_tail.is_empty(),
        &folded_anchor.vault_ranges,
    );

    let mut compacted_messages = vec![ConversationMessage {
        role: MessageRole::System,
        blocks: vec![ContentBlock::Text { text: continuation }],
        usage: None,
        thought_signature: None,
        reasoning_replay: None,
            model: None,
    }];
    compacted_messages.extend(preserved_tail);

    // Atomic seam: hand the compacted transcript to the session so it captures
    // the pre-compaction snapshot BEFORE replacing `messages`. On a persistence
    // Conflict this rolls the whole mutation (messages + metadata) back to the
    // original state instead of leaving memory holding a compacted view that
    // diverges from a peer's newer file.
    compacted_session.apply_compaction_atomic(
        std::sync::Arc::new(compacted_messages),
        summary.clone(),
        removed_message_count,
        Some(folded_anchor),
    );

    CompactionResult {
        summary,
        formatted_summary,
        compacted_session,
        removed_message_count,
    }
}

/// A tool result's raw `(output, images)` borrowed from a persisted message —
/// the restore source keyed by `tool_use_id`.
type RawToolResultBody<'a> = (&'a String, &'a Vec<(String, String)>);

/// Swap microcompact-cleared tool-result bodies in `evicted` back to their raw
/// originals, read from the session's append-only transcript on disk — the step
/// that makes the vault seal truly lossless even for a body a cheaper tier had
/// already trimmed from the live context.
///
/// Microcompact overwrites an old tool-result `output` with
/// [`MICROCOMPACT_PLACEHOLDER`] in memory but never persists that edit, so at
/// seal time — before `record_compaction` rewrites the snapshot — the on-disk
/// JSONL still holds the pre-clear original. Matching is by `tool_use_id` (a
/// required, session-unique field on every tool result, preserved verbatim
/// through microcompact), so a cleared result is paired with its exact original
/// regardless of position; a positional fallback is unnecessary because the id
/// map is built from the whole persisted transcript, so any evicted result
/// whose original still exists on disk is found by id. Only the `ToolResult`
/// body and its out-of-band `images` are restored; a standalone user-pasted
/// image cleared to a text placeholder has no such stable id and is left as-is.
///
/// Strictly best-effort: no persistence path, an unreadable/corrupt transcript,
/// or an id whose original no longer survives on disk all degrade to leaving the
/// placeholder in place — the pre-vault lossy behavior, never worse, and never a
/// failed compaction.
fn restore_microcompacted_bodies_from_disk(evicted: &mut [ConversationMessage], session: &Session) {
    // Only pay the disk read when something was actually cleared — the common
    // case (no microcompact this session) reads nothing.
    let has_cleared = evicted.iter().any(|message| {
        message.blocks.iter().any(|block| {
            matches!(block, ContentBlock::ToolResult { output, .. } if output == MICROCOMPACT_PLACEHOLDER)
        })
    });
    if !has_cleared {
        return;
    }
    let Some(persisted) = session.load_persisted_messages() else {
        return;
    };
    // tool_use_id -> the raw (non-placeholder) body + images from disk. First
    // writer wins; a duplicated id (should not happen) keeps the earliest.
    let mut originals: std::collections::HashMap<&str, RawToolResultBody> =
        std::collections::HashMap::new();
    for message in &persisted {
        for block in &message.blocks {
            if let ContentBlock::ToolResult {
                tool_use_id,
                output,
                images,
                ..
            } = block
            {
                if output != MICROCOMPACT_PLACEHOLDER {
                    originals
                        .entry(tool_use_id.as_str())
                        .or_insert((output, images));
                }
            }
        }
    }
    for message in evicted.iter_mut() {
        for block in &mut message.blocks {
            if let ContentBlock::ToolResult {
                tool_use_id,
                output,
                images,
                ..
            } = block
            {
                if output == MICROCOMPACT_PLACEHOLDER {
                    if let Some(&(original_output, original_images)) =
                        originals.get(tool_use_id.as_str())
                    {
                        output.clone_from(original_output);
                        images.clone_from(original_images);
                    }
                }
            }
        }
    }
}

const STATE_DISTILL_MAX_CHARS: usize = 1_600;

/// Build a deterministic, bounded working-state snapshot without removing any
/// transcript messages.
///
/// This is the cheap "state distill" tier used before full compaction: it gives
/// the model a compact anchor for the current goal/files/next work while the
/// original transcript remains intact. It deliberately reuses the same local
/// extraction helpers as deterministic compaction, so this tier cannot
/// fabricate unseen identifiers and has no provider round-trip.
#[must_use]
pub fn distill_session_state(session: &Session) -> Option<String> {
    let messages = session.messages.as_ref();
    if messages.is_empty() {
        return None;
    }

    let mut lines = vec!["# Distilled working state".to_string()];
    if let Some(current_work) = infer_current_work(messages) {
        lines.push(format!("- Current work: {current_work}"));
    }

    let recent_requests = collect_recent_role_summaries(messages, MessageRole::User, 2);
    if !recent_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        lines.extend(recent_requests.into_iter().map(|request| format!("  - {request}")));
    }

    let pending_work = infer_pending_work(messages);
    if !pending_work.is_empty() {
        lines.push("- Pending/next signals:".to_string());
        lines.extend(pending_work.into_iter().map(|item| format!("  - {item}")));
    }

    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files: {}", key_files.join(", ")));
    }

    if lines.len() == 1 {
        return None;
    }
    Some(truncate_summary(&lines.join("\n"), STATE_DISTILL_MAX_CHARS))
}

/// This is the primary entry point for compaction with a custom summarizer.
#[must_use]
pub fn compact_session_with<S: CompactionSummarizer>(
    session: &Session,
    config: CompactionConfig,
    summarizer: &S,
) -> CompactionResult {
    let Some(plan) = prepare_compaction(session, config) else {
        return CompactionResult {
            summary: String::new(),
            formatted_summary: String::new(),
            compacted_session: session.clone(),
            removed_message_count: 0,
        };
    };
    let raw_summary = summarizer.summarize(&plan.messages_to_compact);
    apply_compaction(plan, &raw_summary)
}

/// Compacts a session by summarizing older messages and preserving the recent tail.
///
/// Uses the [`LocalSummarizer`] for deterministic local summary generation.
#[must_use]
pub fn compact_session(session: &Session, config: CompactionConfig) -> CompactionResult {
    compact_session_with(session, config, &LocalSummarizer)
}

fn compacted_summary_prefix_len(session: &Session) -> usize {
    usize::from(
        session
            .messages
            .first()
            .and_then(extract_existing_compacted_summary)
            .is_some(),
    )
}

fn summarize_messages(messages: &[ConversationMessage]) -> String {
    let user_messages = messages
        .iter()
        .filter(|message| message.role == MessageRole::User)
        .count();
    let assistant_messages = messages
        .iter()
        .filter(|message| message.role == MessageRole::Assistant)
        .count();
    let tool_messages = messages
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .count();

    let mut tool_names = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { name, .. } => Some(name.as_str()),
            ContentBlock::ToolResult { tool_name, .. } => Some(tool_name.as_str()),
            ContentBlock::Text { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. } => None,
        })
        .collect::<Vec<_>>();
    tool_names.sort_unstable();
    tool_names.dedup();

    let mut lines = vec![
        "<summary>".to_string(),
        "Conversation summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={}, assistant={}, tool={}).",
            messages.len(),
            user_messages,
            assistant_messages,
            tool_messages
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    let recent_user_requests = collect_recent_role_summaries(messages, MessageRole::User, 3);
    if !recent_user_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        lines.extend(
            recent_user_requests
                .into_iter()
                .map(|request| format!("  - {request}")),
        );
    }

    let pending_work = infer_pending_work(messages);
    if !pending_work.is_empty() {
        lines.push("- Pending work:".to_string());
        lines.extend(pending_work.into_iter().map(|item| format!("  - {item}")));
    }

    let key_files = collect_key_files(messages);
    if !key_files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", key_files.join(", ")));
    }

    if let Some(current_work) = infer_current_work(messages) {
        lines.push(format!("- Current work: {current_work}"));
    }

    lines.push("- Key timeline:".to_string());
    for message in messages {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let content = message
            .blocks
            .iter()
            .map(summarize_block)
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("  - {role}: {content}"));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

// ── LAVA P1: typed anchor (single source of truth) ─────────────────────────
//
// The model-facing continuation message is RENDERED from the typed
// `AnchorSummary`, which folds in only each round's delta and carries prior
// entries forward verbatim. This replaces the old prose `merge_compact_summaries`
// (which re-parsed + re-truncated the prior summary every round, eroding
// identifiers over a long session — the "summary of a summary" drift).

/// Parse an LLM (or local) compaction summary into the typed [`AnchorSummary`].
///
/// The eight numbered sections of [`COMPACTION_SYSTEM_PROMPT`] are recognized by
/// a leading `N. ` marker; list sections become one item per non-empty line
/// (bullet markers stripped). A summary without the numbered structure (the
/// [`LocalSummarizer`] output, or a legacy continuation message being upgraded
/// on resume) is carried whole into `problem_solving`, so nothing is lost.
fn parse_anchor_from_summary(raw_summary: &str) -> AnchorSummary {
    let body = extract_tag_block(raw_summary, "summary")
        .unwrap_or_else(|| format_compact_summary(raw_summary));
    let sections = split_numbered_sections(&body);
    if sections.is_empty() {
        // No recognizable section structure (LocalSummarizer output, or a legacy
        // continuation being upgraded). Carry every line as its own item so the
        // content survives and a later render→parse round-trip is identity (one
        // multiline blob would otherwise re-split into many items).
        return AnchorSummary {
            problem_solving: section_items(&body),
            ..AnchorSummary::default()
        };
    }
    let section = |n: usize| {
        sections
            .iter()
            .find(|(idx, _)| *idx == n)
            .map(|(_, text)| text.as_str())
            .unwrap_or_default()
    };
    AnchorSummary {
        intent: section(1).trim().to_string(),
        concepts: section_items(section(2)),
        files: section_items(section(3)),
        errors_and_fixes: section_items(section(4)),
        problem_solving: section_items(section(5)),
        user_messages: section_items(section(6)),
        pending_tasks: section_items(section(7)),
        current_work: section(8).trim().to_string(),
        // A summary carries no vault spans; `apply_compaction` appends the round's
        // span to the folded anchor after the seal.
        vault_ranges: Vec::new(),
    }
}

/// Split a summary body into `(section_number, content)` pairs, stripping the
/// `N. Label:` prefix from each section's first line. Content after a marker
/// accumulates until the next `N. ` marker.
fn split_numbered_sections(body: &str) -> Vec<(usize, String)> {
    let mut sections: Vec<(usize, String)> = Vec::new();
    let mut max_section = 0;
    for line in body.lines() {
        let trimmed_start = line.trim_start();
        // Accept a marker only if its number advances (sections run 1..8 in
        // order) AND the remainder carries the `Label:` colon. This rejects
        // numbered prose substeps like "3. run cargo test after patch" that
        // would otherwise be misread as a section header and silently dropped.
        let after_marker = section_marker(trimmed_start)
            .filter(|&number| number > max_section)
            .and_then(|number| {
                trimmed_start
                    .split_once(". ")
                    .map(|(_, rest)| rest)
                    .filter(|rest| rest.contains(':'))
                    .map(|rest| (number, rest))
            });
        if let Some((number, rest)) = after_marker {
            let content = rest
                .split_once(':')
                .map_or("", |(_, value)| value)
                .trim_start()
                .to_string();
            max_section = number;
            sections.push((number, content));
        } else if let Some(last) = sections.last_mut() {
            if !line.trim().is_empty() {
                if !last.1.is_empty() {
                    last.1.push('\n');
                }
                last.1.push_str(line.trim_end());
            }
        }
    }
    sections
}

/// Recognize a `1. ` .. `8. ` section marker at the start of a line.
fn section_marker(line: &str) -> Option<usize> {
    let mut chars = line.chars();
    let digit = chars.next()?;
    if chars.next()? != '.' || chars.next()? != ' ' {
        return None;
    }
    let number = digit.to_digit(10)? as usize;
    (1..=8).contains(&number).then_some(number)
}

/// One item per non-empty line, bullet markers and `None`/`N/A` placeholders stripped.
fn section_items(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| {
            line.trim()
                .trim_start_matches(['-', '*', ' '])
                .trim()
                .to_string()
        })
        .filter(|line| !line.is_empty() && line != "None" && line != "N/A")
        .collect()
}

/// Fold this round's `delta` into the `prior` anchor.
///
/// Every list section accumulates **losslessly** — prior entries carry forward
/// verbatim and the delta is appended with exact-duplicate dedup, so a fact
/// captured in an early round is byte-identical many rounds later. (An earlier
/// design superseded `files` by path, but that silently dropped distinct facts
/// about the *same* file — two different functions in `lib.rs` — so it was
/// removed in favor of strict no-loss accumulation. Semantic supersede of a
/// corrected fact needs an explicit correction marker from the summarizer and
/// is deferred.) `intent` is set once and kept; `current_work` is the only
/// latest-wins field (the prior "current work" is, by definition, now history
/// preserved in `problem_solving` and the raw vault).
///
/// `vault_ranges` are carried forward verbatim from `prior` (the delta parsed
/// from a summary never sets them). The NEW round's span is appended once by
/// [`apply_compaction`] after the seal returns it — never here — so folding can
/// never double-count a span.
fn fold_anchor(prior: Option<&AnchorSummary>, delta: AnchorSummary) -> AnchorSummary {
    let Some(prior) = prior else {
        return delta;
    };
    let mut folded = prior.clone();
    if folded.intent.is_empty() {
        folded.intent = delta.intent;
    }
    folded.concepts = union_dedup(folded.concepts, delta.concepts);
    folded.files = union_dedup(folded.files, delta.files);
    folded.errors_and_fixes = union_dedup(folded.errors_and_fixes, delta.errors_and_fixes);
    folded.problem_solving = union_dedup(folded.problem_solving, delta.problem_solving);
    folded.user_messages = union_dedup(folded.user_messages, delta.user_messages);
    // Pending tasks accumulate rather than replace: a per-round delta only
    // restates the tasks it saw, so replacing would silently drop a task still
    // pending from an earlier round. Lingering-completed is tolerable (the model
    // reconciles against the verbatim recent tail); losing a real pending task
    // is not.
    folded.pending_tasks = union_dedup(folded.pending_tasks, delta.pending_tasks);
    if !delta.current_work.is_empty() {
        folded.current_work = delta.current_work;
    }
    folded
}

fn union_dedup(mut base: Vec<String>, extra: Vec<String>) -> Vec<String> {
    for item in extra {
        if !base.contains(&item) {
            base.push(item);
        }
    }
    base
}

/// Per-section char budget for the accumulating anchor sections. The `fold_anchor`
/// union grows these sections without bound, so a long-lived session's running
/// summary can swell until it dominates the context window — a live 18-hour
/// session reached a ~150k-token anchor that alone filled 60% of a 258k GPT
/// window, leaving compaction only the small compactable tail to reclaim before
/// it re-fired within a few turns. Bounding each accumulating section keeps the
/// summary small; evicted detail stays recoverable via `session_recall` (every
/// evicted original is sealed to the vault). ~3k tokens per section leaves the
/// most recent tens of entries — ample for continuity — while capping the total.
const ANCHOR_SECTION_MAX_CHARS: usize = 12_000;

/// Bound the accumulating anchor sections to the most recent
/// [`ANCHOR_SECTION_MAX_CHARS`] of content each, so the running summary cannot
/// grow to dominate the context window over a long session. Only the union-
/// accumulating sections are trimmed; `intent`/`current_work` are single latest-
/// state strings, `pending_tasks` must never silently drop a still-open item,
/// and `vault_ranges` is location metadata, not prose — all left intact. Applied
/// once per round in [`apply_compaction`], AFTER the raw originals are sealed to
/// the vault, so nothing trimmed here is unrecoverable.
fn bound_anchor(anchor: &mut AnchorSummary) {
    for section in [
        &mut anchor.concepts,
        &mut anchor.files,
        &mut anchor.errors_and_fixes,
        &mut anchor.problem_solving,
        &mut anchor.user_messages,
    ] {
        retain_recent_within_budget(section, ANCHOR_SECTION_MAX_CHARS);
    }
}

/// Drop oldest (front) items until the remaining items' cumulative rendered
/// length fits `budget`, keeping the most recent. Each item's rendered cost
/// includes the `"- "` bullet prefix and trailing newline that
/// [`render_list_section`] adds. The most recent item is always kept even if it
/// alone exceeds `budget` — dropping the newest entry loses more than a slightly
/// over-budget section, and truncating mid-item would corrupt an identifier.
fn retain_recent_within_budget(items: &mut Vec<String>, budget: usize) {
    const PER_ITEM_OVERHEAD: usize = "- \n".len();
    let mut total = 0usize;
    let mut kept = 0usize;
    for item in items.iter().rev() {
        let next = total.saturating_add(item.len() + PER_ITEM_OVERHEAD);
        if next > budget && kept > 0 {
            break;
        }
        total = next;
        kept += 1;
    }
    items.drain(0..items.len() - kept);
}

/// Render the typed anchor back into a `<summary>` block — the single,
/// drift-free source the model-facing continuation message is built from.
fn render_anchor_to_summary_text(anchor: &AnchorSummary) -> String {
    let mut lines = vec!["<summary>".to_string()];
    if !anchor.intent.is_empty() {
        lines.push(format!("1. Primary Request and Intent: {}", anchor.intent));
    }
    render_list_section(&mut lines, "2. Key Technical Concepts", &anchor.concepts);
    render_list_section(&mut lines, "3. Files and Code Sections", &anchor.files);
    render_list_section(&mut lines, "4. Errors and Fixes", &anchor.errors_and_fixes);
    render_list_section(&mut lines, "5. Problem Solving", &anchor.problem_solving);
    render_list_section(&mut lines, "6. All User Messages", &anchor.user_messages);
    render_list_section(&mut lines, "7. Pending Tasks", &anchor.pending_tasks);
    if !anchor.current_work.is_empty() {
        lines.push(format!("8. Current Work: {}", anchor.current_work));
    }
    lines.push("</summary>".to_string());
    lines.join("\n")
}

fn render_list_section(lines: &mut Vec<String>, header: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    lines.push(format!("{header}:"));
    lines.extend(items.iter().map(|item| format!("- {item}")));
}

/// Conservative deterministic faithfulness check (LAVA P1 verifier): does the
/// API summary cite path/code identifiers that appear NOWHERE in the evicted
/// source (nor the prior anchor)? Returns `true` only on egregious fabrication —
/// several identifiers, none grounded — so a faithful-but-paraphrased summary is
/// never rejected. The caller falls back to the non-fabricating
/// [`LocalSummarizer`] when this trips.
#[must_use]
pub fn summary_fabricates_identifiers(
    raw_summary: &str,
    evicted: &[ConversationMessage],
    prior: Option<&AnchorSummary>,
) -> bool {
    let identifiers = extract_verifiable_identifiers(raw_summary);
    if identifiers.len() < 3 {
        return false;
    }
    let haystack = grounding_haystack(evicted, prior);
    let grounded = identifiers
        .iter()
        .filter(|identifier| haystack.contains(identifier.as_str()))
        .count();
    // Reject when the MAJORITY of cited identifiers are ungrounded — so recycling
    // one real path can't immunize many fabricated ones, while a faithful summary
    // that paraphrases a single identifier still passes.
    let ungrounded = identifiers.len() - grounded;
    ungrounded * 2 > identifiers.len()
}

/// Extract high-confidence verifiable identifiers from a summary: single-token
/// backtick spans and path-like bare tokens (`contains('/')` and an extension).
fn extract_verifiable_identifiers(summary: &str) -> Vec<String> {
    let body = format_compact_summary(summary);
    let mut identifiers = Vec::new();
    let mut rest = body.as_str();
    while let Some(open) = rest.find('`') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('`') else { break };
        let span = rest[..close].trim();
        if span.len() >= 3 && !span.contains(char::is_whitespace) {
            identifiers.push(span.to_string());
        }
        rest = &rest[close + 1..];
    }
    for token in body.split(|c: char| c.is_whitespace() || matches!(c, '`' | ',' | '(' | ')' | '"')) {
        let trimmed = token.trim_matches(|c: char| matches!(c, '.' | ',' | ':' | ';'));
        if trimmed.contains('/') && trimmed.contains('.') && trimmed.len() > 4 && !trimmed.contains("://")
        {
            identifiers.push(trimmed.to_string());
        }
    }
    identifiers.sort_unstable();
    identifiers.dedup();
    identifiers
}

/// Concatenated text of the evicted messages plus the prior anchor — the ground
/// truth a summary's identifiers must be verbatim-present in.
fn grounding_haystack(evicted: &[ConversationMessage], prior: Option<&AnchorSummary>) -> String {
    let mut haystack = String::new();
    for message in evicted {
        for block in &message.blocks {
            match block {
                ContentBlock::Text { text } => haystack.push_str(text),
                ContentBlock::ToolUse { name, input, .. } => {
                    haystack.push_str(name);
                    haystack.push(' ');
                    haystack.push_str(input);
                }
                ContentBlock::ToolResult {
                    tool_name, output, ..
                } => {
                    haystack.push_str(tool_name);
                    haystack.push(' ');
                    haystack.push_str(output);
                }
                // Reasoning blocks are internal, not a source of ground-truth
                // identifiers a summary must cite — excluded like images.
                ContentBlock::Image { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. } => {}
            }
            haystack.push('\n');
        }
    }
    if let Some(prior) = prior {
        haystack.push_str(&render_anchor_to_summary_text(prior));
    }
    haystack
}

fn summarize_block(block: &ContentBlock) -> String {
    let raw = match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {name}({input})"),
        ContentBlock::ToolResult {
            tool_name,
            output,
            is_error,
            ..
        } => format!(
            "tool_result {tool_name}: {}{output}",
            if *is_error { "error " } else { "" }
        ),
        ContentBlock::Image { media_type, .. } => format!("[image: {media_type}]"),
        ContentBlock::Thinking { .. } => "[thinking]".to_string(),
        ContentBlock::RedactedThinking { .. } => "[redacted thinking]".to_string(),
    };
    truncate_summary(&raw, 160)
}

fn collect_recent_role_summaries(
    messages: &[ConversationMessage],
    role: MessageRole,
    limit: usize,
) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.role == role)
        .rev()
        .filter_map(|message| first_text_block(message))
        .take(limit)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn infer_pending_work(messages: &[ConversationMessage]) -> Vec<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_block)
        .filter(|text| {
            let lowered = text.to_ascii_lowercase();
            lowered.contains("todo")
                || lowered.contains("next")
                || lowered.contains("pending")
                || lowered.contains("follow up")
                || lowered.contains("remaining")
        })
        .take(3)
        .map(|text| truncate_summary(text, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn collect_key_files(messages: &[ConversationMessage]) -> Vec<String> {
    let mut files = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .map(|block| match block {
            ContentBlock::Text { text } => text.as_str(),
            ContentBlock::ToolUse { input, .. } => input.as_str(),
            ContentBlock::ToolResult { output, .. } => output.as_str(),
            ContentBlock::Image { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. } => "",
        })
        .flat_map(extract_file_candidates)
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    files.into_iter().take(8).collect()
}

fn infer_current_work(messages: &[ConversationMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .filter_map(first_text_block)
        .find(|text| !text.trim().is_empty())
        .map(|text| truncate_summary(text, 200))
}

fn first_text_block(message: &ConversationMessage) -> Option<&str> {
    message.blocks.iter().find_map(|block| match block {
        ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.as_str()),
        ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::Text { .. }
        | ContentBlock::Image { .. }
        | ContentBlock::Thinking { .. }
        | ContentBlock::RedactedThinking { .. } => None,
    })
}

fn has_interesting_extension(candidate: &str) -> bool {
    std::path::Path::new(candidate)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            ["rs", "ts", "tsx", "js", "json", "md"]
                .iter()
                .any(|expected| extension.eq_ignore_ascii_case(expected))
        })
}

fn extract_file_candidates(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|char: char| {
                matches!(char, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/') && has_interesting_extension(candidate) {
                Some(candidate.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn truncate_summary(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut truncated = content.chars().take(max_chars).collect::<String>();
    truncated.push('…');
    truncated
}

/// Per-message overhead added for role markers, message boundaries, and
/// structural framing that the tokenizer produces but raw text does not.
const MESSAGE_OVERHEAD_TOKENS: usize = 4;

fn estimate_message_tokens(message: &ConversationMessage) -> usize {
    let content_tokens: usize = message
        .blocks
        .iter()
        .map(|block| match block {
            // Use byte length / 4 as a fast O(1) token estimate.
            // For ASCII-dominant content (code, English) this matches
            // chars().count()/4 exactly. For CJK/emoji it slightly
            // overestimates (UTF-8 multibyte → higher byte count) which
            // is acceptable for compaction threshold decisions.
            ContentBlock::Text { text } => text.len() / 4 + 1,
            ContentBlock::ToolUse { name, input, .. } => (name.len() + input.len()) / 4 + 8,
            ContentBlock::ToolResult {
                tool_name,
                output,
                images,
                ..
            } => (tool_name.len() + output.len()) / 4 + 8 + images.len() * 1_600,
            // Reasoning replayed to the wire is billed like ordinary tokens;
            // estimate from its byte length the same way as text.
            ContentBlock::Thinking { thinking, signature } => {
                (thinking.len() + signature.len()) / 4 + 1
            }
            ContentBlock::RedactedThinking { data } => data.len() / 4 + 1,
            ContentBlock::Image { .. } => {
                // Anthropic charges image tokens based on pixel dimensions,
                // not base64 data size. A typical image costs 1000–1600
                // tokens. Use a conservative fixed estimate since we don't
                // have pixel dimensions here.
                1_600
            }
        })
        .sum();
    content_tokens + MESSAGE_OVERHEAD_TOKENS
}

/// Placeholder left in place of a cleared tool-result body. Wording matches
/// Claude Code so models recognize the marker as "this result existed but was
/// trimmed", not as live tool output.
pub const MICROCOMPACT_PLACEHOLDER: &str = "[Old tool result content cleared]";

/// Replacement text for an old user-pasted image cleared by microcompact.
pub const MICROCOMPACT_IMAGE_PLACEHOLDER: &str =
    "[Old pasted image cleared to save context — ask the user to re-attach it if needed]";

/// Tool names whose result body is a *mutation record* (the applied diff /
/// written content), not a re-readable observation. Their bodies are exempt
/// from the microcompact trim: unlike a `read_file` or `grep_search` body —
/// which the model can always reproduce by calling the tool again — a cleared
/// `edit_file`/`write_file` diff is unrecoverable from context and erases the
/// evidence that the change was *already applied*. Losing it is the root cause
/// of the "model re-edits or reverts its own prior change after compaction"
/// failure on long sessions. Kept in lockstep with the conversation layer's
/// `is_edit_or_write_tool` (the two must agree on what counts as a mutation).
const EDIT_RESULT_TOOL_NAMES: &[&str] = &[
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "edit_file",
    "write_file",
];

/// File-edit verbs recognized on a *namespaced* tool's leaf — the segment after
/// the last `__` (e.g. an MCP file server's `mcp__fs__write_file`, or a plugin's
/// `myeditor__edit_file`). Deliberately limited to verbs that denote an APPLIED
/// FILE EDIT whose diff body must survive, so a non-file MCP tool
/// (`mcp__memory__search`, `mcp__db__create_entities`, `mcp__fs__move_file`) is
/// never misclassified as a mutation.
const EDIT_RESULT_TOOL_LEAF_VERBS: &[&str] = &["write_file", "edit_file"];

/// Whether a tool result records a file mutation (an applied edit/write) whose
/// body must survive the microcompact trim. Matches the built-in
/// [`EDIT_RESULT_TOOL_NAMES`] exactly, plus any MCP/plugin-namespaced tool whose
/// leaf verb is a recognized file edit (`mcp__<server>__write_file` etc.) — so an
/// edit applied through an MCP file server or editor plugin is preserved the same
/// way a built-in `edit_file` is, instead of being cleared and then reverted.
#[must_use]
pub fn is_edit_result_tool(tool_name: &str) -> bool {
    if EDIT_RESULT_TOOL_NAMES.contains(&tool_name) {
        return true;
    }
    // Leaf matching applies only to *namespaced* tools (containing `__`); a bare
    // name is authoritative via the exact list above, never broadened here.
    let leaf = tool_name.rsplit("__").next().unwrap_or(tool_name);
    leaf != tool_name && EDIT_RESULT_TOOL_LEAF_VERBS.contains(&leaf)
}

/// Extract the distinct file paths mutated by edit/write tool results in
/// `messages`, in first-seen order. Pure and tolerant: each
/// [`is_edit_result_tool`] result's JSON envelope is parsed for its `filePath`
/// (the canonical key the `edit_file`/`write_file` envelopes serialize; `path`
/// / `file_path` are accepted as fallbacks), and anything that does not parse
/// is skipped, never fatal. A microcompact-cleared placeholder body yields no
/// path, so a trimmed result simply contributes nothing.
///
/// This is the bridge that lets both the durable turn trace
/// (`turn_trace::TurnRecord::files_edited`) and the post-compaction reminder
/// name *which files this session already changed*, so the model does not
/// re-apply or revert its own prior edit once the original diff scrolls out of
/// the context window.
#[must_use]
pub fn edited_file_paths(messages: &[ConversationMessage]) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    for message in messages {
        for block in &message.blocks {
            let ContentBlock::ToolResult {
                tool_name, output, ..
            } = block
            else {
                continue;
            };
            if !is_edit_result_tool(tool_name) {
                continue;
            }
            if let Some(path) = edited_path_from_output(output) {
                if !paths.iter().any(|existing| existing == &path) {
                    paths.push(path);
                }
            }
        }
    }
    paths
}

/// Pull the mutated file path out of one edit/write result envelope. Tolerant:
/// returns `None` for a cleared placeholder, non-JSON, or an envelope without a
/// recognizable path key.
fn edited_path_from_output(output: &str) -> Option<String> {
    if output == MICROCOMPACT_PLACEHOLDER {
        return None;
    }
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    let object = value.as_object()?;
    ["filePath", "path", "file_path"]
        .iter()
        .find_map(|key| object.get(*key).and_then(|v| v.as_str()))
        .map(str::to_string)
}

/// Outcome of one [`microcompact_session`] pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrocompactEvent {
    /// Tool-result bodies replaced with [`MICROCOMPACT_PLACEHOLDER`].
    pub cleared_results: usize,
    /// Standalone user-pasted images replaced with
    /// [`MICROCOMPACT_IMAGE_PLACEHOLDER`]. Tool-result images are cleared with
    /// their result and counted there, not here.
    pub cleared_images: usize,
    /// Fast estimate of the context tokens the clearing freed.
    pub estimated_tokens_saved: u64,
}

/// One [`microcompact_session`] pass's candidate selection, computed
/// read-only so it can be shared between the estimate helper (which never
/// mutates) and the actual clear (which consumes it once). Keeping this
/// selection in exactly one place means the two can never drift apart: the
/// batch an estimate reports is, by construction, the same batch a
/// subsequent clear would act on.
struct MicrocompactPlan {
    /// `(message_index, block_index)` of each tool result to blank.
    tool_results: Vec<(usize, usize)>,
    /// `(message_index, block_index)` of each standalone image to placeholder.
    images: Vec<(usize, usize)>,
}

/// Locate every tool result and standalone image [`microcompact_session`]
/// would clear on this pass, keeping the most recent `keep_recent` of each
/// intact. Pure and read-only; see [`microcompact_session`]'s doc for the
/// exact eligibility rules (placeholder/exempt/size checks).
fn plan_microcompact_clears(
    session: &Session,
    keep_recent: usize,
    min_output_bytes: usize,
) -> MicrocompactPlan {
    // Pass 1 (read-only): locate every clearable tool result so "keep the
    // most recent N results" counts across the whole transcript.
    let clearable: Vec<(usize, usize)> =
        session
            .messages
            .iter()
            .enumerate()
            .flat_map(|(message_index, message)| {
                message.blocks.iter().enumerate().filter_map(
                    move |(block_index, block)| match block {
                        ContentBlock::ToolResult {
                            tool_name,
                            output,
                            images,
                            ..
                        } if output != MICROCOMPACT_PLACEHOLDER
                            && !is_edit_result_tool(tool_name)
                            && (output.len() >= min_output_bytes || !images.is_empty()) =>
                        {
                            Some((message_index, block_index))
                        }
                        _ => None,
                    },
                )
            })
            .collect();
    let clear_count = clearable.len().saturating_sub(keep_recent);
    // Standalone (user-pasted) images are their own pass: they are not tool
    // results, so before this they were re-sent as base64 (~1,600 tokens each)
    // every request until FULL compaction evicted the whole message. Keep the
    // most recent `keep_recent` images — those may still be under discussion.
    let clearable_images: Vec<(usize, usize)> = session
        .messages
        .iter()
        .enumerate()
        .flat_map(|(message_index, message)| {
            message
                .blocks
                .iter()
                .enumerate()
                .filter_map(move |(block_index, block)| {
                    matches!(block, ContentBlock::Image { .. })
                        .then_some((message_index, block_index))
                })
        })
        .collect();
    let image_clear_count = clearable_images.len().saturating_sub(keep_recent);
    MicrocompactPlan {
        tool_results: clearable.into_iter().take(clear_count).collect(),
        images: clearable_images.into_iter().take(image_clear_count).collect(),
    }
}

/// Fast token estimate for a [`MicrocompactPlan`] against the session it was
/// planned from: cleared bytes / 4 for each tool result plus 1,600 per image
/// (tool-result-embedded or standalone) — the same arithmetic
/// [`microcompact_session`] used to report `estimated_tokens_saved` before
/// this was factored out. Must be called before any mutation of `session`
/// consumes the plan's indices.
fn plan_estimated_tokens(session: &Session, plan: &MicrocompactPlan) -> u64 {
    let mut estimated = 0u64;
    for &(message_index, block_index) in &plan.tool_results {
        if let ContentBlock::ToolResult { output, images, .. } =
            &session.messages[message_index].blocks[block_index]
        {
            estimated += output.len().saturating_sub(MICROCOMPACT_PLACEHOLDER.len()) as u64 / 4
                + images.len() as u64 * 1_600;
        }
    }
    estimated += plan.images.len() as u64 * 1_600;
    estimated
}

/// Read-only token estimate of what [`microcompact_session`] would free on
/// `session` right now, without mutating anything. Exists so a caller can
/// decide whether firing is even worth it *before* paying for it: a
/// microcompact pass invalidates the prompt cache from its earliest cleared
/// block onward, so every firing re-bills the entire prefix up to that point
/// on the next request — a break-even check needs this number before the
/// clear happens, not after.
#[must_use]
pub fn microcompact_clearable_estimate(
    session: &Session,
    keep_recent: usize,
    min_output_bytes: usize,
) -> u64 {
    let plan = plan_microcompact_clears(session, keep_recent, min_output_bytes);
    plan_estimated_tokens(session, &plan)
}

/// Tier-1 context trim (Claude Code "microcompact" parity): replace the
/// bodies of OLD tool results with a short placeholder, keeping the most
/// recent `keep_recent` results and every `tool_use` block intact — the model
/// still sees that it made each call (and with what input), only the bulky
/// response body is gone. No LLM round-trip, no message removal; strictly
/// cheaper than full compaction, and idempotent (a cleared result is never
/// counted as clearable again).
///
/// Results whose body is already small (`< min_output_bytes`) are left alone
/// unless they carry images — clearing a tiny body saves nothing and costs
/// information, while each image is ~1,600 estimated tokens. Results from
/// file-mutation tools (`edit_file`/`write_file` and friends, per
/// [`is_edit_result_tool`]) are *always* exempt: their body is the applied
/// diff, which — unlike a re-readable `read_file` — cannot be reconstructed and
/// is the only in-context evidence that the change was already made.
///
/// Candidate selection is shared with [`microcompact_clearable_estimate`] via
/// [`plan_microcompact_clears`] (never duplicated), so callers that gate on
/// the estimate before calling this are guaranteed to get exactly the batch
/// they were quoted.
pub fn microcompact_session(
    session: &mut Session,
    keep_recent: usize,
    min_output_bytes: usize,
) -> Option<MicrocompactEvent> {
    let plan = plan_microcompact_clears(session, keep_recent, min_output_bytes);
    if plan.tool_results.is_empty() && plan.images.is_empty() {
        return None;
    }
    let estimated_tokens_saved = plan_estimated_tokens(session, &plan);

    let messages = std::sync::Arc::make_mut(&mut session.messages);
    for &(message_index, block_index) in &plan.tool_results {
        if let ContentBlock::ToolResult { output, images, .. } =
            &mut messages[message_index].blocks[block_index]
        {
            *output = MICROCOMPACT_PLACEHOLDER.to_string();
            images.clear();
        }
    }
    for &(message_index, block_index) in &plan.images {
        let block = &mut messages[message_index].blocks[block_index];
        if matches!(block, ContentBlock::Image { .. }) {
            *block = ContentBlock::Text {
                text: MICROCOMPACT_IMAGE_PLACEHOLDER.to_string(),
            };
        }
    }
    session.mark_transcript_dirty();
    Some(MicrocompactEvent {
        cleared_results: plan.tool_results.len(),
        cleared_images: plan.images.len(),
        estimated_tokens_saved,
    })
}

fn extract_tag_block(content: &str, tag: &str) -> Option<String> {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    let start_index = content.find(&start)? + start.len();
    let end_index = content[start_index..].find(&end)? + start_index;
    Some(content[start_index..end_index].to_string())
}

fn strip_tag_block(content: &str, tag: &str) -> String {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    if let (Some(start_index), Some(end_index_rel)) = (content.find(&start), content.find(&end)) {
        let end_index = end_index_rel + end.len();
        let mut stripped = String::new();
        stripped.push_str(&content[..start_index]);
        stripped.push_str(&content[end_index..]);
        stripped
    } else {
        content.to_string()
    }
}

fn collapse_blank_lines(content: &str) -> String {
    let mut result = String::new();
    let mut last_blank = false;
    for line in content.lines() {
        let is_blank = line.trim().is_empty();
        if is_blank && last_blank {
            continue;
        }
        result.push_str(line);
        result.push('\n');
        last_blank = is_blank;
    }
    result
}

fn extract_existing_compacted_summary(message: &ConversationMessage) -> Option<String> {
    if message.role != MessageRole::System {
        return None;
    }

    let text = first_text_block(message)?;
    let summary = text.strip_prefix(COMPACT_CONTINUATION_PREAMBLE)?;
    let summary = summary
        .split_once(&format!("\n\n{COMPACT_RECENT_MESSAGES_NOTE}"))
        .map_or(summary, |(value, _)| value);
    let summary = summary
        .split_once(&format!("\n{COMPACT_DIRECT_RESUME_INSTRUCTION}"))
        .map_or(summary, |(value, _)| value);
    Some(summary.trim().to_string())
}

#[cfg(test)]
mod tests;
