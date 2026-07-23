//! `session_recall` — cross-session selective read (read-only).
//!
//! Lets the model pull context out of a PRIOR conversation without resuming it:
//!  - **recall** a specific session (`session_ref`), optionally filtered by a
//!    substring `query`, a `role`, and/or a `last_n` tail; or
//!  - **search** across every saved session (omit `session_ref`, pass `query`)
//!    to find which past chat discussed something, with per-session match counts.
//!
//! It never touches the live session — it loads transcripts from disk via
//! `runtime::session_control` and returns only the matched slice as text. This is
//! the selective counterpart to `/resume`, which swaps in an entire prior
//! conversation wholesale.
//!
//! Recall also surfaces the **compaction vault**: the raw messages summarized
//! out of a session are recovered from its append-only `<id>.vault.jsonl`
//! sidecar and folded into the result (tagged `[evicted …]`). That makes
//! compaction non-destructive in practice — the model can pull back an exact
//! detail that was summarized away, which a lossy summary alone cannot provide.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use super::{ToolContext, ToolError};
use runtime::session::{ContentBlock, ConversationMessage, MessageRole, Session};
use runtime::session_control::{
    is_session_reference_alias, list_managed_sessions_for, load_managed_session_for_excluding,
};

#[derive(Debug, Default, Deserialize)]
pub(crate) struct SessionRecallInput {
    /// Which prior session to read: a session id, an alias ("latest" / "last"
    /// / "recent"), or "current" (THIS session — used to pull raw originals a
    /// compaction round sealed to the vault). In a live session, aliases skip the
    /// current session when the current session id is available; "current" does
    /// not skip it. Omit it (or pass "all" / "*") together with `query` to SEARCH
    /// across every saved session instead of reading one.
    pub session_ref: Option<String>,
    /// Case-insensitive substring to match against message text. Required in
    /// search mode; in recall mode, absent means "return the tail".
    pub query: Option<String>,
    /// Restrict to a single role: "user", "assistant", "tool", or "system".
    pub role: Option<String>,
    /// Return only the last N matching messages (most recent kept).
    pub last_n: Option<usize>,
    /// Lower bound (inclusive) of the seq window to recall. Seqs are one
    /// monotonic domain across a session: an evicted vault record's `vault_seq`
    /// and a live message's absolute index (`first_message_index + i`) share it,
    /// so a `[seq_from, seq_to]` window addresses both. Combined with query/role
    /// (logical AND).
    pub seq_from: Option<u32>,
    /// Upper bound (inclusive) of the seq window to recall. `seq_from > seq_to`
    /// is rejected.
    pub seq_to: Option<u32>,
    /// Include tool-result blocks in the rendered output (default true). Set
    /// false to omit them — a message left with no other content is skipped, and
    /// the omission is noted in the header.
    pub include_tool_results: Option<bool>,
    /// SEARCH-mode only: keep sessions modified within the last N days (a recent
    /// lower bound on the window). Fractional days are honored (0.5 = last 12h).
    /// Applied as a prefilter over session mtimes before the scan cap, so the
    /// window also cuts how many transcripts are re-loaded.
    pub since_days: Option<f64>,
    /// SEARCH-mode only: keep sessions modified more than N days ago (an older
    /// upper bound on the window). Pair with `since_days` to bracket a span, e.g.
    /// `since_days: 7, before_days: 2` = "2 to 7 days ago". A window that asks for
    /// sessions both newer than `since_days` and older than `before_days` days
    /// (`since_days < before_days`) is empty and rejected.
    pub before_days: Option<f64>,
}

// Caps — all surfaced in the output when hit, never silently truncated.
const DEFAULT_RECALL_TAIL: usize = 30; // messages returned when neither query nor last_n is given
const MAX_RECALL_MESSAGES: usize = 200; // hard ceiling on recalled messages
const MAX_SEARCH_SESSIONS: usize = 100; // sessions scanned in search mode
const MAX_RENDERED_CHARS: usize = 4000; // per-message render cap
const SNIPPET_CHARS: usize = 160; // search-result snippet length

pub(crate) fn run_session_recall(
    input: SessionRecallInput,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    let base_dir = recall_base_dir(ctx);
    let role = match input.role.as_deref() {
        Some(r) => Some(parse_role(r)?),
        None => None,
    };
    if input.last_n == Some(0) {
        return Err(ToolError::InvalidInput(
            "session_recall: last_n must be >= 1".to_owned(),
        ));
    }
    if let (Some(from), Some(to)) = (input.seq_from, input.seq_to) {
        if from > to {
            return Err(ToolError::InvalidInput(format!(
                "session_recall: seq_from ({from}) must be <= seq_to ({to})"
            )));
        }
    }
    let include_tool_results = input.include_tool_results.unwrap_or(true);
    let query_lc = input
        .query
        .as_deref()
        .map(|q| q.trim().to_lowercase())
        .filter(|q| !q.is_empty());

    let is_search = input
        .session_ref
        .as_deref()
        .map(str::trim)
        .is_none_or(|r| r.is_empty() || r == "all" || r == "*");

    // Time filters narrow which SESSIONS the search scans; a single-session
    // recall already targets one transcript by id/alias, so they have no meaning
    // there — reject rather than silently ignore.
    if !is_search && (input.since_days.is_some() || input.before_days.is_some()) {
        return Err(ToolError::InvalidInput(
            "session_recall: since_days/before_days apply to search mode only (omit `session_ref`)"
                .to_owned(),
        ));
    }
    validate_time_window(input.since_days, input.before_days)?;

    if is_search {
        let query = query_lc.ok_or_else(|| {
            ToolError::InvalidInput(
                "session_recall: provide `query` to search across sessions, or `session_ref` \
                 to recall a specific one"
                    .to_owned(),
            )
        })?;
        run_search(&base_dir, &query, role, input.since_days, input.before_days)
    } else {
        let reference = input.session_ref.unwrap_or_default();
        let reference = reference.trim();
        validate_session_ref(reference)?;
        // "current" resolves to THIS session (no exclusion) so the model can pull
        // back raw originals the just-finished compaction round sealed to this
        // session's vault — the recall affordance the continuation message emits.
        // Every other alias keeps EXCLUDING the current session.
        let (resolved_reference, exclude_session_id) = if reference.eq_ignore_ascii_case("current") {
            let id = ctx.session_id().ok_or_else(|| {
                ToolError::InvalidInput(
                    "session_recall: no active persisted session — pass an explicit session id"
                        .to_owned(),
                )
            })?;
            (id, None)
        } else {
            let exclude = is_session_reference_alias(reference)
                .then(|| ctx.session_id())
                .flatten();
            (reference.to_owned(), exclude)
        };
        run_recall(
            &base_dir,
            &resolved_reference,
            query_lc.as_deref(),
            role,
            input.last_n,
            exclude_session_id.as_deref(),
            input.seq_from,
            input.seq_to,
            include_tool_results,
        )
    }
}

/// Confine recall to a session id or alias resolved under the project's own
/// `.zo/sessions`. Reject path-like references (separators, absolute paths,
/// or `..`) so a model cannot aim the loader at a transcript outside the
/// workspace — unlike `read_file`, this tool has no other boundary check, and
/// transcripts routinely contain pasted secrets. Ids are discoverable via
/// search mode, so dropping raw-path support costs the model nothing.
fn validate_session_ref(reference: &str) -> Result<(), ToolError> {
    let path_like = reference.is_empty()
        || reference.contains('/')
        || reference.contains('\\')
        || reference.contains("..")
        // A bare filename with an extension ("incident.jsonl") has no separators
        // but is still a path the loader would honor, reaching any transcript in
        // the workspace (and its vault). Zo session ids and the aliases have
        // no dots, so rejecting `.` confines recall to real ids/aliases.
        || reference.contains('.')
        || std::path::Path::new(reference).is_absolute();
    if path_like {
        return Err(ToolError::InvalidInput(format!(
            "session_recall: `{reference}` is not a valid session reference — use a session id \
             or \"latest\"/\"last\"/\"recent\" (run a search to discover ids); paths are not allowed"
        )));
    }
    Ok(())
}

/// The directory whose `.zo/sessions` holds the transcripts. Honors the
/// agent's pinned cwd / workspace root (worktree isolation), falling back to the
/// process cwd so a plain run still finds the project's sessions.
fn recall_base_dir(ctx: &ToolContext) -> PathBuf {
    ctx.cwd
        .clone()
        .or_else(|| ctx.workspace_root.clone())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

#[allow(clippy::too_many_arguments)]
fn run_recall(
    base_dir: &std::path::Path,
    reference: &str,
    query_lc: Option<&str>,
    role: Option<MessageRole>,
    last_n: Option<usize>,
    exclude_session_id: Option<&str>,
    seq_from: Option<u32>,
    seq_to: Option<u32>,
    include_tool_results: bool,
) -> Result<String, ToolError> {
    let loaded = load_managed_session_for_excluding(base_dir, reference, exclude_session_id)
        .map_err(|error| ToolError::Execution(format!("session_recall: {error}")))?;
    let session = loaded.session;
    // Raw messages compacted out of this session, recovered from the append-only
    // vault — the context Claude Code permanently loses on compaction. They are
    // chronologically older than the live transcript (which now holds only the
    // summary + recent tail), so they lead the combined view, letting the model
    // pull back an exact detail that was summarized away.
    let vault = session.read_vault();
    let evicted_total = vault.len();
    let total = evicted_total + session.messages.len();
    // Live messages address the same monotonic seq domain the vault records use:
    // a live message at local index `i` has absolute seq `first_message_index + i`.
    let first_live_seq = session.first_message_index();

    let seq_in_range = |seq: u32| {
        seq_from.is_none_or(|from| seq >= from) && seq_to.is_none_or(|to| seq <= to)
    };
    let matches_filters = |msg: &ConversationMessage| {
        role.is_none_or(|want| msg.role == want)
            && query_lc.is_none_or(|q| message_plain(msg).to_lowercase().contains(q))
    };

    // Recovered originals are ALWAYS surfaced (the whole point of recall) —
    // never dropped by the recent-tail cap, which only ever applied to live
    // messages. They lead the output, tagged `[evicted …]`.
    let mut evicted: Vec<&ConversationMessage> = vault
        .iter()
        .filter(|record| seq_in_range(record.vault_seq))
        .map(|record| &record.message)
        .filter(|msg| matches_filters(msg))
        .collect();

    // Live transcript. When the vault recovered the originals, the index-0
    // compaction summary is the LOSSY restatement of those same messages —
    // suppress it so the model is not handed both the raw batch and its summary.
    let suppress_summary = evicted_total > 0;
    let mut live: Vec<&ConversationMessage> = session
        .messages
        .iter()
        .enumerate()
        .filter(|(index, msg)| {
            let seq = first_live_seq.saturating_add(u32::try_from(*index).unwrap_or(u32::MAX));
            !(suppress_summary && *index == 0 && msg.role == MessageRole::System)
                && seq_in_range(seq)
        })
        .map(|(_, msg)| msg)
        .filter(|msg| matches_filters(msg))
        .collect();

    let matched_count = evicted.len() + live.len();

    // The recent-tail / `last_n` cap applies to the LIVE transcript only. With
    // no `last_n` and no query a default tail keeps an un-narrowed recall from
    // dumping the whole transcript; `last_n` (>= 1, validated) is clamped to the
    // hard ceiling.
    let live_limit = match (last_n, query_lc.is_some()) {
        (Some(n), _) => n.min(MAX_RECALL_MESSAGES),
        (None, false) => DEFAULT_RECALL_TAIL,
        (None, true) => MAX_RECALL_MESSAGES,
    };
    let mut notes = String::new();
    if live.len() > live_limit {
        notes = match last_n {
            Some(n) if n > MAX_RECALL_MESSAGES => {
                format!(" (live: requested last {n}, capped at {MAX_RECALL_MESSAGES})")
            }
            Some(n) => format!(" (live: showing last {n})"),
            None if query_lc.is_some() => format!(" (live: capped at {MAX_RECALL_MESSAGES})"),
            None => format!(" (live: showing last {DEFAULT_RECALL_TAIL})"),
        };
        let cut = live.len() - live_limit;
        live.drain(..cut);
    }
    // Bound the recovered set too (keep the most-recent = highest seq) so a
    // pathologically large vault can't dump unbounded.
    if evicted.len() > MAX_RECALL_MESSAGES {
        let cut = evicted.len() - MAX_RECALL_MESSAGES;
        evicted.drain(..cut);
        let _ = write!(notes, " (evicted: showing last {MAX_RECALL_MESSAGES})");
    }

    let filt = describe_filter(query_lc, role, seq_from, seq_to);
    let recovered = if evicted_total > 0 {
        format!(
            " · {} message(s) recovered from the compaction vault",
            evicted.len()
        )
    } else {
        String::new()
    };
    let omitted_note = if include_tool_results {
        ""
    } else {
        " · tool results omitted from render"
    };
    let mut out = format!(
        "Recalled from session `{}` — {matched_count} of {total} message(s) match{filt}{notes}{recovered}{omitted_note}.\n",
        loaded.handle.id,
    );
    if evicted.is_empty() && live.is_empty() {
        out.push_str("\n(no messages matched)");
        return Ok(out);
    }
    // Render each message, skipping any left with no content once tool results
    // are omitted (a tool-result-only message renders to nothing).
    for msg in &evicted {
        if let Some(rendered) = render_message(msg, true, include_tool_results) {
            out.push('\n');
            out.push_str(&rendered);
            out.push('\n');
        }
    }
    for msg in &live {
        if let Some(rendered) = render_message(msg, false, include_tool_results) {
            out.push('\n');
            out.push_str(&rendered);
            out.push('\n');
        }
    }
    Ok(out)
}

fn run_search(
    base_dir: &std::path::Path,
    query_lc: &str,
    role: Option<MessageRole>,
    since_days: Option<f64>,
    before_days: Option<f64>,
) -> Result<String, ToolError> {
    let mut summaries = list_managed_sessions_for(base_dir)
        .map_err(|error| ToolError::Execution(format!("session_recall: {error}")))?;
    // `list_managed_sessions_for` already drops adjacent same-(mtime,id)
    // duplicates (e.g. its doubled search dir), but two distinct files sharing
    // one internal session_id with differing mtimes — a legacy `<id>.json` plus a
    // migrated `<id>.jsonl` — are not adjacent after its (mtime desc) sort and
    // both survive. Dedup by id here so such a session is searched once.
    let mut seen = std::collections::BTreeSet::new();
    summaries.retain(|s| seen.insert(s.id.clone()));

    // Apply the time window on session mtimes BEFORE the scan cap so it also cuts
    // how many transcripts are re-loaded below: an out-of-window session is
    // dropped here instead of consuming a `Session::load_from_path`. `since_days`
    // is a recent lower bound (kept when modified >= now − since_days), `before_days`
    // an older upper bound (kept when modified <= now − before_days).
    let now_ms = now_epoch_millis();
    let since_cutoff = since_days.map(|days| now_ms.saturating_sub(days_to_millis(days)));
    let before_cutoff = before_days.map(|days| now_ms.saturating_sub(days_to_millis(days)));
    summaries.retain(|summary| {
        since_cutoff.is_none_or(|lo| summary.modified_epoch_millis >= lo)
            && before_cutoff.is_none_or(|hi| summary.modified_epoch_millis <= hi)
    });

    let total_sessions = summaries.len();
    let truncated = total_sessions > MAX_SEARCH_SESSIONS;
    summaries.truncate(MAX_SEARCH_SESSIONS);

    let mut hits: Vec<(String, usize, String)> = Vec::new();
    for summary in &summaries {
        let Ok(session) = Session::load_from_path(&summary.path) else {
            continue; // skip unreadable/corrupt transcripts (best-effort search)
        };
        // Search the compaction vault too, so a term that was summarized out of a
        // session is still discoverable — otherwise the lossless recovery is
        // invisible on the primary discovery path.
        let vault = session.read_vault();
        let mut count = 0usize;
        let mut snippet = String::new();
        for msg in vault
            .iter()
            .map(|record| &record.message)
            .chain(session.messages.iter())
        {
            if !role.is_none_or(|want| msg.role == want) {
                continue;
            }
            let plain = message_plain(msg);
            if plain.to_lowercase().contains(query_lc) {
                count += 1;
                if snippet.is_empty() {
                    snippet = make_snippet(&plain, query_lc);
                }
            }
        }
        if count > 0 {
            hits.push((summary.id.clone(), count, snippet));
        }
    }

    let role_note = role
        .map(|r| format!(", role={}", role_label(r)))
        .unwrap_or_default();
    let window_note = describe_time_window(since_days, before_days);
    let scanned = summaries.len();
    let mut out = format!(
        "Searched {scanned} session(s) for \"{query_lc}\"{role_note}{window_note} — {} with matches (most recent first):\n",
        hits.len(),
    );
    if truncated {
        let _ = writeln!(
            out,
            "(scanned the {MAX_SEARCH_SESSIONS} most recent of {total_sessions} sessions)"
        );
    }
    if hits.is_empty() {
        out.push_str("\nNo saved session matched. Try a broader query.");
        return Ok(out);
    }
    for (id, count, snippet) in &hits {
        let _ = write!(out, "\n• `{id}` — {count} match(es): {snippet}");
    }
    let _ = write!(
        out,
        "\n\nRecall full context with session_recall {{\"session_ref\":\"<id>\",\"query\":\"{query_lc}\"}}.",
    );
    Ok(out)
}

const MILLIS_PER_DAY: f64 = 86_400_000.0;

/// Reject a nonsensical time window up front (search mode). Each bound must be a
/// finite, non-negative day count; `since_days < before_days` asks for sessions
/// both newer than `since_days` and older than `before_days` days — an empty
/// window (mirrors the `seq_from > seq_to` guard).
fn validate_time_window(since_days: Option<f64>, before_days: Option<f64>) -> Result<(), ToolError> {
    for (label, value) in [("since_days", since_days), ("before_days", before_days)] {
        if let Some(days) = value {
            if !days.is_finite() || days < 0.0 {
                return Err(ToolError::InvalidInput(format!(
                    "session_recall: {label} ({days}) must be a finite, non-negative number of days"
                )));
            }
        }
    }
    if let (Some(since), Some(before)) = (since_days, before_days) {
        if since < before {
            return Err(ToolError::InvalidInput(format!(
                "session_recall: since_days ({since}) < before_days ({before}) is an empty window \
                 — asks for sessions both newer than {since} and older than {before} days"
            )));
        }
    }
    Ok(())
}

/// Now as epoch milliseconds, matching `ManagedSessionSummary.modified_epoch_millis`.
/// A clock before the epoch (unreachable in practice) yields 0, which disables the
/// lower bound rather than panicking.
fn now_epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or_default()
}

/// Convert a (validated finite, non-negative) day count to milliseconds, rounded.
/// `validate_time_window` already rejected negative/non-finite inputs and the
/// `.max(0.0)` floors any residue, so the cast neither wraps a negative nor
/// truncates meaningfully (day counts are far below `u128`'s range).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn days_to_millis(days: f64) -> u128 {
    (days * MILLIS_PER_DAY).round().max(0.0) as u128
}

/// A short header note describing the active time window, or empty when neither
/// bound is set. `since_days` is the recent edge, `before_days` the older edge.
fn describe_time_window(since_days: Option<f64>, before_days: Option<f64>) -> String {
    match (since_days, before_days) {
        (Some(since), Some(before)) => {
            format!(", modified {before}–{since} day(s) ago")
        }
        (Some(since), None) => format!(", modified within the last {since} day(s)"),
        (None, Some(before)) => format!(", modified more than {before} day(s) ago"),
        (None, None) => String::new(),
    }
}

fn parse_role(role: &str) -> Result<MessageRole, ToolError> {
    match role.trim().to_lowercase().as_str() {
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" => Ok(MessageRole::Tool),
        "system" => Ok(MessageRole::System),
        other => Err(ToolError::InvalidInput(format!(
            "session_recall: unknown role '{other}' (use user/assistant/tool/system)"
        ))),
    }
}

fn role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn describe_filter(
    query_lc: Option<&str>,
    role: Option<MessageRole>,
    seq_from: Option<u32>,
    seq_to: Option<u32>,
) -> String {
    let mut parts = Vec::new();
    if let Some(q) = query_lc {
        parts.push(format!("query \"{q}\""));
    }
    if let Some(r) = role {
        parts.push(format!("role {}", role_label(r)));
    }
    match (seq_from, seq_to) {
        (Some(from), Some(to)) => parts.push(format!("seq {from}-{to}")),
        (Some(from), None) => parts.push(format!("seq >= {from}")),
        (None, Some(to)) => parts.push(format!("seq <= {to}")),
        (None, None) => {}
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" ({})", parts.join(", "))
    }
}

/// Flatten a message's content blocks into searchable/renderable plain text.
/// Tool calls/results and images are condensed to short tags so the search still
/// matches their text without dumping base64 or huge JSON.
fn message_plain(msg: &ConversationMessage) -> String {
    message_plain_with(msg, true)
}

/// Like [`message_plain`], but omits `ToolResult` blocks when
/// `include_tool_results` is false (used by render when the caller asked to drop
/// tool output). With `include_tool_results = true` the output is byte-identical
/// to [`message_plain`], so search/query matching is unchanged.
fn message_plain_with(msg: &ConversationMessage, include_tool_results: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in &msg.blocks {
        match block {
            ContentBlock::Text { text } => parts.push(text.clone()),
            ContentBlock::ToolUse { name, input, .. } => {
                parts.push(format!("[tool call: {name}] {input}"));
            }
            ContentBlock::ToolResult {
                tool_name,
                output,
                is_error,
                ..
            } => {
                if !include_tool_results {
                    continue;
                }
                let tag = if *is_error {
                    "tool error"
                } else {
                    "tool result"
                };
                parts.push(format!("[{tag}: {tool_name}] {output}"));
            }
            ContentBlock::Image { media_type, .. } => {
                parts.push(format!("[image: {media_type}]"));
            }
            // Reasoning blocks are internal; surface only a marker in recall text.
            ContentBlock::Thinking { .. } => parts.push("[thinking]".to_owned()),
            ContentBlock::RedactedThinking { .. } => parts.push("[redacted thinking]".to_owned()),
        }
    }
    parts.join("\n")
}

/// Render one recalled message, or `None` when omitting tool results leaves it
/// with no content (a tool-result-only message asked to drop tool output).
fn render_message(
    msg: &ConversationMessage,
    evicted: bool,
    include_tool_results: bool,
) -> Option<String> {
    let plain = message_plain_with(msg, include_tool_results);
    if plain.trim().is_empty() {
        return None;
    }
    let body = truncate_chars(&plain, MAX_RENDERED_CHARS);
    // Tag vault-recovered originals so the model knows it is reading the exact
    // pre-compaction content, not the lossy summary that replaced it.
    let tag = if evicted { "evicted " } else { "" };
    Some(format!("[{tag}{}] {body}", role_label(msg.role)))
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}… [truncated]")
    }
}

/// A one-line snippet of `plain` centered on the (case-insensitive) match, for
/// search results. The match position is found in the lowercased text, then
/// mapped back to a char offset in the original — because `to_lowercase` is not
/// 1:1 (e.g. Turkish 'İ' expands to two chars), slicing the original at the
/// lowercased offset would mis-center or drop the match. Char-based throughout,
/// so it never splits a UTF-8 boundary; preserves the original casing.
fn make_snippet(plain: &str, query_lc: &str) -> String {
    let collapsed: String = plain.split_whitespace().collect::<Vec<_>>().join(" ");
    let total = collapsed.chars().count();
    let lc = collapsed.to_lowercase();
    let Some((byte, _)) = lc.match_indices(query_lc).next() else {
        // Match spanned whitespace the collapse removed (rare) — show the head.
        return one_line_window(&collapsed, 0, total);
    };
    let lc_match_char = lc[..byte].chars().count();
    // Map the lowercased char offset back onto `collapsed`: walk its chars,
    // accumulating how many lowercased chars each produces, until we reach the
    // match's lowercased offset.
    let mut lc_seen = 0usize;
    let mut match_char = total;
    for (idx, ch) in collapsed.chars().enumerate() {
        if lc_seen >= lc_match_char {
            match_char = idx;
            break;
        }
        lc_seen += ch.to_lowercase().count();
    }
    let start = match_char.saturating_sub(SNIPPET_CHARS / 4);
    one_line_window(&collapsed, start, total)
}

/// A `SNIPPET_CHARS`-wide char window of `text` starting at char `start`, with
/// leading/trailing ellipses when content is elided.
fn one_line_window(text: &str, start: usize, total: usize) -> String {
    let window: String = text.chars().skip(start).take(SNIPPET_CHARS).collect();
    let prefix = if start > 0 { "…" } else { "" };
    let suffix = if total > start + SNIPPET_CHARS {
        "…"
    } else {
        ""
    };
    format!("{prefix}{window}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn msg(role: MessageRole, text: &str) -> ConversationMessage {
        ConversationMessage {
            role,
            blocks: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            usage: None,
            thought_signature: None,
            reasoning_replay: None,
                    model: None,
        }
    }

    /// Write a session transcript under `<base>/.zo/sessions/<id>.jsonl`.
    fn write_session(base: &std::path::Path, id: &str, messages: Vec<ConversationMessage>) {
        let dir = base.join(".zo").join("sessions");
        std::fs::create_dir_all(&dir).expect("mk sessions dir");
        let mut session = Session::new();
        session.session_id = id.to_owned();
        session.messages = Arc::new(messages);
        session
            .save_to_path(dir.join(format!("{id}.jsonl")))
            .expect("save session");
    }

    fn temp_base(tag: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let base = std::env::temp_dir().join(format!("zo-g19-{tag}-{unique}"));
        std::fs::create_dir_all(&base).expect("mk base");
        base
    }

    fn ctx_for(base: &std::path::Path) -> ToolContext {
        ToolContext::new().with_cwd(base.to_path_buf())
    }

    #[test]
    fn recall_filters_by_query_and_role() {
        let base = temp_base("recall");
        write_session(
            &base,
            "sess-A",
            vec![
                msg(MessageRole::User, "how do I fix the parser bug"),
                msg(MessageRole::Assistant, "the parser bug is in tokenize()"),
                msg(MessageRole::User, "thanks, unrelated chatter"),
            ],
        );
        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-A".into()),
                query: Some("parser".into()),
                role: Some("assistant".into()),
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");

        assert!(out.contains("session `sess-A`"), "{out}");
        assert!(
            out.contains("tokenize()"),
            "matched assistant line present: {out}"
        );
        assert!(
            !out.contains("unrelated chatter"),
            "non-matching filtered out: {out}"
        );
        assert!(
            !out.contains("how do I fix"),
            "wrong role filtered out: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn recall_without_query_returns_tail() {
        let base = temp_base("tail");
        let many: Vec<_> = (0..40)
            .map(|i| msg(MessageRole::User, &format!("line {i}")))
            .collect();
        write_session(&base, "sess-tail", many);
        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-tail".into()),
                query: None,
                role: None,
                last_n: Some(3),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");
        assert!(out.contains("line 39") && out.contains("line 37"), "{out}");
        assert!(!out.contains("line 36"), "only last 3: {out}");
        assert!(out.contains("showing last 3"), "tail note: {out}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn recall_recovers_evicted_messages_from_the_vault() {
        // A compacted session: the on-disk transcript holds only a summary, but
        // the raw original was sealed to the vault. Recall must surface the raw
        // content (tagged evicted) — the detail CC would have lost.
        let base = temp_base("vault-recall");
        let dir = base.join(".zo").join("sessions");
        std::fs::create_dir_all(&dir).expect("mk sessions dir");
        let session_path = dir.join("sess-vault.jsonl");

        let mut session = Session::new();
        session.session_id = "sess-vault".to_owned();
        let session = session.with_persistence_path(session_path.clone());
        let _ = session.seal_evicted_to_vault(&[msg(
            MessageRole::User,
            "the exact error was ECONNREFUSED on db/pool.rs:42",
        )]);
        let mut session = session;
        session.messages = Arc::new(vec![msg(
            MessageRole::System,
            "Summary: discussed a database connection issue",
        )]);
        session.save_to_path(&session_path).expect("save");

        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-vault".into()),
                query: Some("ECONNREFUSED".into()),
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");

        assert!(
            out.contains("ECONNREFUSED on db/pool.rs:42"),
            "raw evicted content recovered from vault: {out}"
        );
        assert!(out.contains("evicted"), "recovered content is tagged: {out}");
        assert!(
            out.contains("recovered from the compaction vault"),
            "header notes vault recovery: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn recall_keeps_evicted_even_when_live_tail_is_capped() {
        // Regression (Codex P2 #1): the recent-tail cap must NOT drop recovered
        // evicted content, and the index-0 compaction summary is suppressed when
        // the raw originals are recovered.
        let base = temp_base("vault-tailcut");
        let dir = base.join(".zo").join("sessions");
        std::fs::create_dir_all(&dir).expect("mk sessions dir");
        let session_path = dir.join("sess-tc.jsonl");

        let mut session = Session::new();
        session.session_id = "sess-tc".to_owned();
        let session = session.with_persistence_path(session_path.clone());
        let _ = session.seal_evicted_to_vault(&[msg(MessageRole::User, "EVICTED_MARKER one")]);
        let mut session = session;
        let mut live = vec![msg(MessageRole::System, "Summary: discussed stuff")];
        live.extend((0..40).map(|i| msg(MessageRole::User, &format!("live {i}"))));
        session.messages = Arc::new(live);
        session.save_to_path(&session_path).expect("save");

        // Default recall (no query, no last_n): the live tail is capped, but the
        // recovered evicted message must still appear.
        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-tc".into()),
                query: None,
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");

        assert!(
            out.contains("EVICTED_MARKER one"),
            "evicted content survives the live-tail cap: {out}"
        );
        assert!(out.contains("live 39"), "recent live tail still shown: {out}");
        assert!(!out.contains("live 0"), "old live messages capped out: {out}");
        assert!(
            !out.contains("Summary: discussed stuff"),
            "index-0 compaction summary suppressed when raw recovered: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn alias_recall_skips_current_session_when_session_id_is_known() {
        let base = temp_base("alias-skip-current");
        write_session(
            &base,
            "sess-prior",
            vec![msg(MessageRole::User, "prior session todo marker")],
        );
        std::thread::sleep(Duration::from_millis(5));
        write_session(
            &base,
            "sess-current",
            vec![msg(MessageRole::User, "current empty-session marker")],
        );
        let ctx = ctx_for(&base);
        ctx.set_session_id("sess-current");

        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("latest".into()),
                query: None,
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx,
        )
        .expect("alias recall ok");

        assert!(out.contains("session `sess-prior`"), "{out}");
        assert!(out.contains("prior session todo marker"), "{out}");
        assert!(!out.contains("current empty-session marker"), "{out}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn alias_recall_without_current_session_id_keeps_latest_behavior() {
        let base = temp_base("alias-no-current-id");
        write_session(
            &base,
            "sess-prior",
            vec![msg(MessageRole::User, "prior marker")],
        );
        std::thread::sleep(Duration::from_millis(5));
        write_session(
            &base,
            "sess-current",
            vec![msg(MessageRole::User, "latest marker")],
        );

        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("latest".into()),
                query: None,
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("alias recall ok");

        assert!(out.contains("session `sess-current`"), "{out}");
        assert!(out.contains("latest marker"), "{out}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn search_across_sessions_ranks_by_match() {
        let base = temp_base("search");
        write_session(
            &base,
            "chat-1",
            vec![msg(
                MessageRole::User,
                "the deadlock happens in the mutex guard",
            )],
        );
        write_session(
            &base,
            "chat-2",
            vec![
                msg(MessageRole::User, "a mutex deadlock again"),
                msg(MessageRole::Assistant, "yes the mutex is the cause"),
            ],
        );
        write_session(
            &base,
            "chat-3",
            vec![msg(MessageRole::User, "totally different topic")],
        );

        let out = run_session_recall(
            SessionRecallInput {
                session_ref: None,
                query: Some("mutex".into()),
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("search ok");

        assert!(out.contains("chat-1"), "{out}");
        assert!(out.contains("chat-2"), "{out}");
        assert!(
            !out.contains("chat-3"),
            "non-matching session omitted: {out}"
        );
        assert!(
            out.contains("2 with matches"),
            "two sessions matched: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn search_requires_a_query() {
        let base = temp_base("noquery");
        let err = run_session_recall(
            SessionRecallInput {
                session_ref: None,
                query: None,
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect_err("search without query is rejected");
        assert!(err.to_string().contains("provide `query`"), "{err}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn unknown_session_ref_errors_clearly() {
        let base = temp_base("missing");
        std::fs::create_dir_all(base.join(".zo").join("sessions")).expect("mk");
        let err = run_session_recall(
            SessionRecallInput {
                session_ref: Some("does-not-exist".into()),
                query: None,
                role: None,
                last_n: None,
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect_err("missing session errors");
        assert!(err.to_string().contains("session_recall:"), "{err}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn invalid_role_is_rejected() {
        let err = parse_role("captain").expect_err("bad role");
        assert!(err.to_string().contains("unknown role"), "{err}");
    }

    #[test]
    fn path_like_session_ref_is_rejected() {
        // A model-controlled ref must not escape the workspace via a path; only
        // ids/aliases are allowed (no disk access happens — rejected up front).
        for bad in [
            "../../etc/passwd",
            "/etc/passwd",
            "sub/dir/x",
            "..\\win",
            "incident.jsonl",
            "secrets.json",
        ] {
            let err = run_session_recall(
                SessionRecallInput {
                    session_ref: Some(bad.to_owned()),
                    query: None,
                    role: None,
                    last_n: None,
                    ..Default::default()
                },
                &ToolContext::new(),
            )
            .expect_err("path-like ref rejected");
            assert!(
                err.to_string().contains("not a valid session reference"),
                "ref {bad}: {err}"
            );
        }
    }

    #[test]
    fn last_n_zero_is_rejected() {
        let err = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-x".into()),
                query: None,
                role: None,
                last_n: Some(0),
                ..Default::default()
            },
            &ToolContext::new(),
        )
        .expect_err("last_n=0 rejected");
        assert!(err.to_string().contains("last_n must be >= 1"), "{err}");
    }

    #[test]
    fn dual_cap_note_reports_both_request_and_ceiling() {
        // last_n above the hard ceiling must surface BOTH numbers, not silently
        // relabel as the global cap (the cap-clobber bug).
        let base = temp_base("dualcap");
        let many: Vec<_> = (0..MAX_RECALL_MESSAGES + 5)
            .map(|i| msg(MessageRole::User, &format!("match {i}")))
            .collect();
        write_session(&base, "sess-big", many);
        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-big".into()),
                query: Some("match".into()),
                role: None,
                last_n: Some(MAX_RECALL_MESSAGES + 50),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");
        assert!(
            out.contains(&format!("requested last {}", MAX_RECALL_MESSAGES + 50)),
            "note keeps the requested value: {}",
            out.lines().next().unwrap_or_default()
        );
        assert!(
            out.contains(&format!("capped at {MAX_RECALL_MESSAGES}")),
            "note keeps the ceiling: {}",
            out.lines().next().unwrap_or_default()
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn snippet_centers_on_match_with_case_expanding_chars() {
        // Turkish 'İ' lowercases to TWO chars; the lowercased match offset must be
        // mapped back to the original so the snippet still contains the match.
        let text = format!("{} CRITICALBUG happened here", "İ".repeat(60));
        let snippet = make_snippet(&text, "criticalbug");
        assert!(
            snippet.to_lowercase().contains("criticalbug"),
            "snippet must include the matched term: {snippet}"
        );
    }

    #[test]
    fn message_plain_condenses_tool_and_image_blocks() {
        let m = ConversationMessage {
            role: MessageRole::Assistant,
            blocks: vec![
                ContentBlock::Text {
                    text: "running it".into(),
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: "{\"command\":\"ls\"}".into(),
                },
                ContentBlock::Image {
                    media_type: "image/png".into(),
                    data: "QUJD".into(),
                },
            ],
            usage: None,
            thought_signature: None,
            reasoning_replay: None,
                    model: None,
        };
        let plain = message_plain(&m);
        assert!(plain.contains("running it"));
        assert!(plain.contains("[tool call: bash]"));
        assert!(plain.contains("[image: image/png]"));
        assert!(
            !plain.contains("QUJD"),
            "base64 must not be dumped: {plain}"
        );
    }

    fn tool_result_msg(id: &str, name: &str, output: &str) -> ConversationMessage {
        ConversationMessage {
            role: MessageRole::Tool,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: id.to_owned(),
                tool_name: name.to_owned(),
                output: output.to_owned(),
                is_error: false,
                images: Vec::new(),
            }],
            usage: None,
            thought_signature: None,
            reasoning_replay: None,
                    model: None,
        }
    }

    #[test]
    fn recall_seq_range_filters_across_evicted_and_live_boundary() {
        // A compacted session: seqs 0-2 live in the vault (evicted), seqs 3-5 are
        // the live tail. A [1, 3] window must span the boundary: evicted 1-2 plus
        // live 3, excluding evicted 0 and live 4-5.
        let base = temp_base("seq-range");
        let dir = base.join(".zo").join("sessions");
        std::fs::create_dir_all(&dir).expect("mk sessions dir");
        let path = dir.join("sess-seq.jsonl");

        let mut session = Session::new();
        session.session_id = "sess-seq".to_owned();
        let session = session.with_persistence_path(path.clone());
        let _ = session.seal_evicted_to_vault(&[
            msg(MessageRole::User, "evicted zero"),
            msg(MessageRole::User, "evicted one"),
            msg(MessageRole::User, "evicted two"),
        ]);
        let mut session = session;
        session.messages = Arc::new(vec![
            msg(MessageRole::User, "live three"),
            msg(MessageRole::User, "live four"),
            msg(MessageRole::User, "live five"),
        ]);
        // Advances first_message_index 0 -> 3 and persists the snapshot, so the
        // live tail addresses seqs 3, 4, 5 on reload.
        session.record_compaction("summary of evicted", 3);

        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-seq".into()),
                seq_from: Some(1),
                seq_to: Some(3),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");

        assert!(!out.contains("evicted zero"), "seq 0 excluded: {out}");
        assert!(
            out.contains("evicted one") && out.contains("evicted two"),
            "evicted seqs 1-2 included: {out}"
        );
        assert!(out.contains("live three"), "live seq 3 included: {out}");
        assert!(
            !out.contains("live four") && !out.contains("live five"),
            "live seqs 4-5 excluded: {out}"
        );
        assert!(out.contains("seq 1-3"), "header names the seq window: {out}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn recall_can_omit_tool_results() {
        let base = temp_base("omit-tools");
        write_session(
            &base,
            "sess-omit",
            vec![
                msg(MessageRole::User, "please run the build"),
                tool_result_msg("t1", "bash", "BUILD_OUTPUT_XYZ compiled 42 files"),
            ],
        );

        let with = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-omit".into()),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");
        assert!(
            with.contains("BUILD_OUTPUT_XYZ"),
            "tool result shown by default: {with}"
        );

        let without = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-omit".into()),
                include_tool_results: Some(false),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("recall ok");
        assert!(
            !without.contains("BUILD_OUTPUT_XYZ"),
            "tool result omitted: {without}"
        );
        assert!(
            without.contains("please run the build"),
            "text message still rendered: {without}"
        );
        assert!(
            without.contains("tool results omitted from render"),
            "omission noted in the header: {without}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn current_alias_recalls_this_session_without_excluding_it() {
        // `latest` excludes the current session (see `alias_recall_skips_current…`);
        // `current` must instead load THIS session so the model can pull back its
        // own just-compacted originals.
        let base = temp_base("current-alias");
        write_session(
            &base,
            "sess-current",
            vec![msg(MessageRole::User, "CURRENT_SESSION_MARKER present")],
        );
        let ctx = ctx_for(&base);
        ctx.set_session_id("sess-current");

        let out = run_session_recall(
            SessionRecallInput {
                session_ref: Some("current".into()),
                ..Default::default()
            },
            &ctx,
        )
        .expect("current recall ok");
        assert!(out.contains("session `sess-current`"), "{out}");
        assert!(out.contains("CURRENT_SESSION_MARKER present"), "{out}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn current_alias_without_active_session_id_errors() {
        let base = temp_base("current-no-id");
        let err = run_session_recall(
            SessionRecallInput {
                session_ref: Some("current".into()),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect_err("current with no active session id is rejected");
        assert!(
            err.to_string().contains("no active persisted session"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Stamp a written session file's mtime so the time-window prefilter (which
    /// reads on-disk mtimes via `list_managed_sessions_for`) can be exercised
    /// deterministically without a `filetime` dependency — `File::set_modified`
    /// is std.
    fn set_session_mtime(base: &std::path::Path, id: &str, mtime: SystemTime) {
        let path = base
            .join(".zo")
            .join("sessions")
            .join(format!("{id}.jsonl"));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open session for mtime");
        file.set_modified(mtime).expect("set mtime");
    }

    fn days_ago(days: u64) -> SystemTime {
        SystemTime::now() - Duration::from_secs(days * 86_400)
    }

    /// Write three sessions all matching "widget", aged 0 / 5 / 30 days.
    fn write_timed_widget_sessions(base: &std::path::Path) {
        for (id, age) in [("sess-recent", 0), ("sess-mid", 5), ("sess-old", 30)] {
            write_session(base, id, vec![msg(MessageRole::User, "the widget broke")]);
            set_session_mtime(base, id, days_ago(age));
        }
    }

    #[test]
    fn search_since_days_keeps_only_recent_sessions() {
        let base = temp_base("since-days");
        write_timed_widget_sessions(&base);
        let out = run_session_recall(
            SessionRecallInput {
                query: Some("widget".into()),
                since_days: Some(7.0),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("search ok");
        assert!(out.contains("sess-recent"), "age 0 within 7d: {out}");
        assert!(out.contains("sess-mid"), "age 5 within 7d: {out}");
        assert!(!out.contains("sess-old"), "age 30 excluded: {out}");
        assert!(
            out.contains("within the last 7 day(s)"),
            "header notes the window: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn search_before_days_keeps_only_old_sessions() {
        let base = temp_base("before-days");
        write_timed_widget_sessions(&base);
        let out = run_session_recall(
            SessionRecallInput {
                query: Some("widget".into()),
                before_days: Some(10.0),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("search ok");
        assert!(!out.contains("sess-recent"), "age 0 excluded: {out}");
        assert!(!out.contains("sess-mid"), "age 5 excluded: {out}");
        assert!(out.contains("sess-old"), "age 30 older than 10d: {out}");
        assert!(
            out.contains("more than 10 day(s) ago"),
            "header notes the window: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn search_time_window_band_brackets_a_span() {
        // since_days >= before_days is a valid band: 2..=20 days ago keeps only
        // the 5-day-old session (excludes the 0-day and 30-day ones).
        let base = temp_base("band-days");
        write_timed_widget_sessions(&base);
        let out = run_session_recall(
            SessionRecallInput {
                query: Some("widget".into()),
                since_days: Some(20.0),
                before_days: Some(2.0),
                ..Default::default()
            },
            &ctx_for(&base),
        )
        .expect("search ok");
        assert!(!out.contains("sess-recent"), "age 0 below band: {out}");
        assert!(out.contains("sess-mid"), "age 5 in band: {out}");
        assert!(!out.contains("sess-old"), "age 30 above band: {out}");
        assert!(
            out.contains("modified 2–20 day(s) ago"),
            "header names the band: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn inverted_time_window_is_rejected() {
        // since_days < before_days: newer than 2 days AND older than 20 days is
        // empty — rejected up front (mirrors the seq_from > seq_to guard).
        let err = run_session_recall(
            SessionRecallInput {
                query: Some("widget".into()),
                since_days: Some(2.0),
                before_days: Some(20.0),
                ..Default::default()
            },
            &ToolContext::new(),
        )
        .expect_err("empty window rejected");
        assert!(
            err.to_string().contains("empty window")
                && err.to_string().contains("since_days"),
            "{err}"
        );
    }

    #[test]
    fn negative_days_is_rejected() {
        let err = run_session_recall(
            SessionRecallInput {
                query: Some("widget".into()),
                since_days: Some(-1.0),
                ..Default::default()
            },
            &ToolContext::new(),
        )
        .expect_err("negative days rejected");
        assert!(err.to_string().contains("non-negative"), "{err}");
    }

    #[test]
    fn time_filters_are_rejected_in_recall_mode() {
        // A single-session recall targets one transcript by id; a session-mtime
        // window has no meaning there, so it is rejected rather than ignored.
        let err = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-x".into()),
                since_days: Some(3.0),
                ..Default::default()
            },
            &ToolContext::new(),
        )
        .expect_err("time filter in recall mode rejected");
        assert!(
            err.to_string().contains("search mode only"),
            "{err}"
        );
    }

    #[test]
    fn seq_from_greater_than_seq_to_is_rejected() {
        let err = run_session_recall(
            SessionRecallInput {
                session_ref: Some("sess-x".into()),
                seq_from: Some(10),
                seq_to: Some(5),
                ..Default::default()
            },
            &ToolContext::new(),
        )
        .expect_err("inverted seq window rejected");
        assert!(
            err.to_string().contains("seq_from") && err.to_string().contains("seq_to"),
            "{err}"
        );
    }
}
