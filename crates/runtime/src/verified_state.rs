//! Session-scope verified-state ledger — what the harness ITSELF observed
//! about this session's verification, so the model does not re-derive it.
//!
//! ## The measured waste
//!
//! In long multi-stage sessions the model rebuilds an ad-hoc verification
//! script (a bash heredoc) at every stage. Deep-bench transcripts across four
//! sessions put 37–62% of self-verification volume (11.7k–43.8k chars per
//! task) into re-checks issued while *nothing the previous green check could
//! have covered had been edited since* — an upper-bound estimate, but the
//! shape is unambiguous: the model has no way to know a check already ran
//! green, because nothing in its context says so.
//!
//! ## What this ledger claims, and what it refuses to claim
//!
//! It records exactly two runtime-owned IO facts, in order:
//!
//! 1. a check-shaped `bash` call ran to completion with exit 0 (the executor
//!    recorded the exit — not a model claim about it), and
//! 2. a later `edit_file`/`write_file` succeeded on some path.
//!
//! It deliberately does NOT claim to know which files a check *covers*. That
//! mapping is unknowable to the harness (a `cargo test -p runtime` covers
//! files no argument names), and a harness that guessed it would produce
//! confident nonsense — the failure mode the whole attestation discipline
//! exists to prevent. So the observation lists the edit delta and leaves the
//! judgement to the model: "this ran green, and these files changed since".
//!
//! ## Persistence (why a sidecar and not memory)
//!
//! A bench stage is a fresh process resuming a session, and headless rebuilds
//! the runtime every turn. An in-memory ledger would therefore be empty at
//! exactly the moment it is worth reading — the start of stage N+1. It
//! write-throughs to `<session>.verified-state.json` and reloads on rebind,
//! the same shape as [`crate::file_read_registry::FileReadRegistry`].
//!
//! ## Bounded by construction
//!
//! Only the newest [`STORED_MAX_CHECKS`] commands are kept, an edit that
//! predates the oldest retained check is dropped (it can never be rendered),
//! and both commands and paths dedupe newest-wins — so a session that runs the
//! same check a hundred times stores one row for it.

use std::fs;
use std::path::{Path, PathBuf};

/// One ordered fact the conversation layer hands to the ledger. The classifier
/// that decides what qualifies lives next to the existing `exec_green_checks`
/// vocabulary (`conversation::verified_state`), so both consumers of "is this a
/// check?" stay on one definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedStateEvent {
    /// A check-shaped bash command that exited 0 in the foreground. Already
    /// capped in length by the producer.
    GreenCheck(String),
    /// A successful `edit_file`/`write_file` target path, as the tool reported it.
    Edit(String),
}

/// Newest commands kept in the ledger. Larger than the render cap so a session
/// that alternates between two check sets does not thrash its own history.
const STORED_MAX_CHECKS: usize = 12;
/// Newest edited paths kept. Only paths newer than the oldest retained check
/// survive pruning anyway; this bounds the pathological case (a huge refactor
/// right after one check).
const STORED_MAX_EDITS: usize = 64;
/// Commands reported in one observation block (newest kept).
const RENDER_MAX_CHECKS: usize = 6;
/// Paths listed on one invalidated command's line before it summarizes.
const RENDER_MAX_FILES: usize = 8;

/// Reminder prefix, owned here so the producer and every recognizer bind to one
/// constant. Public because the host-side tests and the conversation layer both
/// match on it.
pub const VERIFIED_STATE_REMINDER_PREFIX: &str = "[zo:verified-state]";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct GreenCheck {
    command: String,
    seq: u64,
    turn: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct EditRecord {
    path: String,
    seq: u64,
    turn: u64,
}

/// The on-disk form. Every field defaults so a sidecar written by an older
/// build (or truncated mid-write) degrades to "less history", never to a parse
/// failure that would silently disable the feature.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct PersistedLedger {
    #[serde(default)]
    seq: u64,
    #[serde(default)]
    turn: u64,
    #[serde(default)]
    checks: Vec<GreenCheck>,
    #[serde(default)]
    edits: Vec<EditRecord>,
}

/// Session-lifetime ledger of green checks and the edits that followed them.
///
/// Both vectors are kept sorted by `seq` ascending: dedupe removes the stale
/// row before pushing the new one, so append order IS sequence order and every
/// "since" question is a single comparison.
#[derive(Debug, Default)]
pub struct VerifiedStateLedger {
    state: PersistedLedger,
    /// `None` = pure in-memory (sub-agent contexts, tests), same convention as
    /// the file-read registry: unbound means no write-through, not an error.
    sidecar: Option<PathBuf>,
}

impl VerifiedStateLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind to a session sidecar and restore its contents. A missing or corrupt
    /// file degrades to an empty ledger — i.e. to byte-neutral behaviour, never
    /// to a wrong observation. Binding itself writes nothing: a resume must not
    /// blank its own sidecar before it has read it.
    pub fn rebind_sidecar(&mut self, sidecar: Option<PathBuf>) {
        self.sidecar = sidecar;
        self.state = self.load_sidecar();
    }

    fn load_sidecar(&self) -> PersistedLedger {
        let Some(path) = self.sidecar.as_ref() else {
            return PersistedLedger::default();
        };
        let Ok(raw) = fs::read(path) else {
            return PersistedLedger::default();
        };
        serde_json::from_slice::<PersistedLedger>(&raw).unwrap_or_default()
    }

    /// Write-through, best effort and atomic (tmp + rename). A failure leaves
    /// the in-memory ledger authoritative for this process; the next stage then
    /// simply observes less, which is the safe direction.
    fn persist(&self) {
        let Some(path) = self.sidecar.as_ref() else {
            return;
        };
        let Ok(payload) = serde_json::to_vec(&self.state) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if fs::write(&tmp, payload).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }

    /// Fold one completed turn's ordered events into the ledger.
    ///
    /// Order within the slice is load-bearing: a check followed by an edit in
    /// the SAME turn must come out as "green, then invalidated", which is only
    /// representable because each event takes its own sequence number.
    ///
    /// An empty slice is a no-op — it neither advances the turn counter nor
    /// touches the disk, so chat-only turns cost nothing.
    pub fn record_turn(&mut self, events: &[VerifiedStateEvent]) {
        if events.is_empty() {
            return;
        }
        self.state.turn += 1;
        let turn = self.state.turn;
        for event in events {
            self.state.seq += 1;
            let seq = self.state.seq;
            match event {
                VerifiedStateEvent::GreenCheck(command) => {
                    let command = command.trim();
                    if command.is_empty() {
                        continue;
                    }
                    self.state.checks.retain(|check| check.command != command);
                    self.state.checks.push(GreenCheck {
                        command: command.to_string(),
                        seq,
                        turn,
                    });
                }
                VerifiedStateEvent::Edit(path) => {
                    let path = path.trim();
                    if path.is_empty() {
                        continue;
                    }
                    self.state.edits.retain(|edit| edit.path != path);
                    self.state.edits.push(EditRecord {
                        path: path.to_string(),
                        seq,
                        turn,
                    });
                }
            }
        }
        self.prune();
        self.persist();
    }

    /// Drop what can never be rendered. An edit older than the oldest retained
    /// check answers no question this ledger can be asked ("was anything edited
    /// AFTER check X?"), so it is not history worth keeping — which is also
    /// what keeps a long session's sidecar small.
    fn prune(&mut self) {
        if self.state.checks.len() > STORED_MAX_CHECKS {
            self.state
                .checks
                .drain(..self.state.checks.len() - STORED_MAX_CHECKS);
        }
        let floor = self.state.checks.first().map_or(u64::MAX, |check| check.seq);
        self.state.edits.retain(|edit| edit.seq > floor);
        if self.state.edits.len() > STORED_MAX_EDITS {
            self.state
                .edits
                .drain(..self.state.edits.len() - STORED_MAX_EDITS);
        }
    }

    /// The observation block for a turn start, or `None` when the ledger holds
    /// no green check at all.
    ///
    /// `None` is the byte-neutrality guarantee: a session that never ran a
    /// check must produce a request byte-identical to one built without this
    /// feature, so an automatic observation can never become an automatic tax.
    ///
    /// `root` shortens display paths only (edit tools report canonical absolute
    /// paths); it never affects what is stored or compared.
    #[must_use]
    pub fn render_observation(&self, root: Option<&Path>) -> Option<String> {
        if self.state.checks.is_empty() {
            return None;
        }
        let start = self.state.checks.len().saturating_sub(RENDER_MAX_CHECKS);
        let mut lines = String::new();
        for check in &self.state.checks[start..] {
            lines.push_str("- `");
            lines.push_str(&first_line(&check.command));
            lines.push_str("` — ");
            lines.push_str(&self.status_for(check.seq, root));
            lines.push('\n');
        }
        Some(format!(
            "{VERIFIED_STATE_REMINDER_PREFIX} <system-reminder>\nVerification already on record \
             for THIS session — each line is a check-shaped bash command the runtime observed \
             exiting 0 (recorded IO facts, not claims), followed by what has been edited since it \
             ran:\n{lines}A command whose line says nothing edited since has a result you already \
             have; re-running it re-derives a fact the harness is holding for you. The harness \
             does NOT know which files a command covers — when files are listed, judge for \
             yourself whether they can affect that command's outcome before re-running \
             it.\n</system-reminder>"
        ))
    }

    /// `"ran green, nothing edited since"` or the invalidated form naming the
    /// paths, capped and newest-first.
    fn status_for(&self, since_seq: u64, root: Option<&Path>) -> String {
        let mut changed: Vec<&EditRecord> = self
            .state
            .edits
            .iter()
            .filter(|edit| edit.seq > since_seq)
            .collect();
        if changed.is_empty() {
            return "ran green, nothing edited since".to_string();
        }
        changed.reverse();
        let overflow = changed.len().saturating_sub(RENDER_MAX_FILES);
        let listed = changed
            .iter()
            .take(RENDER_MAX_FILES)
            .map(|edit| display_path(&edit.path, root))
            .collect::<Vec<_>>()
            .join(", ");
        if overflow == 0 {
            format!("ran green BUT these files changed since: {listed}")
        } else {
            format!("ran green BUT these files changed since: {listed} (+{overflow} more)")
        }
    }

    /// Recorded commands, oldest first — diagnostics and tests.
    #[must_use]
    pub fn green_check_commands(&self) -> Vec<&str> {
        self.state
            .checks
            .iter()
            .map(|check| check.command.as_str())
            .collect()
    }

    /// Turns that contributed at least one recorded event.
    #[must_use]
    pub const fn recorded_turns(&self) -> u64 {
        self.state.turn
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.checks.is_empty() && self.state.edits.is_empty()
    }
}

/// One display line for a possibly multi-line command (heredoc check scripts are
/// routinely 1.9k chars): the first line, marked so the model can tell a
/// truncation from a one-liner rather than mistaking a prefix for the whole
/// command.
fn first_line(command: &str) -> String {
    let mut lines = command.lines();
    let head = lines.next().unwrap_or_default().trim_end();
    if lines.next().is_some() {
        format!("{head} …")
    } else {
        head.to_string()
    }
}

/// Display-only shortening against the session's workspace root.
fn display_path(path: &str, root: Option<&Path>) -> String {
    let Some(root) = root else {
        return path.to_string();
    };
    Path::new(path)
        .strip_prefix(root)
        .map_or_else(|_| path.to_string(), |rest| rest.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        VerifiedStateEvent, VerifiedStateLedger, RENDER_MAX_CHECKS, STORED_MAX_CHECKS,
        VERIFIED_STATE_REMINDER_PREFIX,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-verified-state-{name}-{unique}"));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn green(command: &str) -> VerifiedStateEvent {
        VerifiedStateEvent::GreenCheck(command.to_string())
    }

    fn edit(path: &str) -> VerifiedStateEvent {
        VerifiedStateEvent::Edit(path.to_string())
    }

    /// (a) A green check recorded in one turn is observable at the next turn's
    /// start, and reads as FRESH while nothing has been edited since.
    #[test]
    fn a_green_check_with_no_later_edit_renders_as_fresh() {
        let mut ledger = VerifiedStateLedger::new();
        ledger.record_turn(&[green("cargo test -p runtime")]);
        let block = ledger.render_observation(None).expect("observation");
        assert!(block.starts_with(VERIFIED_STATE_REMINDER_PREFIX));
        assert!(block.contains("`cargo test -p runtime` — ran green, nothing edited since"));
        assert!(!block.contains("changed since"));
    }

    /// (b) An edit recorded AFTER the check invalidates it, and the changed
    /// paths are named — the harness reports the delta and judges nothing.
    #[test]
    fn an_edit_after_the_check_marks_it_invalidated_and_names_the_files() {
        let mut ledger = VerifiedStateLedger::new();
        ledger.record_turn(&[
            green("cargo test -p runtime"),
            edit("/repo/crates/runtime/src/a.rs"),
        ]);
        let block = ledger
            .render_observation(Some(std::path::Path::new("/repo")))
            .expect("observation");
        assert!(
            block.contains(
                "`cargo test -p runtime` — ran green BUT these files changed since: \
                 crates/runtime/src/a.rs"
            ),
            "{block}"
        );
    }

    /// An edit that PRECEDES the check does not invalidate it — the check saw
    /// that edit's tree. This is the ordering the event slice encodes.
    #[test]
    fn an_edit_before_the_check_leaves_it_fresh() {
        let mut ledger = VerifiedStateLedger::new();
        ledger.record_turn(&[edit("/repo/a.rs"), green("cargo test")]);
        let block = ledger.render_observation(None).expect("observation");
        assert!(block.contains("ran green, nothing edited since"), "{block}");
        // The now-unanswerable edit is pruned rather than carried forever.
        assert!(ledger.state.edits.is_empty());
    }

    /// Re-running an invalidated check restores it to fresh, and drops the edit
    /// that can no longer invalidate anything.
    #[test]
    fn rerunning_a_check_restores_freshness_and_prunes_the_stale_edit() {
        let mut ledger = VerifiedStateLedger::new();
        ledger.record_turn(&[green("cargo test"), edit("/repo/a.rs")]);
        assert!(ledger
            .render_observation(None)
            .expect("observation")
            .contains("changed since"));
        ledger.record_turn(&[green("cargo test")]);
        let block = ledger.render_observation(None).expect("observation");
        assert!(block.contains("ran green, nothing edited since"), "{block}");
        assert_eq!(ledger.green_check_commands(), vec!["cargo test"]);
        assert!(ledger.state.edits.is_empty());
    }

    /// (c) No green check anywhere = no observation at all. This is the
    /// byte-neutrality pin: an edit-only session must render nothing.
    #[test]
    fn without_any_green_check_the_observation_is_empty() {
        let mut ledger = VerifiedStateLedger::new();
        assert!(ledger.render_observation(None).is_none());
        ledger.record_turn(&[edit("/repo/a.rs"), edit("/repo/b.rs")]);
        assert!(
            ledger.render_observation(None).is_none(),
            "edits alone must not produce an observation"
        );
        ledger.record_turn(&[]);
        assert_eq!(ledger.recorded_turns(), 1, "an empty turn records nothing");
    }

    /// (d) Sidecar round trip: a ledger written by one process is restored by a
    /// freshly rebound one, and still renders the same observation. This is the
    /// stage-boundary guarantee.
    #[test]
    fn sidecar_roundtrip_survives_a_new_ledger() {
        let dir = temp_dir("roundtrip");
        let sidecar = dir.join("session.verified-state.json");

        let mut writer = VerifiedStateLedger::new();
        writer.rebind_sidecar(Some(sidecar.clone()));
        writer.record_turn(&[green("cargo test -p runtime"), edit("/repo/a.rs")]);
        assert!(sidecar.exists(), "record must write through");

        let mut resumed = VerifiedStateLedger::new();
        resumed.rebind_sidecar(Some(sidecar.clone()));
        let block = resumed.render_observation(None).expect("observation");
        assert!(block.contains("cargo test -p runtime"), "{block}");
        assert!(block.contains("/repo/a.rs"), "{block}");
        assert_eq!(resumed.recorded_turns(), 1);

        // A later stage keeps counting from where the previous one stopped.
        resumed.record_turn(&[green("cargo clippy -p runtime")]);
        assert_eq!(resumed.recorded_turns(), 2);
        let mut third = VerifiedStateLedger::new();
        third.rebind_sidecar(Some(sidecar));
        assert_eq!(
            third.green_check_commands(),
            vec!["cargo test -p runtime", "cargo clippy -p runtime"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A corrupt sidecar degrades to empty (= byte-neutral), never to a wrong
    /// observation.
    #[test]
    fn a_corrupt_sidecar_degrades_to_empty() {
        let dir = temp_dir("corrupt");
        let sidecar = dir.join("session.verified-state.json");
        std::fs::write(&sidecar, b"{not json").expect("corrupt seed");

        let mut ledger = VerifiedStateLedger::new();
        ledger.rebind_sidecar(Some(sidecar));
        assert!(ledger.is_empty());
        assert!(ledger.render_observation(None).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A conversation reset is a REBIND, not a wipe: the new session reads its
    /// own empty sidecar while the previous session's file is left intact, so
    /// `/resume` still finds the history it was promised. (Wiping in place is
    /// the documented trap on the read-registry swap.)
    #[test]
    fn rebinding_to_a_new_session_leaves_the_old_sidecar_intact() {
        let dir = temp_dir("rebind");
        let old = dir.join("old.verified-state.json");
        let new = dir.join("new.verified-state.json");

        let mut ledger = VerifiedStateLedger::new();
        ledger.rebind_sidecar(Some(old.clone()));
        ledger.record_turn(&[green("cargo test -p runtime")]);

        ledger.rebind_sidecar(Some(new));
        assert!(ledger.is_empty(), "the fresh session starts empty");
        assert!(ledger.render_observation(None).is_none());

        let mut resumed = VerifiedStateLedger::new();
        resumed.rebind_sidecar(Some(old));
        assert_eq!(resumed.green_check_commands(), vec!["cargo test -p runtime"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// (e) Both caps keep the NEWEST evidence: storage prunes the oldest
    /// commands, and the render shows only the most recent ones.
    #[test]
    fn caps_keep_the_newest_checks() {
        let mut ledger = VerifiedStateLedger::new();
        for index in 0..(STORED_MAX_CHECKS + 4) {
            ledger.record_turn(&[green(&format!("cargo test case{index}"))]);
        }
        let stored = ledger.green_check_commands();
        assert_eq!(stored.len(), STORED_MAX_CHECKS);
        assert_eq!(stored[0], "cargo test case4");
        assert_eq!(stored[STORED_MAX_CHECKS - 1], "cargo test case15");

        let block = ledger.render_observation(None).expect("observation");
        assert_eq!(
            block.matches("ran green").count(),
            RENDER_MAX_CHECKS,
            "render cap keeps the block short: {block}"
        );
        assert!(block.contains("cargo test case15"), "newest is kept");
        assert!(!block.contains("cargo test case9"), "oldest is dropped");
    }

    /// The per-line file cap summarizes the tail instead of printing a wall.
    #[test]
    fn the_file_list_is_capped_and_summarized() {
        let mut ledger = VerifiedStateLedger::new();
        let mut events = vec![green("cargo test")];
        for index in 0..12 {
            events.push(edit(&format!("/repo/f{index}.rs")));
        }
        ledger.record_turn(&events);
        let block = ledger.render_observation(None).expect("observation");
        assert!(block.contains("(+4 more)"), "{block}");
        // Newest-first: the last edit leads the list.
        assert!(
            block.contains("changed since: /repo/f11.rs, /repo/f10.rs"),
            "{block}"
        );
        assert!(!block.contains("/repo/f3.rs"), "{block}");
    }

    /// A heredoc check script renders as ONE line — the observation must stay
    /// compact no matter how large the command that produced it was.
    #[test]
    fn a_multiline_command_renders_on_a_single_line() {
        let mut ledger = VerifiedStateLedger::new();
        ledger.record_turn(&[green("python3 - <<'PY'\nassert 1\nPY")]);
        let block = ledger.render_observation(None).expect("observation");
        assert!(block.contains("`python3 - <<'PY' …` — ran green"), "{block}");
        assert!(!block.contains("assert 1"), "{block}");
    }
}
