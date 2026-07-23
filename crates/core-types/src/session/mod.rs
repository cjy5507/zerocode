//! Persisted conversational state for the runtime and CLI session manager.
//!
//! The [`Session`] aggregate and its append-only JSONL persistence live here.
//! The conversation content value objects are split into focused siblings:
//! [`message`] ([`MessageRole`]/[`ContentBlock`]/[`ConversationMessage`]),
//! [`compaction`] ([`SessionCompaction`]), [`fork`] ([`SessionFork`]), and
//! [`error`] ([`SessionError`]); [`json_field`] holds the shared JSON-field
//! extraction helpers. Re-exports below keep the crate's public paths
//! (`core_types::session::*` and the `core_types::*` re-exports) unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::json::JsonValue;

mod compaction;
mod error;
mod fork;
mod json_field;
mod message;

pub use compaction::{AnchorSummary, SessionCompaction};
pub use error::SessionError;
pub use fork::SessionFork;
pub use message::{ContentBlock, ConversationMessage, MessageRole};

use json_field::{
    i64_from_u64, required_string, required_u32, required_u64, required_u64_from_value,
};

const SESSION_VERSION: u32 = 1;
/// Synthetic `tool_result` body injected for a `tool_use` whose turn was
/// cancelled before the tool produced a result. Keeps the next Anthropic
/// request valid (every `tool_use` must be followed by a `tool_result`).
const INTERRUPTED_TOOL_RESULT: &str =
    "[Interrupted: the previous turn was cancelled before this tool produced a result.]";
const ROTATE_AFTER_BYTES: u64 = 256 * 1024;
const MAX_ROTATED_FILES: usize = 3;
static SESSION_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Identity + freshness of a session file as this writer last observed it.
/// Captured when a `Session` is bound to a path (load or bind), and refreshed
/// after every successful bound write. A mismatch against the current on-disk
/// file means a peer rewrote it, which is the stale-write signal.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileFingerprint {
    /// File length in bytes.
    len: u64,
    /// A cheap content digest over the whole file, so a same-length rewrite
    /// (e.g. an edited-in-place snapshot of identical size) is still detected.
    digest: u64,
}

impl FileFingerprint {
    /// Fingerprint the bytes we are about to publish (post-write expected state).
    fn of_bytes(bytes: &[u8]) -> Self {
        Self {
            len: bytes.len() as u64,
            digest: digest_bytes(bytes),
        }
    }

    /// Fingerprint the current on-disk file. `Ok(None)` when the file is absent
    /// (a legitimate "no prior state" that a first writer expects).
    fn of_path(path: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read(path) {
            Ok(bytes) => Ok(Some(Self::of_bytes(&bytes))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }
}

/// Stable 64-bit content digest (`FNV-1a`). Not cryptographic — it only needs to
/// catch concurrent rewrites, and collisions merely weaken (never falsely
/// trigger) the stale-write check.
fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Cross-`Session`-clone writer state for one bound persistence path. A single
/// `Session` and all its clones share one `Arc<Mutex<WriterState>>`, so they
/// cooperate as one logical writer; an independently loaded `Session` gets its
/// own `WriterState` and contends via the OS advisory lock instead.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FullSnapshotState {
    version: u32,
    session_id: String,
    name: Option<String>,
    created_at_ms: u64,
    compaction: Option<SessionCompaction>,
    fork: Option<SessionFork>,
    session_goal: Option<String>,
    first_message_index: u32,
}

#[derive(Debug, Default)]
struct WriterState {
    /// The fingerprint the next bound write expects to find on disk. `None`
    /// until the path is bound; `Some(None)` conceptually is folded into this
    /// `Option<FileFingerprint>` where the *outer* `expected_captured` flag
    /// records whether we have observed the file at all.
    expected: Option<FileFingerprint>,
    /// Whether `expected` has been initialized (distinguishes "absent file"
    /// from "never looked").
    expected_captured: bool,
    /// Header/compaction state written by the most recent full snapshot. The
    /// per-message append stream carries `updated_at_ms`, so that hot counter is
    /// intentionally excluded.
    full_snapshot_state: Option<FullSnapshotState>,
    /// In-memory transcript edits that did not travel through `push_message`
    /// require one healing full snapshot at the next ordinary-turn persist.
    transcript_dirty: bool,
    /// The held exclusive advisory lock (Unix). Acquired lazily on the first
    /// bound write and retained for the lifetime of the shared state so a peer
    /// process cannot interleave. `None` on non-Unix or before first write.
    /// Dropping the `Flock` releases the OS lock.
    #[cfg(unix)]
    lock: Option<nix::fcntl::Flock<std::fs::File>>,
}

#[derive(Debug, Clone)]
struct SessionPersistence {
    path: PathBuf,
    secure_regular_file: bool,
    /// Clone-shared writer lease + fingerprint. Cloning a `Session` clones this
    /// `Arc` (shared lease); loading a fresh `Session` from the same path makes
    /// a new one (contending lease).
    writer: Arc<Mutex<WriterState>>,
}

impl SessionPersistence {
    fn new(path: PathBuf, secure_regular_file: bool) -> Self {
        Self {
            path,
            secure_regular_file,
            writer: Arc::new(Mutex::new(WriterState::default())),
        }
    }
}

// Identity of a bound persistence is its path + secure flag; the runtime writer
// lease is intentionally excluded from equality (it is not session content).
impl PartialEq for SessionPersistence {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.secure_regular_file == other.secure_regular_file
    }
}

impl Eq for SessionPersistence {}

/// Persisted conversational state for the runtime and CLI session manager.
#[derive(Debug, Clone)]
pub struct Session {
    pub version: u32,
    pub session_id: String,
    pub name: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    /// 대화 메시지 이력. `Arc<Vec<_>>` 로 보관해 `build_request` 의 요청
    /// 스냅샷이 전체 메시지를 deep clone 하지 않고 `Arc::clone`(포인터 복사)
    /// 으로 공유한다. 변경(push/pop/truncate)은 [`Arc::make_mut`] 로 COW —
    /// 공유 중일 때만 1회 복제되고 단독 보유 시 in-place.
    pub messages: Arc<Vec<ConversationMessage>>,
    pub compaction: Option<SessionCompaction>,
    pub fork: Option<SessionFork>,
    /// Session-scoped goal (`/goal`) — the standing objective surfaced to the
    /// model each turn and exposed to `TurnEnd` (Stop) hooks for completion
    /// gating. Optional in the persisted header, so older session files load
    /// unchanged.
    pub session_goal: Option<String>,
    first_message_index: u32,
    persistence: Option<SessionPersistence>,
}

impl PartialEq for Session {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.session_id == other.session_id
            && self.name == other.name
            && self.created_at_ms == other.created_at_ms
            && self.updated_at_ms == other.updated_at_ms
            && self.messages == other.messages
            && self.compaction == other.compaction
            && self.fork == other.fork
            && self.session_goal == other.session_goal
            && self.first_message_index == other.first_message_index
    }
}

impl Eq for Session {}

impl Session {
    #[must_use]
    pub fn new() -> Self {
        let now = current_time_millis();
        Self {
            version: SESSION_VERSION,
            session_id: generate_session_id(),
            name: None,
            created_at_ms: now,
            updated_at_ms: now,
            messages: Arc::new(Vec::new()),
            compaction: None,
            fork: None,
            session_goal: None,
            first_message_index: 0,
            persistence: None,
        }
    }

    #[must_use]
    pub fn with_persistence_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.persistence = Some(SessionPersistence::new(path.into(), false));
        self.capture_initial_fingerprint();
        self
    }

    /// Persist through descriptor-validated regular files. Agent transcripts
    /// use this mode because their sibling paths can be influenced by persisted
    /// manifests and must never follow links during load or append.
    #[must_use]
    pub fn with_secure_persistence_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.persistence = Some(SessionPersistence::new(path.into(), true));
        self.capture_initial_fingerprint();
        self
    }

    /// Record the on-disk fingerprint at bind time so the first bound write can
    /// tell whether the file is still what this `Session` was built from. A
    /// read error here is non-fatal: the first write's lock+recheck still
    /// guards correctness, and a bind should not fail just because the file is
    /// momentarily unreadable.
    fn capture_initial_fingerprint(&self) {
        let Some(persistence) = self.persistence.as_ref() else {
            return;
        };
        let observed = FileFingerprint::of_path(&persistence.path);
        let snapshot_state = self.full_snapshot_state();
        if let Ok(mut state) = persistence.writer.lock() {
            if !state.expected_captured {
                if let Ok(fingerprint) = observed {
                    state.expected = fingerprint;
                    state.expected_captured = true;
                    state.full_snapshot_state = Some(snapshot_state);
                    state.transcript_dirty = false;
                }
            }
        }
    }

    fn full_snapshot_state(&self) -> FullSnapshotState {
        FullSnapshotState {
            version: self.version,
            session_id: self.session_id.clone(),
            name: self.name.clone(),
            created_at_ms: self.created_at_ms,
            compaction: self.compaction.clone(),
            fork: self.fork.clone(),
            session_goal: self.session_goal.clone(),
            first_message_index: self.first_message_index,
        }
    }

    #[must_use]
    pub fn persistence_path(&self) -> Option<&Path> {
        self.persistence.as_ref().map(|value| value.path.as_path())
    }

    /// The bound path's trailing `projects/<slug>/sessions/<file>` shape
    /// re-rooted onto each current global config root. Paths that do not have
    /// the canonical store shape yield no candidates, so ad-hoc stores (tests,
    /// explicit exports) never rebind implicitly.
    fn moved_store_candidates(bound: &Path, roots: &[PathBuf]) -> Vec<PathBuf> {
        let mut tail: Vec<_> = bound
            .components()
            .rev()
            .take(4)
            .map(|component| component.as_os_str().to_os_string())
            .collect();
        tail.reverse();
        let [projects, _slug, sessions, _file] = tail.as_slice() else {
            return Vec::new();
        };
        if projects != "projects" || sessions != "sessions" {
            return Vec::new();
        }
        let suffix: PathBuf = tail.iter().collect();
        roots
            .iter()
            .map(|root| root.join(&suffix))
            .filter(|candidate| candidate.as_path() != bound)
            .collect()
    }

    /// Whether `contents` are the byte-identical transcript this writer last
    /// observed for THIS session: the fingerprint must match and the first
    /// (meta) record must carry the same `session_id`. A stale same-id copy or
    /// same-named different session must never be adopted — either would fork
    /// history.
    fn candidate_matches_expected_store(
        &self,
        contents: &[u8],
        expected: Option<&FileFingerprint>,
    ) -> bool {
        let observed = Some(FileFingerprint::of_bytes(contents));
        if observed.as_ref() != expected {
            return false;
        }
        let Ok(contents) = std::str::from_utf8(contents) else {
            return false;
        };
        let Some(first) = contents.lines().next() else {
            return false;
        };
        JsonValue::parse(first)
            .ok()
            .and_then(|value| {
                value
                    .as_object()
                    .and_then(|object| object.get("session_id"))
                    .and_then(JsonValue::as_str)
                    .map(str::to_string)
            })
            .is_some_and(|id| id == self.session_id)
    }

    /// One-shot recovery for a store that moved out from under a live session
    /// (a home-root migration): when the bound file is gone, look for the
    /// byte-identical transcript with the same session id at the bound path's
    /// canonical tail under each current global config root and rebind to the
    /// first match. Returns whether a rebind happened. The multi-writer
    /// fingerprint guard is re-armed against the new path — never bypassed.
    fn rebind_moved_store(&mut self) -> bool {
        self.rebind_moved_store_with_roots(&crate::paths::zo_global_config_roots())
    }

    fn rebind_moved_store_with_roots(&mut self, roots: &[PathBuf]) -> bool {
        let Some(persistence) = self.persistence.as_ref() else {
            return false;
        };
        if persistence.path.exists() {
            return false;
        }
        let bound = persistence.path.clone();
        let secure = persistence.secure_regular_file;
        let expected = {
            let Ok(state) = persistence.writer.lock() else {
                return false;
            };
            if !state.expected_captured {
                return false;
            }
            state.expected.clone()
        };
        let Some(candidate) = Self::moved_store_candidates(&bound, roots)
            .into_iter()
            .find(|candidate| {
                fs::read(candidate).is_ok_and(|contents| {
                    self.candidate_matches_expected_store(&contents, expected.as_ref())
                })
            })
        else {
            return false;
        };
        eprintln!(
            "[zo] session store moved: rebinding {} -> {}",
            bound.display(),
            candidate.display()
        );
        let rebound = SessionPersistence::new(candidate, secure);
        {
            let Ok(mut state) = rebound.writer.lock() else {
                return false;
            };
            state.expected = expected;
            state.expected_captured = true;
        }
        self.persistence = Some(rebound);
        true
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), SessionError> {
        let path = path.as_ref();
        let snapshot = self.render_jsonl_snapshot()?;
        // A full snapshot that targets the session's own bound path must not
        // clobber a peer's newer state: guard it with the writer lease +
        // fingerprint. Exporting to any *other* path keeps plain overwrite
        // semantics (an explicit, caller-chosen destination).
        if self.is_bound_path(path) {
            return self.guarded_bound_write(path, snapshot.as_bytes(), |target| {
                rotate_session_file_if_needed(target)?;
                write_atomic(target, &snapshot)?;
                cleanup_rotated_logs(target)?;
                Ok(())
            });
        }
        rotate_session_file_if_needed(path)?;
        write_atomic(path, &snapshot)?;
        cleanup_rotated_logs(path)?;
        Ok(())
    }

    /// Persist state after an ordinary turn whose new messages already landed
    /// through [`Self::push_message`]. Clean append-only turns need no write;
    /// header/compaction changes or an explicitly marked transcript rewrite
    /// fall back to the same guarded full snapshot as [`Self::save_to_path`].
    pub fn persist_appended_state_to_path(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), SessionError> {
        let path = path.as_ref();
        let Some(persistence) = self
            .persistence
            .as_ref()
            .filter(|persistence| persistence.path == path)
        else {
            return self.save_to_path(path);
        };
        let secure = persistence.secure_regular_file;
        let mut state = persistence
            .writer
            .lock()
            .map_err(|_| SessionError::Conflict("session writer lease poisoned".to_string()))?;
        acquire_writer_lock(path, &mut state)?;

        let on_disk = FileFingerprint::of_path(path).map_err(SessionError::Io)?;
        if state.expected_captured && on_disk != state.expected {
            return Err(SessionError::Conflict(format!(
                "session file {} changed on disk since this writer last observed it; refusing to accept stale append state",
                path.display()
            )));
        }
        let nonempty_file = if state.expected.is_some() {
            if secure {
                open_secure_regular_file(path, false)?.metadata()?.len() > 0
            } else {
                fs::metadata(path)?.len() > 0
            }
        } else {
            false
        };
        let can_skip = state.expected_captured
            && nonempty_file
            && !state.transcript_dirty
            && state.full_snapshot_state.as_ref() == Some(&self.full_snapshot_state());
        drop(state);

        if can_skip {
            Ok(())
        } else if secure {
            self.save_to_secure_path(path)
        } else {
            self.save_to_path(path)
        }
    }

    /// Mark a transcript mutation that did not use [`Self::push_message`]. The
    /// next append-aware persist will heal the JSONL with one full snapshot.
    pub fn mark_transcript_dirty(&self) {
        let Some(persistence) = self.persistence.as_ref() else {
            return;
        };
        if let Ok(mut state) = persistence.writer.lock() {
            state.transcript_dirty = true;
        }
    }

    fn save_to_secure_path(&self, path: &Path) -> Result<(), SessionError> {
        let snapshot = self.render_jsonl_snapshot()?;
        if self.is_bound_path(path) {
            return self.guarded_bound_write(path, snapshot.as_bytes(), |target| {
                write_atomic_secure(target, &snapshot)
            });
        }
        write_atomic_secure(path, &snapshot)
    }

    /// True when `path` is the persistence path this `Session` is bound to.
    fn is_bound_path(&self, path: &Path) -> bool {
        let Some(persistence) = self.persistence.as_ref() else {
            return false;
        };
        paths_identify_same_file(&persistence.path, path)
    }

    /// Finalize a persistence-bound full-snapshot mutation (`compaction` /
    /// `rewind`) from a callsite that cannot return an error.
    ///
    /// Ordinary IO failures stay best-effort: the in-memory state is still
    /// authoritative and will be re-persisted on the next write, so the earlier
    /// mutation is kept. A [`SessionError::Conflict`], however, means a peer
    /// owns the file and this snapshot was deliberately NOT written — keeping
    /// the mutation would silently diverge memory from disk. So on Conflict we
    /// roll the mutated fields back to their pre-mutation snapshot and emit an
    /// explicit diagnostic. Returns `true` iff a conflict was rolled back.
    fn finish_bound_mutation(
        &mut self,
        operation: &str,
        path: &Path,
        result: Result<(), SessionError>,
        rollback: MutationRollback,
    ) -> bool {
        if let Err(SessionError::Conflict(reason)) = result {
            rollback.restore(self);
            eprintln!(
                "[zo] warning: {operation} did not persist session {} (rolled back in-memory to avoid divergence): {reason}",
                path.display()
            );
            return true;
        }
        false
    }

    /// Perform a persistence-bound write under the clone-shared writer lease.
    ///
    /// Steps, in order: (1) lazily acquire the exclusive advisory lock on a
    /// sibling `<file>.lock` and hold it for the shared state's lifetime; (2)
    /// recheck that the current on-disk fingerprint matches what this writer
    /// expects — if not, a peer rewrote the file, so return
    /// [`SessionError::Conflict`] **before any write or rename**; (3) run the
    /// caller's write closure (which does the atomic temp+rename); (4) on
    /// success, refresh the expected fingerprint to the bytes just published.
    ///
    /// The user-supplied bytes are fingerprinted, never a caller callback run
    /// while holding the lock beyond the write itself.
    fn guarded_bound_write(
        &self,
        path: &Path,
        published_bytes: &[u8],
        write: impl FnOnce(&Path) -> Result<(), SessionError>,
    ) -> Result<(), SessionError> {
        let persistence = self
            .persistence
            .as_ref()
            .expect("guarded_bound_write requires bound persistence");
        let mut state = persistence
            .writer
            .lock()
            .map_err(|_| SessionError::Conflict("session writer lease poisoned".to_string()))?;

        acquire_writer_lock(path, &mut state)?;

        // Freshness recheck under the lock: compare disk to expected.
        let on_disk = FileFingerprint::of_path(path)
            .map_err(SessionError::Io)?;
        if state.expected_captured && on_disk != state.expected {
            return Err(SessionError::Conflict(format!(
                "session file {} changed on disk since this writer last observed it; refusing to overwrite newer state",
                path.display()
            )));
        }

        write(path)?;

        // Publish succeeded: the file now holds exactly `published_bytes`.
        state.expected = Some(FileFingerprint::of_bytes(published_bytes));
        state.expected_captured = true;
        state.full_snapshot_state = Some(self.full_snapshot_state());
        state.transcript_dirty = false;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        Self::load_from_path_from_turn(path, None)
    }

    pub fn load_from_secure_path(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref();
        let contents = read_secure_regular_file(path)?;
        Ok(Self::load_from_contents(&contents, None)?
            .with_secure_persistence_path(path.to_path_buf()))
    }

    pub fn load_from_path_from_turn(
        path: impl AsRef<Path>,
        from_turn: Option<u32>,
    ) -> Result<Self, SessionError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)?;
        Ok(Self::load_from_contents(&contents, from_turn)?
            .with_persistence_path(path.to_path_buf()))
    }

    fn load_from_contents(contents: &str, from_turn: Option<u32>) -> Result<Self, SessionError> {
        match JsonValue::parse(contents) {
            Ok(value)
                if value
                    .as_object()
                    .is_some_and(|object| object.contains_key("messages")) =>
            {
                if from_turn.is_some() {
                    return Err(SessionError::Format(
                        "--from-turn requires JSONL session records with turn_index".to_string(),
                    ));
                }
                Self::from_json(&value)
            }
            Err(_) | Ok(_) => Self::from_jsonl(contents, from_turn),
        }
    }

    pub fn push_message(&mut self, message: ConversationMessage) -> Result<(), SessionError> {
        self.touch();
        Arc::make_mut(&mut self.messages).push(message);
        let persist_result = {
            let message_ref = self.messages.last().ok_or_else(|| {
                SessionError::Format("message was just pushed but missing".to_string())
            })?;
            self.append_persisted_message(message_ref)
        };
        if let Err(error) = persist_result {
            // A moved store is recoverable: rebind once to the relocated
            // transcript (identity-checked) and retry. Real peer-writer
            // conflicts and everything else still fail loudly.
            if self.rebind_moved_store() {
                let retried = {
                    let message_ref = self.messages.last().ok_or_else(|| {
                        SessionError::Format("message was just pushed but missing".to_string())
                    })?;
                    self.append_persisted_message(message_ref)
                };
                if let Err(retry_error) = retried {
                    Arc::make_mut(&mut self.messages).pop();
                    return Err(retry_error);
                }
            } else {
                Arc::make_mut(&mut self.messages).pop();
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn push_user_text(&mut self, text: impl Into<String>) -> Result<(), SessionError> {
        self.push_message(ConversationMessage::user_text(text))
    }

    /// Push a user message that contains image attachments.
    ///
    /// Each image is a `(media_type, base64_data)` pair. The images appear
    /// before the text block in the message, matching the Anthropic API
    /// convention.
    pub fn push_user_with_images(
        &mut self,
        text: impl Into<String>,
        images: Vec<(String, String)>,
    ) -> Result<(), SessionError> {
        self.push_message(ConversationMessage::user_with_images(text, images))
    }

    /// Return a message view in which every `tool_use` block is immediately
    /// followed by a matching `tool_result`.
    ///
    /// A turn cancelled mid-flight — e.g. Ctrl+C after the assistant's
    /// `tool_use` was committed to the session but before the tool produced
    /// its result — leaves an *orphan* `tool_use`. Anthropic then rejects the
    /// next request with `400 invalid_request_error: tool_use ids were found
    /// without tool_result blocks`, which bricks the session for good. This
    /// seals each orphan with a synthetic error result so the outgoing request
    /// stays well-formed, *without* mutating or re-persisting stored history
    /// (the seal is a per-request view; a later real result still appends
    /// cleanly).
    ///
    /// Cheap on the happy path: when every `tool_use` already has a result it
    /// shares the existing `Arc` with zero allocation.
    #[must_use]
    pub fn tool_consistent_messages(&self) -> Arc<Vec<ConversationMessage>> {
        let mut used_ids: Vec<&str> = Vec::new();
        let mut satisfied: BTreeSet<&str> = BTreeSet::new();
        for block in self.messages.iter().flat_map(|message| &message.blocks) {
            match block {
                ContentBlock::ToolUse { id, .. } => used_ids.push(id.as_str()),
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    satisfied.insert(tool_use_id.as_str());
                }
                _ => {}
            }
        }
        if used_ids.iter().all(|id| satisfied.contains(id)) {
            return Arc::clone(&self.messages);
        }

        // Inject each orphan's synthetic result immediately after the message
        // that introduced it — Anthropic requires the result to *directly*
        // follow the use, even across an interleaved user message.
        let mut sealed = Vec::with_capacity(self.messages.len() + 1);
        for message in self.messages.iter() {
            let orphans: Vec<(String, String)> = message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, .. } if !satisfied.contains(id.as_str()) => {
                        Some((id.clone(), name.clone()))
                    }
                    _ => None,
                })
                .collect();
            sealed.push(message.clone());
            for (id, name) in orphans {
                sealed.push(ConversationMessage::tool_result(
                    id,
                    name,
                    INTERRUPTED_TOOL_RESULT,
                    true,
                ));
            }
        }
        Arc::new(sealed)
    }

    /// Convenience wrapper: reconcile this session's messages against the
    /// `known_tool_names` the upcoming request will advertise. See
    /// [`reconcile_tool_history`].
    #[must_use]
    pub fn messages_reconciled_for_tools(
        &self,
        known_tool_names: &BTreeSet<String>,
    ) -> Arc<Vec<ConversationMessage>> {
        reconcile_tool_history(&self.messages, known_tool_names)
    }

    pub fn record_compaction(&mut self, summary: impl Into<String>, removed_message_count: usize) {
        self.record_compaction_with_anchor(summary, removed_message_count, None);
    }

    /// Like [`record_compaction`](Self::record_compaction) but also stores the
    /// typed [`AnchorSummary`] (LAVA P1) the continuation message was rendered
    /// from, so the next round folds against the typed anchor instead of
    /// re-parsing the prior round's prose (which eroded identifiers over a long
    /// session).
    pub fn record_compaction_with_anchor(
        &mut self,
        summary: impl Into<String>,
        removed_message_count: usize,
        anchor: Option<AnchorSummary>,
    ) {
        // Snapshot at THIS point so a persistence Conflict can roll the fields
        // this method mutates back to a state consistent with what remains on
        // disk. NOTE: the snapshot captures `messages` as they are *now* — this
        // method does not itself replace the transcript, so if a caller already
        // swapped `messages` to the compacted set before calling here, a
        // rollback restores that compacted set, NOT the pre-compaction one. For
        // an atomic replace+record that also restores the original transcript
        // on Conflict, use [`apply_compaction_atomic`](Self::apply_compaction_atomic).
        let rollback = MutationRollback::capture(self);
        self.commit_compaction(summary, removed_message_count, anchor, rollback);
    }

    /// Atomically replace the transcript with the compacted set and record the
    /// compaction metadata as one seam, capturing the pre-mutation snapshot
    /// *before* the messages are replaced.
    ///
    /// This is the production compaction seam: on a persistence
    /// [`SessionError::Conflict`] it rolls back the *entire* mutation —
    /// messages, `updated_at`, compaction anchor, and `first_message_index` —
    /// to the exact pre-compaction state, so a stale writer never leaves memory
    /// holding a compacted view that diverges from a peer's newer file. Unlike
    /// [`record_compaction_with_anchor`](Self::record_compaction_with_anchor),
    /// the caller must NOT pre-assign `self.messages`; it passes the replacement
    /// so the rollback snapshot still knows the original transcript.
    pub fn apply_compaction_atomic(
        &mut self,
        compacted_messages: Arc<Vec<ConversationMessage>>,
        summary: impl Into<String>,
        removed_message_count: usize,
        anchor: Option<AnchorSummary>,
    ) {
        // Capture BEFORE replacing messages so a Conflict restores the original
        // pre-compaction transcript, not the compacted set.
        let rollback = MutationRollback::capture(self);
        self.messages = compacted_messages;
        self.commit_compaction(summary, removed_message_count, anchor, rollback);
    }

    /// Shared core of the compaction seam: advance the compaction metadata and
    /// `first_message_index`, then re-persist the full snapshot once, rolling
    /// back to `rollback` on a persistence Conflict. The caller owns capturing
    /// `rollback` at the correct point (see the two public entry points).
    fn commit_compaction(
        &mut self,
        summary: impl Into<String>,
        removed_message_count: usize,
        anchor: Option<AnchorSummary>,
        rollback: MutationRollback,
    ) {
        self.touch();
        let count = self.compaction.as_ref().map_or(1, |value| value.count + 1);
        let first_kept_message_index = self
            .first_message_index
            .saturating_add(u32::try_from(removed_message_count).unwrap_or(u32::MAX));
        self.first_message_index = first_kept_message_index;
        self.compaction = Some(SessionCompaction {
            count,
            removed_message_count,
            summary: summary.into(),
            first_kept_message_index: Some(first_kept_message_index),
            anchor,
        });
        // Persistence is append-only, so the on-disk JSONL still holds the
        // pre-compaction messages and would diverge from memory after a crash.
        // Re-persist the full compacted snapshot once, here at the single
        // compaction mutation point (mirrors `rewind_turns`).
        if let Some(ref persistence) = self.persistence {
            let path = persistence.path.clone();
            let result = if persistence.secure_regular_file {
                self.save_to_secure_path(&path)
            } else {
                self.save_to_path(&path)
            };
            self.finish_bound_mutation("compaction", &path, result, rollback);
        }
    }

    /// Append the about-to-be-evicted messages to the append-only Raw Vault
    /// sidecar before [`record_compaction`](Self::record_compaction) destroys
    /// them with its full-snapshot rewrite. The vault is the lossless ground
    /// truth that makes compaction recoverable: it is NEVER rewritten, and the
    /// rotation/cleanup that prunes the working transcript skips it (its name
    /// does not match the `*.rot-*.jsonl` prefix), so a message sealed here
    /// survives every later compaction round and a cold resume.
    ///
    /// `vault_seq` is a stable id stamped per message, counting up from
    /// `first_message_index` at seal time. In the steady state it equals the
    /// record's append position (both `first_message_index` and the vault count
    /// advance by the removed count each round, so ranges are contiguous), but a
    /// reader must resolve a message by its `vault_seq` *field*, not by physical
    /// line position — a session compacted before vault support starts its vault
    /// at a non-zero seq, and the duplication cases below mean position is not a
    /// reliable index. The seq does not shift when `first_message_index` later
    /// advances, so it stays valid across compaction rounds (unlike a live turn
    /// index, which `record_compaction` rebases).
    ///
    /// Call this on the compacted session *before* `record_compaction`, while
    /// `first_message_index` still holds its pre-compaction value.
    ///
    /// Best-effort: a vault write failure must never fail the compaction itself
    /// (that would brick the session), so on failure the round degrades to the
    /// pre-vault lossy behavior — never worse. No-ops for an unpersisted session
    /// (in-memory tests, `zo -p` without a store) and when
    /// `ZO_DISABLE_RAW_VAULT` is set.
    ///
    /// The seal and the destructive `record_compaction` rewrite are not atomic,
    /// so a crash in between (or two processes compacting the same session)
    /// can re-seal a batch, appending the same `vault_seq` range twice. This is
    /// duplication, never loss; the vault reader (added with `RecallArchive`)
    /// deduplicates by `vault_seq` (last record wins) and drops a torn trailing
    /// line. Cross-process locking is deferred to that reader phase.
    ///
    /// Returns the inclusive `(lo, hi)` `vault_seq` span this call sealed — the
    /// contiguous range `[first_message_index, first_message_index + len - 1]`
    /// (LAVA P1, so the continuation can advertise the exact recall range) — or
    /// `None` when nothing was sealed (empty batch, vault disabled, unpersisted
    /// session, or an append failure that degraded to lossy behavior).
    #[must_use]
    pub fn seal_evicted_to_vault(&self, evicted: &[ConversationMessage]) -> Option<(u32, u32)> {
        if evicted.is_empty() || std::env::var_os("ZO_DISABLE_RAW_VAULT").is_some() {
            return None;
        }
        let session_path = self.persistence_path()?;
        let vault_path = vault_path_for(session_path);
        if let Err(error) = append_vault_records(&vault_path, evicted, self.first_message_index) {
            eprintln!(
                "zo: raw vault seal failed; this compaction round falls back to lossy behavior: {error}"
            );
            return None;
        }
        // Mirror the per-message stamping in `append_vault_records` (base_seq +
        // offset, saturating): the highest seq is base + (len - 1).
        let lo = self.first_message_index;
        let count = u32::try_from(evicted.len()).unwrap_or(u32::MAX);
        let hi = lo.saturating_add(count.saturating_sub(1));
        Some((lo, hi))
    }

    /// Load the messages this session persisted to disk, if any — the lossless
    /// counterpart to the in-memory `messages`, used by compaction to recover a
    /// microcompact-cleared tool-result body before it is sealed to the vault.
    ///
    /// The transcript is append-only, so before `record_compaction` rewrites the
    /// snapshot it still holds every message verbatim, including bodies a cheaper
    /// microcompact tier cleared in memory (that trim is never persisted).
    /// Returns `None` for an unpersisted session or an unreadable/corrupt file —
    /// callers treat that as "no recovery source" and degrade, never fail.
    #[must_use]
    pub fn load_persisted_messages(&self) -> Option<Vec<ConversationMessage>> {
        let path = self.persistence_path()?;
        let session = Self::load_from_path(path).ok()?;
        Some(session.messages.to_vec())
    }

    /// The absolute turn index of this session's first live message — the base
    /// of the monotonic seq domain that `vault_seq` also counts in, so a caller
    /// (e.g. `session_recall`) can address a live message and an evicted vault
    /// record on the same axis. Advances by the removed count each compaction
    /// round (see [`record_compaction_with_anchor`](Self::record_compaction_with_anchor)).
    #[must_use]
    pub fn first_message_index(&self) -> u32 {
        self.first_message_index
    }

    /// Read the Raw Vault sidecar back into messages — the read side that makes
    /// compaction recoverable (used by cold-resume page-back and the
    /// `RecallArchive` tool).
    ///
    /// Records are deduplicated by `vault_seq` (last record wins): a crash
    /// between [`seal_evicted_to_vault`](Self::seal_evicted_to_vault) and the
    /// destructive rewrite, or two processes compacting the same session, can
    /// append the same seq range twice — duplication, never loss. An unparseable
    /// line (a torn trailing append, or interior corruption) is skipped, never
    /// fatal, so one bad line cannot hide the rest of the vault. Returns records
    /// ascending by `vault_seq`; empty when the session is unpersisted or has no
    /// vault yet.
    #[must_use]
    pub fn read_vault(&self) -> Vec<VaultRecord> {
        let Some(session_path) = self.persistence_path() else {
            return Vec::new();
        };
        read_vault_records(&vault_path_for(session_path))
    }

    #[must_use]
    pub fn fork(&self, branch_name: Option<String>) -> Self {
        let now = current_time_millis();
        Self {
            version: self.version,
            session_id: generate_session_id(),
            name: self.name.clone(),
            created_at_ms: now,
            updated_at_ms: now,
            messages: self.messages.clone(),
            compaction: self.compaction.clone(),
            fork: Some(SessionFork {
                parent_session_id: self.session_id.clone(),
                branch_name: normalize_optional_string(branch_name),
            }),
            session_goal: self.session_goal.clone(),
            first_message_index: self.first_message_index,
            persistence: None,
        }
    }

    pub fn to_json(&self) -> Result<JsonValue, SessionError> {
        let mut object = BTreeMap::new();
        object.insert(
            "version".to_string(),
            JsonValue::Number(i64::from(self.version)),
        );
        object.insert(
            "session_id".to_string(),
            JsonValue::String(self.session_id.clone()),
        );
        if let Some(name) = &self.name {
            object.insert("name".to_string(), JsonValue::String(name.clone()));
        }
        object.insert(
            "created_at_ms".to_string(),
            JsonValue::Number(i64_from_u64(self.created_at_ms, "created_at_ms")?),
        );
        object.insert(
            "updated_at_ms".to_string(),
            JsonValue::Number(i64_from_u64(self.updated_at_ms, "updated_at_ms")?),
        );
        object.insert(
            "messages".to_string(),
            JsonValue::Array(
                self.messages
                    .iter()
                    .map(ConversationMessage::to_json)
                    .collect(),
            ),
        );
        if let Some(compaction) = &self.compaction {
            object.insert("compaction".to_string(), compaction.to_json()?);
        }
        if let Some(fork) = &self.fork {
            object.insert("fork".to_string(), fork.to_json());
        }
        if let Some(goal) = &self.session_goal {
            object.insert("session_goal".to_string(), JsonValue::String(goal.clone()));
        }
        Ok(JsonValue::Object(object))
    }

    pub fn from_json(value: &JsonValue) -> Result<Self, SessionError> {
        let object = value
            .as_object()
            .ok_or_else(|| SessionError::Format("session must be an object".to_string()))?;
        let version = object
            .get("version")
            .and_then(JsonValue::as_i64)
            .ok_or_else(|| SessionError::Format("missing version".to_string()))?;
        let version = u32::try_from(version)
            .map_err(|_| SessionError::Format("version out of range".to_string()))?;
        let messages = object
            .get("messages")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SessionError::Format("missing messages".to_string()))?
            .iter()
            .map(ConversationMessage::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        let now = current_time_millis();
        let session_id = object
            .get("session_id")
            .and_then(JsonValue::as_str)
            .map_or_else(generate_session_id, ToOwned::to_owned);
        let name = object
            .get("name")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned);
        let created_at_ms = object
            .get("created_at_ms")
            .map(|value| required_u64_from_value(value, "created_at_ms"))
            .transpose()?
            .unwrap_or(now);
        let updated_at_ms = object
            .get("updated_at_ms")
            .map(|value| required_u64_from_value(value, "updated_at_ms"))
            .transpose()?
            .unwrap_or(created_at_ms);
        let compaction = object
            .get("compaction")
            .map(SessionCompaction::from_json)
            .transpose()?;
        let fork = object.get("fork").map(SessionFork::from_json).transpose()?;
        let session_goal = object
            .get("session_goal")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned);
        let first_message_index = compaction
            .as_ref()
            .and_then(|value| value.first_kept_message_index)
            .unwrap_or(0);
        Ok(Self {
            version,
            session_id,
            name,
            created_at_ms,
            updated_at_ms,
            messages: Arc::new(messages),
            compaction,
            fork,
            session_goal,
            first_message_index,
            persistence: None,
        })
    }

    // Cohesive JSONL deserializer: one pass building nine accumulators from the
    // record stream, then assembling `Self`. Splitting it would thread that
    // mutable state through helpers for no readability gain.
    #[allow(clippy::too_many_lines)]
    fn from_jsonl(contents: &str, from_turn: Option<u32>) -> Result<Self, SessionError> {
        let mut version = SESSION_VERSION;
        let mut session_id = None;
        let mut name = None;
        let mut created_at_ms = None;
        let mut updated_at_ms = None;
        let mut messages = Vec::new();
        let mut compaction = None;
        let mut fork = None;
        let mut session_goal = None;
        let mut first_message_index = None;

        // A crash/OOM/power-loss between an incremental append's write and its
        // newline can leave the final record torn; persistence is append-only,
        // so only the physically-last line can be partial.
        let last_content_line = last_non_blank_line_index(contents);
        // Count of records actually parsed before the current line. Torn-trailing
        // recovery only "preserves the history before the torn line" — so it is
        // legitimate only when there IS prior valid history. A file whose only
        // content line is corrupt has nothing to preserve and must be rejected as
        // corrupt rather than silently recovered into an empty session.
        let mut parsed_records = 0usize;

        for (line_index, raw_line) in contents.lines().enumerate() {
            let record = match parse_jsonl_record(raw_line, line_index + 1) {
                Ok(record) => record,
                Err(error) => {
                    if parsed_records > 0
                        && is_torn_trailing_line(line_index, last_content_line, raw_line)
                    {
                        eprintln!(
                            "zo: recovered session by dropping a torn trailing line ({error})"
                        );
                        break;
                    }
                    return Err(error);
                }
            };
            let Some(record) = record else {
                continue;
            };
            parsed_records += 1;
            match record {
                SessionJsonlRecord::Meta {
                    record_version,
                    record_session_id,
                    record_name,
                    record_created_at_ms,
                    record_updated_at_ms,
                    record_fork,
                    record_session_goal,
                } => {
                    version = record_version;
                    session_id = Some(record_session_id);
                    name = record_name;
                    created_at_ms = Some(record_created_at_ms);
                    updated_at_ms = Some(record_updated_at_ms);
                    fork = record_fork;
                    session_goal = record_session_goal;
                }
                SessionJsonlRecord::Message {
                    turn_index,
                    record_updated_at_ms,
                    message,
                    line_number,
                } => {
                    if let Some(record_updated_at_ms) = record_updated_at_ms {
                        updated_at_ms = Some(record_updated_at_ms);
                    }
                    if let Some(from_turn) = from_turn {
                        let Some(turn_index) = turn_index else {
                            return Err(SessionError::Format(format!(
                                "JSONL record at line {line_number} missing turn_index required by --from-turn"
                            )));
                        };
                        if turn_index < from_turn {
                            continue;
                        }
                    }
                    if first_message_index.is_none() {
                        first_message_index = turn_index;
                    }
                    messages.push(message);
                }
                SessionJsonlRecord::Compaction(record) => compaction = Some(record),
            }
        }

        if let (Some(from_turn), Some(compaction)) = (from_turn, compaction.as_ref()) {
            if let Some(first_kept) = compaction.first_kept_message_index {
                if from_turn < first_kept {
                    return Err(SessionError::Format(format!(
                        "--from-turn {from_turn} predates compacted history; first kept turn is {first_kept}"
                    )));
                }
            }
        }
        if from_turn.is_some()
            && messages
                .first()
                .is_some_and(message_starts_with_tool_result)
        {
            return Err(SessionError::Format(
                "--from-turn would start at an orphaned tool_result; choose an earlier turn boundary"
                    .to_string(),
            ));
        }

        let now = current_time_millis();
        let first_message_index = first_message_index
            .or_else(|| {
                compaction
                    .as_ref()
                    .and_then(|value| value.first_kept_message_index)
            })
            .unwrap_or(0);
        Ok(Self {
            version,
            session_id: session_id.unwrap_or_else(generate_session_id),
            name,
            created_at_ms: created_at_ms.unwrap_or(now),
            updated_at_ms: updated_at_ms.unwrap_or(created_at_ms.unwrap_or(now)),
            messages: Arc::new(messages),
            compaction,
            fork,
            session_goal,
            first_message_index,
            persistence: None,
        })
    }

    fn render_jsonl_snapshot(&self) -> Result<String, SessionError> {
        let mut lines = vec![self.meta_record()?.render()];
        if let Some(compaction) = &self.compaction {
            lines.push(compaction.to_jsonl_record()?.render());
        }
        lines.extend(self.messages.iter().enumerate().map(|(index, message)| {
            message_record(message, self.absolute_turn_index(index), None).render()
        }));
        let mut rendered = lines.join("\n");
        rendered.push('\n');
        Ok(rendered)
    }

    fn append_persisted_message(&self, message: &ConversationMessage) -> Result<(), SessionError> {
        let Some(persistence) = self.persistence.as_ref() else {
            return Ok(());
        };
        let path = persistence.path.clone();
        let secure = persistence.secure_regular_file;
        let turn_index = self.absolute_turn_index(self.messages.len().saturating_sub(1));
        let line = message_record(message, turn_index, Some(self.updated_at_ms)).render();

        // The append shares the same writer lease + fingerprint as full
        // snapshots, so an append and a concurrent full-snapshot rewrite can
        // never race into a lost update: whichever writer does not hold the
        // lease is refused, and a stale writer whose expected fingerprint no
        // longer matches disk gets a Conflict before appending.
        let mut state = persistence
            .writer
            .lock()
            .map_err(|_| SessionError::Conflict("session writer lease poisoned".to_string()))?;
        acquire_writer_lock(&path, &mut state)?;

        // Bootstrap (create or replace an empty file) is a full snapshot, which
        // needs the fingerprint recheck+refresh; delegate to the guarded save
        // while holding the lease already (re-entrant on the same thread would
        // deadlock, so drop the guard's lock first and let save re-take it).
        let needs_bootstrap = if secure {
            match open_secure_regular_file(&path, true) {
                Ok(file) => file.metadata()?.len() == 0,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                Err(error) => return Err(error.into()),
            }
        } else {
            !path.exists() || fs::metadata(&path)?.len() == 0
        };

        if needs_bootstrap {
            let snapshot = self.render_jsonl_snapshot()?;
            if state.expected_captured {
                let on_disk = FileFingerprint::of_path(&path).map_err(SessionError::Io)?;
                if on_disk != state.expected {
                    return Err(SessionError::Conflict(format!(
                        "session file {} changed on disk before bootstrap; refusing to overwrite",
                        path.display()
                    )));
                }
            }
            if secure {
                write_atomic_secure(&path, &snapshot)?;
            } else {
                rotate_session_file_if_needed(&path)?;
                write_atomic(&path, &snapshot)?;
                cleanup_rotated_logs(&path)?;
            }
            state.expected = Some(FileFingerprint::of_bytes(snapshot.as_bytes()));
            state.expected_captured = true;
            state.full_snapshot_state = Some(self.full_snapshot_state());
            state.transcript_dirty = false;
            return Ok(());
        }

        // Non-bootstrap append: recheck the file has not been rewritten out from
        // under us, then append one line and re-fingerprint the whole file.
        if state.expected_captured {
            let on_disk = FileFingerprint::of_path(&path).map_err(SessionError::Io)?;
            if on_disk != state.expected {
                return Err(SessionError::Conflict(format!(
                    "session file {} changed on disk since this writer last observed it; refusing to append onto newer state",
                    path.display()
                )));
            }
        }

        if secure {
            let mut file = open_secure_regular_file(&path, true)?;
            writeln!(file, "{line}")?;
            file.sync_all()?;
        } else {
            let mut file = OpenOptions::new().append(true).open(&path)?;
            // Re-assert owner-only access (best-effort) so a transcript created
            // by an older, pre-restriction build is tightened on next append.
            let _ = crate::paths::restrict_permissions_owner_only(&path);
            writeln!(file, "{line}")?;
        }
        state.expected = FileFingerprint::of_path(&path).map_err(SessionError::Io)?;
        state.expected_captured = true;
        Ok(())
    }

    fn absolute_turn_index(&self, local_index: usize) -> u32 {
        self.first_message_index
            .saturating_add(u32::try_from(local_index).unwrap_or(u32::MAX))
    }

    fn meta_record(&self) -> Result<JsonValue, SessionError> {
        let mut object = BTreeMap::new();
        object.insert(
            "type".to_string(),
            JsonValue::String("session_meta".to_string()),
        );
        object.insert(
            "version".to_string(),
            JsonValue::Number(i64::from(self.version)),
        );
        object.insert(
            "session_id".to_string(),
            JsonValue::String(self.session_id.clone()),
        );
        if let Some(name) = &self.name {
            object.insert("name".to_string(), JsonValue::String(name.clone()));
        }
        object.insert(
            "created_at_ms".to_string(),
            JsonValue::Number(i64_from_u64(self.created_at_ms, "created_at_ms")?),
        );
        object.insert(
            "updated_at_ms".to_string(),
            JsonValue::Number(i64_from_u64(self.updated_at_ms, "updated_at_ms")?),
        );
        if let Some(fork) = &self.fork {
            object.insert("fork".to_string(), fork.to_json());
        }
        if let Some(goal) = &self.session_goal {
            object.insert("session_goal".to_string(), JsonValue::String(goal.clone()));
        }
        Ok(JsonValue::Object(object))
    }

    /// Remove the last `steps` assistant turns from the session.
    ///
    /// A "turn" is one assistant message plus any immediately following
    /// user-role tool-result messages that belong to it, **and** the user
    /// message that preceded the assistant reply. Returns the number of
    /// messages actually removed.
    pub fn rewind_turns(&mut self, steps: usize) -> usize {
        if steps == 0 || self.messages.is_empty() {
            return 0;
        }

        // Snapshot BEFORE any mutation so a persistence Conflict restores the
        // exact pre-rewind messages/metadata, not the already-truncated state.
        let rollback = MutationRollback::capture(self);
        let mut removed = 0usize;
        let mut turns_removed = 0usize;

        while turns_removed < steps && !self.messages.is_empty() {
            // Walk backward: skip trailing tool-result messages first.
            while self.messages.last().is_some_and(|m| {
                m.role == MessageRole::User
                    && m.blocks
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            }) {
                Arc::make_mut(&mut self.messages).pop();
                removed += 1;
            }
            // Remove the assistant message.
            if self
                .messages
                .last()
                .is_some_and(|m| m.role == MessageRole::Assistant)
            {
                Arc::make_mut(&mut self.messages).pop();
                removed += 1;
            }
            // Remove the preceding user prompt.
            if self
                .messages
                .last()
                .is_some_and(|m| m.role == MessageRole::User)
            {
                Arc::make_mut(&mut self.messages).pop();
                removed += 1;
            }
            turns_removed += 1;
        }

        if removed > 0 {
            self.touch();
            // Re-persist the full snapshot so the on-disk state matches.
            if let Some(ref persistence) = self.persistence {
                let path = persistence.path.clone();
                let result = self.save_to_path(&path);
                // On a persistence Conflict the on-disk file belongs to a newer
                // peer; roll the removed messages back into memory so the
                // session does not diverge, and report nothing was removed.
                if self.finish_bound_mutation("rewind", &path, result, rollback) {
                    return 0;
                }
            }
        }
        removed
    }

    fn touch(&mut self) {
        self.updated_at_ms = current_time_millis();
    }
}

/// Rewrite history that references tools the current request will not advertise.
///
/// A stored `tool_use` whose `name` is not in `known_tool_names` — an MCP server
/// disconnected, `--allowedTools` was narrowed, or the model switched to a
/// provider with a different toolset — makes the converted request invalid: the
/// OpenAI-compatible path hard-400s on a `tool_use` naming a tool that is not in
/// the request's tool list. This rewrites each such `tool_use` (and its paired
/// `tool_result`) into a plain `Text` block that preserves the call/result as
/// readable context, so the request stays well-formed across a toolset change.
///
/// Expects orphan `tool_use`s to already be sealed (see
/// [`Session::tool_consistent_messages`]); a synthetic seal for an unknown tool
/// is itself rewritten here because its `tool_use_id` matches the rewritten use.
///
/// Cheap on the happy path: when every `tool_use` name is still known it shares
/// the existing `Arc` with zero allocation.
#[must_use]
pub fn reconcile_tool_history(
    messages: &Arc<Vec<ConversationMessage>>,
    known_tool_names: &BTreeSet<String>,
) -> Arc<Vec<ConversationMessage>> {
    let unknown_ids: BTreeSet<&str> = messages
        .iter()
        .flat_map(|message| &message.blocks)
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, .. } if !known_tool_names.contains(name) => {
                Some(id.as_str())
            }
            // A tool_result naming a gone tool must be rewritten even if its paired
            // tool_use is absent (a degenerate orphan result), or it reaches the
            // wire as an unpaired tool_result and 400s the OpenAI-compatible path.
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                ..
            } if !known_tool_names.contains(tool_name) => Some(tool_use_id.as_str()),
            _ => None,
        })
        .collect();
    if unknown_ids.is_empty() {
        return Arc::clone(messages);
    }

    let rebuilt: Vec<ConversationMessage> = messages
        .iter()
        .map(|message| {
            let blocks: Vec<ContentBlock> = message
                .blocks
                .iter()
                .flat_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input }
                        if unknown_ids.contains(id.as_str()) =>
                    {
                        vec![ContentBlock::Text {
                            text: format!(
                                "[prior tool call: {name} — tool no longer available] {input}"
                            ),
                        }]
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        tool_name,
                        output,
                        images,
                        ..
                    } if unknown_ids.contains(tool_use_id.as_str()) => {
                        // Rewrite the now-unpaired result to text, but PRESERVE any
                        // out-of-band images (G10) as standalone Image blocks so the
                        // model still sees them once the tool linkage is dropped.
                        let mut out = vec![ContentBlock::Text {
                            text: format!(
                                "[prior tool result: {tool_name} — tool no longer available] {output}"
                            ),
                        }];
                        out.extend(images.iter().map(|(media_type, data)| ContentBlock::Image {
                            media_type: media_type.clone(),
                            data: data.clone(),
                        }));
                        out
                    }
                    other => vec![other.clone()],
                })
                .collect();
            ConversationMessage {
                role: message.role,
                blocks,
                usage: message.usage,
                thought_signature: message.thought_signature.clone(),
                reasoning_replay: message.reasoning_replay.clone(),
                            model: None,
            }
        })
        .collect();
    Arc::new(rebuilt)
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

enum SessionJsonlRecord {
    Meta {
        record_version: u32,
        record_session_id: String,
        record_name: Option<String>,
        record_created_at_ms: u64,
        record_updated_at_ms: u64,
        record_fork: Option<SessionFork>,
        record_session_goal: Option<String>,
    },
    Message {
        turn_index: Option<u32>,
        record_updated_at_ms: Option<u64>,
        message: ConversationMessage,
        line_number: usize,
    },
    Compaction(SessionCompaction),
}

/// Index of the last non-blank line in `contents`, or `None` if every line is
/// blank. Identifies the only record a torn append could have damaged.
fn last_non_blank_line_index(contents: &str) -> Option<usize> {
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, _)| index)
        .last()
}

/// Whether a parse failure on `line_index` is a recoverable torn *trailing*
/// line rather than real corruption. A torn append truncates the final record
/// mid-write, leaving incomplete (unparseable) JSON on the physically-last
/// line; that line is dropped on load. A line that parses as JSON but violates
/// the record schema is genuine corruption (or a forward-incompatible version)
/// and is NOT recovered — nor is any interior line.
fn is_torn_trailing_line(
    line_index: usize,
    last_content_line: Option<usize>,
    raw_line: &str,
) -> bool {
    Some(line_index) == last_content_line && JsonValue::parse(raw_line.trim()).is_err()
}

fn parse_jsonl_record(
    raw_line: &str,
    line_number: usize,
) -> Result<Option<SessionJsonlRecord>, SessionError> {
    let line = raw_line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let value = JsonValue::parse(line).map_err(|error| {
        SessionError::Format(format!(
            "invalid JSONL record at line {line_number}: {error}"
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        SessionError::Format(format!(
            "JSONL record at line {line_number} must be an object"
        ))
    })?;
    let record = match object
        .get("type")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            SessionError::Format(format!("JSONL record at line {line_number} missing type"))
        })? {
        "session_meta" => SessionJsonlRecord::Meta {
            record_version: required_u32(object, "version")?,
            record_session_id: required_string(object, "session_id")?,
            record_name: object
                .get("name")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned),
            record_created_at_ms: required_u64(object, "created_at_ms")?,
            record_updated_at_ms: required_u64(object, "updated_at_ms")?,
            record_fork: object.get("fork").map(SessionFork::from_json).transpose()?,
            record_session_goal: object
                .get("session_goal")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned),
        },
        "message" => SessionJsonlRecord::Message {
            turn_index: parse_turn_index(object, line_number)?,
            record_updated_at_ms: object
                .get("updated_at_ms")
                .map(|_| required_u64(object, "updated_at_ms"))
                .transpose()?,
            message: ConversationMessage::from_json(object.get("message").ok_or_else(|| {
                SessionError::Format(format!(
                    "JSONL record at line {line_number} missing message"
                ))
            })?)?,
            line_number,
        },
        "compaction" => SessionJsonlRecord::Compaction(SessionCompaction::from_json(
            &JsonValue::Object(object.clone()),
        )?),
        other => {
            return Err(SessionError::Format(format!(
                "unsupported JSONL record type at line {line_number}: {other}"
            )));
        }
    };
    Ok(Some(record))
}

fn parse_turn_index(
    object: &BTreeMap<String, JsonValue>,
    line_number: usize,
) -> Result<Option<u32>, SessionError> {
    object
        .get("turn_index")
        .map(|value| {
            value
                .as_i64()
                .and_then(|raw| u32::try_from(raw).ok())
                .ok_or_else(|| {
                    SessionError::Format(format!(
                        "JSONL record at line {line_number} has invalid turn_index"
                    ))
                })
        })
        .transpose()
}

fn message_record(
    message: &ConversationMessage,
    turn_index: u32,
    updated_at_ms: Option<u64>,
) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert("type".to_string(), JsonValue::String("message".to_string()));
    object.insert(
        "turn_index".to_string(),
        JsonValue::Number(i64::from(turn_index)),
    );
    if let Some(updated_at_ms) = updated_at_ms {
        object.insert(
            "updated_at_ms".to_string(),
            JsonValue::Number(i64::try_from(updated_at_ms).unwrap_or(i64::MAX)),
        );
    }
    object.insert("message".to_string(), message.to_json());
    JsonValue::Object(object)
}

/// Path of the Raw Vault sidecar for a session transcript: `<id>.jsonl` →
/// `<id>.vault.jsonl`, alongside the transcript so it travels with `--resume`.
/// The `.vault.` infix keeps it clear of the `<id>.rot-*.jsonl` rotation prefix
/// so [`cleanup_rotated_logs`] never deletes it.
fn vault_path_for(session_path: &Path) -> PathBuf {
    let stem = session_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    session_path.with_file_name(format!("{stem}.vault.jsonl"))
}

/// Append `evicted` to the append-only vault, stamping each with a `vault_seq`
/// counting up from `base_seq` (the session's pre-compaction `first_message_index`).
///
/// The whole batch is rendered into one buffer and written with a single
/// `write_all` to an `O_APPEND` handle: that collapses N per-message torn-write
/// windows into one and lets the OS append the batch atomically in the common
/// (small-batch) case. A crash mid-write can still leave a torn trailing line,
/// which the vault reader recovers from (drop the torn last line) — the same
/// contract the session transcript uses.
fn append_vault_records(
    path: &Path,
    evicted: &[ConversationMessage],
    base_seq: u32,
) -> Result<(), SessionError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut batch = String::new();
    for (offset, message) in evicted.iter().enumerate() {
        let seq = base_seq.saturating_add(u32::try_from(offset).unwrap_or(u32::MAX));
        batch.push_str(&vault_record(message, seq).render());
        batch.push('\n');
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    // Match the transcript's owner-only (0o600) hardening — the vault holds the
    // same raw prompts and file contents. Best-effort, like `write_atomic`.
    let _ = crate::paths::restrict_permissions_owner_only(path);
    file.write_all(batch.as_bytes())?;
    Ok(())
}

fn vault_record(message: &ConversationMessage, vault_seq: u32) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert("type".to_string(), JsonValue::String("vault".to_string()));
    object.insert(
        "vault_seq".to_string(),
        JsonValue::Number(i64::from(vault_seq)),
    );
    object.insert("message".to_string(), message.to_json());
    JsonValue::Object(object)
}

/// A raw message recovered from the Raw Vault, with its stable `vault_seq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultRecord {
    pub vault_seq: u32,
    pub message: ConversationMessage,
}

/// Read and normalize a vault file: dedup by `vault_seq` (last wins), skip
/// unparseable lines (torn/corrupt — quarantine, never abort), sorted ascending.
fn read_vault_records(path: &Path) -> Vec<VaultRecord> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut by_seq: BTreeMap<u32, ConversationMessage> = BTreeMap::new();
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((seq, message)) = parse_vault_line(line) {
            by_seq.insert(seq, message);
        }
    }
    by_seq
        .into_iter()
        .map(|(vault_seq, message)| VaultRecord { vault_seq, message })
        .collect()
}

fn parse_vault_line(line: &str) -> Option<(u32, ConversationMessage)> {
    let value = JsonValue::parse(line).ok()?;
    let object = value.as_object()?;
    if object.get("type").and_then(JsonValue::as_str) != Some("vault") {
        return None;
    }
    let seq = object
        .get("vault_seq")
        .and_then(JsonValue::as_i64)
        .and_then(|raw| u32::try_from(raw).ok())?;
    let message = ConversationMessage::from_json(object.get("message")?).ok()?;
    Some((seq, message))
}

fn message_starts_with_tool_result(message: &ConversationMessage) -> bool {
    message
        .blocks
        .first()
        .is_some_and(|block| matches!(block, ContentBlock::ToolResult { .. }))
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

fn generate_session_id() -> String {
    let millis = current_time_millis();
    let counter = SESSION_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("session-{millis}-{counter}")
}

fn open_secure_regular_file(path: &Path, append: bool) -> std::io::Result<fs::File> {
    let mut options = OpenOptions::new();
    if append {
        options.append(true);
    } else {
        options.read(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    validate_secure_regular_file(&file)?;
    Ok(file)
}

fn validate_secure_regular_file(file: &fs::File) -> std::io::Result<()> {
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "session persistence target is not a regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.nlink() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "session persistence target has multiple hard links",
            ));
        }
    }
    Ok(())
}

fn read_secure_regular_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;
    let mut file = open_secure_regular_file(path, false)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

/// Lazily acquire the exclusive advisory writer lock for a bound session path
/// and retain it in `state` for the shared writer state's lifetime.
///
/// On Unix this is a nonblocking `flock(LOCK_EX | LOCK_NB)` on a sibling
/// `<file>.lock`, created owner-only and `O_NOFOLLOW` (symlink-safe). If a peer
/// process already holds it, we return [`SessionError::Conflict`] instead of
/// blocking — the caller must not silently proceed. Once held, all clones of
/// this `Session` share the lock (same `Arc<Mutex<WriterState>>`), so they do
/// not contend with each other, only with independently loaded writers.
///
/// On non-Unix there is no advisory lock; correctness there rests on the
/// optimistic fingerprint recheck the callers already perform (a narrower,
/// documented guarantee: it catches sequential stale rewrites but not a truly
/// simultaneous interleave).
fn acquire_writer_lock(path: &Path, state: &mut WriterState) -> Result<(), SessionError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        if state.lock.is_some() {
            return Ok(());
        }
        let lock_path = lock_sibling_path(path);
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(nix::libc::O_NOFOLLOW)
            .open(&lock_path)?;
        // Nonblocking exclusive advisory lock via nix's safe wrapper (the
        // workspace forbids `unsafe`). flock is released automatically when the
        // descriptor closes — including on process crash — so a crashed peer
        // never leaves a wedged lock.
        match nix::fcntl::Flock::lock(file, nix::fcntl::FlockArg::LockExclusiveNonblock) {
            Ok(flock) => {
                state.lock = Some(flock);
                Ok(())
            }
            Err((_file, errno))
                if errno == nix::errno::Errno::EWOULDBLOCK
                    || errno == nix::errno::Errno::EAGAIN =>
            {
                Err(SessionError::Conflict(format!(
                    "session file {} is being written by another process (writer lease held)",
                    path.display()
                )))
            }
            Err((_file, errno)) => Err(SessionError::Io(std::io::Error::from(errno))),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, state);
        Ok(())
    }
}

/// Snapshot of the in-memory `Session` fields that `record_compaction_with_anchor`
/// and `rewind_turns` mutate, captured before the mutation so a persistence
/// Conflict can restore them. `messages` is an `Arc`, so capture is a pointer
/// clone, not a deep copy.
struct MutationRollback {
    updated_at_ms: u64,
    messages: Arc<Vec<ConversationMessage>>,
    compaction: Option<SessionCompaction>,
    first_message_index: u32,
}

impl MutationRollback {
    fn capture(session: &Session) -> Self {
        Self {
            updated_at_ms: session.updated_at_ms,
            messages: Arc::clone(&session.messages),
            compaction: session.compaction.clone(),
            first_message_index: session.first_message_index,
        }
    }

    fn restore(self, session: &mut Session) {
        session.updated_at_ms = self.updated_at_ms;
        session.messages = self.messages;
        session.compaction = self.compaction;
        session.first_message_index = self.first_message_index;
    }
}

/// Whether `bound` and `candidate` name the same session file, robust to path
/// aliasing (relative vs absolute, `./` segments, symlinks). A plain string
/// equality lets a caller bypass the writer lease/fingerprint by targeting the
/// bound file through an alias, so we compare by resolved filesystem identity.
///
/// Rules that preserve existing contracts:
/// - When both paths resolve on disk, compare OS-level identity (device+inode
///   on Unix; canonicalized path elsewhere). This catches every alias of an
///   existing bound file.
/// - When the candidate cannot be resolved (e.g. a not-yet-created export
///   target), fall back to normalized string comparison. A genuinely different,
///   nonexistent export therefore stays unguarded (existing overwrite
///   semantics), while an alias that merely lacks a canonical form still matches
///   the bound path by string.
///
/// This only selects whether the lease guard applies; it performs no write and
/// does not relax the secure path's own `O_NOFOLLOW` symlink policy.
fn paths_identify_same_file(bound: &Path, candidate: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if let (Ok(bound_meta), Ok(candidate_meta)) =
            (std::fs::metadata(bound), std::fs::metadata(candidate))
        {
            return bound_meta.dev() == candidate_meta.dev()
                && bound_meta.ino() == candidate_meta.ino();
        }
    }
    #[cfg(not(unix))]
    {
        if let (Ok(bound_canon), Ok(candidate_canon)) =
            (std::fs::canonicalize(bound), std::fs::canonicalize(candidate))
        {
            return bound_canon == candidate_canon;
        }
    }
    // Fall back to a lexical comparison when the filesystem cannot resolve one
    // side (typically the export target does not exist yet).
    normalize_lexical(bound) == normalize_lexical(candidate)
}

/// Lexically normalize a path for the fallback comparison: make it absolute
/// against the current directory when relative, and collapse `.`/`..` segments
/// without touching the filesystem. Purely syntactic — used only when identity
/// cannot be resolved on disk.
fn normalize_lexical(path: &Path) -> PathBuf {
    use std::path::Component;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// Sibling advisory-lock path for a session file: `<file>.lock` in the same
/// directory, so the lock shares the session's access controls and directory.
fn lock_sibling_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    path.with_file_name(format!("{file_name}.lock"))
}

fn write_atomic_secure(path: &Path, contents: &str) -> Result<(), SessionError> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // Reject a target that a peer or attacker has turned into a symlink or a
    // non-regular file: `rename` replaces the name in *this* directory, and we
    // only ever write to a fresh O_EXCL temp (below), so this guards the
    // final publish location's type. We deliberately do NOT gate on the
    // target's `nlink` here: under concurrent writers the target inode is being
    // atomically replaced by another writer's rename, so a stat of it can
    // transiently observe `nlink` of 0 (old inode already unlinked) or 2 (mid
    // link/unlink) even though nothing is wrong. That transient was a spurious
    // multi-process failure. The meaningful hard-link guard is on the temp we
    // exclusively create and actually write into (`validate_secure_regular_file`
    // below), whose `nlink` is authoritative because O_EXCL guarantees we made
    // it.
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if !metadata.file_type().is_file() {
            return Err(SessionError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "session persistence target is not a regular file",
            )));
        }
    }
    let (mut file, temp_path) = create_temp_exclusive(path, |options| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .mode(0o600)
                .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
        }
        #[cfg(not(unix))]
        {
            let _ = options;
        }
    })?;
    let result = (|| -> Result<(), SessionError> {
        validate_secure_regular_file(&file)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        fs::rename(&temp_path, path)?;
        // fsync the directory so the rename that published the snapshot survives
        // a crash, not just the file's own data. Best-effort (see helper).
        sync_parent_dir(path);
        Ok(())
    })();
    if result.is_err() {
        // Only ever remove the temp this call exclusively created (create_new),
        // never a path a peer writer might own.
        let _ = fs::remove_file(&temp_path);
    }
    result
}

/// fsync the directory containing `path` so a completed rename is durable.
/// Best-effort by design: some platforms/filesystems reject opening a directory
/// for sync, and the snapshot's own bytes are already fsynced before the rename,
/// so a directory-sync failure must never fail an otherwise-successful save.
fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = fs::File::open(parent).and_then(|dir| dir.sync_all());
    }
}

fn write_atomic(path: &Path, contents: &str) -> Result<(), SessionError> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // Exclusive create (O_EXCL) via a unique per-writer name: two processes
    // saving in the same millisecond can no longer open and truncate the same
    // temp, which previously let one process's rename steal the other's file.
    let (mut file, temp_path) = create_temp_exclusive(path, |_options| {})?;
    let result = (|| -> Result<(), SessionError> {
        file.write_all(contents.as_bytes())?;
        // Persist the snapshot's bytes before the rename so a crash cannot leave
        // a renamed-but-empty transcript. Session transcripts hold prompts and
        // file contents, so lock the file to the owner (0o600) before publishing
        // it via the rename. Best-effort permissions: a privacy hardening must
        // never fail the save itself — on a filesystem without POSIX
        // permissions, or a session directory the process does not own, the
        // write still succeeds. The session *directory* is restricted where
        // zo creates it (see `runtime::session_control`), not against
        // arbitrary parents here.
        file.sync_all()?;
        let _ = crate::paths::restrict_permissions_owner_only(&temp_path);
        fs::rename(&temp_path, path)?;
        sync_parent_dir(path);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

/// Global monotonic sequence for temp-file names. Distinct from
/// `SESSION_ID_COUNTER` so that bumping temp uniqueness never perturbs session
/// id allocation. Combined with the process id below, this makes a temp name
/// unique across every writer on the host: process id separates processes,
/// this counter separates concurrent writers inside one process, and `attempt`
/// separates retries within one write.
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Build a collision-free temp path in the target's own directory. The name
/// carries the process id, a millisecond timestamp, a process-global monotonic
/// sequence, and the retry `attempt`. Two Zo processes can no longer land on
/// the same `<file>.tmp-<millis>-<counter>` name just because they first saved
/// in the same millisecond — the process id and per-attempt salt keep every
/// writer's temp distinct. Staying in the same directory preserves the
/// same-filesystem rename that makes the publish atomic.
fn temporary_path_for(path: &Path, attempt: u32) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    path.with_file_name(format!(
        "{file_name}.tmp-{}-{}-{}-{}",
        std::process::id(),
        current_time_millis(),
        TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        attempt
    ))
}

/// Number of distinct temp names to try before giving up. Each attempt draws a
/// fresh name, so an [`std::io::ErrorKind::AlreadyExists`] — whether from a live
/// concurrent writer that happened to collide or a stale temp left by a crashed
/// process — is retried with a new name rather than clobbering an in-flight
/// file. The bound keeps a pathological directory from spinning forever.
const TEMP_CREATE_MAX_ATTEMPTS: u32 = 16;

/// Exclusively create a fresh temp file next to `path`, retrying with a new
/// unique name on `AlreadyExists`. Both the secure and non-secure write paths
/// funnel through this so neither can ever open a temp another writer is using:
/// `create_new` (`O_EXCL`) guarantees the returned file is one this call alone
/// created, so a later cleanup that removes it can never delete a peer's temp.
/// `configure` applies path-specific open flags (owner-only mode, `O_NOFOLLOW`)
/// before the exclusive create.
fn create_temp_exclusive(
    path: &Path,
    configure: impl Fn(&mut OpenOptions),
) -> Result<(fs::File, PathBuf), SessionError> {
    let mut last_error: Option<std::io::Error> = None;
    for attempt in 0..TEMP_CREATE_MAX_ATTEMPTS {
        let temp_path = temporary_path_for(path, attempt);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        configure(&mut options);
        match options.open(&temp_path) {
            Ok(file) => return Ok((file, temp_path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_error = Some(error);
            }
            Err(error) => return Err(SessionError::Io(error)),
        }
    }
    Err(SessionError::Io(last_error.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "exhausted unique session temp-file names",
        )
    })))
}

fn rotate_session_file_if_needed(path: &Path) -> Result<(), SessionError> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(());
    };
    if metadata.len() < ROTATE_AFTER_BYTES {
        return Ok(());
    }
    let rotated_path = rotated_log_path(path);
    fs::rename(path, rotated_path)?;
    Ok(())
}

fn rotated_log_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    path.with_file_name(format!("{stem}.rot-{}.jsonl", current_time_millis()))
}

fn cleanup_rotated_logs(path: &Path) -> Result<(), SessionError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    let prefix = format!("{stem}.rot-");
    let mut rotated_paths = fs::read_dir(parent)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|entry_path| {
            entry_path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| {
                    name.starts_with(&prefix)
                        && Path::new(name)
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
                })
        })
        .collect::<Vec<_>>();

    rotated_paths.sort_by_key(|entry_path| {
        fs::metadata(entry_path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH)
    });

    let remove_count = rotated_paths.len().saturating_sub(MAX_ROTATED_FILES);
    for stale_path in rotated_paths.into_iter().take(remove_count) {
        fs::remove_file(stale_path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
