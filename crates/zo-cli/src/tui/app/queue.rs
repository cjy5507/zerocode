//! Queued user inputs: steering-message queue, pending pasted images,
//! clipboard image ring, and the pending external-editor request.

use std::path::PathBuf;

use runtime::message_stream::SystemLevel;

use super::App;
use super::types::{
    AgentResultMeta, ImageAttachment, QueueLimitError, QueuedMessage, TranscriptViewRequest,
};

/// Maximum number of composed prompts retained while a turn is running.
///
/// The queue is a type-ahead convenience, not an unbounded transcript store.
pub(crate) const MAX_QUEUED_MESSAGES: usize = 64;

/// Maximum number of base64 clipboard images retained before the next submit.
///
/// Images are the riskiest entries because each `data` field can be multi-MB.
pub(crate) const MAX_PENDING_IMAGES: usize = 8;

impl App {
    /// Drain and return all messages queued while input was disabled.
    ///
    /// The caller (turn controller) calls this after a turn finishes
    /// to submit queued prompts in order.
    pub fn take_queued_messages(&mut self) -> Vec<QueuedMessage> {
        self.queued_messages.drain(..).collect()
    }

    /// Queue a prompt to run as the next user turn.
    pub fn queue_message(&mut self, message: impl Into<String>) -> Result<(), QueueLimitError> {
        self.ensure_can_queue_message()?;
        self.queued_messages.push_back(QueuedMessage {
            text: message.into(),
            images: Vec::new(),
            goal_owned: false,
            loop_id: None,
            agent_result: None,
            steered: false,
        });
        Ok(())
    }

    /// Queue a re-injected **background sub-agent result** as the next user turn.
    /// The `text` still submits as a normal user-role turn so the model reads the
    /// agent's result, but it is tagged so the REPL renders a collapsible
    /// agent-result card (authored by the agent) instead of an amber `You`
    /// message. See [`QueuedMessage::agent_result`].
    pub fn queue_agent_result_message(
        &mut self,
        message: impl Into<String>,
        meta: AgentResultMeta,
    ) -> Result<(), QueueLimitError> {
        self.ensure_can_queue_message()?;
        self.queued_messages.push_back(QueuedMessage {
            text: message.into(),
            images: Vec::new(),
            goal_owned: false,
            loop_id: None,
            agent_result: Some(meta),
            steered: false,
        });
        Ok(())
    }

    /// Queue a `/goal` action/repair prompt, tagged so the REPL latches goal
    /// ownership when *this* message pops (not when it was dispatched). Keeps a
    /// user message typed ahead of it from consuming the goal's verifier verdict.
    pub fn queue_goal_message(
        &mut self,
        message: impl Into<String>,
    ) -> Result<(), QueueLimitError> {
        self.ensure_can_queue_message()?;
        self.queued_messages.push_back(QueuedMessage {
            text: message.into(),
            images: Vec::new(),
            goal_owned: true,
            loop_id: None,
            agent_result: None,
            steered: false,
        });
        Ok(())
    }

    /// Queue a `/loop`-owned run, tagged with its loop id so the controller can
    /// gate (and drop) it at *pop* time — see [`QueuedMessage::loop_id`]. This is
    /// what makes a fixed-count loop stoppable instead of fire-and-forget.
    pub fn queue_loop_message(
        &mut self,
        message: impl Into<String>,
        loop_id: impl Into<String>,
    ) -> Result<(), QueueLimitError> {
        self.ensure_can_queue_message()?;
        self.queued_messages.push_back(QueuedMessage {
            text: message.into(),
            images: Vec::new(),
            goal_owned: false,
            loop_id: Some(loop_id.into()),
            agent_result: None,
            steered: false,
        });
        Ok(())
    }

    /// `true` when another composed prompt can be retained without growing the
    /// queue past its hard cap.
    #[must_use]
    pub fn can_queue_message(&self) -> bool {
        self.queued_messages.len() < MAX_QUEUED_MESSAGES
    }

    /// Return an overflow error when the composed prompt queue is full.
    pub fn ensure_can_queue_message(&self) -> Result<(), QueueLimitError> {
        if self.can_queue_message() {
            Ok(())
        } else {
            Err(QueueLimitError::QueuedMessagesFull {
                limit: MAX_QUEUED_MESSAGES,
            })
        }
    }

    pub(crate) fn queue_composed_message(
        &mut self,
        message: QueuedMessage,
    ) -> Result<(), QueueLimitError> {
        self.ensure_can_queue_message()?;
        self.queued_messages.push_back(message);
        Ok(())
    }

    /// Surface a bounded-queue rejection without changing queue/input state.
    pub fn report_queue_limit_error(&mut self, error: QueueLimitError) {
        self.push_diff_note(SystemLevel::Warn, format!("Input queue full: {error}"));
    }

    /// Stage images from an already-accepted queued message for the submit path.
    ///
    /// This is not a new paste admission point: the queued entry was bounded
    /// when it was composed, and these attachments are moved rather than cloned.
    pub fn stage_queued_images_for_submit(&mut self, images: Vec<ImageAttachment>) {
        debug_assert!(
            self.pending_images.is_empty(),
            "queued image replay expects a clean pending-image staging area"
        );
        for image in images {
            self.pending_images.push(image);
            self.input.add_image();
        }
    }

    /// Stage the agent-result provenance of a popped [`QueuedMessage`] so the
    /// next submit renders an agent-result card. Mirrors
    /// [`Self::stage_queued_images_for_submit`]; consumed once by
    /// [`Self::take_pending_agent_result`] at submit.
    pub fn stage_queued_agent_result_for_submit(&mut self, meta: Option<AgentResultMeta>) {
        self.pending_agent_result = meta;
    }

    /// Take the staged agent-result provenance for the current submit, clearing
    /// it. `Some` only for a re-injected background agent result; `None` for
    /// ordinary user input (which submits as a `You` message).
    pub fn take_pending_agent_result(&mut self) -> Option<AgentResultMeta> {
        self.pending_agent_result.take()
    }

    /// Pop and return the oldest queued message (FIFO), or `None` when empty.
    ///
    /// Called at the top of the REPL loop so each queued prompt is
    /// auto-submitted as its own turn — preserving send-order while
    /// keeping `take_queued_messages` available for callers that want
    /// to flush the whole queue at once.
    pub fn pop_next_queued_message(&mut self) -> Option<QueuedMessage> {
        self.queued_messages.pop_front()
    }

    /// Remove and return every queued **agent-result** re-injection, preserving
    /// their relative order and leaving user/goal/loop messages in their FIFO
    /// slots. The REPL calls this after popping an agent-result head so a batch
    /// of background completions folds into ONE follow-up turn instead of
    /// dispatching one near-identical alarm turn per completion.
    pub fn drain_queued_agent_results(&mut self) -> Vec<QueuedMessage> {
        let mut drained = Vec::new();
        self.queued_messages.retain_mut(|message| {
            if message.agent_result.is_some() {
                drained.push(std::mem::take(message));
                false
            } else {
                true
            }
        });
        drained
    }

    /// `true` when the queue holds at least one pending prompt.
    #[must_use]
    pub fn has_queued_messages(&self) -> bool {
        !self.queued_messages.is_empty()
    }

    /// Number of messages currently sitting in the queue.
    #[must_use]
    pub fn queued_message_count(&self) -> usize {
        self.queued_messages.len()
    }

    /// Drain any images that were pasted from the clipboard since the last
    /// submit. The caller attaches them to the outbound user message.
    pub fn take_pending_images(&mut self) -> Vec<ImageAttachment> {
        std::mem::take(&mut self.pending_images)
    }

    /// Record that the `/memory` command wants the host to suspend the TUI and
    /// open `path` in `$EDITOR`. The host drains this with
    /// [`Self::take_pending_editor_file`] after slash dispatch.
    pub fn request_file_edit(&mut self, path: PathBuf) {
        self.pending_editor_file = Some(path);
    }

    /// Drain a pending external-editor request, if any.
    pub fn take_pending_editor_file(&mut self) -> Option<PathBuf> {
        self.pending_editor_file.take()
    }

    /// Record that the `/dump` command wants the host to suspend the TUI and
    /// open the transcript dump at `path` in `$PAGER` (`edit` → `$EDITOR`).
    /// The host drains this with [`Self::take_pending_transcript_view`] after
    /// slash dispatch.
    pub fn request_transcript_view(&mut self, path: PathBuf, edit: bool) {
        self.pending_transcript_view = Some(TranscriptViewRequest { path, edit });
    }

    /// Drain a pending `/dump` external-viewer request, if any.
    pub fn take_pending_transcript_view(&mut self) -> Option<TranscriptViewRequest> {
        self.pending_transcript_view.take()
    }

    /// `true` when there are clipboard images awaiting submission.
    #[must_use]
    pub fn has_pending_images(&self) -> bool {
        !self.pending_images.is_empty()
    }

    /// Push a clipboard image attachment for inclusion in the next submit.
    /// Also updates the input widget to show a visual badge.
    pub fn push_clipboard_image(
        &mut self,
        media_type: String,
        data: String,
    ) -> Result<(), QueueLimitError> {
        if self.pending_images.len() >= MAX_PENDING_IMAGES {
            return Err(QueueLimitError::PendingImagesFull {
                limit: MAX_PENDING_IMAGES,
            });
        }
        self.pending_images
            .push(ImageAttachment { media_type, data });
        self.input.add_image();
        Ok(())
    }

    /// Remove the most recently added clipboard image.
    /// Returns `true` if an image was removed.
    pub fn pop_clipboard_image(&mut self) -> bool {
        if self.input.remove_last_image() {
            self.pending_images.pop();
            true
        } else {
            false
        }
    }

    /// Remove the first queued *text* message whose content matches `steer`
    /// (trim-equal). Called when that content was delivered into the live turn
    /// via steering, so the queue must not run it again as its own turn.
    pub fn remove_queued_message_matching(&mut self, steer: &str) -> bool {
        let needle = steer.trim();
        if let Some(pos) = self
            .queued_messages
            .iter()
            .position(|message| message.images.is_empty() && message.text.trim() == needle)
        {
            self.queued_messages.remove(pos).is_some()
        } else {
            false
        }
    }
}
