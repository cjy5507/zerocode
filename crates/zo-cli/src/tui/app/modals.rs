//! Modal lifecycle: every `open_*` entry point, modal exit, and the
//! workflow/rewind viewer surfaces.

use super::types::AppMode;
use super::{
    AgentsViewerModal, ApiKeyConnectInfo, ApiKeyModal, App, AppAction, ChoicePickerModal,
    CustomProviderWizardModal, DeepTierModal, DeepTierView, DiffViewerModal, EffortPickerModal,
    FilePickerModal, Modal, ModalResult, ModalSelection,
    ModelPickerEntry, ModelPickerModal, PermissionPickerModal, PermissionPrompt,
    ReportViewerBlock, ReportViewerModal, RewindViewerModal,
    ReviewModal,
    RuntimePermissionMode, SmartSettingsModal, TeamInboxViewerModal, ToolToggleModal, ToolToggleRow,
    UsageDashboardModal, UserQuestionModal, UserQuestionPrompt, WorkflowView, WorkflowViewerModal,
    RemoteOnboardingModal, RemoteOnboardingView,
    collect_workspace_files,
    new_scan_cancel_token,
};
use crate::tui::modals::workflow_viewer::WorkflowAgentRow;
use crate::tui::workflow_progress::AgentRowsSnapshot;
use runtime::TeamInboxSnapshot;

impl App {
    /// Open the permission modal for `prompt`.
    ///
    /// Stores the prompt (so the oneshot responder stays alive) and
    /// switches the app into [`AppMode::ModalChoice`]. The host loop
    /// is responsible for wiring the selected [`PermissionDecision`]
    /// through `prompt.responder` when the modal resolves.
    pub fn open_permission_modal(&mut self, prompt: PermissionPrompt) {
        self.modals.workflow = None;
        self.modals.agents = None;
        self.modals.team_inbox = None;
        self.workflow_view_cache = crate::tui::workflow_progress::WorkflowViewCache::default();
        self.modals.usage_dashboard = None;
        // A permission prompt is ingested from the async agent loop regardless
        // of the current mode, so it can land while a slot modal (e.g. the arg
        // picker) is open; clear the slot so it cannot shadow this prompt.
        self.active_modal = None;
        self.active_user_question = None;
        // Focus the safe default (Deny) so a reflexive Enter denies, not allows.
        self.permission_selected = crate::tui::blocks::permission::default_selected_index(&prompt);
        self.active_prompt = Some(prompt);
        self.mode = AppMode::ModalChoice;
    }

    /// Open the `AskUserQuestion` modal for `prompt`.
    pub fn open_user_question_modal(&mut self, prompt: UserQuestionPrompt) {
        // `set_active_modal` → `exit_modal` clears any prior modal (and its
        // pending responder) before this one takes the slot, so a question
        // ingested from the async agent loop cannot be shadowed by an open modal.
        let modal = UserQuestionModal::from_prompt(&prompt);
        self.set_active_modal(Box::new(modal), AppMode::ModalQuestion);
        // Hold the prompt (with its oneshot responder) until the modal resolves.
        self.active_user_question = Some(prompt);
    }

    /// Open the `/model` picker modal.
    pub fn open_model_modal(&mut self, entries: Vec<ModelPickerEntry>) {
        self.set_active_modal(Box::new(ModelPickerModal::new(entries)), AppMode::ModalModel);
    }

    /// Open the `/permissions` picker modal.
    pub fn open_permission_picker_modal(&mut self, mode: RuntimePermissionMode) {
        self.set_active_modal(
            Box::new(PermissionPickerModal::with_selected(mode)),
            AppMode::ModalPermissions,
        );
    }

    /// Open the `/resume` session picker modal.
    ///
    /// `labels` are the display strings; `ids` are the corresponding
    /// session identifiers used when loading the chosen session.
    pub fn open_session_modal(&mut self, labels: Vec<String>, ids: Vec<String>) {
        self.set_active_modal(
            Box::new(ChoicePickerModal::new("Resume session", labels)),
            AppMode::ModalSession,
        );
        // Set after `set_active_modal`, which clears the choice side-lists via
        // `exit_modal`; the parallel `session_ids` back the picker's indices.
        self.choice_modals.session_ids = ids;
    }

    /// Open the `/login` · `/connect` provider picker modal.
    ///
    /// `labels` are the display strings; `ids` are the matching command/provider
    /// tokens. The chosen entry is re-submitted via the token's command prefix,
    /// so `/login` starts OAuth while `/connect` remains a status check.
    pub fn open_login_modal(
        &mut self,
        title: impl Into<String>,
        labels: Vec<String>,
        ids: Vec<String>,
    ) {
        self.set_active_modal(
            Box::new(ChoicePickerModal::new(title, labels)),
            AppMode::ModalLogin,
        );
        // Set after `set_active_modal`, which clears the choice side-lists via
        // `exit_modal`; the parallel `login_provider_ids` back the picker's
        // indices (each is a `command:provider` token).
        self.choice_modals.login_provider_ids = ids;
    }

    /// Open the `/connect` API-key setup modal for a cloud adapter preset.
    pub fn open_connect_api_key_modal(&mut self, info: ApiKeyConnectInfo) {
        self.set_active_modal(Box::new(ApiKeyModal::new(info)), AppMode::ModalApiKey);
    }

    /// Open the `/connect` custom OpenAI-compatible provider wizard.
    pub fn open_custom_provider_modal(&mut self) {
        self.set_active_modal(
            Box::new(CustomProviderWizardModal::new()),
            AppMode::ModalCustomProvider,
        );
    }

    /// Open a generic fixed-choice argument picker for `command`.
    ///
    /// Mirrors the `/login` modal pattern: the chosen option is re-submitted
    /// as `/<command> <label>`, so the command's existing text handler runs
    /// unchanged and there is no per-command modal to maintain. `title` is the
    /// border caption (e.g. `"/theme"`); `options` are the selectable labels.
    pub fn open_arg_picker(
        &mut self,
        command: &str,
        title: impl Into<String>,
        options: Vec<String>,
    ) {
        // Set the command before `set_active_modal` (which clears the sibling
        // modals + choice side-lists via `exit_modal`, but not this field) so
        // `apply_choice_selection` can rebuild `/{command} {label}` on Enter.
        self.choice_modals.arg_picker_command = command.to_string();
        self.set_active_modal(
            Box::new(ChoicePickerModal::new(title, options)),
            AppMode::ModalArgPick,
        );
    }

    /// Open the `/effort` slider modal, pre-positioned on the step that
    /// best matches `current_budget` (`None` ⇒ first step).
    pub fn open_effort_modal(&mut self, current_budget: Option<u32>) {
        self.set_active_modal(
            Box::new(EffortPickerModal::with_budget(current_budget)),
            AppMode::ModalEffort,
        );
    }

    /// Open the interactive `/diff` viewer over the given per-file diffs.
    pub fn open_diff_viewer(&mut self, files: Vec<runtime::message_stream::DiffView>) {
        self.active_prompt = None;
        self.active_user_question = None;
        self.modals.usage_dashboard = None;
        self.modals.team_inbox = None;
        self.choice_modals.session_ids.clear();
        self.modals.diff_viewer = Some(DiffViewerModal::new(files));
        self.mode = AppMode::ModalDiff;
    }

    /// Open the `@`-triggered fuzzy file-reference picker.
    ///
    /// Scans the current working directory for files (bounded, skipping
    /// VCS/build/dependency dirs) and hands the relative paths to the
    /// Open the `@`-triggered fuzzy [`FilePickerModal`].
    ///
    /// The cwd scan is **not** run inline: it would block the UI thread on
    /// a large repo (the BFS visits up to 4000 dirs). Instead the modal
    /// opens instantly in a "scanning…" state and the walk runs on a
    /// `spawn_blocking` worker; [`App::poll_file_scan`] lands the result
    /// from the run loop once it finishes. When no tokio runtime is
    /// available (unit tests) the scan falls back to a synchronous call so
    /// the picker is still populated deterministically.
    pub fn open_file_picker(&mut self) {
        self.cancel_file_scan();
        // Open empty + loading first so the UI is responsive immediately.
        self.open_file_picker_with(Vec::new());
        if let Some(modal) = self.active_modal_as::<FilePickerModal>() {
            modal.set_loading(true);
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let cancel = new_scan_cancel_token();
            let worker_cancel = std::sync::Arc::clone(&cancel);
            self.scans.file_cancel = Some(cancel);
            self.scans.file_task =
                Some(handle.spawn_blocking(move || collect_workspace_files(&worker_cancel)));
        } else {
            // No async runtime (tests): scan synchronously and land it now.
            let cancel = new_scan_cancel_token();
            let items = collect_workspace_files(&cancel);
            if let Some(modal) = self.active_modal_as::<FilePickerModal>() {
                modal.set_items(items);
            }
        }
    }

    /// Poll the in-flight workspace scan and, when it has finished, land
    /// its result into the open file picker. Returns `true` when the
    /// result was applied (so the caller can request a redraw). Called
    /// each tick from the run loop; a no-op when no scan is pending.
    pub fn poll_file_scan(&mut self) -> bool {
        let Some(task) = &mut self.scans.file_task else {
            return false;
        };
        if !task.is_finished() {
            return false;
        }
        // Take the handle so we only land the result once.
        let task = self.scans.file_task.take().expect("checked Some above");
        self.scans.file_cancel.take();
        let items = match futures_util::future::FutureExt::now_or_never(task) {
            Some(Ok(items)) => items,
            // Join error (panic/cancel) → fall back to an empty list so the
            // picker stops claiming it's scanning.
            _ => Vec::new(),
        };
        // Only apply if the picker is still open; otherwise discard.
        if let Some(modal) = self.active_modal_as::<FilePickerModal>() {
            modal.set_items(items);
            true
        } else {
            false
        }
    }

    /// Open the file picker over an explicit list of paths. Splitting
    /// this out from [`App::open_file_picker`] keeps the cwd scan out of
    /// the state-transition logic so unit tests can seed a deterministic
    /// list.
    fn open_file_picker_with(&mut self, items: Vec<String>) {
        self.set_active_modal(Box::new(FilePickerModal::new(items)), AppMode::ModalFile);
    }

    /// Test-only: seed the `@` file picker with an explicit path list so
    /// the selection path can be exercised without depending on the
    /// process working directory.
    #[cfg(test)]
    pub(super) fn open_file_picker_for_test(&mut self, items: Vec<String>) {
        self.open_file_picker_with(items);
    }

    /// Splice a selected file path into the input buffer as an `@path`
    /// reference token. Inserts a leading space when the buffer already
    /// ends in a non-space character so the mention stays a distinct
    /// token, and always appends a trailing space so the user can keep
    /// typing immediately after the reference.
    pub(super) fn insert_file_reference(&mut self, path: &str) {
        let needs_leading_space = self
            .input
            .text()
            .chars()
            .last()
            .is_some_and(|c| !c.is_whitespace());
        if needs_leading_space {
            self.input.insert_char(' ');
        }
        self.input.insert_char('@');
        self.input.insert_text(path);
        self.input.insert_char(' ');
    }

    /// Read-only access to the currently-open permission prompt, if
    /// any. Used by the modal widget to render the prompt text.
    #[must_use]
    pub const fn active_prompt(&self) -> Option<&PermissionPrompt> {
        self.active_prompt.as_ref()
    }

    /// Return to the normal view.
    pub fn exit_modal(&mut self) {
        self.active_prompt = None;
        self.active_user_question = None;
        self.modals.diff_viewer = None;
        self.modals.rewind = None;
        self.rewind_confirm = None;
        self.modals.workflow = None;
        self.modals.agents = None;
        self.modals.team_inbox = None;
        self.modals.usage_dashboard = None;
        self.cancel_file_scan();
        self.active_modal = None;
        self.choice_modals.session_ids.clear();
        self.choice_modals.login_provider_ids.clear();
        self.mode = AppMode::Normal;
    }

    /// Install `modal` as the single active overlay in the unified [`Modal`]
    /// slot and switch to `mode`. Routes through [`Self::exit_modal`] first, so
    /// the "exactly one modal open" invariant is enforced in one place — every
    /// legacy field, the slot, and the side-state are cleared before the new
    /// modal takes the foreground.
    pub fn set_active_modal(&mut self, modal: Box<dyn Modal>, mode: AppMode) {
        self.exit_modal();
        self.active_modal = Some(modal);
        self.mode = mode;
    }

    /// Downcast the active slot modal to a concrete type `T`, for the few host
    /// paths that must reach a specific modal after it is on the slot: the file
    /// picker's async scan lands rows via `set_items`, and the API-key modal
    /// receives a clipboard paste via `paste_text`. `None` when the slot is
    /// empty or holds a different modal.
    pub(super) fn active_modal_as<T: 'static>(&mut self) -> Option<&mut T> {
        self.active_modal.as_mut()?.as_any_mut().downcast_mut::<T>()
    }

    /// Apply the outcome of a slot modal's `handle_key` to the App. This is the
    /// single per-selection outcome match that replaces the ~11 near-identical
    /// dispatch arms; `None` keeps the modal open, `Cancelled` closes it, and a
    /// `Selected` routes to the right [`AppAction`].
    pub(super) fn apply_modal_outcome(&mut self, result: Option<ModalResult>) -> AppAction {
        match result {
            None => AppAction::None,
            Some(ModalResult::Cancelled) => {
                self.exit_modal();
                AppAction::None
            }
            Some(ModalResult::Selected(selection)) => self.apply_modal_selection(selection),
        }
    }

    fn apply_modal_selection(&mut self, selection: ModalSelection) -> AppAction {
        match selection {
            ModalSelection::Model(model) => {
                self.exit_modal();
                AppAction::SelectModel(model)
            }
            ModalSelection::Permission(mode) => {
                self.exit_modal();
                AppAction::SelectPermission(mode)
            }
            ModalSelection::Effort { budget, .. } => {
                self.exit_modal();
                // Re-enter the text path so the same budget validator runs once.
                AppAction::Submit(format!("/effort {budget}"))
            }
            ModalSelection::ApiKey { provider, api_key } => {
                self.exit_modal();
                AppAction::ConnectApiKey { provider, api_key }
            }
            ModalSelection::CustomProvider(draft) => {
                self.exit_modal();
                AppAction::ConnectCustomProvider(draft)
            }
            // Tool toggles keep the modal open so several tools can be flipped.
            ModalSelection::ToolToggle { name, enabled } => AppAction::ToggleTool { name, enabled },
            ModalSelection::SmartSettings(commit) => {
                self.exit_modal();
                AppAction::SaveSmartSettings(commit)
            }
            ModalSelection::DeepTier(action) => AppAction::DeepTier(action),
            ModalSelection::RemoteCommand(command) => {
                self.exit_modal();
                AppAction::Submit(command)
            }
            // Copy keeps the popup open so the user can keep reading; the host
            // arm writes the text to the clipboard and toasts the outcome.
            ModalSelection::CopyText(text) => AppAction::ClipboardCopyBlock(text),
            ModalSelection::QuestionAnswer(answers) => {
                if let Some(prompt) = self.active_user_question.take() {
                    let _ = prompt.responder.send(answers);
                }
                self.exit_modal();
                AppAction::None
            }
            // A generic Choice is disambiguated by the current `mode`, since
            // ChoicePickerModal backs several roles (session/login/file/arg).
            ModalSelection::Choice { index, label } => {
                self.apply_choice_selection(index, &label)
            }
        }
    }

    fn apply_choice_selection(&mut self, index: usize, label: &str) -> AppAction {
        // A generic `Choice` is disambiguated by the current mode, since several
        // roles are backed by `ChoicePickerModal` on the slot. Each reads its
        // parallel side-list at `index` *before* `exit_modal` clears it.
        match self.mode {
            AppMode::ModalArgPick => {
                let command = self.choice_modals.arg_picker_command.clone();
                self.exit_modal();
                // Re-enter the text path so the command's own handler validates
                // and applies the choice (mirrors /login, /effort — one apply
                // path, no duplicated logic).
                AppAction::Submit(format!("/{command} {label}"))
            }
            AppMode::ModalSession => {
                let session_id = self
                    .choice_modals
                    .session_ids
                    .get(index)
                    .cloned()
                    .unwrap_or_default();
                self.exit_modal();
                AppAction::SelectSession(session_id)
            }
            AppMode::ModalLogin => {
                let token = self
                    .choice_modals
                    .login_provider_ids
                    .get(index)
                    .cloned()
                    .unwrap_or_default();
                self.exit_modal();
                let (command, provider) = token
                    .split_once(':')
                    .map_or(("login", token.as_str()), |(command, provider)| {
                        (command, provider)
                    });
                if command == "connect-key" {
                    if let Some(info) = connect_api_key_info(provider) {
                        self.open_connect_api_key_modal(info);
                        return AppAction::None;
                    }
                }
                if command == "connect-custom" {
                    self.open_custom_provider_modal();
                    return AppAction::None;
                }
                // Re-enter the text path so the provider-specific slash handler
                // runs: `/login` starts OAuth; `/connect` checks local state.
                AppAction::Submit(format!("/{command} {provider}"))
            }
            AppMode::ModalFile => {
                // The `@` file picker splices the chosen path into the input as
                // an `@path` reference token rather than emitting an AppAction.
                self.exit_modal();
                self.insert_file_reference(label);
                AppAction::None
            }
            _ => {
                self.exit_modal();
                AppAction::None
            }
        }
    }

    /// Open the interactive snapshot rewind viewer with a timeline the host
    /// precomputed from the git snapshot stack. Clears any other open modal.
    pub fn open_rewind_viewer(&mut self, modal: RewindViewerModal) {
        self.active_prompt = None;
        self.active_user_question = None;
        self.modals.diff_viewer = None;
        self.cancel_file_scan();
        self.modals.workflow = None;
        self.modals.agents = None;
        self.modals.team_inbox = None;
        self.modals.usage_dashboard = None;
        self.choice_modals.session_ids.clear();
        self.modals.rewind = Some(modal);
        self.mode = AppMode::ModalRewind;
    }

    /// Open the session-scoped `/hunks` attribution review modal.
    pub fn open_hunks_modal(&mut self, modal: ReviewModal) {
        self.set_active_modal(Box::new(modal), AppMode::ModalHunks);
    }

    /// Open the live workflow progress viewer with a snapshot the host
    /// precomputed from the workflow progress file + per-agent manifests.
    /// Clears any other open modal. While open, the host calls
    /// [`Self::refresh_workflow_viewer`] each poll tick so the tree stays live.
    pub fn open_workflow_viewer(&mut self, mut modal: WorkflowViewerModal) {
        self.active_prompt = None;
        self.active_user_question = None;
        self.modals.diff_viewer = None;
        self.modals.rewind = None;
        self.cancel_file_scan();
        self.modals.usage_dashboard = None;
        self.modals.agents = None;
        self.modals.team_inbox = None;
        self.choice_modals.session_ids.clear();
        modal.attach_plan_items(&self.hud_state.todo_items);
        self.modals.workflow = Some(modal);
        self.workflow_viewer_empty_refreshes = 0;
        self.mode = AppMode::ModalWorkflow;
    }

    /// Open the runtime tool toggle modal. Clears any other open modal.
    pub fn open_tool_toggle_modal(&mut self, rows: Vec<ToolToggleRow>) {
        self.set_active_modal(Box::new(ToolToggleModal::new(rows)), AppMode::ModalTools);
    }

    /// Open the graphical `/usage` dashboard over a precomputed snapshot.
    pub fn open_usage_dashboard_modal(&mut self, modal: UsageDashboardModal) {
        self.active_prompt = None;
        self.active_user_question = None;
        self.modals.diff_viewer = None;
        self.modals.rewind = None;
        self.modals.workflow = None;
        self.modals.agents = None;
        self.modals.team_inbox = None;
        self.cancel_file_scan();
        self.choice_modals.session_ids.clear();
        self.modals.usage_dashboard = Some(modal);
        self.mode = AppMode::ModalUsage;
    }

    /// Open the large `/smart` Smart Router settings dashboard.
    pub fn open_smart_settings_modal(&mut self, modal: SmartSettingsModal) {
        self.set_active_modal(Box::new(modal), AppMode::ModalSmartSettings);
    }

    /// Open the generic slash-command report popup (`CommandOutput::Popup`).
    /// The modal pre-renders its copy text against the current theme, so it is
    /// constructed here rather than at the dispatch site.
    pub fn open_report_modal(&mut self, title: String, blocks: Vec<ReportViewerBlock>) {
        let modal = ReportViewerModal::new(title, blocks, &self.theme);
        self.set_active_modal(Box::new(modal), AppMode::ModalReport);
    }

    /// Open the `/remote` onboarding and status modal from a local-safe snapshot.
    pub fn open_remote_onboarding_modal(&mut self, view: RemoteOnboardingView) {
        self.set_active_modal(
            Box::new(RemoteOnboardingModal::new(view)),
            AppMode::ModalRemoteOnboarding,
        );
    }

    pub fn open_deep_tier_modal(&mut self, view: DeepTierView) {
        self.set_active_modal(Box::new(DeepTierModal::new(view)), AppMode::ModalDeepTier);
    }

    pub fn apply_deep_tier_update(
        &mut self,
        view: Option<DeepTierView>,
        result: Result<String, String>,
    ) {
        if self.mode != AppMode::ModalDeepTier {
            return;
        }
        if let Some(modal) = self.active_modal_as::<DeepTierModal>() {
            modal.apply_update(view, result);
        }
    }

    /// Feed a fresh workflow snapshot into the open viewer (no-op if it is not
    /// open). Selection/scroll are preserved across the refresh.
    pub fn refresh_workflow_viewer(&mut self, view: WorkflowView) {
        if let Some(modal) = &mut self.modals.workflow {
            modal.refresh(view, &self.hud_state.todo_items);
        }
    }

    /// `true` when the live workflow viewer is the active modal — the host uses
    /// this to decide whether to keep polling the progress snapshot.
    #[must_use]
    pub fn workflow_viewer_open(&self) -> bool {
        self.mode == AppMode::ModalWorkflow && self.modals.workflow.is_some()
    }

    /// One animation tick of the live workflow viewer. Disk-backed progress
    /// refreshes are scheduled by the host loop through
    /// [`Self::Workflow_viewer_refresh_due`] and applied via
    /// [`Self::apply_workflow_viewer_snapshot`], so this per-frame path only
    /// advances in-memory animation state.
    pub fn tick_workflow_viewer(&mut self) -> bool {
        if !self.workflow_viewer_open() {
            return false;
        }
        // Under reduce-motion the phase/agent spinners hold frame 0, so there is
        // no per-frame animation to advance — report no work and stop driving
        // idle redraws of the viewer.
        if crate::tui::term::reduce_motion_enabled() {
            return false;
        }
        if let Some(modal) = self.modals.workflow.as_mut() {
            modal.advance_spinner();
        }
        true
    }

    #[must_use]
    pub fn workflow_viewer_refresh_due(&self) -> bool {
        self.workflow_viewer_open() && self.tick.is_multiple_of(10)
    }

    #[must_use]
    pub fn workflow_viewer_snapshot_scope(&self) -> Option<(u64, Option<String>)> {
        self.workflow_viewer_open().then(|| {
            (
                self.agent_manifest_started_after,
                self.agent_manifest_session_id.clone(),
            )
        })
    }

    pub fn apply_workflow_viewer_snapshot(&mut self, view: Option<WorkflowView>) {
        if !self.workflow_viewer_open() {
            return;
        }
        if let Some(view) = view {
            self.workflow_viewer_empty_refreshes = 0;
            self.refresh_workflow_viewer(view);
        } else {
            // A single empty read can be transient — a manifest or progress doc
            // caught mid-rewrite — and closing on it made the freshly opened
            // viewer vanish under the user. Keep the last tree for one more
            // refresh cycle; only two consecutive empties (the run is genuinely
            // gone and its manifests aged out) close the viewer.
            self.workflow_viewer_empty_refreshes =
                self.workflow_viewer_empty_refreshes.saturating_add(1);
            if self.workflow_viewer_empty_refreshes >= 2 {
                self.exit_modal();
            }
        }
        // Tail refresh is mtime-gated and capped to 16 KiB; keep it at the
        // snapshot boundary, never in the draw path.
        if let Some(modal) = self.modals.workflow.as_mut() {
            modal.refresh_output_tail();
        }
    }


    // ── TeamInbox viewer (/inbox) ───────────────────────────────────

    /// Open the `TeamInbox` viewer over a caller-provided snapshot. Keeping the
    /// data source outside the modal lets tests inject fixtures and keeps all
    /// store access in the session/runtime layer.
    pub fn open_team_inbox_viewer(&mut self, snapshot: TeamInboxSnapshot) {
        self.exit_modal();
        self.modals.team_inbox = Some(TeamInboxViewerModal::new(snapshot));
        self.mode = AppMode::ModalTeamInbox;
    }

    #[must_use]
    pub fn team_inbox_viewer_open(&self) -> bool {
        self.mode == AppMode::ModalTeamInbox && self.modals.team_inbox.is_some()
    }

    pub fn apply_team_inbox_snapshot(&mut self, snapshot: TeamInboxSnapshot) {
        if !self.team_inbox_viewer_open() {
            return;
        }
        if let Some(modal) = self.modals.team_inbox.as_mut() {
            modal.refresh(snapshot);
        }
    }

    // ── Agents viewer (Ctrl+G) ──────────────────────────────────────

    /// The agents snapshot for this session's scope, with the HUD's in-memory
    /// fleet as the spawn-window fallback (manifests not on disk yet), so an
    /// early Ctrl+G shows the fleet instead of an empty list.
    fn read_agents_snapshot(&self, include_history: bool) -> AgentRowsSnapshot {
        let mut snapshot = crate::tui::workflow_progress::read_agent_rows_since(
            self.agent_manifest_started_after,
            self.agent_manifest_session_id.as_deref(),
            include_history,
        );
        if snapshot.rows.is_empty() && !self.hud_state.agents.is_empty() {
            snapshot.rows = self.hud_state.agents.iter().map(hud_agent_to_row).collect();
        }
        snapshot
    }

    /// Open the Ctrl+G agents viewer over a fresh snapshot — the structured
    /// replacement for the old raw-text agents pager. No live gate: finished
    /// fleets stay browsable.
    pub fn open_agents_viewer(&mut self) {
        let snapshot = self.read_agents_snapshot(false);
        let mut modal = AgentsViewerModal::new(snapshot);
        modal.set_turn_active(self.turn_activity.is_some());
        self.exit_modal();
        self.modals.agents = Some(modal);
        self.mode = AppMode::ModalAgents;
    }

    /// Open the agents viewer pre-selected to `agent_id` (a clicked pinned
    /// panel row whose agent is missing from the workflow view).
    pub fn open_agents_viewer_focused(&mut self, agent_id: &str) {
        self.open_agents_viewer();
        if let Some(modal) = self.modals.agents.as_mut() {
            let _ = modal.select_agent_by_id(agent_id);
        }
    }

    /// `true` when the agents viewer is the active modal — the host uses this
    /// to keep polling the manifest snapshot.
    #[must_use]
    pub fn agents_viewer_open(&self) -> bool {
        self.mode == AppMode::ModalAgents && self.modals.agents.is_some()
    }

    /// Deliver a Ctrl+G message-box send and surface the outcome in the modal
    /// footer. A steer is a queue push and a resume detaches onto its own OS
    /// thread, so this never blocks the render loop. A resumed agent's reply
    /// arrives through the normal background-completion re-injection.
    pub(super) fn send_agent_message_from_viewer(&mut self, target: &str, message: &str) {
        // A user-initiated resume still respects the session's privilege: the
        // HUD badge mirrors `LiveCli.permission_mode`, so map it back onto the
        // runtime ladder for the spawn clamp (Plan is a read-only gate).
        let session_mode = match self.hud_state.perm_mode {
            crate::tui::hud::PermissionMode::ReadOnly | crate::tui::hud::PermissionMode::Plan => {
                runtime::PermissionMode::ReadOnly
            }
            crate::tui::hud::PermissionMode::Workspace => runtime::PermissionMode::WorkspaceWrite,
            crate::tui::hud::PermissionMode::All => runtime::PermissionMode::DangerFullAccess,
        };
        let outcome = tools::send_agent_message(target, message, Some(session_mode));
        let Some(modal) = self.modals.agents.as_mut() else {
            return;
        };
        let (text, is_error) = match outcome {
            tools::AgentSendOutcome::Steered { name } => {
                (format!("✉ delivered to {name} mid-run"), false)
            }
            tools::AgentSendOutcome::Resumed { name } => (
                format!("✉ {name} resumed — its reply will arrive in the conversation"),
                false,
            ),
            tools::AgentSendOutcome::NotFound => ("no matching agent".to_string(), true),
            tools::AgentSendOutcome::Unreachable { name } => (
                format!("{name} is between turns — retry in a moment"),
                true,
            ),
            tools::AgentSendOutcome::Failed { name, error } => {
                (format!("{name}: {error}"), true)
            }
        };
        modal.set_feedback(text, is_error);
    }

    /// Synchronous snapshot re-read after the history toggle (the manifest
    /// listing is cached, so this is cheap enough for a key press).
    pub(super) fn reload_agents_viewer(&mut self) {
        let Some(include_history) = self
            .modals
            .agents
            .as_ref()
            .map(AgentsViewerModal::show_history)
        else {
            return;
        };
        let snapshot = self.read_agents_snapshot(include_history);
        let turn_active = self.turn_activity.is_some();
        if let Some(modal) = self.modals.agents.as_mut() {
            modal.refresh(snapshot);
            modal.set_turn_active(turn_active);
        }
    }

    #[must_use]
    pub fn agents_viewer_refresh_due(&self) -> bool {
        self.agents_viewer_open() && self.tick.is_multiple_of(10)
    }

    /// Scope for a background snapshot read: `(started_after, session_id,
    /// include_history)`. `None` when the viewer is closed.
    #[must_use]
    pub fn agents_viewer_snapshot_scope(&self) -> Option<(u64, Option<String>, bool)> {
        self.agents_viewer_open().then(|| {
            (
                self.agent_manifest_started_after,
                self.agent_manifest_session_id.clone(),
                self.modals
                    .agents
                    .as_ref()
                    .is_some_and(AgentsViewerModal::show_history),
            )
        })
    }

    /// Land a background snapshot into the open viewer. Unlike the workflow
    /// viewer there is no empty-close: finished manifests persist on disk, so
    /// an empty read means "genuinely nothing" and the empty state shows.
    pub fn apply_agents_viewer_snapshot(&mut self, snapshot: AgentRowsSnapshot) {
        if !self.agents_viewer_open() {
            return;
        }
        let snapshot = if snapshot.rows.is_empty() && !self.hud_state.agents.is_empty() {
            AgentRowsSnapshot {
                rows: self.hud_state.agents.iter().map(hud_agent_to_row).collect(),
                ..snapshot
            }
        } else {
            snapshot
        };
        let turn_active = self.turn_activity.is_some();
        if let Some(modal) = self.modals.agents.as_mut() {
            modal.refresh(snapshot);
            modal.set_turn_active(turn_active);
        }
    }

    /// One animation tick of the agents viewer (running-row spinners).
    pub fn tick_agents_viewer(&mut self) -> bool {
        if !self.agents_viewer_open() {
            return false;
        }
        if crate::tui::term::reduce_motion_enabled() {
            return false;
        }
        if let Some(modal) = self.modals.agents.as_mut() {
            modal.advance_spinner();
        }
        true
    }
}

/// One HUD in-memory agent summary → a viewer row, for the spawn window before
/// manifests land on disk. Mirrors the pinned panel's `summary_to_tree_row` so
/// the two spawn-window fallbacks describe the same fleet.
fn hud_agent_to_row(agent: &crate::tui::hud::AgentTaskSummary) -> WorkflowAgentRow {
    WorkflowAgentRow {
        id: agent.id.clone(),
        name: agent.name.clone(),
        status: agent.status.clone(),
        model: agent.model.clone(),
        subagent_type: agent.subagent_type.clone(),
        current_tool: agent.current_tool.clone(),
        current_phase: agent.current_phase.clone(),
        tool_calls: agent.tool_calls,
        tokens: agent.tokens,
        elapsed_secs: agent.elapsed_secs,
        output_tail: agent.output_tail.clone(),
        route_reason: agent.route_reason.clone(),
        ..WorkflowAgentRow::default()
    }
}

/// Preset [`ApiKeyConnectInfo`] for the cloud adapters reachable via the
/// `connect-key:<provider>` entries in the `/connect` login picker. Returns
/// `None` for providers that use OAuth or a plain status check instead.
///
/// Lives beside the login apply path (its only caller) rather than in the
/// keyboard-dispatch module.
fn connect_api_key_info(provider: &str) -> Option<ApiKeyConnectInfo> {
    match provider {
        "deepseek" => Some(ApiKeyConnectInfo {
            provider: "deepseek".to_string(),
            label: "DeepSeek".to_string(),
            auth_env: "DEEPSEEK_API_KEY".to_string(),
            models: vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()],
        }),
        "kimi" => Some(ApiKeyConnectInfo {
            provider: "kimi".to_string(),
            label: "Kimi (Moonshot)".to_string(),
            auth_env: "MOONSHOT_API_KEY".to_string(),
            models: vec!["kimi-k2-0905-preview".to_string(), "moonshot-v1-32k".to_string()],
        }),
        "qwen" => Some(ApiKeyConnectInfo {
            provider: "qwen".to_string(),
            label: "Qwen (DashScope)".to_string(),
            auth_env: "DASHSCOPE_API_KEY".to_string(),
            models: vec!["qwen-max".to_string(), "qwen-plus".to_string(), "qwen-turbo".to_string()],
        }),
        "nvidia" => Some(ApiKeyConnectInfo {
            provider: "nvidia".to_string(),
            label: "NVIDIA NIM".to_string(),
            auth_env: "NVIDIA_API_KEY".to_string(),
            models: vec!["meta/llama-3.1-8b-instruct".to_string(), "z-ai/glm-5.2".to_string()],
        }),
        "openrouter" => Some(ApiKeyConnectInfo {
            provider: "openrouter".to_string(),
            label: "OpenRouter".to_string(),
            auth_env: "OPENROUTER_API_KEY".to_string(),
            models: vec!["openrouter/auto".to_string()],
        }),
        _ => None,
    }
}
