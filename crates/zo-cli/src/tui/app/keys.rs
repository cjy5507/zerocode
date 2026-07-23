use super::{
    AgentCommand, AgentsViewerAction, App, AppAction, AppMode, ClipboardCopyTarget,
    DiffViewerAction, InputCommand, Instant, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    PermissionMode, QueuedMessage, RewindViewerAction,
    RuntimePermissionMode, TuiError, UsageDashboardAction, WorkflowViewerAction, apply_mention,
    mention_trigger, slash_completion_for,
};
use crate::tui::modals::TeamInboxViewerAction;
use crate::tui::startup::{
    STARTUP_LOGIN_CLAUDE_COMMAND, STARTUP_LOGIN_OPENAI_COMMAND, STARTUP_PERMISSIONS_COMMAND,
    STARTUP_SUMMARIZE_REPO_PROMPT,
};

impl App {
    /// Handle a single key event and emit a semantic action.
    ///
    /// This is separated from [`App::run`] so unit tests can drive the
    /// state machine without spinning up an async runtime.
    ///
    pub fn handle_key(&mut self, key: KeyEvent) -> Result<AppAction, TuiError> {
        if key.kind != KeyEventKind::Press {
            return Ok(AppAction::None);
        }

        if let Some(action) = self.handle_ctrl_c(key)? {
            return Ok(action);
        }

        if let Some(action) = self.dispatch_mode_key(key) {
            return Ok(action);
        }
        if let Some(action) = self.handle_normal_shortcuts(key) {
            return Ok(action);
        }
        if let Some(action) = self.handle_slash_hint_key(key) {
            return Ok(action);
        }
        if let Some(action) = self.handle_mention_hint_key(key) {
            return Ok(action);
        }
        if let Some(action) = self.handle_normal_esc(key) {
            return Ok(action);
        }
        if let Some(action) = self.handle_input_key(key) {
            return Ok(action);
        }
        if let Some(action) = self.handle_queued_input_key(key) {
            return Ok(action);
        }
        Ok(AppAction::None)
    }

    /// Ctrl+C: double-tap quits, a single tap cancels the in-flight turn.
    ///
    /// Returns `Ok(Some(action))` when the key was a Ctrl+C (fully consumed),
    /// `Ok(None)` to let the dispatcher continue with the remaining stages.
    fn handle_ctrl_c(&mut self, key: KeyEvent) -> Result<Option<AppAction>, TuiError> {
        if !(matches!(key.code, KeyCode::Char('c'))
            && key.modifiers.contains(KeyModifiers::CONTROL))
        {
            return Ok(None);
        }

        let now = Instant::now();
        let double_tapped = self
            .last_ctrl_c
            .is_some_and(|prev| now.duration_since(prev) <= Self::CTRL_C_DOUBLE_TAP_WINDOW);
        self.last_ctrl_c = Some(now);

        if double_tapped {
            // Best-effort shutdown signal. If the agent task is
            // already gone we still exit the loop.
            let _ = self.cmd_tx.try_send(AgentCommand::Quit);
            self.should_quit = true;
            return Ok(Some(AppAction::Quit));
        }
        self.cmd_tx
            .try_send(AgentCommand::CancelTurn)
            .map_err(|_| TuiError::CommandChannelClosed)?;
        Ok(Some(AppAction::None))
    }

    fn dispatch_mode_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        match self.mode {
            AppMode::Normal => None,
            AppMode::Search | AppMode::Focus | AppMode::Pager => self.dispatch_nav_key(key),
            _ => self.dispatch_modal_key(key),
        }
    }

    /// Keys for the active modal overlay (`Modal*` modes). One arm per modal
    /// keeps each modal's apply path explicit; split out of `dispatch_mode_key`.
    #[allow(clippy::too_many_lines)] // one arm per modal keeps the apply path explicit
    fn dispatch_modal_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        // Text-entry modals delegate clipboard paste to the host (they cannot
        // read the clipboard themselves), so intercept the paste chord before
        // the slot consumes the key. This is the one host-clipboard concern that
        // does not fit the generic `ModalResult` flow.
        if matches!(
            self.mode,
            AppMode::ModalApiKey | AppMode::ModalCustomProvider | AppMode::ModalModel
        ) && secret_modal_paste_key(&key)
        {
            return Some(AppAction::ClipboardPaste);
        }
        // A modal installed on the unified [`Modal`] slot owns its own key
        // handling and routes through the single `apply_modal_outcome`; the
        // legacy per-mode arms below only serve not-yet-migrated modals.
        if self.active_modal.is_some() {
            let result = self.active_modal.as_mut().and_then(|modal| modal.handle_key(key));
            return Some(self.apply_modal_outcome(result));
        }
        match self.mode {
            AppMode::ModalConfirmRewind => match key.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                    self.exit_modal();
                    return Some(AppAction::ConfirmRewind);
                }
                // Anything that is not an explicit yes cancels — `n`, Esc, and
                // any stray key all leave the latest turn untouched, so a
                // reflexive Esc burst can never confirm by accident.
                _ => {
                    self.exit_modal();
                    return Some(AppAction::None);
                }
            },
            AppMode::ModalChoice => {
                if matches!(key.code, KeyCode::Esc) {
                    self.exit_modal();
                    return Some(AppAction::None);
                }
            }
            AppMode::ModalDiff => {
                if let Some(modal) = &mut self.modals.diff_viewer {
                    // Navigation / view-toggle keys return None; Esc/q close,
                    // `r` requests reverting the selected file to HEAD.
                    match modal.handle_key(key) {
                        Some(DiffViewerAction::Close) => self.exit_modal(),
                        Some(DiffViewerAction::RevertFile(path)) => self.revert_diff_file(&path),
                        None => {}
                    }
                    return Some(AppAction::None);
                }
            }
            AppMode::ModalRewind => {
                if let Some(modal) = &mut self.modals.rewind {
                    // Navigation / diff-toggle keys return None; Esc/q close,
                    // Enter requests rewinding the worktree to the selection.
                    match modal.handle_key(key) {
                        Some(RewindViewerAction::Close) => self.exit_modal(),
                        Some(RewindViewerAction::RewindTo(index)) => {
                            self.exit_modal();
                            return Some(AppAction::RewindTo(index));
                        }
                        None => {}
                    }
                    return Some(AppAction::None);
                }
            }
            AppMode::ModalWorkflow => {
                // Mirror the keys that open these surfaces: Ctrl+O toggles the
                // viewer closed (Claude-Code style), Ctrl+G swaps it for the
                // agents viewer. Both were dead keys here before.
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    if matches!(key.code, KeyCode::Char('o')) {
                        self.exit_modal();
                        return Some(AppAction::None);
                    }
                    if matches!(key.code, KeyCode::Char('g')) {
                        self.open_agents_viewer();
                        return Some(AppAction::None);
                    }
                }
                if let Some(modal) = &mut self.modals.workflow {
                    let modal_consumes_key = workflow_modal_consumes_key(&key);
                    if modal_consumes_key {
                        // Read-only live monitor: navigation returns None,
                        // Esc/Ctrl+C close. Printable chars pass through to the
                        // composer so the user can steer while watching agents.
                        match modal.handle_key(key) {
                            Some(WorkflowViewerAction::Close) => self.exit_modal(),
                            None => {}
                        }
                        return Some(AppAction::None);
                    }
                }
            }
            AppMode::ModalAgents => {
                // Ctrl+G toggles the viewer closed (the key that opened it);
                // Ctrl+O swaps to the live workflow viewer.
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    if matches!(key.code, KeyCode::Char('g')) {
                        self.exit_modal();
                        return Some(AppAction::None);
                    }
                    if matches!(key.code, KeyCode::Char('o')) {
                        self.exit_modal();
                        return Some(AppAction::OpenWorkflowViewer);
                    }
                }
                // History toggle needs a disk re-read, so it lives here rather
                // than inside the modal (which never touches the disk). While
                // the message box is open, `a` is a character being typed, so
                // it falls through to the modal instead.
                if matches!(key.code, KeyCode::Char('a'))
                    && key.modifiers.is_empty()
                    && key.kind == KeyEventKind::Press
                    && self
                        .modals
                        .agents
                        .as_ref()
                        .is_some_and(|modal| !modal.input_active())
                {
                    if let Some(modal) = self.modals.agents.as_mut() {
                        let _ = modal.toggle_history();
                        self.reload_agents_viewer();
                    }
                    return Some(AppAction::None);
                }
                // Unlike the workflow monitor, this is a fully interactive
                // browser: it consumes every key (j/k/g/G navigate) instead of
                // passing printable chars to the composer.
                if let Some(modal) = self.modals.agents.as_mut() {
                    match modal.handle_key(key) {
                        Some(AgentsViewerAction::Close) => self.exit_modal(),
                        Some(AgentsViewerAction::Send { target, message }) => {
                            self.send_agent_message_from_viewer(&target, &message);
                        }
                        None => {}
                    }
                    return Some(AppAction::None);
                }
            }
            AppMode::ModalTeamInbox => {
                if let Some(modal) = &mut self.modals.team_inbox {
                    match modal.handle_key(key) {
                        Some(TeamInboxViewerAction::Close) => self.exit_modal(),
                        Some(TeamInboxViewerAction::Refresh) => {
                            return Some(AppAction::RefreshTeamInboxViewer);
                        }
                        Some(TeamInboxViewerAction::Ack(update_id)) => {
                            return Some(AppAction::AckTeamInboxUpdate(update_id));
                        }
                        Some(TeamInboxViewerAction::Include(text)) => {
                            self.exit_modal();
                            return Some(AppAction::IncludeTeamInboxUpdate(text));
                        }
                        None => {}
                    }
                    return Some(AppAction::None);
                }
            }
            AppMode::ModalUsage => {
                if let Some(modal) = &mut self.modals.usage_dashboard {
                    match modal.handle_key(key) {
                        Some(UsageDashboardAction::Close) => self.exit_modal(),
                        None => {}
                    }
                    return Some(AppAction::None);
                }
            }
            _ => {}
        }

        None
    }

    /// Keys for the read-only navigation modes: transcript search, focus-scroll,
    /// and the full-screen pager. Split out of `dispatch_mode_key`.
    fn dispatch_nav_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        match self.mode {
            AppMode::Search => match key.code {
                KeyCode::Esc => {
                    self.exit_search();
                    return Some(AppAction::None);
                }
                // Enter / Down jump to the next match, Up to the previous.
                // The bar stays open so the user can keep cycling or
                // refine the query; Esc returns to Normal.
                KeyCode::Enter | KeyCode::Down => {
                    self.search_next();
                    return Some(AppAction::None);
                }
                KeyCode::Up => {
                    self.search_prev();
                    return Some(AppAction::None);
                }
                KeyCode::Backspace => {
                    self.search.query.pop();
                    self.refresh_search();
                    return Some(AppAction::None);
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.search.query.push(ch);
                    self.refresh_search();
                    return Some(AppAction::None);
                }
                _ => return Some(AppAction::None),
            },
            AppMode::Focus => match key.code {
                KeyCode::Esc | KeyCode::F(11) => {
                    self.mode = AppMode::Normal;
                    return Some(AppAction::None);
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.transcript.scroll_up(1);
                    return Some(AppAction::None);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.transcript.scroll_down(1);
                    return Some(AppAction::None);
                }
                KeyCode::PageUp => {
                    self.transcript.scroll_up(Self::HALF_PAGE_SCROLL_ROWS);
                    return Some(AppAction::None);
                }
                KeyCode::PageDown => {
                    self.transcript.scroll_down(Self::HALF_PAGE_SCROLL_ROWS);
                    return Some(AppAction::None);
                }
                _ => return Some(AppAction::None),
            },
            AppMode::Pager => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    self.exit_pager();
                    return Some(AppAction::None);
                }
                // The keys that open the agent surfaces keep working while a
                // pager (long tool output, help) is up: Ctrl+O switches to the
                // live workflow viewer and Ctrl+G to the agents viewer.
                KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.exit_pager();
                    return Some(AppAction::OpenWorkflowViewer);
                }
                KeyCode::Char('g') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.exit_pager();
                    self.open_agents_viewer();
                    return Some(AppAction::None);
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.pager_scroll_up(1);
                    return Some(AppAction::None);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.pager_scroll_down(1);
                    return Some(AppAction::None);
                }
                KeyCode::PageUp => {
                    self.pager_scroll_up(Self::HALF_PAGE_SCROLL_ROWS);
                    return Some(AppAction::None);
                }
                KeyCode::PageDown => {
                    self.pager_scroll_down(Self::HALF_PAGE_SCROLL_ROWS);
                    return Some(AppAction::None);
                }
                KeyCode::Home => {
                    self.pager_scroll = 0;
                    return Some(AppAction::None);
                }
                KeyCode::End => {
                    self.pager_scroll = u16::MAX;
                    return Some(AppAction::None);
                }
                _ => return Some(AppAction::None),
            },
            _ => {}
        }

        None
    }

    fn handle_normal_shortcuts(&mut self, key: KeyEvent) -> Option<AppAction> {
        if !matches!(self.mode, AppMode::Normal) {
            return None;
        }

        // Shift+Tab in Normal mode routes a permission-mode cycle
        // request through the session so the real runtime
        // `PermissionMode` is the single source of truth. The HUD is
        // refreshed from the session state after the outer loop
        // applies the change — we do not mutate `self.hud_state` here.
        //
        // Shift+Tab reaches us as `BackTab` on legacy terminals, but as
        // `Tab` + SHIFT once the Kitty keyboard protocol's
        // DISAMBIGUATE_ESCAPE_CODES flag is active (kitty, foot, WezTerm,
        // Ghostty, recent iTerm2…) — which `init_terminal` pushes whenever
        // the terminal advertises support. Match both encodings so the
        // permission cycle works regardless of protocol; matching here (the
        // first dispatch stage) also keeps Shift+Tab a permission cycle even
        // when a slash/mention hint is open.
        let is_shift_tab = matches!(key.code, KeyCode::BackTab)
            || (matches!(key.code, KeyCode::Tab) && key.modifiers.contains(KeyModifiers::SHIFT));
        if is_shift_tab {
            // Cycle order: ReadOnly → Plan → Workspace → All → ReadOnly.
            // `Plan` and `ReadOnly` are both runtime read-only; the
            // `plan_mode_active` flag carries the distinction so the next
            // snapshot refresh resolves the read-only runtime mode to the
            // right HUD badge (see `apply_session_snapshot`).
            //
            // The plan-gate mutation below happens before the host loop applies
            // the runtime permission change, so arm a rollback first: if that
            // change fails the loop restores this snapshot, keeping the UI Plan
            // flag from diverging from the runtime.
            self.arm_plan_cycle_rollback();
            let next = match self.hud_state.perm_mode {
                PermissionMode::ReadOnly => {
                    self.set_plan_mode_active(true);
                    RuntimePermissionMode::ReadOnly
                }
                PermissionMode::Plan => {
                    self.set_plan_mode_active(false);
                    RuntimePermissionMode::WorkspaceWrite
                }
                PermissionMode::Workspace => {
                    self.set_plan_mode_active(false);
                    RuntimePermissionMode::DangerFullAccess
                }
                PermissionMode::All => {
                    self.set_plan_mode_active(false);
                    RuntimePermissionMode::ReadOnly
                }
            };
            return Some(AppAction::SelectPermission(next));
        }

        if let Some(action) = self.handle_normal_top_shortcuts(key) {
            return Some(action);
        }

        if let Some(action) = self.handle_history_nav_keys(key) {
            return Some(action);
        }

        let arrows_scroll_transcript = self.input.lines().len() == 1
            && !self.slash_hint_active()
            && !key.modifiers.contains(KeyModifiers::CONTROL)
            && !key.modifiers.contains(KeyModifiers::ALT);
        if let Some(action) = self.handle_normal_scroll_shortcuts(key, arrows_scroll_transcript) {
            return Some(action);
        }
        if let Some(action) = self.handle_normal_control_shortcuts(key) {
            return Some(action);
        }
        if let Some(action) = self.handle_normal_modifier_scroll_shortcuts(key) {
            return Some(action);
        }

        // Transcript block navigation — the keybinding overlay advertises Tab
        // (focus next block) and Enter (expand/collapse the focused block).
        // Both are gated so they never shadow the composer, which owns Enter
        // (submit) and Tab (slash completion) in later dispatch stages:
        //   • Enter acts here only while a block is actually focused; with no
        //     focus it falls through to submit.
        //   • Tab acts only while the composer is empty (so no slash/mention
        //     completion is pending) and a focusable block exists; otherwise it
        //     falls through.
        // Esc clears the focus (see `handle_normal_esc`), restoring submit.
        if key.modifiers.is_empty() {
            if matches!(key.code, KeyCode::Enter) && self.transcript.focused_idx().is_some() {
                self.transcript.toggle_expanded();
                return Some(AppAction::Redraw);
            }
            if matches!(key.code, KeyCode::Tab)
                && self.input.is_text_empty()
                && self.transcript.focus_next()
            {
                return Some(AppAction::Redraw);
            }
        }

        None
    }

    /// Claude Code CLI parity: Up/Down browse the prompt history when the input
    /// cursor is at the edge of the buffer.
    ///
    /// Up recalls the previous (older) prompt only when the cursor sits on the
    /// first line; Down steps toward more-recent prompts (and finally restores
    /// the stashed draft) only when the cursor sits on the last line. Inside a
    /// multi-line draft the arrows still move between lines, and when there is
    /// no history to recall this returns `None` so the key falls through to
    /// transcript scrolling. Modifier-held arrows (Ctrl/Alt/Shift) are reserved
    /// for scrolling and never browse history.
    fn handle_history_nav_keys(&mut self, key: KeyEvent) -> Option<AppAction> {
        // Only in the live composer, in Normal mode, with no modifiers. A hint
        // popup (slash/`@`-mention) owns the arrows while it is open.
        if !self.input_enabled
            || !matches!(self.mode, AppMode::Normal)
            || !key.modifiers.is_empty()
            || self.slash_hint_active()
            || self.mention_hint_active()
        {
            return None;
        }

        let (row, _col) = self.input.cursor();
        let last_row = self.input.lines().len().saturating_sub(1);
        // Once history browsing is active the arrows belong to it entirely:
        // even when `history_prev`/`history_next` report "nothing further" we
        // consume the key (a no-op) rather than fall through to transcript
        // scrolling, which would reset `history_cursor` and strand the browse.
        let browsing = self.history_cursor.is_some();
        match key.code {
            KeyCode::Up if row == 0 => {
                let moved = self.history_prev();
                (browsing || moved).then_some(AppAction::None)
            }
            KeyCode::Down if row == last_row => {
                let moved = self.history_next();
                (browsing || moved).then_some(AppAction::None)
            }
            _ => None,
        }
    }

    fn prefill_startup_command(&mut self, command: &str) -> bool {
        if self.startup.screen.is_none() || !self.input_enabled || !self.input.is_empty() {
            return false;
        }
        self.set_input_text(command);
        true
    }

    fn startup_prefill_command_for_key(key: KeyEvent) -> Option<&'static str> {
        if !key.modifiers.contains(KeyModifiers::ALT) {
            return None;
        }
        match key.code {
            KeyCode::Char('s') => Some(STARTUP_SUMMARIZE_REPO_PROMPT),
            KeyCode::Char('c') => Some(STARTUP_LOGIN_CLAUDE_COMMAND),
            KeyCode::Char('o') => Some(STARTUP_LOGIN_OPENAI_COMMAND),
            KeyCode::Char('p') => Some(STARTUP_PERMISSIONS_COMMAND),
            _ => None,
        }
    }

    fn handle_normal_top_shortcuts(&mut self, key: KeyEvent) -> Option<AppAction> {
        // Startup launchpad shortcuts prefill the composer without submitting.
        // Chords are consumed globally so they never fall through as literal
        // text when the launchpad is hidden or the composer already has a
        // draft/pasted image.
        if let Some(command) = Self::startup_prefill_command_for_key(key) {
            if self.startup.screen.is_some() {
                self.prefill_startup_command(command);
            }
            return Some(AppAction::None);
        }

        // F3 — open the model picker without typing `/model`.
        if matches!(key.code, KeyCode::F(3)) && key.modifiers.is_empty() {
            return Some(AppAction::Submit("/model".to_string()));
        }
        // Alt+1 — open the model picker without typing `/model`. Routed as
        // a Submit so the existing `/model` dispatch builds the provider-grouped
        // entries (which need the live `cli`, unavailable here in `App`).
        if matches!(key.code, KeyCode::Char('1')) && key.modifiers.contains(KeyModifiers::ALT) {
            return Some(AppAction::Submit("/model".to_string()));
        }
        // Alt+2 — open Smart Model Router settings, the single owner of
        // sub-agent model routing now that hidden session-wide overrides are removed.
        if matches!(key.code, KeyCode::Char('2')) && key.modifiers.contains(KeyModifiers::ALT) {
            return Some(AppAction::Submit("/smart".to_string()));
        }
        if matches!(key.code, KeyCode::Char('v')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(AppAction::ClipboardPaste);
        }
        // Ctrl+Y splits by composer state, same as Ctrl+A/E: an empty buffer
        // keeps it as "copy last message" (Ctrl+Shift+C stays a full-time
        // alias), a non-empty buffer lets it reach the composer as the
        // readline yank that completes the Ctrl+K/U/W kill set.
        if matches!(key.code, KeyCode::Char('y'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && self.input.is_empty()
        {
            return Some(AppAction::ClipboardCopy(ClipboardCopyTarget::Last));
        }
        if matches!(key.code, KeyCode::Char('y')) && key.modifiers.contains(KeyModifiers::ALT) {
            return Some(AppAction::ClipboardCopy(ClipboardCopyTarget::All));
        }
        // Ctrl+Shift+C — discoverable alias for "copy last message".
        if matches!(key.code, KeyCode::Char('C'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && key.modifiers.contains(KeyModifiers::SHIFT)
        {
            return Some(AppAction::ClipboardCopy(ClipboardCopyTarget::Last));
        }
        // `?` on an empty prompt opens the keybinding help overlay — but only
        // between turns. Mid-turn (input disabled) the composer is in
        // queued-message mode, so a literal `?` must reach the input widget
        // instead of stealing focus into the help pager and dropping whatever
        // the user is queuing.
        if matches!(key.code, KeyCode::Char('?'))
            && key.modifiers.is_empty()
            && self.input_enabled
            && self.input.is_empty()
        {
            self.open_pager(crate::tui::keybindings::help_text(!self.theme.no_color));
            return Some(AppAction::None);
        }

        None
    }

    fn handle_normal_scroll_shortcuts(
        &mut self,
        key: KeyEvent,
        arrows_scroll_transcript: bool,
    ) -> Option<AppAction> {
        match key.code {
            KeyCode::PageUp => {
                self.prepare_user_scroll();
                self.transcript.scroll_up(Self::HALF_PAGE_SCROLL_ROWS);
                self.transcript_view.follow_output = false;
                Some(AppAction::None)
            }
            KeyCode::PageDown => {
                self.prepare_user_scroll();
                self.transcript.scroll_down(Self::HALF_PAGE_SCROLL_ROWS);
                self.refresh_follow_output();
                Some(AppAction::None)
            }
            KeyCode::Home => {
                self.transcript.scroll_to_top();
                self.transcript_view.follow_output = false;
                Some(AppAction::None)
            }
            KeyCode::End => {
                self.transcript.scroll_to_bottom();
                self.transcript_view.follow_output = true;
                Some(AppAction::None)
            }
            KeyCode::Down if arrows_scroll_transcript => {
                self.history_cursor = None;
                self.history_stash.clear();
                self.hints.slash_cursor = None;
                self.hints.mention_cursor = None;
                self.prepare_user_scroll();
                self.transcript.scroll_down(Self::KEY_SCROLL_ROWS);
                self.refresh_follow_output();
                Some(AppAction::None)
            }
            KeyCode::Up if arrows_scroll_transcript => {
                self.history_cursor = None;
                self.history_stash.clear();
                self.hints.slash_cursor = None;
                self.hints.mention_cursor = None;
                self.prepare_user_scroll();
                self.transcript.scroll_up(Self::KEY_SCROLL_ROWS);
                self.transcript_view.follow_output = false;
                Some(AppAction::None)
            }
            KeyCode::Up if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.prepare_user_scroll();
                self.transcript.scroll_up(Self::KEY_SCROLL_ROWS);
                self.transcript_view.follow_output = false;
                Some(AppAction::None)
            }
            KeyCode::Down if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.prepare_user_scroll();
                self.transcript.scroll_down(Self::KEY_SCROLL_ROWS);
                self.refresh_follow_output();
                Some(AppAction::None)
            }
            KeyCode::Char('u')
                if self.input.is_empty() && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.prepare_user_scroll();
                self.transcript.scroll_up(Self::HALF_PAGE_SCROLL_ROWS);
                self.transcript_view.follow_output = false;
                Some(AppAction::None)
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.prepare_user_scroll();
                self.transcript.scroll_down(Self::HALF_PAGE_SCROLL_ROWS);
                self.refresh_follow_output();
                Some(AppAction::None)
            }
            _ => None,
        }
    }

    fn handle_normal_control_shortcuts(&mut self, key: KeyEvent) -> Option<AppAction> {
        match key.code {
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.toggle_sidebar();
                Some(AppAction::None)
            }
            KeyCode::Char('a')
                if self.input.is_empty() && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.toggle_sidebar_agents();
                Some(AppAction::None)
            }
            KeyCode::Char('g') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.open_agents_viewer();
                Some(AppAction::None)
            }
            KeyCode::Char('e')
                if self.input.is_empty() && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                Some(AppAction::Editor)
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.enter_search();
                Some(AppAction::None)
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+P mirrors typing `/`.
                if self.input_enabled && self.input.is_text_empty() {
                    self.input.insert_char('/');
                }
                Some(AppAction::None)
            }
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(AppAction::OpenRewindViewer)
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(AppAction::OpenWorkflowViewer)
            }
            // Ctrl+X — expand/collapse tool detail in the transcript (Claude
            // Code's ctrl+o verbose toggle; Ctrl+O here already opens the
            // workflow viewer, so eXpand takes the mnemonic next door).
            KeyCode::Char('x') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let expanded = self.transcript.toggle_tool_groups_disabled();
                self.push_diff_note(
                    runtime::message_stream::SystemLevel::Info,
                    if expanded {
                        "Tool detail expanded — every call/result shown (Ctrl+X to re-collapse)."
                    } else {
                        "Tool detail collapsed back into compact groups."
                    }
                    .to_string(),
                );
                Some(AppAction::None)
            }
            KeyCode::F(11) => {
                self.toggle_focus_mode();
                Some(AppAction::None)
            }
            _ => None,
        }
    }

    fn handle_normal_modifier_scroll_shortcuts(&mut self, key: KeyEvent) -> Option<AppAction> {
        match key.code {
            KeyCode::Up
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.prepare_user_scroll();
                self.transcript.scroll_up(1);
                self.transcript_view.follow_output = false;
                Some(AppAction::None)
            }
            KeyCode::Down
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.prepare_user_scroll();
                self.transcript.scroll_down(1);
                self.refresh_follow_output();
                Some(AppAction::None)
            }
            _ => None,
        }
    }

    fn handle_slash_hint_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        // Slash-hint popup: Up/Down move the highlighted row, Tab completes the
        // highlighted command (or the first visible command), and Esc hides the
        // hint until the input text changes.
        if self.slash_hint_active() {
            match key.code {
                KeyCode::Up
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.select_prev_slash_hint();
                    return Some(AppAction::None);
                }
                KeyCode::Down
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.select_next_slash_hint();
                    return Some(AppAction::None);
                }
                KeyCode::Tab
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    if let Some((cmd, _requires_arg)) = self.slash_hint_candidate() {
                        self.set_input_text(&format!("{cmd} "));
                        self.hide_slash_hint_for_current_input();
                        return Some(AppAction::None);
                    }
                }
                KeyCode::Enter if self.hints.slash_cursor.is_some() => {
                    if let Some((cmd, requires_arg)) = self.slash_hint_selected() {
                        self.hints.slash_cursor = None;
                        if requires_arg {
                            // Required `<arg>`: drop the command into the input
                            // and wait for the user to type the argument.
                            self.set_input_text(&format!("{cmd} "));
                            self.hide_slash_hint_for_current_input();
                            return Some(AppAction::None);
                        }
                        // No required argument: run immediately. Pickers like
                        // /model open their own modal from the command handler.
                        // The hint only shows while input is enabled, so this
                        // path always submits rather than queues.
                        self.record_command_usage(&cmd);
                        self.input.clear();
                        self.hints.slash_hidden_for = None;
                        return Some(AppAction::Submit(cmd));
                    }
                }
                KeyCode::Esc => {
                    self.hide_slash_hint_for_current_input();
                    return Some(AppAction::None);
                }
                _ => {
                    self.hints.slash_cursor = None;
                }
            }
        }

        None
    }

    fn handle_mention_hint_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        // `@`-mention popup: Tab inserts the highlighted path (or the first
        // visible path), Enter keeps the selected-row behavior for callers
        // that set a cursor explicitly, and plain arrows scroll the transcript.
        if self.mention_hint_active() {
            match key.code {
                KeyCode::Tab
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    if let Some(path) = self
                        .mention_hint_selected_path()
                        .or_else(|| self.mention_hint_suggestions().into_iter().next())
                    {
                        let line = self.input.text();
                        if let Some((at, _)) = mention_trigger(&line) {
                            self.set_input_text(&apply_mention(&line, at, &path));
                            let _ = self.mention_history.record(&path);
                        }
                        self.hints.mention_cursor = None;
                        return Some(AppAction::None);
                    }
                }
                KeyCode::Enter if self.hints.mention_cursor.is_some() => {
                    if let Some(path) = self.mention_hint_selected_path() {
                        let line = self.input.text();
                        if let Some((at, _)) = mention_trigger(&line) {
                            self.set_input_text(&apply_mention(&line, at, &path));
                            let _ = self.mention_history.record(&path);
                        }
                        self.hints.mention_cursor = None;
                        return Some(AppAction::None);
                    }
                }
                KeyCode::Esc => {
                    self.hide_mention_hint_for_current_input();
                    return Some(AppAction::None);
                }
                _ => {
                    self.hints.mention_cursor = None;
                }
            }
        }

        None
    }

    fn handle_normal_esc(&mut self, key: KeyEvent) -> Option<AppAction> {
        // Claude Code Esc semantics, in priority order (modal/search/pager and
        // hint-popup Esc are consumed earlier in the dispatch):
        //   1. a turn is running  → interrupt it (the spinner has always
        //      advertised `esc to interrupt`; only Ctrl+C was wired),
        //   2. the composer holds a draft → clear it,
        //   3. otherwise Esc-Esc rewinds the previous turn's conversation
        //      *and* code together (one combined checkpoint step). The window
        //      is short (600 ms) so a lone Esc decays quickly.
        if matches!(self.mode, AppMode::Normal) && matches!(key.code, KeyCode::Esc) {
            if self.turn_activity.is_some() {
                self.last_esc = None;
                let _ = self.cmd_tx.try_send(AgentCommand::CancelTurn);
                return Some(AppAction::None);
            }
            // A focused transcript block claims Esc next: dropping the focus
            // restores composer-driven Enter (submit) / Tab semantics.
            if self.transcript.clear_focus() {
                self.last_esc = None;
                return Some(AppAction::Redraw);
            }
            if !self.input.is_empty() {
                self.last_esc = None;
                while self.input.image_count() > 0 {
                    self.pop_clipboard_image();
                }
                self.input.clear();
                // A cleared draft also ends any history browse cleanly.
                self.history_cursor = None;
                self.history_stash.clear();
                return Some(AppAction::None);
            }
            let now = Instant::now();
            let double_tapped = self
                .last_esc
                .is_some_and(|prev| now.duration_since(prev) <= Self::ESC_DOUBLE_TAP_WINDOW);
            if double_tapped {
                self.last_esc = None;
                return Some(AppAction::RewindCheckpoint);
            }
            self.last_esc = Some(now);
            return Some(AppAction::None);
        }

        None
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        // `ModalWorkflow` is the one modal designed to coexist with the composer
        // (a read-only live monitor — see `dispatch_modal_key`). The mirror guard
        // in `handle_queued_input_key` already routes composer keys here while the
        // turn streams (`input_enabled == false`); without this `ModalWorkflow`
        // case, once the turn ends and input is re-enabled the viewer stays open
        // and every plain Enter/char is swallowed (no handler claims it), so the
        // composer can neither submit nor clear until the user presses Esc.
        if self.input_enabled && matches!(self.mode, AppMode::Normal | AppMode::ModalWorkflow) {
            // Backspace on empty text buffer: remove the last pending image.
            if matches!(key.code, KeyCode::Backspace)
                && self.input.cursor() == (0, 0)
                && self.input.image_count() > 0
            {
                self.pop_clipboard_image();
                return Some(AppAction::None);
            }

            // `@` opens the fuzzy file-reference picker instead of
            // landing a literal `@` in the buffer. The chosen path is
            // spliced back in as an `@path ` token when the modal
            // resolves (see the `AppMode::ModalFile` dispatch above).
            if matches!(key.code, KeyCode::Char('@'))
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT)
            {
                // Coexist (opencode `autocomplete.tsx` @ arm): a bare `@` at
                // the line start opens the full fuzzy picker modal; an `@`
                // typed after existing text starts an inline mention popup.
                if self.mention_opens_inline() {
                    self.input.insert_char('@');
                    self.ensure_workspace_files();
                    self.hints.mention_cursor = None;
                    return Some(AppAction::None);
                }
                self.open_file_picker();
                return Some(AppAction::None);
            }

            // Slash-command autocomplete: when the user is mid-way
            // through typing a slash command and presses Space or
            // Tab, expand the current token to the top suggestion
            // before the key reaches the input widget. We only
            // trigger when the input cursor is parked at the end of
            // a single-line `/token` buffer so we never rewrite past
            // edits or in-flight multi-line prompts.
            if matches!(key.code, KeyCode::Char(' ') | KeyCode::Tab)
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT)
                && !self.input.has_collapsed_paste()
            {
                if let Some(completed) =
                    slash_completion_for(self.input.text().as_str(), &self.prompt_commands)
                {
                    self.input.clear();
                    for ch in completed.chars() {
                        self.input.insert_char(ch);
                    }
                    return Some(AppAction::None);
                }
            }

            let input_revision_before_key = self.input.content_revision();
            if let Some(command) = self.input.handle_key(key) {
                // Reset history browsing when user submits or cancels.
                self.history_cursor = None;
                self.history_stash.clear();
                self.hints.slash_hidden_for = None;
                return Some(match command {
                    InputCommand::Submit(text) => {
                        // If the user presses Enter on a partial slash
                        // command that has exactly one completion, auto-
                        // complete and submit the full command.
                        let submit_text =
                            slash_completion_for(text.as_str(), &self.prompt_commands)
                                .unwrap_or(text);
                        // Submitting a fresh message from the read-only workflow
                        // viewer ends the monitor session: close it so the new
                        // turn is not hidden behind the viewer's `Clear` overlay.
                        // The mid-turn steer path (input disabled) deliberately
                        // leaves the modal open while the live turn runs.
                        if matches!(self.mode, AppMode::ModalWorkflow) {
                            self.exit_modal();
                        }
                        self.transcript.scroll_to_bottom();
                        self.transcript_view.follow_output = true;
                        AppAction::Submit(submit_text)
                    }
                    InputCommand::Cancel => AppAction::None,
                });
            }
            let _ = self.input.strip_sgr_mouse_sequences();

            if self.input.content_revision() != input_revision_before_key {
                self.hints.slash_hidden_for = None;
            }

            // Reset history browsing after any key routed through the input.
            if self.history_cursor.is_some() {
                self.history_cursor = None;
                self.history_stash.clear();
            }
        }

        None
    }

    fn handle_queued_input_key(&mut self, key: KeyEvent) -> Option<AppAction> {
        // Queued-message mode: when input is disabled (turn in progress)
        // but the app is in Normal mode, allow typing into the input
        // widget and treat Enter as "queue this message for later".
        if !self.input_enabled && matches!(self.mode, AppMode::Normal | AppMode::ModalWorkflow) {
            // Shift/Alt+Enter inserts a newline so the user can compose a
            // multi-line message while a turn is in flight — exactly as the
            // input-enabled path does (see `InputWidget::handle_key`). Only a
            // *plain* Enter commits the composed entry to the queue; without
            // this guard every Shift+Enter would split one multi-line message
            // into several separate queued turns.
            let enter_with_modifier = key.code == KeyCode::Enter
                && (key.modifiers.contains(KeyModifiers::SHIFT)
                    || key.modifiers.contains(KeyModifiers::ALT));
            if key.code == KeyCode::Enter && !enter_with_modifier {
                let text = self.input.text();
                let trimmed = text.trim();
                let has_images = self.input.image_count() > 0;
                // Nothing to queue: blank text and no pasted images.
                if trimmed.is_empty() && !has_images {
                    return Some(AppAction::None);
                }
                // Claude Code CLI parity: every composed entry — plain text,
                // slash command, or pasted image(s) — is parked in the queue
                // and shown via the "queued" badge. Plain text additionally
                // rides the steering channel so the *live* turn folds it in at
                // its next tool boundary (the "type to steer" reflex) — the
                // queued entry is removed when the `⤷ steering` delivery echo
                // lands, and any steer the turn never folded auto-submits as
                // its own next turn in FIFO order. Slash commands and entries
                // carrying images always wait for their own turn.
                //
                // FIFO guard: only steer while every earlier queued entry
                // itself rode the steering channel. A steer cuts into the
                // *live* turn, so steering a message that sits behind an
                // unsteered entry (a slash command, an image turn) would
                // deliver it before that earlier entry — breaking the strict
                // FIFO order Claude Code preserves. Earlier steered plain
                // texts are fine: the SteeringQueue is itself FIFO, so this
                // message folds after them. (The old empty-queue guard made
                // every message after the first wait out the whole turn.)
                let earlier_all_steered =
                    self.queued_messages.iter().all(|message| message.steered);
                if let Err(error) = self.ensure_can_queue_message() {
                    self.report_queue_limit_error(error);
                    return Some(AppAction::None);
                }
                let images = self.take_pending_images();
                let plain_text = !trimmed.starts_with('/') && images.is_empty();
                let steer_text = if plain_text && earlier_all_steered {
                    Some(trimmed.to_string())
                } else {
                    None
                };
                if let Err(error) = self.queue_composed_message(QueuedMessage {
                    text,
                    images,
                    goal_owned: false,
                    loop_id: None,
                    agent_result: None,
                    steered: steer_text.is_some(),
                }) {
                    self.report_queue_limit_error(error);
                    return Some(AppAction::None);
                }
                if let Some(steer_text) = steer_text {
                    let _ = self.cmd_tx.try_send(AgentCommand::Steer(steer_text));
                }
                self.input.clear();
                return Some(AppAction::None);
            }
            // Backspace on an empty buffer removes the last pending image,
            // mirroring the input-enabled path above — otherwise a pasted
            // image can't be removed while a turn is in flight (the plain
            // `handle_key` backspace only touches the text buffer).
            if matches!(key.code, KeyCode::Backspace)
                && self.input.cursor() == (0, 0)
                && self.input.image_count() > 0
            {
                self.pop_clipboard_image();
                return Some(AppAction::None);
            }
            // Forward all other keys (chars, backspace, arrows,
            // etc.) to the input widget so the user can compose
            // their next message while waiting.
            let _ = self.input.handle_key(key);
            let _ = self.input.strip_sgr_mouse_sequences();
            return Some(AppAction::None);
        }

        None
    }
}

fn secret_modal_paste_key(key: &KeyEvent) -> bool {
    if key.kind != KeyEventKind::Press {
        return false;
    }
    (matches!(key.code, KeyCode::Char('v')) && key.modifiers.contains(KeyModifiers::CONTROL))
        || (matches!(key.code, KeyCode::Insert) && key.modifiers.contains(KeyModifiers::SHIFT))
}

fn workflow_modal_consumes_key(key: &KeyEvent) -> bool {
    if key.kind != KeyEventKind::Press {
        return true;
    }
    if matches!(key.code, KeyCode::Char('c' | 'e'))
        && key.modifiers.contains(KeyModifiers::CONTROL)
    {
        return true;
    }
    matches!(
        key.code,
        KeyCode::Esc
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Left
            | KeyCode::Right
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::PageUp
            | KeyCode::PageDown
    )
}

#[cfg(test)]
mod secret_paste_key_tests {
    use super::*;
    use crossterm::event::KeyEventState;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn secret_modal_paste_key_accepts_clipboard_shortcuts() {
        assert!(secret_modal_paste_key(&key(
            KeyCode::Char('v'),
            KeyModifiers::CONTROL,
        )));
        assert!(secret_modal_paste_key(&key(
            KeyCode::Insert,
            KeyModifiers::SHIFT,
        )));
    }

    #[test]
    fn secret_modal_paste_key_rejects_plain_typing() {
        assert!(!secret_modal_paste_key(&key(
            KeyCode::Char('v'),
            KeyModifiers::NONE,
        )));
    }
}
