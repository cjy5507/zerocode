use std::sync::Arc;

use zo_cli::tui::modals::Effort;

use super::runtime_bridge::LiveAsyncApiClient;
use super::BuiltRuntime;
use crate::cli_args::AllowedToolSet;

/// Shared host-side turn scaffolding for the CLI/TUI entry paths.
///
/// The runtime crate owns the model/tool loop; this harness deliberately stays
/// at the session-host layer and exposes only the pieces that are byte-for-byte
/// common across entry paths. Callers still choose their path-specific ordering
/// and policies (TUI host pre-spawn vs. headless model-led routing, long-lived
/// serve stale-gate clearing vs. per-turn runtime rebuilds, text/json no stream
/// client vs. streaming paths with a live client).
pub(crate) struct TurnHarness;

impl TurnHarness {
    pub(crate) fn route_reminder_for_model_led_turn(
        input: &str,
        effort: Option<Effort>,
        session_tokens: usize,
        system_prompt: &[String],
    ) -> Option<String> {
        let context_tokens = estimate_route_context_tokens(session_tokens, system_prompt);
        super::auto_fanout::build_route_hint(input, effort, context_tokens).model_led_reminder()
    }

    pub(crate) fn setup_model_led_turn(
        runtime: &mut BuiltRuntime,
        options: ModelLedTurnSetup<'_>,
    ) -> TurnSetupToken {
        if options.clear_stale_reactive_gate {
            Self::clear_stale_reactive_gate(runtime);
        }
        let route_reminder = Self::route_reminder_for_model_led_turn(
            options.input,
            options.effort,
            options.session_tokens,
            options.system_prompt,
        );
        Self::apply_route_reminder(runtime, route_reminder.as_deref());
        Self::install_turn_agent_policy(runtime, options.input);
        TurnSetupToken { route_reminder }
    }

    /// Install this turn's [`tools::TurnAgentPolicy`] onto the shared tool
    /// context so the `Agent` dispatch guard can fold a wasteful same-model
    /// simple-implementation spawn to inline. Centralized here so every
    /// non-interactive model-led entry (text/JSON/NDJSON, serve/socket) shares
    /// one install point; the interactive path installs its own. Signals come
    /// from the corpus-pinned SSOT classifiers on the raw user turn.
    fn install_turn_agent_policy(runtime: &mut BuiltRuntime, input: &str) {
        let orchestration = tools::assess_turn_orchestration(input);
        let policy = tools::TurnAgentPolicy {
            user_complexity: tools::assess_turn_complexity(input),
            user_shape: orchestration.shape,
            user_need_count: orchestration.need_count,
            user_requested_delegation: orchestration.user_requested_delegation,
        };
        if let Some(inner) = runtime.try_runtime_mut() {
            inner
                .tool_executor_mut()
                .tool_registry_mut()
                .context()
                .set_turn_agent_policy(Some(policy));
        }
    }

    pub(crate) fn apply_route_reminder(runtime: &mut BuiltRuntime, reminder: Option<&str>) {
        if let Some(inner) = runtime.try_runtime_mut() {
            inner.replace_transient_system_reminder_by_prefix(
                super::auto_fanout::ROUTE_HINT_REMINDER_PREFIX,
                reminder,
            );
        }
    }

    pub(crate) fn build_live_client(
        runtime: &BuiltRuntime,
        allowed_tools: Option<AllowedToolSet>,
        thinking: Option<api::ThinkingConfig>,
        named_effort: Option<api::EffortLevel>,
        effort_band_ceiling: Option<api::EffortLevel>,
    ) -> Arc<LiveAsyncApiClient> {
        let api_client = runtime.api_client();
        Arc::new(LiveAsyncApiClient::new(
            api_client.client(),
            api_client.model().to_string(),
            api_client.auth_route(),
            api_client.enable_tools(),
            allowed_tools,
            api_client.tool_registry(),
            thinking,
            named_effort,
            effort_band_ceiling,
        ))
    }

    pub(crate) fn install_automation_plan_gate_if_needed(
        input: &str,
        runtime: &mut BuiltRuntime,
    ) -> DeepGateRestore {
        let previous = runtime.deep_gate().cloned();
        // Delegate the install/restore decision to automation.rs so every turn
        // entry path shares the same one-turn plan gate semantics.
        let Some(change) = super::automation::automation_plan_gate_change(input, previous.as_ref())
        else {
            return DeepGateRestore::not_installed();
        };
        if let Some(config) = change.install {
            if let Some(inner) = runtime.try_runtime_mut() {
                inner.set_deep_gate(Some(config));
            }
        }
        DeepGateRestore::installed(change.restore)
    }

    pub(crate) fn install_reactive_verify_gate_if_coding(
        input: &str,
        runtime: &mut BuiltRuntime,
    ) -> DeepGateRestore {
        if !Self::reactive_verify_gate_wanted(
            input,
            Self::auto_verify_opted_out(),
            runtime.deep_gate().is_some(),
        ) {
            return DeepGateRestore::not_installed();
        }
        let previous = runtime.deep_gate().cloned();
        if let Some(inner) = runtime.try_runtime_mut() {
            inner.set_deep_gate(Some(runtime::DeepGateConfig {
                mode: runtime::DeepMode::Reactive,
                check_command: Self::headless_objective_check_command(),
                max_attempts: 2,
            }));
        }
        DeepGateRestore::installed(previous)
    }

    pub(crate) fn restore_deep_gate(runtime: &mut BuiltRuntime, restore: DeepGateRestore) {
        if let DeepGateRestore::Installed { previous } = restore {
            if let Some(inner) = runtime.try_runtime_mut() {
                inner.set_deep_gate(previous);
            }
        }
    }

    /// Install a one-turn read-only permission gate for an unattended
    /// (`/loop`/`/goal` schedule) automation turn, mirroring
    /// [`Self::install_automation_plan_gate_if_needed`]. Delegates the policy to
    /// [`super::automation::automation_permission_gate_change`], so a user-typed
    /// turn and an opted-in (`--allow-writes`) automation turn keep the session's
    /// permission untouched. When the gate fires it (a) forces read-only if the
    /// session is write-capable, and (b) grants the propose-only allowlist so the
    /// read-only turn can still record its proposal into the team inbox — both
    /// undone by [`Self::restore_automation_permission_gate`].
    pub(crate) fn install_automation_permission_gate_if_needed(
        input: &str,
        runtime: &mut BuiltRuntime,
    ) -> AutomationPermissionGate {
        let Some(inner) = runtime.try_runtime_mut() else {
            return AutomationPermissionGate::NotInstalled;
        };
        let current = inner.active_permission_mode();
        let Some(change) =
            super::automation::automation_permission_gate_change(input, current)
        else {
            return AutomationPermissionGate::NotInstalled;
        };
        if let Some(mode) = change.downgrade_to {
            inner.set_active_permission_mode(mode);
        }
        let rules = super::automation::automation_read_only_allow_rules();
        let grant = inner.add_temporary_permission_allow_rules(&rules);
        AutomationPermissionGate::Installed {
            previous_mode: change.restore,
            grant,
        }
    }

    pub(crate) fn restore_automation_permission_gate(
        runtime: &mut BuiltRuntime,
        gate: AutomationPermissionGate,
    ) {
        if let AutomationPermissionGate::Installed {
            previous_mode,
            grant,
        } = gate
        {
            if let Some(inner) = runtime.try_runtime_mut() {
                inner.remove_temporary_permission_allow_rules(grant);
                // A no-op when the mode was never downgraded (already read-only),
                // since `previous_mode` then equals the live mode.
                inner.set_active_permission_mode(previous_mode);
            }
        }
    }

    /// Clear a **reactive** auto-verify gate that a prior long-lived-runtime turn
    /// left installed. Per-turn rebuilt runtimes do not need this, so callers opt
    /// in only for serve/TUI-style long-lived entry paths.
    pub(crate) fn clear_stale_reactive_gate(runtime: &mut BuiltRuntime) {
        if matches!(
            runtime.deep_gate().map(|gate| gate.mode),
            Some(runtime::DeepMode::Reactive)
        ) {
            if let Some(inner) = runtime.try_runtime_mut() {
                inner.set_deep_gate(None);
            }
        }
    }

    pub(crate) fn reactive_verify_gate_wanted(
        input: &str,
        opted_out: bool,
        has_gate: bool,
    ) -> bool {
        !opted_out && !has_gate && crate::main_dispatch::prompt_is_coding_task(input)
    }

    pub(crate) fn auto_verify_opted_out() -> bool {
        std::env::var("ZO_AUTO_VERIFY")
            .map(|value| value == "0" || value.eq_ignore_ascii_case("off"))
            .unwrap_or(false)
    }

    pub(crate) fn headless_objective_check_command() -> Option<String> {
        std::env::var("ZO_AUTO_VERIFY_CMD")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ModelLedTurnSetup<'a> {
    pub(crate) input: &'a str,
    pub(crate) effort: Option<Effort>,
    pub(crate) session_tokens: usize,
    pub(crate) system_prompt: &'a [String],
    pub(crate) clear_stale_reactive_gate: bool,
}

#[derive(Debug)]
pub(crate) struct TurnSetupToken {
    #[allow(dead_code)]
    pub(crate) route_reminder: Option<String>,
}

/// One-turn deep-gate restore token. A plain `Option<Option<_>>` would
/// conflate "nothing was installed" with "installed over an absent gate",
/// so the three states get named cases instead.
#[derive(Debug)]
pub(crate) enum DeepGateRestore {
    /// No gate was installed this turn; [`TurnHarness::restore_deep_gate`]
    /// is a no-op.
    NotInstalled,
    /// A gate was installed; restore whatever was in place before
    /// (possibly no gate at all).
    Installed {
        previous: Option<runtime::DeepGateConfig>,
    },
}

impl DeepGateRestore {
    fn not_installed() -> Self {
        Self::NotInstalled
    }

    fn installed(previous: Option<runtime::DeepGateConfig>) -> Self {
        Self::Installed { previous }
    }
}

/// One-turn restore token for the unattended-automation read-only permission
/// gate. Carries both the previous mode and the transient propose-only allow
/// grant so [`TurnHarness::restore_automation_permission_gate`] can undo the full
/// change (never leaking the read-only override or the allowlist past the turn).
#[derive(Debug)]
pub(crate) enum AutomationPermissionGate {
    /// No gate was installed this turn (user turn or an `--allow-writes` opt-in);
    /// restore is a no-op.
    NotInstalled,
    /// The gate fired: restore the recorded mode and drop the transient grant.
    Installed {
        previous_mode: runtime::PermissionMode,
        grant: runtime::TemporaryAllowGrant,
    },
}

/// Estimate accumulated context size for the route classifier. Mirrors the TUI
/// approximation so headless/serve route nudges see the same signal.
fn estimate_route_context_tokens(session_tokens: usize, system_prompt: &[String]) -> usize {
    system_prompt.iter().fold(session_tokens, |acc, section| {
        acc.saturating_add(section.len() / 4 + 4)
    })
}

#[cfg(test)]
mod tests {
    use super::{ModelLedTurnSetup, TurnHarness};

    /// Set a dummy API key for the test's lifetime, restoring the previous
    /// value on drop. Callers must hold [`crate::test_env_lock`] first so the
    /// mutation cannot race sibling tests that read or write the same var.
    struct ApiKeyGuard {
        previous: Option<std::ffi::OsString>,
    }

    impl ApiKeyGuard {
        fn set_dummy() -> Self {
            let previous = std::env::var_os("ANTHROPIC_API_KEY");
            std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-turn-harness");
            Self { previous }
        }
    }

    impl Drop for ApiKeyGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var("ANTHROPIC_API_KEY", value),
                None => std::env::remove_var("ANTHROPIC_API_KEY"),
            }
        }
    }

    #[test]
    fn model_led_setup_applies_route_reminder_and_clears_stale_reactive_gate() {
        let _env_lock = crate::test_env_lock();
        crate::isolate_global_zo_home_for_tests();
        let _api_key = ApiKeyGuard::set_dummy();

        let (mut runtime, _hook_abort_monitor) = crate::session::LiveCli::new(
            "sonnet".to_string(),
            true,
            None,
            runtime::PermissionMode::ReadOnly,
        )
        .expect("live cli should build")
        .prepare_turn_runtime_for_test(false)
        .expect("runtime should build");
        if let Some(inner) = runtime.try_runtime_mut() {
            inner.set_deep_gate(Some(runtime::DeepGateConfig {
                mode: runtime::DeepMode::Reactive,
                check_command: None,
                max_attempts: 2,
            }));
        }

        let token = TurnHarness::setup_model_led_turn(
            &mut runtime,
            ModelLedTurnSetup {
                input: "이 failing test 하나 원인 찾아줘",
                effort: Some(zo_cli::tui::modals::Effort::High),
                session_tokens: 1_000,
                system_prompt: &[],
                clear_stale_reactive_gate: true,
            },
        );

        assert!(
            token.route_reminder.as_deref().is_some_and(|reminder| reminder
                .starts_with(super::super::auto_fanout::ROUTE_HINT_REMINDER_PREFIX)),
            "headless setup should compute and apply a route reminder"
        );
        assert!(
            runtime.deep_gate().is_none(),
            "headless setup should clear a stale reactive gate when requested"
        );
    }

    #[test]
    fn reactive_verify_gate_policy_is_shared_by_headless_and_serve_callers() {
        for &(prompt, opted_out, has_gate, want) in &[
            ("fix the bug in src/click/core.py", false, false, true),
            (
                "analyze and summarize the architecture",
                false,
                false,
                false,
            ),
            ("fix the bug in src/click/core.py", false, true, false),
            ("fix the bug in src/click/core.py", true, false, false),
        ] {
            assert_eq!(
                TurnHarness::reactive_verify_gate_wanted(prompt, opted_out, has_gate),
                want,
                "prompt={prompt:?} opted_out={opted_out} has_gate={has_gate}"
            );
        }
    }
}
