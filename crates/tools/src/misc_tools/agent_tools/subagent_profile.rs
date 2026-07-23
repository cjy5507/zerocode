//! Sub-agent profile resolution: harness type, system prompt, and model routing
//! for a spawn. Behaviour-preserving split out of `agent_tools.rs`; the entry
//! points stay reachable via re-export so call sites are unchanged.

use runtime::load_system_prompt;

use super::custom::CustomAgent;
use super::{AGENT_MODEL_ENV, DEFAULT_AGENT_MODEL, normalize_subagent_type};

const MEDIUM_EFFORT_TOKENS: u32 = 4_096;
const HIGH_EFFORT_TOKENS: u32 = 10_000;
const XHIGH_EFFORT_TOKENS: u32 = 16_000;

const AGENT_COMPLETION_CONTRACT: &str = "Stop condition: once you have enough evidence to answer \
the delegated task, stop calling tools and return the final result. Keep the investigation \
bounded; do not try to exhaust the repository. If two consecutive searches or reads add no new \
evidence, pivot once, then report the blocker instead of looping.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AgentModelSelection {
    pub model: String,
    pub thinking_budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AgentModelSelectionError {
    ExplicitCrossProvider {
        requested: String,
        parent: String,
    },
    CustomCrossProvider {
        requested: String,
        parent: String,
    },
}

impl std::fmt::Display for AgentModelSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExplicitCrossProvider { requested, parent } => write!(
                f,
                "explicit sub-agent model \"{requested}\" is outside the active parent model family \"{parent}\"; refusing to silently inherit the parent model. If the USER explicitly asked for this model, retry the same spawn with \"allow_cross_provider\": true — do NOT substitute a different model or change session settings. Otherwise switch the session to that provider or set ZO_AGENT_MODEL to intentionally override all agents."
            ),
            Self::CustomCrossProvider { requested, parent } => write!(
                f,
                "custom sub-agent model \"{requested}\" is outside the active parent model family \"{parent}\"; refusing to silently inherit the parent model. Switch the session to that provider, use trusted Smart routing, or set ZO_AGENT_MODEL to intentionally override all agents."
            ),
        }
    }
}

/// Resolve the harness type for a spawn. An explicit, non-empty type is
/// normalized to its canonical form; an absent or blank type is inferred from
/// the task text so callers (and the model) get the best-fit harness without
/// having to name it.
pub(crate) fn resolve_subagent_type(
    explicit: Option<&str>,
    description: &str,
    prompt: &str,
) -> String {
    match explicit.map(str::trim) {
        Some(token) if !token.is_empty() => normalize_subagent_type(Some(token)),
        _ => infer_subagent_type(description, prompt).to_string(),
    }
}

/// Heuristically pick the best built-in harness for a task from its text.
/// Order encodes precedence: the most specific specialist wins, falling back
/// through verification → refactor → planning → research → analysis →
/// exploration → `general-purpose`. Callers (and the model) can always name a
/// `subagent_type` explicitly to bypass inference.
#[allow(clippy::too_many_lines)] // a flat keyword table, clearer unsplit
fn infer_subagent_type(description: &str, prompt: &str) -> &'static str {
    // `to_ascii_lowercase` leaves non-ASCII (Korean) untouched, so the Korean
    // keywords below match verbatim — a Korean task gets a proper specialist type
    // (and harness) instead of falling through to the generic default the way
    // English-only matching did, which made a "...분석" fan-out spawn generic
    // agents that then routed to the cheap Fast tier.
    let haystack = format!("{description}\n{prompt}").to_ascii_lowercase();
    let has_any = |needles: &[&str]| needles.iter().any(|needle| haystack.contains(needle));

    // Code review (read-only critique) — before Verification so "review the
    // code" isn't mistaken for a test run.
    if has_any(&[
        "code review",
        "review the code",
        "review this code",
        "review the change",
        "review the diff",
        "review the pr",
        "review my",
        "critique the",
        "코드 리뷰",
        "코드리뷰",
        "리뷰",
        "검토",
    ]) {
        return "code-reviewer";
    }
    let has_verification_intent = has_any(&[
        "run the test",
        "run tests",
        "unit test",
        "integration test",
        "verify",
        "verification",
        "cargo test",
        "type-check",
        "typecheck",
        "lint",
        "검증",
        "테스트 실행",
        "테스트를 실행",
        "검사",
    ]);
    let contains_ordered_pair = |firsts: &[&str], seconds: &[&str]| {
        firsts.iter().any(|first| {
            haystack.find(first).is_some_and(|first_idx| {
                let after_first = &haystack[first_idx + first.len()..];
                seconds.iter().any(|second| after_first.contains(second))
            })
        })
    };
    let has_mixed_verify_and_edit_intent = has_verification_intent
        && (has_any(&[
            "verify and fix",
            "verify then fix",
            "test and fix",
            "run tests and fix",
            "fix and verify",
            "fix and test",
        ]) || contains_ordered_pair(
            &["검증", "테스트", "검사"],
            &["구현", "수정", "고쳐", "변경", "패치"],
        ));
    if has_verification_intent && !has_mixed_verify_and_edit_intent {
        return "Verification";
    }

    // Debugging (reproduce → root-cause → fix) — before Verification so an
    // explicit debug task edits rather than only running tests.
    if has_any(&[
        "debug",
        "root cause",
        "root-cause",
        "stack trace",
        "stacktrace",
        "traceback",
        "segfault",
        "panic",
        "fix the bug",
        "fix this bug",
        "why does it crash",
        "why is it failing",
        "디버그",
        "디버깅",
        "버그 수정",
        "재현",
    ]) {
        return "debugger";
    }
    // Implementation/fix intent must beat a verification word when the same
    // delegated task asks to "verify and fix". Korean continuation prompts often
    // say "적대적검증 하고 ... 수정"; routing that to a Verification harness drops
    // the edit intent before the worker sees it. Pure verification still falls
    // through to the branch below.
    if has_any(&[
        "implement",
        "fix the",
        "fix this",
        "fix it",
        "fix bug",
        "apply the fix",
        "and fix",
        "fix and",
        "edit the code",
        "patch the code",
        "change the code",
        "modify",
        "구현",
        "수정",
        "고쳐",
        "변경",
        "패치",
    ]) {
        return "general-purpose";
    }
    if has_any(&[
        "run the test",
        "run tests",
        "unit test",
        "integration test",
        "verify",
        "verification",
        "cargo test",
        "build fails",
        "compile error",
        "type-check",
        "typecheck",
        "lint",
        "reproduce the bug",
        "reproduce the failure",
        "검증",
        "테스트 실행",
        "테스트를 실행",
        "검사",
    ]) {
        return "Verification";
    }
    // Behavior-preserving structural change.
    if has_any(&[
        "refactor",
        "restructure",
        "clean up the",
        "deduplicate",
        "extract a",
        "extract the",
        "rename the",
        "simplify the code",
        "tidy up",
    ]) {
        return "refactor";
    }
    if has_any(&[
        "make a plan",
        "write a plan",
        "design ",
        "architect",
        "approach",
        "strategy",
        "trade-off",
        "tradeoff",
        "proposal",
        "how should we",
        "how would you structure",
        "분석",
        "추론",
        "계획",
        "설계",
        "아키텍처",
    ]) {
        return "Plan";
    }
    // Multi-pass research/synthesis — a heavier Explore that also reads the web.
    if has_any(&[
        "deep research",
        "deep-research",
        "comprehensive analysis",
        "thoroughly research",
        "in-depth investigation",
        "survey the",
        "literature review",
        "research how",
        "research the",
    ]) {
        return "deep-research";
    }
    // Data/log/metric analysis.
    if has_any(&[
        "analyze the data",
        "analyse the data",
        "data analysis",
        "analyze the log",
        "analyse the log",
        "log analysis",
        "parse the logs",
        "dataset",
        "metrics",
        "statistics",
    ]) {
        return "data-analyst";
    }
    if has_any(&[
        "explore",
        "search for",
        "find where",
        "locate",
        "where is",
        "where are",
        "investigate",
        "map the",
        "understand how",
        "trace ",
        "look for",
        "which files",
        "audit the codebase",
    ]) {
        return "Explore";
    }
    "general-purpose"
}

/// Per-type harness guidance appended to a built-in agent's base prompt.
/// This is the "best harness": each type gets role-specific instructions
/// instead of one generic line, so an `Explore` agent behaves like a search
/// specialist and a `Verification` agent like a test runner.
pub(super) fn builtin_harness_instruction(subagent_type: &str) -> String {
    let role = match subagent_type {
        "Explore" => {
            "You are a read-only exploration agent. Search broadly to answer the delegated question, read enough to be certain, and return a concise conclusion with the key `file:line` references — not raw file dumps. Do not edit files."
        }
        "Plan" => {
            "You are a software-architect agent. Investigate the relevant code, then return a concrete, step-by-step implementation plan: the files to touch, the approach, and the trade-offs. Do not write or edit code."
        }
        "Verification" => {
            "You are a verification agent. Build, test, lint, or otherwise check the work as delegated, and report pass/fail with the exact command output as evidence. Never paper over a failure."
        }
        "deep-research" => {
            "You are a deep-research agent. Investigate the question in multiple passes across the codebase and the web — search broadly, read the primary sources, and cross-check competing answers — then synthesize a structured, cited conclusion with the key `file:line` references and URLs. Do not edit files."
        }
        "code-reviewer" => {
            "You are an adversarial code-review agent. Scrutinize the delegated change or code for correctness, security, edge cases, and maintainability — actively try to break it. Report concrete `file:line` findings with a severity and a suggested fix. Do not edit files; your output is the review."
        }
        "debugger" => {
            "You are a debugging agent. Reproduce the failure first, then isolate the root cause with evidence (not a guess), apply the minimal fix, and re-run to confirm it is resolved. Report the root cause and the exact fix."
        }
        "data-analyst" => {
            "You are a data-analysis agent. Inspect the delegated data, logs, or metrics, compute the relevant aggregates, and report findings with concrete numbers and the queries/commands used as evidence. Prefer precise figures over impressions."
        }
        "refactor" => {
            "You are a refactoring agent. Improve the structure and clarity of the delegated code without changing its behavior. Make surgical, mechanical edits, keep the tests green before and after, and report what changed and why."
        }
        "zo-guide" => {
            "You are a zo guide agent. Answer questions about this project's behavior and conventions using the actual code as the source of truth."
        }
        "statusline-setup" => {
            "You are a statusline-setup agent. Configure the user's status line per their request and confirm the resulting setting."
        }
        _ => {
            "You are a general-purpose agent. Complete the delegated task end-to-end using the tools available to you."
        }
    };
    format!(
        "{role} Work only on the delegated task, use only the tools available to you, do not ask the user questions, and finish with a concise result. {AGENT_COMPLETION_CONTRACT}"
    )
}

pub(super) fn build_agent_system_prompt(
    subagent_type: &str,
    custom: Option<&CustomAgent>,
) -> Result<Vec<String>, String> {
    // The classifier harness is a micro prompt, not the full system prompt.
    // Its callers (fan-out triage / decomposition) run one-shot classification
    // on a CHEAP model from a different family than the parent — a separate
    // prompt-cache namespace — so the ~4.4k-token full prompt was re-written
    // there per session with zero cache benefit, and its coding/delegation
    // guidance is noise for a one-sentence JSON answer. The identity line must
    // stay first: the Claude Max OAuth path rejects any other first system
    // block as a fingerprint mismatch (surfaced as a 429).
    if custom.is_none() && subagent_type == "classifier" {
        return Ok(vec![
            runtime::CLAUDE_CODE_IDENTITY.to_string(),
            format!(
                "You are a fast classification agent. Answer the delegated question about the \
                 given text directly and return ONLY the requested output format — no preamble, \
                 no tool exploration, no questions. {AGENT_COMPLETION_CONTRACT}"
            ),
        ]);
    }
    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    // Sub-agents get the live local date, same source as the main session's
    // prompt — a hardcoded date here once froze every sub-agent's "today".
    let mut prompt = load_system_prompt(
        cwd,
        core_types::date::current_local_date(),
        std::env::consts::OS,
        "unknown",
    )
    .map_err(|error| error.to_string())?;
    if let Some(custom) = custom {
        let header = if custom.description.is_empty() {
            format!("You are the `{}` sub-agent.", custom.name)
        } else {
            format!(
                "You are the `{}` sub-agent — {}.",
                custom.name,
                custom.description.trim_end_matches('.')
            )
        };
        prompt.push(format!(
            "{header} Work only on the delegated task, use only the tools available to you, do not ask the user questions, and finish with a concise result. {AGENT_COMPLETION_CONTRACT}"
        ));
        prompt.push(custom.system_prompt.clone());
    } else {
        prompt.push(builtin_harness_instruction(subagent_type));
    }
    Ok(prompt)
}

/// Pick the model for a sub-agent. Precedence (first non-empty wins):
/// 1. `ZO_AGENT_MODEL` env override — an explicit user-level override for
///    all sub-agents,
/// 2. the parent/session model — automatic agents stay in the active model
///    family,
/// 3. the process fallback ([`DEFAULT_AGENT_MODEL`]) for non-live harnesses
///    that cannot supply a parent model.
pub(super) fn resolve_agent_model(
    _model: Option<&str>,
    _custom_model: Option<&str>,
    _subagent_type: &str,
    parent_model: Option<&str>,
) -> String {
    let non_empty = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    non_empty(std::env::var(AGENT_MODEL_ENV).ok().as_deref())
        .or_else(|| non_empty(parent_model))
        .unwrap_or_else(|| DEFAULT_AGENT_MODEL.to_string())
}

/// Pick the concrete model and optional reasoning budget for a spawned
/// sub-agent. This keeps `resolve_agent_model` as the plain inheritance helper
/// while the live spawn path inherits the parent/session model and uses task
/// difficulty only to tune reasoning budget/effort.
pub(super) fn try_resolve_agent_model_selection(
    model: Option<&str>,
    cross_provider_allowed: bool,
    custom_model: Option<&str>,
    subagent_type: &str,
    parent_model: Option<&str>,
    description: &str,
    prompt: &str,
) -> Result<AgentModelSelection, AgentModelSelectionError> {
    // 1. A global env override wins over everything.
    if let Some(env_model) = non_empty(std::env::var(AGENT_MODEL_ENV).ok().as_deref()) {
        return Ok(AgentModelSelection {
            model: env_model,
            thinking_budget_tokens: None,
        });
    }
    // 2. An explicit per-agent `model` is honored within the active
    //    parent/session provider family. `SpawnMultiAgent` is on the default
    //    wire now, so model-authored per-agent overrides must not silently jump
    //    a sub-agent across providers — crossing requires the spawn input's
    //    `allow_cross_provider: true`, the explicit, transcript-visible escape
    //    hatch for when the USER asked for that model (the live incident this
    //    fixes: an explicit "opus" request was refused, and the session had no
    //    legitimate path left, so the ask silently ended as another model).
    if let Some(explicit) = non_empty(model) {
        let resolved = resolve_untrusted_spawn_model(&explicit, "explicit");
        if cross_provider_allowed || same_provider_family(parent_model, &resolved) {
            return Ok(AgentModelSelection {
                model: resolved,
                thinking_budget_tokens: None,
            });
        }
        if let Some(parent) = non_empty(parent_model) {
            return Err(AgentModelSelectionError::ExplicitCrossProvider {
                requested: explicit,
                parent,
            });
        }
    }
    // 3. A custom agent's `model:` frontmatter is honored, but only within the
    //    parent's provider family, so a custom Claude agent under a GPT session
    //    doesn't silently dial Anthropic (BUG-R17 — previously parsed but ignored).
    if let Some(custom) = non_empty(custom_model) {
        let resolved = resolve_untrusted_spawn_model(&custom, "custom");
        if same_provider_family(parent_model, &resolved) {
            return Ok(AgentModelSelection {
                model: resolved,
                thinking_budget_tokens: None,
            });
        }
        if let Some(parent) = non_empty(parent_model) {
            return Err(AgentModelSelectionError::CustomCrossProvider {
                requested: custom,
                parent,
            });
        }
    }
    // 4. Inherit the parent model verbatim (CC parity). The old automatic
    //    difficulty tiering re-routed within the provider family (e.g. a
    //    claude-fable-5 parent spawning opus/haiku sub-agents, a gpt-5.5-fast
    //    parent spawning gpt-5.5/spark), which broke the user's contract that
    //    sub-agents run on the model the session runs on (BUG 2026-06-11:
    //    "지금 부모모델로 sub-agent 해야하는데?"). Only the *reasoning budget*
    //    still scales with task difficulty — that tunes how hard the inherited
    //    model thinks, not which model runs.
    let difficulty = classify_agent_task_difficulty(subagent_type, description, prompt);
    if let Some(parent) = non_empty(parent_model) {
        // The difficulty-scaled reasoning budget is provider-neutral: it tunes
        // how hard the inherited model thinks, not which model runs, so it
        // applies to every parent family (Anthropic, GPT, Gemini, xAI, Ollama,
        // and custom OpenAI-compatible). The budget→effort derivation and each
        // backend's wire clamp live downstream in the single provider-client
        // seam (`provider_client.rs`: `effort_level_for_budget` +
        // `anthropic_for_model`/`gpt_for_model`/`gemini`), exactly as the
        // foreground turn does in `runtime_bridge::build_message_request` — so a
        // non-Anthropic/GPT sub-agent is no longer a second-class citizen that
        // silently runs at the provider default effort. `Fast` work still maps
        // to `None`, keeping quick agents free of extended-thinking overhead.
        return Ok(AgentModelSelection {
            model: parent,
            thinking_budget_tokens: thinking_budget_for_difficulty(difficulty),
        });
    }
    // 5. No parent (non-live harness): the process fallback.
    Ok(AgentModelSelection {
        model: resolve_agent_model(model, custom_model, subagent_type, parent_model),
        thinking_budget_tokens: None,
    })
}

fn resolve_untrusted_spawn_model(requested: &str, source: &str) -> String {
    let permissive = api::resolve_model_alias(requested);
    let registered = api::resolve_registered_model_alias(requested);
    if registered != permissive {
        eprintln!(
            "[zo] snapped unregistered {source} sub-agent model {requested:?} to registered {registered:?}"
        );
    }
    registered
}

/// Legacy/test helper that preserves the historical fallback behavior: a
/// cross-family explicit/custom model is ignored and the parent is inherited.
/// Live spawn paths must call [`try_resolve_agent_model_selection`] so a
/// user-visible `model` request never silently runs on another provider.
#[cfg(test)]
pub(super) fn resolve_agent_model_selection(
    model: Option<&str>,
    custom_model: Option<&str>,
    subagent_type: &str,
    parent_model: Option<&str>,
    description: &str,
    prompt: &str,
) -> AgentModelSelection {
    match try_resolve_agent_model_selection(
        model,
        false,
        custom_model,
        subagent_type,
        parent_model,
        description,
        prompt,
    ) {
        Ok(selection) => selection,
        Err(AgentModelSelectionError::ExplicitCrossProvider { .. }) => {
            try_resolve_agent_model_selection(
                None,
                false,
                custom_model,
                subagent_type,
                parent_model,
                description,
                prompt,
            )
            .expect("selection without explicit model cannot fail")
        }
        Err(AgentModelSelectionError::CustomCrossProvider { .. }) => {
            try_resolve_agent_model_selection(
                model,
                false,
                None,
                subagent_type,
                parent_model,
                description,
                prompt,
            )
            .expect("selection without custom model cannot fail")
        }
    }
}

/// Honor a TRUSTED Smart-route model for a spawned sub-agent. A global
/// `ZO_AGENT_MODEL` override still wins first because it is the user's
/// explicit all-agents override. Otherwise, unlike the untrusted on-wire
/// `model` path, the trusted route is NOT re-gated by [`same_provider_family`]:
/// it came from the host's config-driven Smart router and
/// [`route_model`](runtime::route_model) already constrained it to the connected
/// inventory, so a deliberate cross-provider route — the whole point of
/// `/smart` diversity for a Verifier/Reviewer/Judge role — is honored verbatim.
pub(super) fn smart_routed_model_selection(
    routed_model: &str,
    subagent_type: &str,
    description: &str,
    prompt: &str,
) -> AgentModelSelection {
    if let Some(env_model) = non_empty(std::env::var(AGENT_MODEL_ENV).ok().as_deref()) {
        return AgentModelSelection {
            model: env_model,
            thinking_budget_tokens: None,
        };
    }
    let difficulty = classify_agent_task_difficulty(subagent_type, description, prompt);
    AgentModelSelection {
        model: routed_model.to_string(),
        thinking_budget_tokens: thinking_budget_for_difficulty(difficulty),
    }
}

/// Reasoning budget for an inherited sub-agent model: quick work runs without
/// extended thinking; complex/hard work gets a high budget; deep research the
/// top budget. Provider-neutral — the caller applies it to every parent family,
/// and the provider-client seam maps the budget onto each backend's effort wire
/// (Anthropic `output_config.effort`, OpenAI `reasoning_effort`, Gemini
/// `thinkingLevel`), clamping to that model's ceiling.
const fn thinking_budget_for_difficulty(difficulty: AgentTaskDifficulty) -> Option<u32> {
    match difficulty {
        AgentTaskDifficulty::Fast => None,
        AgentTaskDifficulty::Standard => Some(MEDIUM_EFFORT_TOKENS),
        AgentTaskDifficulty::Complex | AgentTaskDifficulty::Hard => Some(HIGH_EFFORT_TOKENS),
        AgentTaskDifficulty::Deep => Some(XHIGH_EFFORT_TOKENS),
    }
}

/// Whether `parent_model` is a genuine OpenAI model. Consumed by
/// [`same_provider_family`] to gate a custom agent's `model:` frontmatter to the
/// parent's provider family (the difficulty-scaled reasoning budget itself is
/// now provider-neutral and no longer keys off this predicate).
///
/// Detection is purely **by model name** (the builtin GPT families plus the
/// `gpt`/`o3`/`o4`/`codex` prefixes), deliberately *not* [`api::detect_provider_kind`]:
/// that one falls back to ambient env (e.g. `OPENAI_API_KEY` present ⇒ every
/// unknown model id is "OpenAI"), which rerouted a **custom OpenAI-compatible
/// provider** parent (say `deepseek-chat`) to a hardcoded builtin GPT id,
/// ignoring the user's actual provider/model (BUG-R14). Name-based detection
/// still recognizes a real `gpt-*`/`o3` parent while leaving custom providers on
/// inheritance, and is deterministic (no env reads).
fn is_openai_parent_model(parent_model: Option<&str>) -> bool {
    let Some(model) = non_empty(parent_model) else {
        return false;
    };
    api::openai_gpt_model_family(&model).is_some()
        || matches!(
            api::explicit_non_claude_provider_kind(&model),
            Some(api::ProviderKind::OpenAi)
        )
}

/// Whether `parent_model` is an Anthropic Claude model — detected purely by name
/// (alias or canonical id), so the difficulty tiering never depends on ambient
/// env and never misfires for a custom provider (cf. [`is_openai_parent_model`]).
fn is_anthropic_parent_model(parent_model: Option<&str>) -> bool {
    let Some(model) = non_empty(parent_model) else {
        return false;
    };
    let lower = model.to_ascii_lowercase();
    lower == "opus"
        || lower == "sonnet"
        || lower == "haiku"
        || lower.starts_with("claude-")
        || api::resolve_model_alias(&lower)
            .to_ascii_lowercase()
            .starts_with("claude-")
}

/// Whether `candidate` belongs to the same provider family as `parent_model`.
/// Used to gate a per-agent `model` override or custom agent's `model:`
/// frontmatter so it only applies when it matches the active provider (a Claude
/// id under a GPT session is ignored). The detection is deterministic and
/// name-based; custom/Ollama ids with no unambiguous provider prefix are not
/// treated as same-family because zo cannot safely prove they share a backend.
fn same_provider_family(parent_model: Option<&str>, candidate: &str) -> bool {
    let Some(parent_kind) = model_provider_family(parent_model) else {
        return false;
    };
    let Some(candidate_kind) = model_provider_family(Some(candidate)) else {
        return false;
    };
    parent_kind == candidate_kind
}

fn model_provider_family(model: Option<&str>) -> Option<api::ProviderKind> {
    let model = non_empty(model)?;
    if is_anthropic_parent_model(Some(&model)) {
        return Some(api::ProviderKind::Anthropic);
    }
    if is_openai_parent_model(Some(&model)) {
        return Some(api::ProviderKind::OpenAi);
    }
    match api::explicit_non_claude_provider_kind(&model) {
        Some(api::ProviderKind::Google) => Some(api::ProviderKind::Google),
        Some(api::ProviderKind::Xai) => Some(api::ProviderKind::Xai),
        _ => None,
    }
}

/// One step down the capability/latency ladder for a model that is starving on
/// rate-limits, returned as a model *alias* (callers resolve aliases to
/// canonical ids downstream, exactly like the difficulty tiering below). `None`
/// when no lower tier exists — a bottom-tier model (haiku, a `-fast`/`flash`
/// variant) or a family with no catalog-verified downtier (xAI/Ollama/custom),
/// so the caller gives up honestly instead of switching to a nonexistent id.
///
/// Ladders (each a single step; the caller loops until `None`):
/// - Anthropic: opus → sonnet → haiku (the 2026-06-10 starvation incident).
/// - OpenAI GPT: legacy gpt-5.5(카탈로그 퇴역)는 terra로 세대 상향 이관 —
///   서빙 티어 보존(5.5→terra, 5.5-fast→terra[fast]). 이후는 5.6 사다리.
/// - GPT-5.6 (sol/terra/luna): capability-derived in-family downtier — see
///   [`gpt56_family_downtier`]. Sol/Terra step to Luna; Luna is the family's
///   floor and gives up honestly (no cross-generation fallback into
///   gpt-5.5-fast).
/// - Gemini: a `pro` tier → `gemini-3.5-flash`, which is itself the bottom.
///
/// All fallback targets are real catalog aliases (`api::providers` entries), so
/// a switched spawn always resolves to a spawnable model.
pub(super) fn starvation_demotion(model: &str) -> Option<&'static str> {
    let lower = model.to_ascii_lowercase();
    // Anthropic ladder (unchanged): opus → sonnet → haiku.
    if lower.contains("opus") {
        return Some("sonnet");
    }
    if lower.contains("sonnet") {
        return Some("haiku");
    }
    // GPT-5.6 family: capability-derived in-family downtier, checked before the
    // legacy gpt-5.5 literal match below so a dated/suffixed 5.6 id is never
    // mistaken for a 5.5 one.
    if let Some(demoted) = gpt56_family_downtier(&lower) {
        return Some(demoted);
    }
    // OpenAI GPT ladder: legacy gpt-5.5는 카탈로그에서 퇴역했으므로(사용자
    // 확정 2026-07-11, terra가 대체) 세대를 올려 이관한다 — **서빙 티어는
    // 보존**: 5.5 표준 → terra 표준, 5.5-fast(fast on) → terra[fast].
    // 티어를 하드코딩하지 않고 원래 모델의 fast 상태를 따른다(사용자 정책).
    // `[fast]` 브래킷 id는 model_id_matches_family/chatgpt_backend가 일급
    // 파싱한다.
    if lower == "gpt-5.5" || lower == "gpt-5.5-2026-04-23" {
        return Some("gpt-5.6-terra");
    }
    if lower == "gpt-5.5-fast" {
        return Some("gpt-5.6-terra[fast]");
    }
    // Gemini ladder: a `pro` tier → flash. A `flash` model is already the bottom.
    if lower.contains("gemini") && lower.contains("pro") {
        return Some("gemini-3.5-flash");
    }
    None
}

/// The current GPT-5.6 sibling ids this in-family downtier ladder covers. Any
/// rung, including one with no further downtier of its own (Luna), is a
/// member — [`suppress_cross_family_premium_fast_fallback`] reads this same
/// membership fact so the two concerns share one source of truth instead of a
/// second, independent hardcoded family check.
const GPT56_FAMILY: &[&str] = &["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"];

/// `true` when `model` is `prefix` itself, or `prefix` followed by a segment
/// boundary (`-`, `@`, `[`) — mirrors `api::types::model_id_matches_family`
/// (not reachable from `tools`, hence duplicated) so a dated/suffixed id
/// (`gpt-5.6-sol-2026-07-09`) matches its bare family while an id that merely
/// shares the family as a SUBSTRING elsewhere (`custom/gpt-5.6-sol`, which
/// does not itself *start with* the family) does not.
fn matches_gpt56_prefix(model: &str, prefix: &str) -> bool {
    model == prefix
        || model
            .strip_prefix(prefix)
            .is_some_and(|suffix| matches!(suffix.as_bytes().first(), Some(b'-' | b'@' | b'[')))
}

/// Whether `model` (already lowercased) is a member of [`GPT56_FAMILY`].
pub(super) fn is_gpt56_family_member(lower_model: &str) -> bool {
    GPT56_FAMILY.iter().any(|id| matches_gpt56_prefix(lower_model, id))
}

/// Capability-derived in-family downtier for the GPT-5.6 trio: the sibling
/// with the highest `api::max_supported_effort` ceiling that still ranks
/// STRICTLY below `model`'s own ceiling. Sol/Terra (`Ultra`) step down to Luna
/// (`Max`); Luna has no lower-ceiling sibling, so it returns `None` (honest
/// give-up) rather than falling back to a different GPT generation. Reads the
/// SAME ceiling SSOT `model_inventory::tiers_for_model` uses for Deep-tier
/// promotion, so a newly announced higher-ceiling sibling demotes correctly
/// once added to [`GPT56_FAMILY`] — no new pairwise match arm required.
fn gpt56_family_downtier(lower_model: &str) -> Option<&'static str> {
    let current = GPT56_FAMILY.iter().find(|id| matches_gpt56_prefix(lower_model, id))?;
    let current_rank = effort_ceiling_rank(api::max_supported_effort(current));
    GPT56_FAMILY
        .iter()
        .filter(|candidate| *candidate != current)
        .filter(|candidate| effort_ceiling_rank(api::max_supported_effort(candidate)) < current_rank)
        .max_by_key(|candidate| effort_ceiling_rank(api::max_supported_effort(candidate)))
        .copied()
}

/// Ranks `api::EffortLevel` for ceiling comparison. `api::EffortLevel`
/// intentionally has no `Ord` (it is a provider-neutral wire enum, not a
/// general-purpose ranking type), so this mirrors
/// `session::runtime_bridge::effort_rank`'s tier-max ordering — duplicated
/// here rather than shared cross-crate since `tools` cannot depend on the CLI
/// crate.
const fn effort_ceiling_rank(level: api::EffortLevel) -> u8 {
    match level {
        api::EffortLevel::Low => 0,
        api::EffortLevel::Medium => 1,
        api::EffortLevel::High => 2,
        api::EffortLevel::Xhigh => 3,
        api::EffortLevel::Max => 4,
        api::EffortLevel::Ultra => 5,
    }
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentTaskDifficulty {
    Fast,
    Standard,
    Complex,
    Hard,
    Deep,
}

fn classify_agent_task_difficulty(
    subagent_type: &str,
    description: &str,
    prompt: &str,
) -> AgentTaskDifficulty {
    let haystack = format!("{subagent_type}\n{description}\n{prompt}").to_ascii_lowercase();
    let has_any = |needles: &[&str]| needles.iter().any(|needle| haystack.contains(needle));

    if matches!(subagent_type, "deep-research") || has_any(&["ultracode", "매우", "깊게"]) {
        return AgentTaskDifficulty::Deep;
    }
    if matches!(subagent_type, "debugger" | "refactor" | "code-reviewer")
        || has_any(&[
            "debug",
            "root cause",
            "root-cause",
            "refactor",
            "race condition",
            "security",
            "correctness",
            "review",
            "복잡",
            "어려",
            "분석",
            "설계",
            "리팩",
            "디버",
            "감사",
            "최적",
        ])
    {
        return AgentTaskDifficulty::Hard;
    }
    if matches!(subagent_type, "Plan")
        || has_any(&["architecture", "architect", "design", "proposal", "설계"])
    {
        return AgentTaskDifficulty::Complex;
    }
    if matches!(subagent_type, "data-analyst")
        || has_any(&[
            "analyze", "analyse", "metrics", "dataset", "logs", "통계", "로그",
        ])
    {
        return AgentTaskDifficulty::Standard;
    }
    AgentTaskDifficulty::Fast
}

#[cfg(test)]
mod tests {
    use super::{
        AGENT_MODEL_ENV, HIGH_EFFORT_TOKENS, XHIGH_EFFORT_TOKENS, build_agent_system_prompt,
        is_openai_parent_model, resolve_agent_model_selection, smart_routed_model_selection,
        starvation_demotion,
    };

    #[test]
    fn classifier_harness_is_a_micro_prompt_with_identity_first() {
        // Fan-out triage / decomposition runs on a cheap model outside the
        // parent's cache namespace: the full system prompt (~4.4k tok) would be
        // re-billed there for a one-sentence JSON answer. The identity line
        // must stay first — the Claude Max OAuth path 429s on any other first
        // system block.
        let prompt = build_agent_system_prompt("classifier", None).expect("classifier prompt");
        assert_eq!(prompt[0], runtime::CLAUDE_CODE_IDENTITY);
        let total_chars: usize = prompt.iter().map(String::len).sum();
        assert!(
            total_chars < 1_000,
            "classifier prompt must stay micro, got {total_chars} chars"
        );
        assert!(
            !prompt.iter().any(|s| s.contains("# Environment context")),
            "no full-prompt sections in the classifier harness"
        );

        // The classifier needs no tools beyond the forced StructuredOutput.
        let allowed = crate::misc_tools::allowed_tools_for_subagent("classifier");
        assert_eq!(
            allowed.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["StructuredOutput"]
        );
    }

    #[test]
    fn builtin_gpt_parents_are_recognized() {
        // Used to gate the reasoning-budget assignment and the custom-agent
        // frontmatter family check for genuine builtin GPT-family parents.
        assert!(is_openai_parent_model(Some("gpt-5.5")));
        assert!(is_openai_parent_model(Some("gpt-5.6-luna")));
        assert!(is_openai_parent_model(Some("gpt-5.3-codex-spark")));
        assert!(is_openai_parent_model(Some("gpt-5.5-2026-04-23")));
        assert!(is_openai_parent_model(Some("gpt-5.5-fast")));
    }

    /// CC parity (2026-06-11): a sub-agent runs on the parent's model verbatim.
    /// The old difficulty tiering re-routed within the family (fable → opus,
    /// gpt-5.5-fast → spark/5.5) and broke the inheritance contract; only the
    /// reasoning budget may scale with task difficulty now.
    #[test]
    fn subagents_inherit_the_parent_model_verbatim() {
        if std::env::var("ZO_AGENT_MODEL").is_ok() {
            return; // env override active in this environment — contract not observable
        }
        // Anthropic parent: same model for easy and hard work alike.
        let easy = resolve_agent_model_selection(
            None,
            None,
            "Explore",
            Some("claude-fable-5"),
            "list files",
            "list files quickly",
        );
        assert_eq!(easy.model, "claude-fable-5");
        assert!(easy.thinking_budget_tokens.is_none());
        let hard = resolve_agent_model_selection(
            None,
            None,
            "debugger",
            Some("claude-fable-5"),
            "debug the race condition",
            "find the root cause",
        );
        assert_eq!(hard.model, "claude-fable-5");
        assert_eq!(hard.thinking_budget_tokens, Some(HIGH_EFFORT_TOKENS));
        // OpenAI parent: inherits verbatim too (no Spark/mini/5.5 re-routing).
        let gpt = resolve_agent_model_selection(
            None,
            None,
            "debugger",
            Some("gpt-5.5-fast"),
            "debug the race condition",
            "find the root cause",
        );
        assert_eq!(gpt.model, "gpt-5.5-fast");
        assert_eq!(gpt.thinking_budget_tokens, Some(HIGH_EFFORT_TOKENS));
    }

    /// A TRUSTED Smart-route model is applied VERBATIM — a deliberate
    /// cross-provider diversity route included — because it came from the host's
    /// config-driven router (already gated to the connected inventory), unlike an
    /// on-wire user `model` which stays fenced to the parent's provider family.
    #[test]
    fn smart_routed_model_is_applied_verbatim_across_provider_families() {
        if std::env::var(AGENT_MODEL_ENV).is_ok() {
            return; // env override intentionally wins over Smart routes
        }
        // Cross-provider route under an Anthropic-flavored context: honored as-is,
        // NOT re-gated to the parent family (the whole point of /smart Verifier/
        // Reviewer diversity), with a difficulty-scaled, provider-neutral budget.
        let routed = smart_routed_model_selection(
            "gpt-5.5-fast",
            "debugger",
            "debug the race condition",
            "find the root cause",
        );
        assert_eq!(routed.model, "gpt-5.5-fast");
        assert_eq!(routed.thinking_budget_tokens, Some(HIGH_EFFORT_TOKENS));

        // Contrast: the SAME cross-family model supplied as an on-wire user
        // `model` under a Claude parent is fenced out and inherits the parent —
        // proving the routed path deliberately bypasses `same_provider_family`.
        let gated = resolve_agent_model_selection(
            Some("gpt-5.5-fast"),
            None,
            "debugger",
            Some("claude-sonnet"),
            "debug the race condition",
            "find the root cause",
        );
        assert_eq!(
            gated.model, "claude-sonnet",
            "an on-wire cross-family user model is fenced to the parent, unlike a route"
        );

        // The difficulty budget is provider-neutral: a trivial routed task runs
        // without extended thinking overhead.
        let trivial =
            smart_routed_model_selection("gpt-5.5-fast", "Explore", "list files", "list files quickly");
        assert_eq!(trivial.model, "gpt-5.5-fast");
        assert!(trivial.thinking_budget_tokens.is_none());
    }

    #[test]
    fn custom_openai_compatible_provider_is_not_rerouted_to_builtin_openai() {
        // A custom OpenAI-compatible provider's model is not a builtin GPT
        // family, so the predicate must be false regardless of ambient
        // OPENAI_API_KEY (BUG-R14: the old `detect_provider_kind` env fallback
        // rerouted it to a hardcoded builtin GPT family).
        assert!(!is_openai_parent_model(Some("deepseek-chat")));
        assert!(!is_openai_parent_model(Some("llama-3.3-70b-instruct")));
        assert!(!is_openai_parent_model(Some("my-self-hosted-model")));
        assert!(!is_openai_parent_model(None));

        // …and the sub-agent therefore inherits the user's actual model instead
        // of a builtin GPT id (env-key-free path: no ZO_AGENT_MODEL set).
        if std::env::var("ZO_AGENT_MODEL").is_err() {
            let selection = resolve_agent_model_selection(
                None,
                None,
                "general-purpose",
                Some("deepseek-chat"),
                "summarize the module",
                "summarize the module",
            );
            assert_eq!(selection.model, "deepseek-chat");
            assert!(selection.thinking_budget_tokens.is_none());
        }
    }

    /// fallback 사다리: Anthropic opus→sonnet→haiku (별칭/정식 id 모두), 그리고
    /// Gap C로 추가된 GPT(legacy gpt-5.5 non-fast → gpt-5.5-fast)·Gemini(pro → flash)
    /// 한 단계. GPT-5.6은 capability-derived in-family downtier: sol/terra→luna,
    /// luna는 바닥(family floor)이라 None. 바닥 티어(haiku/fast/flash)·카탈로그
    /// 미검증 패밀리(xAI/custom)는 None으로 정직하게 포기한다. 모든 fallback
    /// 대상은 실재 카탈로그 별칭.
    #[test]
    fn starvation_demotion_walks_each_provider_ladder() {
        // Anthropic (unchanged).
        assert_eq!(starvation_demotion("opus"), Some("sonnet"));
        assert_eq!(starvation_demotion("claude-opus-4-8"), Some("sonnet"));
        assert_eq!(starvation_demotion("claude-sonnet-4-6"), Some("haiku"));
        assert_eq!(starvation_demotion("haiku"), None);
        assert_eq!(starvation_demotion("claude-haiku-4-5-20251001"), None);
        // OpenAI GPT: legacy gpt-5.5(카탈로그 퇴역)는 terra로 세대 상향 이관,
        // 서빙 티어는 보존(표준→표준, fast→[fast]) — 티어 하드코딩 없음.
        assert_eq!(starvation_demotion("gpt-5.5"), Some("gpt-5.6-terra"));
        assert_eq!(
            starvation_demotion("gpt-5.5-2026-04-23"),
            Some("gpt-5.6-terra")
        );
        assert_eq!(
            starvation_demotion("gpt-5.5-fast"),
            Some("gpt-5.6-terra[fast]")
        );
        // GPT-5.6: capability-derived in-family downtier — sol/terra (Ultra
        // ceiling) step down to luna (Max ceiling); luna is the family's floor
        // (no cross-generation fallback into gpt-5.5-fast).
        assert_eq!(starvation_demotion("gpt-5.6-sol"), Some("gpt-5.6-luna"));
        assert_eq!(starvation_demotion("gpt-5.6-terra"), Some("gpt-5.6-luna"));
        assert_eq!(starvation_demotion("gpt-5.6-luna"), None);
        // A future/unregistered GPT sibling this ladder does not know about is
        // an honest give-up, not a guess.
        assert_eq!(
            starvation_demotion("gpt-5.7-nova"),
            None,
            "no catalog-verified downtier for an unrecognized GPT sibling"
        );
        // Gemini: a pro tier demotes to flash; flash is the bottom.
        assert_eq!(
            starvation_demotion("gemini-3.1-pro-preview"),
            Some("gemini-3.5-flash")
        );
        assert_eq!(starvation_demotion("gemini-pro"), Some("gemini-3.5-flash"));
        assert_eq!(starvation_demotion("gemini-3.5-flash"), None);
        assert_eq!(starvation_demotion("gemini-3-flash"), None);
        // No catalog-verified downtier → honest give-up (unchanged).
        assert_eq!(starvation_demotion("grok-3"), None);
        assert_eq!(starvation_demotion("deepseek-chat"), None);
        assert_eq!(starvation_demotion("ollama"), None);
    }

    /// Gap A (CC ultracode parity): the difficulty-scaled reasoning budget is
    /// provider-neutral. A Gemini, xAI/Ollama, or custom OpenAI-compatible
    /// parent — none of which `is_openai_parent_model`/`is_anthropic_parent_model`
    /// match — must still get the same hard-task budget an Anthropic/GPT parent
    /// gets, instead of silently running at the provider default effort. The
    /// model is still inherited verbatim; only the budget is asserted here.
    #[test]
    fn non_anthropic_non_gpt_parents_get_difficulty_scaled_budget() {
        if std::env::var("ZO_AGENT_MODEL").is_ok() {
            return; // env override active — inheritance/budget contract not observable
        }
        // (a) Gemini parent + hard (`debugger`) task → High budget, model verbatim.
        let gemini = resolve_agent_model_selection(
            None,
            None,
            "debugger",
            Some("gemini-3-pro"),
            "debug the race condition",
            "find the root cause",
        );
        assert_eq!(gemini.model, "gemini-3-pro");
        assert_eq!(gemini.thinking_budget_tokens, Some(HIGH_EFFORT_TOKENS));
        // (b) Custom OpenAI-compatible parent + deep-research task → Xhigh budget.
        let custom = resolve_agent_model_selection(
            None,
            None,
            "deep-research",
            Some("deepseek-chat"),
            "thoroughly research the failure modes",
            "survey the literature and synthesize",
        );
        assert_eq!(custom.model, "deepseek-chat");
        assert_eq!(custom.thinking_budget_tokens, Some(XHIGH_EFFORT_TOKENS));
        // (c) Same families on a trivial Fast task → no extended-thinking overhead.
        let gemini_fast = resolve_agent_model_selection(
            None,
            None,
            "general-purpose",
            Some("gemini-3-flash"),
            "list files",
            "list files quickly",
        );
        assert_eq!(gemini_fast.model, "gemini-3-flash");
        assert!(gemini_fast.thinking_budget_tokens.is_none());
        let custom_fast = resolve_agent_model_selection(
            None,
            None,
            "general-purpose",
            Some("llama-3.3-70b-instruct"),
            "list files",
            "list files quickly",
        );
        assert_eq!(custom_fast.model, "llama-3.3-70b-instruct");
        assert!(custom_fast.thinking_budget_tokens.is_none());
    }
}
