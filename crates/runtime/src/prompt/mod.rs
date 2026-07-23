use std::fmt::Write as _;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::{ConfigError, ConfigLoader, RuntimeConfig};
use crate::git_snapshot::read_git_root;

pub mod output_style;
mod sections;

use sections::{
    get_actions_section, get_clarification_section, get_context_trust_label_section,
    get_default_coding_harness_section, get_delegation_section, get_grounding_in_code_section,
    get_memory_protocol_section, get_responding_section, get_response_style_contract_section,
    get_simple_doing_tasks_section, get_simple_intro_section, get_simple_system_section,
    get_skills_and_docs_section, get_turn_discipline_section, render_config_section,
    render_memory_index,
};

/// Which interaction surface the prompt is built for. The behavioral contract
/// is identical across models; only the turn-ending discipline varies —
/// interactive sessions may hand a genuine decision back to the user, an
/// autonomous surface (headless one-shot) must not ask mid-task questions,
/// and sub-agents carry their own completion contract in the agent profile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PromptMode {
    /// A human is present at the terminal (REPL/TUI sessions).
    #[default]
    Interactive,
    /// Nobody can answer mid-task (headless `-p` one-shots, CI runs).
    Autonomous,
    /// Delegated sub-agents and informational surfaces.
    Subagent,
}

/// Errors raised while assembling the final system prompt.
#[derive(Debug)]
pub enum PromptBuildError {
    Io(std::io::Error),
    Config(ConfigError),
}

impl std::fmt::Display for PromptBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Config(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PromptBuildError {}

impl From<std::io::Error> for PromptBuildError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ConfigError> for PromptBuildError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

/// Marker separating static prompt scaffolding from dynamic runtime context.
pub const SYSTEM_PROMPT_DYNAMIC_BOUNDARY: &str = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__";
/// Human-readable default frontier model name embedded into generated prompts.
pub const FRONTIER_MODEL_NAME: &str = "Claude Opus 4.8";
const MAX_INSTRUCTION_FILE_CHARS: usize = 4_000;
const MAX_TOTAL_INSTRUCTION_CHARS: usize = 12_000;
const MAX_SKILL_INDEX_ENTRIES: usize = 32;
/// Cap on the `git status` snapshot embedded in the system prompt. A large
/// monorepo, or a branch with thousands of untracked/modified paths, can emit a
/// `git status --short` of tens of thousands of lines. Embedding it verbatim on
/// **every** turn balloons the request — worst case a near-1M-token prefill that
/// the provider rejects with `overloaded_error` (the "다른 폴더에서 hi만 쳐도
/// Overloaded" report). The model can always run `git status` itself for the
/// full list, so a bounded preview is sufficient.
const MAX_GIT_STATUS_LINES: usize = 80;
const MAX_GIT_STATUS_CHARS: usize = 4_000;

/// Contents of an instruction file included in prompt construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
    pub globs: Vec<String>,
    pub always_apply: bool,
}

impl ContextFile {
    fn new(path: PathBuf, content: String) -> Self {
        Self {
            path,
            content,
            globs: Vec::new(),
            always_apply: true,
        }
    }

    fn instruction(path: PathBuf, content: &str) -> Self {
        let (frontmatter, body) = parse_frontmatter(content).map_or_else(
            || (Vec::new(), content.trim().to_string()),
            |(fields, body)| (fields, body.trim().to_string()),
        );
        let globs = parse_globs(&frontmatter);
        let always_apply = parse_always_apply(&frontmatter).unwrap_or(globs.is_empty());
        Self {
            path,
            content: body,
            globs,
            always_apply,
        }
    }

    fn is_always_instruction(&self) -> bool {
        self.always_apply || self.globs.is_empty()
    }
}

/// How a skill may be invoked when its triggers match a turn. Mirrors the
/// Claude Code "control who invokes a skill" idea: `Manual` is recommended only
/// on an explicit mention, `Suggest` (default) is nudged whenever triggers
/// match, and `Auto` is the strongest nudge. The recommendation is always
/// advisory — it never force-loads a skill body — so even `Auto` only asks the
/// model to call the `Skill` tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkillInvocationMode {
    /// Recommend only when the user names the skill explicitly.
    Manual,
    /// Recommend whenever triggers match (default).
    #[default]
    Suggest,
    /// Strongest recommendation when triggers match.
    Auto,
}

impl SkillInvocationMode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "manual" => Some(Self::Manual),
            "suggest" => Some(Self::Suggest),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

/// Optional Codex-style implicit-invocation metadata parsed from a skill's
/// frontmatter. All fields default to empty, so a skill without trigger
/// metadata is simply never auto-recommended (it stays listed in the prompt's
/// `# Available skills` index exactly as before).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillTriggers {
    pub keywords: Vec<String>,
    pub paths: Vec<String>,
    pub excludes: Vec<String>,
}

impl SkillTriggers {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keywords.is_empty() && self.paths.is_empty() && self.excludes.is_empty()
    }
}

/// Compact metadata for a project-local skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillIndexEntry {
    pub name: String,
    pub description: Option<String>,
    pub path: PathBuf,
    /// Implicit-invocation mode (default [`SkillInvocationMode::Suggest`]).
    pub invocation_mode: SkillInvocationMode,
    /// Trigger metadata for keyword/path matching. Empty for legacy skills.
    pub triggers: SkillTriggers,
}

impl SkillIndexEntry {
    /// Convenience constructor for callers (and tests) that only care about the
    /// prompt-index fields; trigger/invocation metadata default to empty.
    #[must_use]
    pub fn new(name: String, description: Option<String>, path: PathBuf) -> Self {
        Self {
            name,
            description,
            path,
            invocation_mode: SkillInvocationMode::default(),
            triggers: SkillTriggers::default(),
        }
    }
}

/// Project-local context injected into the rendered system prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectContext {
    pub cwd: PathBuf,
    pub project_root: Option<PathBuf>,
    pub current_date: String,
    pub git_status: Option<String>,
    pub git_diff: Option<String>,
    pub instruction_files: Vec<ContextFile>,
    /// Global per-project `memory/MEMORY.md` index for `cwd`, if any.
    /// Only the compact index is carried into the prompt — full entries stay
    /// on disk and the model reads them on demand, keeping token cost low.
    pub memory_index: Option<ContextFile>,
    /// Zo skills discovered from this repo's `.zo/skills/` plus Zo's
    /// global skill directory.
    /// Only frontmatter metadata is carried; the skill body is loaded on demand.
    pub skills_index: Vec<SkillIndexEntry>,
}

impl ProjectContext {
    pub fn discover(
        cwd: impl Into<PathBuf>,
        current_date: impl Into<String>,
    ) -> std::io::Result<Self> {
        let cwd = cwd.into();
        let instruction_files = discover_instruction_files(&cwd)?;
        let memory_index = discover_memory_index(&cwd);
        let skills_index = discover_skills_index(&cwd);
        Ok(Self {
            cwd,
            project_root: None,
            current_date: current_date.into(),
            git_status: None,
            git_diff: None,
            instruction_files,
            memory_index,
            skills_index,
        })
    }

    pub fn discover_with_git(
        cwd: impl Into<PathBuf>,
        current_date: impl Into<String>,
    ) -> std::io::Result<Self> {
        let mut context = Self::discover(cwd, current_date)?;
        // Each git fact is a separate `git` subprocess whose fork+exec latency
        // dominates, so gather the independent reads concurrently — this collapses
        // startup's largest synchronous cost from N spawn latencies to ~1. The
        // full working-tree diff is intentionally NOT gathered: it never reaches
        // the system prompt (`render_project_context` omits it; the model runs
        // `git diff` on demand), so spawning it would be pure startup tax.
        let git_cwd = context.cwd.clone();
        let (root, status) = std::thread::scope(|scope| {
            let root = scope.spawn(|| read_git_root(&git_cwd));
            let status = scope.spawn(|| read_git_status(&git_cwd));
            (
                root.join().ok().flatten(),
                status.join().ok().flatten(),
            )
        });
        context.project_root = root;
        context.git_status = status;
        Ok(context)
    }
}

/// Builder for the runtime system prompt and dynamic environment sections.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemPromptBuilder {
    output_style_name: Option<String>,
    output_style_prompt: Option<String>,
    os_name: Option<String>,
    os_version: Option<String>,
    append_sections: Vec<String>,
    project_context: Option<ProjectContext>,
    config: Option<RuntimeConfig>,
    mode: PromptMode,
}

impl SystemPromptBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_output_style(mut self, name: impl Into<String>, prompt: impl Into<String>) -> Self {
        self.output_style_name = Some(name.into());
        self.output_style_prompt = Some(prompt.into());
        self
    }

    #[must_use]
    pub fn with_os(mut self, os_name: impl Into<String>, os_version: impl Into<String>) -> Self {
        self.os_name = Some(os_name.into());
        self.os_version = Some(os_version.into());
        self
    }

    #[must_use]
    pub fn with_project_context(mut self, project_context: ProjectContext) -> Self {
        self.project_context = Some(project_context);
        self
    }

    #[must_use]
    pub fn with_runtime_config(mut self, config: RuntimeConfig) -> Self {
        self.config = Some(config);
        self
    }

    #[must_use]
    pub fn append_section(mut self, section: impl Into<String>) -> Self {
        self.append_sections.push(section.into());
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: PromptMode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub fn build(&self) -> Vec<String> {
        let mut sections = Vec::new();
        sections.push(get_simple_intro_section(self.output_style_name.is_some()));
        sections.push(get_response_style_contract_section());
        if let (Some(name), Some(prompt)) = (&self.output_style_name, &self.output_style_prompt) {
            sections.push(format!("# Output Style: {name}\n{prompt}"));
        }
        sections.push(get_simple_system_section());
        sections.push(get_responding_section());
        sections.push(get_clarification_section());
        sections.push(get_default_coding_harness_section());
        sections.push(get_grounding_in_code_section());
        sections.push(get_simple_doing_tasks_section());
        sections.push(get_delegation_section());
        sections.push(get_actions_section());
        if let Some(discipline) = get_turn_discipline_section(self.mode) {
            sections.push(discipline);
        }
        sections.push(get_skills_and_docs_section());
        sections.push(get_memory_protocol_section());
        sections.push(get_context_trust_label_section());
        sections.push(SYSTEM_PROMPT_DYNAMIC_BOUNDARY.to_string());
        sections.push(self.environment_section());
        if let Some(project_context) = &self.project_context {
            sections.push(render_project_context(project_context));
            let always_instruction_files = project_context
                .instruction_files
                .iter()
                .filter(|file| file.is_always_instruction())
                .cloned()
                .collect::<Vec<_>>();
            let scoped_instruction_files = project_context
                .instruction_files
                .iter()
                .filter(|file| !file.is_always_instruction())
                .cloned()
                .collect::<Vec<_>>();
            if !always_instruction_files.is_empty() {
                sections.push(render_instruction_files(&always_instruction_files));
            }
            if !scoped_instruction_files.is_empty() {
                sections.push(render_scoped_instruction_index(&scoped_instruction_files));
            }
            if !project_context.skills_index.is_empty() {
                sections.push(render_skills_index(&project_context.skills_index));
            }
            if let Some(memory) = &project_context.memory_index {
                sections.push(render_memory_index(memory));
            }
        }
        if let Some(config) = &self.config {
            sections.push(render_config_section(config));
        }
        sections.extend(self.append_sections.iter().cloned());
        sections
    }

    #[must_use]
    pub fn render(&self) -> String {
        self.build().join("\n\n")
    }

    fn environment_section(&self) -> String {
        let cwd = self.project_context.as_ref().map_or_else(
            || "unknown".to_string(),
            |context| context.cwd.display().to_string(),
        );
        let date = self.project_context.as_ref().map_or_else(
            || "unknown".to_string(),
            |context| context.current_date.clone(),
        );
        let mut lines = vec!["# Environment context".to_string()];
        lines.extend(prepend_bullets(vec![
            format!("Model family: {FRONTIER_MODEL_NAME}"),
            format!("Working directory: {cwd}"),
            format!("Date: {date}"),
            format!(
                "Platform: {} {}",
                self.os_name.as_deref().unwrap_or("unknown"),
                self.os_version.as_deref().unwrap_or("unknown")
            ),
        ]));
        lines.join("\n")
    }
}

/// Formats each item as an indented bullet for prompt sections.
#[must_use]
pub fn prepend_bullets(items: Vec<String>) -> Vec<String> {
    items.into_iter().map(|item| format!(" - {item}")).collect()
}

/// The Claude Code identity line. On the Anthropic OAuth (Claude Max) path the
/// `system` array's FIRST text block must be exactly this string — any other
/// first block is rejected as a client-fingerprint mismatch, which the API
/// surfaces as a 429 rate-limit error.
pub const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Lower a joined system prompt into wire-level [`api::SystemBlock`]s with the
/// Claude Code identity isolated and cache breakpoints placed for maximal reuse.
///
/// The prompt is cut into up to three blocks:
/// 1. the Claude Code identity line — kept verbatim as the first block with
///    **no** `cache_control` (Claude Max OAuth fingerprint requirement, see
///    [`CLAUDE_CODE_IDENTITY`]);
/// 2. the static scaffolding (everything up to [`SYSTEM_PROMPT_DYNAMIC_BOUNDARY`])
///    as its own 1h-cached block, identical across turns and sessions;
/// 3. the dynamic context after the boundary — stable within a session.
///
/// The boundary marker itself is removed so it never reaches the model.
///
/// This is the single source of truth for *every* Anthropic-bound request —
/// the foreground TUI/headless turn and background sub-agents alike. The
/// sub-agent path previously sent the whole prompt as one plain block
/// (identity not isolated, no cache breakpoints), which the OAuth path
/// rejects as a fingerprint mismatch: the zo-only agent 429s.
#[must_use]
pub fn split_system_with_identity(system_text: &str) -> Vec<api::SystemBlock> {
    let mut blocks = Vec::with_capacity(3);
    let body = if let Some(rest) = system_text.strip_prefix(CLAUDE_CODE_IDENTITY) {
        blocks.push(api::SystemBlock::Text {
            text: CLAUDE_CODE_IDENTITY.to_string(),
            cache_control: None,
        });
        rest
    } else {
        system_text
    };
    match body.split_once(SYSTEM_PROMPT_DYNAMIC_BOUNDARY) {
        Some((static_part, dynamic_part)) => {
            push_cache_block(&mut blocks, static_part);
            push_cache_block(&mut blocks, dynamic_part);
        }
        None => push_cache_block(&mut blocks, body),
    }
    blocks
}

/// Push `text` (trimmed of surrounding blank lines) as a 1h-cached system
/// block, skipping it when empty.
fn push_cache_block(blocks: &mut Vec<api::SystemBlock>, text: &str) {
    let text = text.trim_matches('\n');
    if !text.is_empty() {
        blocks.push(api::SystemBlock::Text {
            text: text.to_string(),
            cache_control: Some(api::CacheControl::ephemeral_1h()),
        });
    }
}

fn discover_instruction_files(cwd: &Path) -> std::io::Result<Vec<ContextFile>> {
    let mut directories = Vec::new();
    let mut cursor = Some(cwd);
    while let Some(dir) = cursor {
        directories.push(dir.to_path_buf());
        cursor = dir.parent();
    }
    directories.reverse();

    let mut files = Vec::new();
    for dir in directories {
        for candidate in [
            dir.join("context.md"),
            dir.join("CONTEXT.md"),
            dir.join("context.local.md"),
            dir.join("CONTEXT.local.md"),
            dir.join(".zo").join("context.md"),
            dir.join(".zo").join("CONTEXT.md"),
        ] {
            push_context_file(&mut files, candidate)?;
        }
    }
    Ok(dedupe_instruction_files(files))
}

/// Find the persistent-memory index for `cwd` in Zo's global per-project
/// store (`memory/MEMORY.md`) and then the machine-local overlay
/// (`memory.local/MEMORY.md`). Read-only and best-effort: any IO error or
/// absent file yields `None` so a missing memory store never blocks prompt
/// assembly. Only the index is loaded as a locator; query-aware merged recall of
/// the global stores happens in `memory::recall`, and individual entries stay on
/// disk for on-demand reads, keeping the per-turn token cost to the small index.
fn discover_memory_index(cwd: &Path) -> Option<ContextFile> {
    for path in crate::memory::paths::memory_index_candidates(cwd) {
        if let Ok(content) = fs::read_to_string(&path) {
            if !content.trim().is_empty() {
                return Some(ContextFile::new(path, content));
            }
        }
    }
    None
}

fn discover_skills_index(cwd: &Path) -> Vec<SkillIndexEntry> {
    discover_skills_index_inner(cwd)
}

/// Discover the active project + global skills for `cwd` — the same set the
/// system prompt's `# Available skills` index lists (proposed skills excluded).
/// Public so the per-turn skill router can re-scan for implicit invocation
/// without rebuilding the whole prompt; re-scanning each turn matches Claude
/// Code/Codex live change detection.
#[must_use]
pub fn discover_skills(cwd: &Path) -> Vec<SkillIndexEntry> {
    discover_skills_index_inner(cwd)
}

fn discover_skills_index_inner(cwd: &Path) -> Vec<SkillIndexEntry> {
    let mut entries = Vec::new();
    for skills_dir in skill_search_roots(cwd) {
        push_skill_entries_from_dir(&mut entries, &skills_dir);
        if entries.len() >= MAX_SKILL_INDEX_ENTRIES {
            entries.truncate(MAX_SKILL_INDEX_ENTRIES);
            return entries;
        }
    }
    entries
}

/// Every directory skills are discovered from for `cwd`, in precedence order:
/// walk-up `.zo/skills` → the Zo global homes. Only Zo roots are read; skills
/// from another tool must be placed (or symlinked) under a Zo root. The single source of
/// truth shared by the prompt index, the per-turn skill router, and the
/// `Skill` tool's loader — the loader once kept its own copy of this walk,
/// drifted, and answered "unknown skill" for a skill the index had just
/// advertised.
#[must_use]
pub fn skill_search_roots(cwd: &Path) -> Vec<PathBuf> {
    let project_root = read_git_root(cwd);
    let mut directories = Vec::new();
    let mut cursor = Some(cwd);
    while let Some(dir) = cursor {
        directories.push(dir.to_path_buf());
        if project_root.as_ref().is_some_and(|root| dir == root) {
            break;
        }
        cursor = dir.parent();
    }
    if let Some(project_root) = project_root {
        if !directories.iter().any(|dir| dir == &project_root) {
            directories.push(project_root);
        }
    }

    let mut skill_roots = Vec::new();
    for dir in &directories {
        push_unique_path(&mut skill_roots, dir.join(".zo").join("skills"));
    }
    for dir in zo_global_skill_roots() {
        push_unique_path(&mut skill_roots, dir);
    }
    skill_roots
}

fn zo_global_skill_roots() -> Vec<PathBuf> {
    // Resolve the global skill homes through the single source of truth so the
    // lookup order (`ZO_CONFIG_HOME` → `ZO_HOME` → `~/.zo`) stays
    // identical to sessions, agents, and MCP discovery.
    crate::config::zo_global_config_roots()
        .into_iter()
        .map(|root| root.join("skills"))
        .collect()
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn push_skill_entries_from_dir(entries: &mut Vec<SkillIndexEntry>, skills_dir: &Path) {
    let Ok(children) = fs::read_dir(skills_dir) else {
        return;
    };

    let mut skill_files = children
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("SKILL.md"))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    skill_files.sort();

    for path in skill_files {
        if entries.len() >= MAX_SKILL_INDEX_ENTRIES {
            break;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        if let Some(entry) = parse_skill_index_entry(path, &content) {
            push_unique_skill_entry(entries, entry);
        }
    }
}

fn parse_skill_index_entry(path: PathBuf, content: &str) -> Option<SkillIndexEntry> {
    let mut name = None;
    let mut description = None;
    let mut state = None;
    let mut invocation_mode = SkillInvocationMode::default();
    let mut triggers = SkillTriggers::default();

    let fields = parse_frontmatter_fields(content);
    for (key, value) in &fields {
        match key.as_str() {
            "name" => name = Some(value.clone()),
            "description" => description = Some(value.clone()),
            "state" => state = Some(value.clone()),
            "invocation" | "invocation_mode" | "invocationMode" => {
                if let Some(mode) = SkillInvocationMode::parse(value) {
                    invocation_mode = mode;
                }
            }
            "keyword" | "keywords" | "triggers.keywords" | "trigger_keywords" => {
                push_trigger_values(&mut triggers.keywords, value);
            }
            "trigger_path" | "trigger_paths" | "triggers.paths" | "paths" => {
                push_trigger_values(&mut triggers.paths, value);
            }
            "exclude" | "excludes" | "triggers.excludes" | "trigger_excludes" => {
                push_trigger_values(&mut triggers.excludes, value);
            }
            _ => {}
        }
    }
    if state
        .as_deref()
        .is_some_and(|value| value.eq_ignore_ascii_case("proposed"))
    {
        return None;
    }

    let fallback_name = path.parent()?.file_name()?.to_string_lossy().into_owned();
    let name = name
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(fallback_name);
    let description = description.filter(|value| !value.trim().is_empty());
    Some(SkillIndexEntry {
        name,
        description,
        path,
        invocation_mode,
        triggers,
    })
}

/// Append the frontmatter value(s) for a trigger key, skipping blanks. Reuses
/// [`split_frontmatter_list_value`] so inline `[a, b]` arrays and repeated
/// `- item` list lines (already flattened to repeated key/value pairs by
/// [`parse_frontmatter`]) are handled identically.
fn push_trigger_values(target: &mut Vec<String>, value: &str) {
    for item in split_frontmatter_list_value(value) {
        let trimmed = item.trim();
        if !trimmed.is_empty() && !target.iter().any(|existing| existing == trimmed) {
            target.push(trimmed.to_string());
        }
    }
}

fn parse_frontmatter_fields(content: &str) -> Vec<(String, String)> {
    parse_frontmatter(content).map_or_else(Vec::new, |(fields, _body)| fields)
}

fn parse_frontmatter(content: &str) -> Option<(Vec<(String, String)>, &str)> {
    let after_open = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))?;
    let mut fields = Vec::new();
    let mut current_list_key: Option<String> = None;
    let mut body_start = content.len();
    let open_len = content.len() - after_open.len();
    let mut consumed = 0;

    for line in after_open.split_inclusive('\n') {
        consumed += line.len();
        let trimmed = line.trim();
        if trimmed == "---" {
            body_start = open_len + consumed;
            break;
        }
        if let Some(item) = trimmed.strip_prefix("- ") {
            if let Some(key) = &current_list_key {
                fields.push((
                    key.clone(),
                    trim_frontmatter_scalar(item.trim()).to_string(),
                ));
            }
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            current_list_key = None;
            continue;
        };
        let key = key.trim().to_string();
        let value = value.trim();
        if value.is_empty() {
            current_list_key = Some(key);
        } else {
            fields.push((key, trim_frontmatter_scalar(value).to_string()));
            current_list_key = None;
        }
    }

    if body_start == content.len() {
        return None;
    }
    Some((fields, &content[body_start..]))
}

fn trim_frontmatter_scalar(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|stripped| stripped.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|stripped| stripped.strip_suffix('\''))
        })
        .unwrap_or(value)
}

fn parse_globs(fields: &[(String, String)]) -> Vec<String> {
    fields
        .iter()
        .filter(|(key, _)| matches!(key.as_str(), "glob" | "globs"))
        .flat_map(|(_, value)| split_frontmatter_list_value(value))
        .filter(|value| !value.trim().is_empty())
        .collect()
}

fn parse_always_apply(fields: &[(String, String)]) -> Option<bool> {
    fields
        .iter()
        .rev()
        .find(|(key, _)| matches!(key.as_str(), "alwaysApply" | "always_apply"))
        .and_then(
            |(_, value)| match value.trim().to_ascii_lowercase().as_str() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            },
        )
}

fn split_frontmatter_list_value(value: &str) -> Vec<String> {
    let trimmed = value.trim();
    let list = trimmed
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(trimmed);
    list.split(',')
        .map(|item| trim_frontmatter_scalar(item.trim()).to_string())
        .collect()
}

fn push_unique_skill_entry(entries: &mut Vec<SkillIndexEntry>, entry: SkillIndexEntry) {
    if entries
        .iter()
        .any(|existing| existing.name == entry.name || existing.path == entry.path)
    {
        return;
    }
    entries.push(entry);
}

fn push_context_file(files: &mut Vec<ContextFile>, path: PathBuf) -> std::io::Result<()> {
    match fs::read_to_string(&path) {
        Ok(content) if !content.trim().is_empty() => {
            let expanded = expand_context_imports(&content, &path);
            files.push(ContextFile::instruction(path, &expanded));
            Ok(())
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Maximum `@path` import recursion depth, matching Claude Code's 5-hop limit.
/// Beyond this an unexpanded `@path` token is left verbatim so a deep (or
/// cyclic) chain degrades gracefully rather than looping or exploding the prompt.
const MAX_CONTEXT_IMPORT_DEPTH: usize = 5;

/// Inline `@path` import references inside a `context.md` instruction file.
/// A whitespace-
/// delimited `@<path>` token on a non-code line is replaced with the (recursively
/// expanded) contents of the referenced file, resolved relative to the importing
/// file's directory (absolute `@/...` paths are honored too). The expansion is
/// deterministic and best-effort: a missing/unreadable target, a cycle, or a
/// depth-limit hit leaves the original `@path` token untouched. Tokens inside
/// fenced code blocks or inline code spans are not treated as imports.
fn expand_context_imports(content: &str, path: &Path) -> String {
    let mut visited = Vec::new();
    if let Ok(canonical) = fs::canonicalize(path) {
        visited.push(canonical);
    }
    expand_context_imports_inner(content, path, &mut visited, 0)
}

fn expand_context_imports_inner(
    content: &str,
    source_path: &Path,
    visited: &mut Vec<PathBuf>,
    depth: usize,
) -> String {
    if !content.contains('@') {
        return content.to_string();
    }
    let base_dir = source_path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = String::with_capacity(content.len());
    let mut in_fenced_block = false;
    for (index, line) in content.lines().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        if is_code_fence_line(line) {
            in_fenced_block = !in_fenced_block;
            out.push_str(line);
            continue;
        }
        if in_fenced_block {
            out.push_str(line);
            continue;
        }
        out.push_str(&expand_import_line(line, base_dir, visited, depth));
    }
    // `lines()` drops a trailing newline; preserve it so re-joined content keeps
    // the source file's final-newline shape (keeps the expansion deterministic).
    if content.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Returns true for a fenced code-block delimiter (triple-backtick or
/// triple-tilde, optionally indented and followed by an info string), which
/// toggles code-block state.
fn is_code_fence_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

/// Expand every resolvable `@path` import token on a single non-code line.
fn expand_import_line(
    line: &str,
    base_dir: &Path,
    visited: &mut Vec<PathBuf>,
    depth: usize,
) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_inline_code = false;
    let mut chars = line.char_indices();
    while let Some((offset, ch)) = chars.next() {
        if ch == '`' {
            in_inline_code = !in_inline_code;
            out.push(ch);
            continue;
        }
        // An import token starts at `@` either at line start or after whitespace,
        // so mid-word `@` (e.g. an email address) is never treated as an import.
        let at_token_boundary = offset == 0
            || line[..offset]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace);
        if in_inline_code || ch != '@' || !at_token_boundary {
            out.push(ch);
            continue;
        }
        let rest = &line[offset + ch.len_utf8()..];
        let token: String = rest
            .chars()
            .take_while(|c| !c.is_whitespace())
            .collect();
        match resolve_import(&token, base_dir, visited, depth) {
            Some(inlined) => {
                out.push_str(&inlined);
                // Skip the path token we just consumed from the iterator.
                for _ in 0..token.chars().count() {
                    chars.next();
                }
            }
            None => out.push(ch),
        }
    }
    out
}

/// Read and recursively expand the file named by an `@path` token, returning
/// `None` (so the caller leaves the token verbatim) when the token is not a
/// usable import: empty, depth-exhausted, missing/unreadable, or a cycle.
fn resolve_import(
    token: &str,
    base_dir: &Path,
    visited: &mut Vec<PathBuf>,
    depth: usize,
) -> Option<String> {
    if token.is_empty() || depth >= MAX_CONTEXT_IMPORT_DEPTH {
        return None;
    }
    let raw = Path::new(token);
    let candidate = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        base_dir.join(raw)
    };
    // Canonicalize for the cycle guard; this also fails fast on missing files.
    let canonical = fs::canonicalize(&candidate).ok()?;
    if visited.contains(&canonical) {
        return None;
    }
    let imported = fs::read_to_string(&canonical).ok()?;
    visited.push(canonical.clone());
    let expanded = expand_context_imports_inner(&imported, &canonical, visited, depth + 1);
    // Pop so sibling imports of the same file on later lines still expand once
    // each; only an active ancestor chain is treated as a cycle.
    visited.pop();
    Some(expanded)
}

fn read_git_status(cwd: &Path) -> Option<String> {
    let output = Command::new("git")
        .args([
            "--no-optional-locks",
            "status",
            "--short",
            "--branch",
            "--",
            ":(exclude)target",
            ":(exclude)node_modules",
            ":(exclude).build",
        ])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(truncate_git_status_snapshot(trimmed))
    }
}

/// Clamp a `git status --short --branch` snapshot to a bounded size so a repo
/// with a huge working tree cannot blow up every request's system prompt. Keeps
/// the leading lines (the `## branch...ahead/behind` header sorts first, so it
/// is always retained) up to the line/char budget and appends a one-line summary
/// of what was dropped.
fn truncate_git_status_snapshot(text: &str) -> String {
    let total_lines = text.lines().count();
    let mut out = String::new();
    let mut kept = 0usize;
    for line in text.lines() {
        // +1 for the joining newline; compare on chars so multibyte paths are
        // measured the same way the budget is expressed.
        if kept >= MAX_GIT_STATUS_LINES
            || out.chars().count() + line.chars().count() + 1 > MAX_GIT_STATUS_CHARS
        {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        kept += 1;
    }
    let omitted = total_lines.saturating_sub(kept);
    if omitted > 0 {
        use std::fmt::Write as _;
        // `write!` into a String is infallible; ignore the formatter Result.
        let _ = write!(
            out,
            "\n… ({omitted} more changed path(s) omitted; run `git status` for the full list)"
        );
    }
    out
}

fn render_project_context(project_context: &ProjectContext) -> String {
    let mut lines = vec!["# Project context".to_string()];
    let mut bullets = vec![
        format!("Today's date is {}.", project_context.current_date),
        format!("Working directory: {}", project_context.cwd.display()),
    ];
    if let Some(project_root) = &project_context.project_root {
        if project_root != &project_context.cwd {
            bullets.push(format!("Project root: {}.", project_root.display()));
        }
    }
    if !project_context.instruction_files.is_empty() {
        bullets.push(format!(
            "Project instruction files discovered: {}.",
            project_context.instruction_files.len()
        ));
    }
    if !project_context.skills_index.is_empty() {
        bullets.push(format!(
            "Project skills discovered: {}.",
            project_context.skills_index.len()
        ));
    }
    lines.extend(prepend_bullets(bullets));
    if let Some(status) = &project_context.git_status {
        lines.push(String::new());
        lines.push("Git status snapshot:".to_string());
        lines.push(status.clone());
    }
    // The full working-tree diff is intentionally absent from the prompt — it
    // burns input tokens every turn (the model runs `git diff` on demand), so
    // `discover_with_git` no longer even gathers it (`git_diff` stays `None`).
    lines.join("\n")
}

fn render_instruction_files(files: &[ContextFile]) -> String {
    let mut sections = vec!["# Project instructions".to_string()];
    let mut remaining_chars = MAX_TOTAL_INSTRUCTION_CHARS;
    for file in files {
        if remaining_chars == 0 {
            sections.push(
                "_Additional instruction content omitted after reaching the prompt budget._"
                    .to_string(),
            );
            break;
        }

        let rendered_content = render_instruction_content_for_file(file, remaining_chars);
        let consumed = rendered_content.chars().count().min(remaining_chars);
        remaining_chars = remaining_chars.saturating_sub(consumed);

        sections.push(format!("## {}", describe_instruction_file(file, files)));
        sections.push(rendered_content);
    }
    sections.join("\n\n")
}

fn render_scoped_instruction_index(files: &[ContextFile]) -> String {
    let mut lines = vec![
        "# Scoped instructions".to_string(),
        "These instruction files declare path globs and are not always applied. Read the file when the task touches a matching path; do not assume the rule applies globally.".to_string(),
    ];
    lines.extend(prepend_bullets(
        files
            .iter()
            .map(|file| {
                format!(
                    "{} (globs: {}; path: {})",
                    display_context_path(&file.path),
                    file.globs.join(", "),
                    file.path.display()
                )
            })
            .collect(),
    ));
    lines.join("\n")
}

fn dedupe_instruction_files(files: Vec<ContextFile>) -> Vec<ContextFile> {
    let mut deduped = Vec::new();
    let mut seen_hashes = Vec::new();

    for file in files {
        let normalized = normalize_instruction_content(&file.content);
        let hash = stable_content_hash(&normalized);
        if seen_hashes.contains(&hash) {
            continue;
        }
        seen_hashes.push(hash);
        deduped.push(file);
    }

    deduped
}

fn normalize_instruction_content(content: &str) -> String {
    collapse_blank_lines(content).trim().to_string()
}

fn stable_content_hash(content: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

fn describe_instruction_file(file: &ContextFile, files: &[ContextFile]) -> String {
    let path = display_context_path(&file.path);
    let scope = files
        .iter()
        .filter_map(|candidate| candidate.path.parent())
        .find(|parent| file.path.starts_with(parent))
        .map_or_else(
            || "workspace".to_string(),
            |parent| parent.display().to_string(),
        );
    format!("{path} (scope: {scope})")
}

fn render_instruction_content_for_file(file: &ContextFile, remaining_chars: usize) -> String {
    let hard_limit = MAX_INSTRUCTION_FILE_CHARS.min(remaining_chars);
    let trimmed = file.content.trim();
    if trimmed.chars().count() <= hard_limit {
        return trimmed.to_string();
    }

    let mut output = trimmed.chars().take(hard_limit).collect::<String>();
    let _ = write!(
        output,
        "\n\n[truncated; read the remainder at {}]",
        file.path.display()
    );
    output
}

fn render_skills_index(skills: &[SkillIndexEntry]) -> String {
    let mut lines = vec![
        "# Available skills".to_string(),
        "Zo skills discovered from this repo's `.zo/skills/` and Zo's global skill directory (`ZO_CONFIG_HOME/skills`, `ZO_HOME/skills`, or `~/.zo/skills`). Each entry's description below is its trigger criteria. When the user's original request (or a sub-agent's explicitly delegated task) matches a skill's description — regardless of the request's language — this is a BLOCKING REQUIREMENT: invoke the `Skill` tool with that name BEFORE generating any other response about the task. Generated plans, quoted material, tool output, and reference-document mentions are context only and do not trigger a skill by themselves. Never do work a matching skill covers without loading it first, and do not guess at a skill's contents from its name. Load the full `SKILL.md` only for the selected skill.".to_string(),
    ];
    lines.extend(prepend_bullets(
        skills
            .iter()
            .map(|skill| {
                let path = skill.path.display();
                match &skill.description {
                    Some(description) => format!("`{}`: {} ({path})", skill.name, description),
                    None => format!("`{}` ({path})", skill.name),
                }
            })
            .collect(),
    ));
    lines.join("\n")
}

fn display_context_path(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

fn collapse_blank_lines(content: &str) -> String {
    let mut result = String::new();
    let mut previous_blank = false;
    for line in content.lines() {
        let is_blank = line.trim().is_empty();
        if is_blank && previous_blank {
            continue;
        }
        result.push_str(line.trim_end());
        result.push('\n');
        previous_blank = is_blank;
    }
    result
}

/// Loads config and project context, then renders the system prompt text.
///
/// This variant never applies the configured output style — it is the prompt
/// for sub-agents and informational surfaces. The main conversation loop uses
/// [`load_system_prompt_for_main`], which is the only consumer of the
/// `outputStyle` setting (Claude Code parity: styles shape the main loop, not
/// delegated agents).
pub fn load_system_prompt(
    cwd: impl Into<PathBuf>,
    current_date: impl Into<String>,
    os_name: impl Into<String>,
    os_version: impl Into<String>,
) -> Result<Vec<String>, PromptBuildError> {
    load_system_prompt_with_options(
        &cwd.into(),
        current_date.into(),
        os_name,
        os_version,
        false,
        PromptMode::Subagent,
    )
}

/// [`load_system_prompt`] plus the configured `outputStyle` (settings key or
/// `/output-style`), resolved via [`output_style::resolve`].
pub fn load_system_prompt_for_main(
    cwd: impl Into<PathBuf>,
    current_date: impl Into<String>,
    os_name: impl Into<String>,
    os_version: impl Into<String>,
) -> Result<Vec<String>, PromptBuildError> {
    load_system_prompt_for_main_with_mode(
        cwd,
        current_date,
        os_name,
        os_version,
        PromptMode::Interactive,
    )
}

/// [`load_system_prompt_for_main`] with an explicit [`PromptMode`], for
/// surfaces where nobody can answer mid-task questions (headless one-shots).
pub fn load_system_prompt_for_main_with_mode(
    cwd: impl Into<PathBuf>,
    current_date: impl Into<String>,
    os_name: impl Into<String>,
    os_version: impl Into<String>,
    mode: PromptMode,
) -> Result<Vec<String>, PromptBuildError> {
    load_system_prompt_with_options(&cwd.into(), current_date.into(), os_name, os_version, true, mode)
}

fn load_system_prompt_with_options(
    cwd: &Path,
    current_date: String,
    os_name: impl Into<String>,
    os_version: impl Into<String>,
    apply_output_style: bool,
    mode: PromptMode,
) -> Result<Vec<String>, PromptBuildError> {
    let project_context = ProjectContext::discover_with_git(cwd, current_date)?;
    let config = ConfigLoader::default_for(cwd).load()?;
    let style = apply_output_style
        .then(|| {
            config
                .get("outputStyle")
                .and_then(|value| value.as_str())
                .and_then(|name| output_style::resolve(cwd, name))
        })
        .flatten();
    let mut builder = SystemPromptBuilder::new()
        .with_os(os_name, os_version)
        .with_project_context(project_context)
        .with_runtime_config(config)
        .with_mode(mode);
    if let Some((name, prompt)) = style {
        builder = builder.with_output_style(name, prompt);
    }
    Ok(builder.build())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod import_tests {
    use super::{
        expand_context_imports, push_context_file, ContextFile, MAX_CONTEXT_IMPORT_DEPTH,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "runtime-prompt-import-{}-{nanos}-{counter}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn inlines_relative_at_path_reference() {
        let dir = temp_dir();
        fs::write(dir.join("conventions.md"), "Use tabs, not spaces.")
            .expect("write conventions");
        let root = dir.join("context.md");
        fs::write(&root, "Project rules:\n@./conventions.md\nDone.").expect("write root");

        let content = fs::read_to_string(&root).expect("read root");
        let expanded = expand_context_imports(&content, &root);

        assert!(
            expanded.contains("Use tabs, not spaces."),
            "relative @path import should be inlined: {expanded}"
        );
        assert!(
            !expanded.contains("@./conventions.md"),
            "the @path token should be consumed once inlined: {expanded}"
        );
    }

    #[test]
    fn missing_import_target_left_verbatim() {
        let dir = temp_dir();
        let root = dir.join("context.md");
        let content = "See @./does-not-exist.md for details.";
        let expanded = expand_context_imports(content, &root);
        assert_eq!(
            expanded, content,
            "an unresolvable @path must degrade to the original line"
        );
    }

    #[test]
    fn cycle_does_not_loop() {
        let dir = temp_dir();
        let a = dir.join("a.md");
        let b = dir.join("b.md");
        fs::write(&a, "A imports @./b.md").expect("write a");
        fs::write(&b, "B imports @./a.md").expect("write b");

        let content = fs::read_to_string(&a).expect("read a");
        // Must terminate (no infinite recursion) and break the cycle by leaving
        // the back-reference to the root verbatim.
        let expanded = expand_context_imports(&content, &a);
        assert!(
            expanded.contains("B imports"),
            "the first hop should inline: {expanded}"
        );
        assert!(
            expanded.contains("@./a.md"),
            "the cyclic back-reference must be left verbatim: {expanded}"
        );
    }

    #[test]
    fn depth_is_bounded() {
        let dir = temp_dir();
        // Chain of files each importing the next, longer than the depth limit.
        let chain_len = MAX_CONTEXT_IMPORT_DEPTH + 3;
        for i in 0..chain_len {
            let path = dir.join(format!("d{i}.md"));
            if i + 1 < chain_len {
                fs::write(&path, format!("level {i} @./d{}.md", i + 1)).expect("write level");
            } else {
                fs::write(&path, format!("level {i} END")).expect("write tail");
            }
        }

        let root = dir.join("d0.md");
        let content = fs::read_to_string(&root).expect("read root");
        let expanded = expand_context_imports(&content, &root);

        // Exactly MAX_CONTEXT_IMPORT_DEPTH hops expand; the file at the depth
        // limit keeps its @path token rather than recursing further.
        assert!(
            expanded.contains(&format!("level {MAX_CONTEXT_IMPORT_DEPTH}")),
            "should expand up to the depth limit: {expanded}"
        );
        assert!(
            expanded.contains(&format!("@./d{}.md", MAX_CONTEXT_IMPORT_DEPTH + 1)),
            "the token at the depth limit must be left verbatim: {expanded}"
        );
        assert!(
            !expanded.contains("END"),
            "expansion past the depth limit must not occur: {expanded}"
        );
    }

    #[test]
    fn import_inside_code_block_not_expanded() {
        let dir = temp_dir();
        fs::write(dir.join("secret.md"), "INLINED").expect("write secret");
        let root = dir.join("context.md");
        let content = "```\n@./secret.md\n```\nplain @./secret.md";
        let expanded = expand_context_imports(content, &root);

        // The fenced occurrence stays literal; the plain one is inlined.
        assert!(
            expanded.contains("@./secret.md\n```"),
            "fenced @path must not be expanded: {expanded}"
        );
        assert!(
            expanded.contains("plain INLINED"),
            "non-fenced @path must be expanded: {expanded}"
        );
    }

    #[test]
    fn push_context_file_expands_imports_into_stored_content() {
        let dir = temp_dir();
        fs::write(dir.join("conventions.md"), "House style applies.")
            .expect("write conventions");
        let root = dir.join("context.md");
        fs::write(&root, "Top.\n@./conventions.md").expect("write root");

        let mut files: Vec<ContextFile> = Vec::new();
        push_context_file(&mut files, root).expect("push should succeed");

        let stored = &files.first().expect("one context file").content;
        assert!(
            stored.contains("House style applies."),
            "stored ContextFile content must carry the inlined import: {stored}"
        );
    }
}
