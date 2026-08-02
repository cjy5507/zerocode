use std::fs;
use std::path::{Path, PathBuf};

// Shared project defaults. Written to `.zo/settings.json`, the canonical path
// read by `ConfigLoader::discover` in crates/runtime/src/config/mod.rs.
const STARTER_SETTINGS_JSON: &str = concat!(
    "{\n",
    "  \"$schema\": \"SettingsSchema\",\n",
    "  \"permissions\": {\n",
    "    \"defaultMode\": \"dontAsk\"\n",
    "  }\n",
    "}\n",
);
const GITIGNORE_COMMENT: &str = "# zo local artifacts";
// Canonical local-only paths under `.zo/` (settings loader + session control).
// Persistent memory is global per-project state under the Zo config home, not
// a repository-local `.zo/` artifact.
const GITIGNORE_ENTRIES: [&str; 4] = [
    "**/.zo/settings.local.json",
    "**/.zo/sessions/",
    "**/.zo/session-prefs/",
    "**/.zo/cache/",
];

// Custom sub-agent harness stubs. Parsed by
// `crates/tools/src/misc_tools/agent_tools/custom.rs`; frontmatter keys and the
// `permissionMode` vocabulary mirror that loader. These names are not built-in
// subagent types, so the files take effect rather than being shadowed.
const AGENT_REVIEWER_MD: &str = concat!(
    "---\n",
    "name: reviewer\n",
    "description: Adversarial code review of a diff or file — surface bugs, security issues, and missing tests without editing.\n",
    "tools: read_file, grep_search, glob_search, bash\n",
    "permissionMode: read-only\n",
    "---\n",
    "\n",
    "You are a meticulous code reviewer for this repository.\n",
    "\n",
    "- Review only what was asked and ground every finding in the current source you read.\n",
    "- Prioritize correctness, security, and missing test coverage over style.\n",
    "- Do not edit files — report findings with `file:line` references and concrete fixes.\n",
    "- See `.zo/docs/architecture.md` for module boundaries and `.zo/docs/testing.md` for verification commands.\n",
);

const AGENT_TESTER_MD: &str = concat!(
    "---\n",
    "name: tester\n",
    "description: Run and extend the test suite, diagnose failures, and report results faithfully.\n",
    "tools: bash, read_file, grep_search, glob_search, edit_file, write_file\n",
    "permissionMode: workspace-write\n",
    "---\n",
    "\n",
    "You are a testing specialist for this repository.\n",
    "\n",
    "- Use the build/test/verify commands documented in `.zo/docs/testing.md`.\n",
    "- Reproduce failures before changing code; diagnose the root cause first.\n",
    "- Keep changes scoped to tests and the code under test; avoid unrelated refactors.\n",
    "- Report outcomes honestly: if a check fails or was not run, say so explicitly.\n",
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InitStatus {
    Created,
    Updated,
    Skipped,
}

impl InitStatus {
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Updated => "updated",
            Self::Skipped => "skipped (already exists)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InitArtifact {
    pub(crate) name: &'static str,
    pub(crate) status: InitStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InitReport {
    pub(crate) project_root: PathBuf,
    pub(crate) artifacts: Vec<InitArtifact>,
}

impl InitReport {
    #[must_use]
    pub(crate) fn render(&self) -> String {
        let mut lines = vec![
            "Init".to_string(),
            format!("  Project          {}", self.project_root.display()),
        ];
        for artifact in &self.artifacts {
            lines.push(format!(
                "  {:<16} {}",
                artifact.name,
                artifact.status.label()
            ));
        }
        lines.push("  Next step        Review and tailor the generated guidance".to_string());
        lines.join("\n")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
struct RepoDetection {
    rust_workspace: bool,
    rust_root: bool,
    python: bool,
    package_json: bool,
    typescript: bool,
    nextjs: bool,
    react: bool,
    vite: bool,
    nest: bool,
    src_dir: bool,
    tests_dir: bool,
    rust_dir: bool,
}

pub(crate) fn initialize_repo(cwd: &Path) -> Result<InitReport, Box<dyn std::error::Error>> {
    let mut artifacts = Vec::new();

    // Project config root that the runtime config loader and multi-agent
    // surfaces actually read (`.zo/settings.json`, `.zo/agents/`,
    // `.zo/docs/`).
    let zo_dir = cwd.join(".zo");
    artifacts.push(InitArtifact {
        name: ".zo/",
        status: ensure_dir(&zo_dir)?,
    });

    let settings_json = zo_dir.join("settings.json");
    artifacts.push(InitArtifact {
        name: ".zo/settings.json",
        status: write_file_if_missing(&settings_json, STARTER_SETTINGS_JSON)?,
    });

    let gitignore = cwd.join(".gitignore");
    artifacts.push(InitArtifact {
        name: ".gitignore",
        status: ensure_gitignore_entries(&gitignore)?,
    });

    // The instruction file is the shared, lightweight entry point every agent
    // (root and sub-agents) sees via the instruction-file discovery walk
    // (`crates/runtime/src/prompt/mod.rs`). Detailed, role-specific guidance
    // lives in the `.zo/docs/` files it points to.
    //
    // Zo scaffolds the model-agnostic `context.md` so the project works under
    // any model/CLI. Unrelated legacy instruction files are ignored and left
    // untouched; only `context.md` participates in the current contract.
    let instruction_path = cwd.join("context.md");
    let content = render_init_context_md(cwd);
    artifacts.push(InitArtifact {
        name: "context.md",
        status: write_file_if_missing(&instruction_path, &content)?,
    });

    // Topic docs that sub-agents reference instead of re-reading one monolithic
    // file. Generated only as starter scaffolding; never overwritten.
    let docs_dir = zo_dir.join("docs");
    ensure_dir(&docs_dir)?;
    artifacts.push(InitArtifact {
        name: ".zo/docs/architecture.md",
        status: write_file_if_missing(
            &docs_dir.join("architecture.md"),
            &render_architecture_doc(cwd),
        )?,
    });
    artifacts.push(InitArtifact {
        name: ".zo/docs/testing.md",
        status: write_file_if_missing(&docs_dir.join("testing.md"), &render_testing_doc(cwd))?,
    });

    // Custom sub-agent harness stubs. Loaded on demand by
    // `crates/tools/src/misc_tools/agent_tools/custom.rs` when an agent of that
    // name is spawned; harmless until used.
    let agents_dir = zo_dir.join("agents");
    ensure_dir(&agents_dir)?;
    artifacts.push(InitArtifact {
        name: ".zo/agents/reviewer.md",
        status: write_file_if_missing(&agents_dir.join("reviewer.md"), AGENT_REVIEWER_MD)?,
    });
    artifacts.push(InitArtifact {
        name: ".zo/agents/tester.md",
        status: write_file_if_missing(&agents_dir.join("tester.md"), AGENT_TESTER_MD)?,
    });

    Ok(InitReport {
        project_root: cwd.to_path_buf(),
        artifacts,
    })
}

fn ensure_dir(path: &Path) -> Result<InitStatus, std::io::Error> {
    if path.is_dir() {
        return Ok(InitStatus::Skipped);
    }
    fs::create_dir_all(path)?;
    Ok(InitStatus::Created)
}

fn write_file_if_missing(path: &Path, content: &str) -> Result<InitStatus, std::io::Error> {
    if path.exists() {
        return Ok(InitStatus::Skipped);
    }
    fs::write(path, content)?;
    Ok(InitStatus::Created)
}

fn ensure_gitignore_entries(path: &Path) -> Result<InitStatus, std::io::Error> {
    if !path.exists() {
        let mut lines = vec![GITIGNORE_COMMENT.to_string()];
        lines.extend(GITIGNORE_ENTRIES.iter().map(|entry| (*entry).to_string()));
        fs::write(path, format!("{}\n", lines.join("\n")))?;
        return Ok(InitStatus::Created);
    }

    let existing = fs::read_to_string(path)?;
    let mut lines = existing.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    let mut changed = false;

    if !lines.iter().any(|line| line == GITIGNORE_COMMENT) {
        lines.push(GITIGNORE_COMMENT.to_string());
        changed = true;
    }

    for entry in GITIGNORE_ENTRIES {
        if !lines.iter().any(|line| line == entry) {
            lines.push(entry.to_string());
            changed = true;
        }
    }

    if !changed {
        return Ok(InitStatus::Skipped);
    }

    fs::write(path, format!("{}\n", lines.join("\n")))?;
    Ok(InitStatus::Updated)
}

pub(crate) fn render_init_context_md(cwd: &Path) -> String {
    let detection = detect_repo(cwd);
    let mut lines = vec![
        "# context.md".to_string(),
        String::new(),
        "This file provides guidance to zo when working with code in this repository."
            .to_string(),
        String::new(),
    ];

    let detected_languages = detected_languages(&detection);
    let detected_frameworks = detected_frameworks(&detection);
    lines.push("## Detected stack".to_string());
    if detected_languages.is_empty() {
        lines.push("- No specific language markers were detected yet; document the primary language and verification commands once the project structure settles.".to_string());
    } else {
        lines.push(format!("- Languages: {}.", detected_languages.join(", ")));
    }
    if detected_frameworks.is_empty() {
        lines.push("- Frameworks: none detected from the supported starter markers.".to_string());
    } else {
        lines.push(format!(
            "- Frameworks/tooling markers: {}.",
            detected_frameworks.join(", ")
        ));
    }
    lines.push(String::new());

    let verification_lines = verification_lines(cwd, &detection);
    if !verification_lines.is_empty() {
        lines.push("## Verification".to_string());
        lines.extend(verification_lines);
        lines.push(String::new());
    }

    let structure_lines = repository_shape_lines(&detection);
    if !structure_lines.is_empty() {
        lines.push("## Repository shape".to_string());
        lines.extend(structure_lines);
        lines.push(String::new());
    }

    let framework_lines = framework_notes(&detection);
    if !framework_lines.is_empty() {
        lines.push("## Framework notes".to_string());
        lines.extend(framework_lines);
        lines.push(String::new());
    }

    lines.push("## Project guidance & sub-agents".to_string());
    lines.push("This file is the shared, lightweight entry point that every agent — the root agent and each spawned sub-agent — sees. Keep it short and link out to detailed docs rather than growing it into one monolith.".to_string());
    lines.push("- `.zo/docs/architecture.md` — module boundaries, key directories, and how the pieces fit together.".to_string());
    lines.push(
        "- `.zo/docs/testing.md` — build/test/verify commands and the definition of done."
            .to_string(),
    );
    lines.push("- `.zo/agents/*.md` — custom sub-agent harnesses (e.g. `reviewer`, `tester`). Each has its own focused system prompt and tool/permission scope and points at the docs it needs.".to_string());
    lines.push(
        "- `.zo/skills/<name>/SKILL.md` — reusable procedures the agent can load on demand."
            .to_string(),
    );
    lines.push("- Durable cross-session memory is stored by Zo under its global per-project state directory; use `MemoryWrite` for hard-won gotchas or user/project facts that should survive future sessions without dirtying this repository.".to_string());
    lines.push(String::new());

    lines.push("## Working agreement".to_string());
    lines.push("- Think before coding: call out important assumptions and ask when ambiguity changes the implementation.".to_string());
    lines.push("- Simplicity first: prefer the smallest correct change and avoid speculative abstractions or configuration.".to_string());
    lines.push("- Surgical changes: touch only files needed for the request and avoid unrelated refactors, formatting, or cleanup.".to_string());
    lines.push("- Goal-driven execution: define success criteria for non-trivial work and verify with focused checks when practical.".to_string());
    lines.push("- Prefer small, reviewable changes and keep generated bootstrap files aligned with actual repo workflows.".to_string());
    lines.push("- Keep shared defaults in `.zo/settings.json`; reserve `.zo/settings.local.json` for machine-local overrides (gitignored).".to_string());
    lines.push("- Do not overwrite existing `context.md` content automatically; update it intentionally when repo workflows change.".to_string());
    lines.push(String::new());

    lines.join("\n")
}

/// Render the starter `.zo/docs/architecture.md`. Detection-aware so the
/// scaffold names the repository's actual top-level surfaces; the prose is
/// starter guidance meant to be edited.
fn render_architecture_doc(cwd: &Path) -> String {
    let detection = detect_repo(cwd);
    let mut lines = vec![
        "# Architecture".to_string(),
        String::new(),
        "Starter notes on how this repository is laid out. Sub-agents read this instead of re-deriving structure each turn — keep it accurate and concise.".to_string(),
        String::new(),
        "## Layout".to_string(),
    ];
    let shape = repository_shape_lines(&detection);
    if shape.is_empty() {
        lines.push(
            "- Document the primary directories and their responsibilities here.".to_string(),
        );
    } else {
        lines.extend(shape);
    }
    lines.push(String::new());
    lines.push("## Conventions".to_string());
    lines.push(
        "- Record module boundaries, the source-of-truth for each concern, and patterns to follow."
            .to_string(),
    );
    lines.push("- Note anything a contributor (or sub-agent) must not break.".to_string());
    lines.push(String::new());
    lines.join("\n")
}

/// Render the starter `.zo/docs/testing.md`, reusing the same detection that
/// drives the context.md verification section so the commands stay consistent.
fn render_testing_doc(cwd: &Path) -> String {
    let detection = detect_repo(cwd);
    let mut lines = vec![
        "# Testing & verification".to_string(),
        String::new(),
        "How to verify a change in this repository. The `tester` sub-agent and any contributor should follow these before declaring work done.".to_string(),
        String::new(),
        "## Commands".to_string(),
    ];
    let verification = verification_lines(cwd, &detection);
    if verification.is_empty() {
        lines.push(
            "- Document the build, test, and lint commands for this project here.".to_string(),
        );
    } else {
        lines.extend(verification);
    }
    lines.push(String::new());
    lines.push("## Definition of done".to_string());
    lines.push("- Behavior change is covered by a focused test.".to_string());
    lines.push(
        "- All commands above pass, or any skipped/failing check is reported explicitly."
            .to_string(),
    );
    lines.push(String::new());
    lines.join("\n")
}

fn detect_repo(cwd: &Path) -> RepoDetection {
    let package_json_contents = fs::read_to_string(cwd.join("package.json"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    RepoDetection {
        rust_workspace: cwd.join("rust").join("Cargo.toml").is_file(),
        rust_root: cwd.join("Cargo.toml").is_file(),
        python: cwd.join("pyproject.toml").is_file()
            || cwd.join("requirements.txt").is_file()
            || cwd.join("setup.py").is_file(),
        package_json: cwd.join("package.json").is_file(),
        typescript: cwd.join("tsconfig.json").is_file()
            || package_json_contents.contains("typescript"),
        nextjs: package_json_contents.contains("\"next\""),
        react: package_json_contents.contains("\"react\""),
        vite: package_json_contents.contains("\"vite\""),
        nest: package_json_contents.contains("@nestjs"),
        src_dir: cwd.join("src").is_dir(),
        tests_dir: cwd.join("tests").is_dir(),
        rust_dir: cwd.join("rust").is_dir(),
    }
}

fn detected_languages(detection: &RepoDetection) -> Vec<&'static str> {
    let mut languages = Vec::new();
    if detection.rust_workspace || detection.rust_root {
        languages.push("Rust");
    }
    if detection.python {
        languages.push("Python");
    }
    if detection.typescript {
        languages.push("TypeScript");
    } else if detection.package_json {
        languages.push("JavaScript/Node.js");
    }
    languages
}

fn detected_frameworks(detection: &RepoDetection) -> Vec<&'static str> {
    let mut frameworks = Vec::new();
    if detection.nextjs {
        frameworks.push("Next.js");
    }
    if detection.react {
        frameworks.push("React");
    }
    if detection.vite {
        frameworks.push("Vite");
    }
    if detection.nest {
        frameworks.push("NestJS");
    }
    frameworks
}

fn verification_lines(cwd: &Path, detection: &RepoDetection) -> Vec<String> {
    let mut lines = Vec::new();
    if detection.rust_workspace {
        lines.push("- Run Rust verification from `rust/`: `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`".to_string());
    } else if detection.rust_root {
        lines.push("- Run Rust verification from the repo root: `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`".to_string());
    }
    if detection.python {
        if cwd.join("pyproject.toml").is_file() {
            lines.push("- Run the Python project checks declared in `pyproject.toml` (for example: `pytest`, `ruff check`, and `mypy` when configured).".to_string());
        } else {
            lines.push(
                "- Run the repo's Python test/lint commands before shipping changes.".to_string(),
            );
        }
    }
    if detection.package_json {
        lines.push("- Run the JavaScript/TypeScript checks from `package.json` before shipping changes (`npm test`, `npm run lint`, `npm run build`, or the repo equivalent).".to_string());
    }
    if detection.tests_dir && detection.src_dir {
        lines.push("- `src/` and `tests/` are both present; update both surfaces together when behavior changes.".to_string());
    }
    lines
}

fn repository_shape_lines(detection: &RepoDetection) -> Vec<String> {
    let mut lines = Vec::new();
    if detection.rust_dir {
        lines.push(
            "- `rust/` contains the Rust workspace and active CLI/runtime implementation."
                .to_string(),
        );
    }
    if detection.src_dir {
        lines.push("- `src/` contains source files that should stay consistent with generated guidance and tests.".to_string());
    }
    if detection.tests_dir {
        lines.push("- `tests/` contains validation surfaces that should be reviewed alongside code changes.".to_string());
    }
    lines
}

fn framework_notes(detection: &RepoDetection) -> Vec<String> {
    let mut lines = Vec::new();
    if detection.nextjs {
        lines.push("- Next.js detected: preserve routing/data-fetching conventions and verify production builds after changing app structure.".to_string());
    }
    if detection.react && !detection.nextjs {
        lines.push("- React detected: keep component behavior covered with focused tests and avoid unnecessary prop/API churn.".to_string());
    }
    if detection.vite {
        lines.push("- Vite detected: validate the production bundle after changing build-sensitive configuration or imports.".to_string());
    }
    if detection.nest {
        lines.push("- NestJS detected: keep module/provider boundaries explicit and verify controller/service wiring after refactors.".to_string());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{initialize_repo, render_init_context_md, GITIGNORE_ENTRIES};
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> std::path::PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_millis();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "zo-init-{}-{millis}-{counter}",
            std::process::id()
        ))
    }

    #[test]
    fn initialize_repo_creates_expected_files_and_gitignore_entries() {
        let root = temp_dir();
        fs::create_dir_all(root.join("rust")).expect("create rust dir");
        fs::write(root.join("rust").join("Cargo.toml"), "[workspace]\n").expect("write cargo");

        let report = initialize_repo(&root).expect("init should succeed");
        let rendered = report.render();
        assert!(rendered.contains(".zo/             created"));
        assert!(rendered.contains(".zo/settings.json created"));
        assert!(rendered.contains(".gitignore       created"));
        assert!(rendered.contains("context.md       created"));
        assert!(rendered.contains(".zo/docs/architecture.md created"));
        assert!(rendered.contains(".zo/docs/testing.md created"));
        assert!(rendered.contains(".zo/agents/reviewer.md created"));
        assert!(rendered.contains(".zo/agents/tester.md created"));
        assert!(root.join(".zo").is_dir());
        // `.zo/settings.json` is the canonical project settings path.
        assert!(root.join(".zo").join("settings.json").is_file());
        assert!(root.join("context.md").is_file());
        assert!(root
            .join(".zo")
            .join("docs")
            .join("architecture.md")
            .is_file());
        assert!(root
            .join(".zo")
            .join("docs")
            .join("testing.md")
            .is_file());
        assert!(root
            .join(".zo")
            .join("agents")
            .join("reviewer.md")
            .is_file());
        assert!(root
            .join(".zo")
            .join("agents")
            .join("tester.md")
            .is_file());
        assert_eq!(
            fs::read_to_string(root.join(".zo").join("settings.json"))
                .expect("read settings json"),
            concat!(
                "{\n",
                "  \"$schema\": \"SettingsSchema\",\n",
                "  \"permissions\": {\n",
                "    \"defaultMode\": \"dontAsk\"\n",
                "  }\n",
                "}\n",
            )
        );
        let gitignore = fs::read_to_string(root.join(".gitignore")).expect("read gitignore");
        assert!(gitignore.contains("**/.zo/settings.local.json"));
        assert!(gitignore.contains("**/.zo/sessions/"));
        assert!(gitignore.contains("**/.zo/session-prefs/"));
        assert!(gitignore.contains("**/.zo/cache/"));
        // The recursive cache ignore must not be narrowed back to the
        // prompt-cache subdir, and must never shadow persistent user memory
        // (a sibling of `cache/`, not a descendant of it).
        assert!(!gitignore.contains("**/.zo/cache/prompt-cache/"));
        assert!(!gitignore.contains(".zo/memory/"));
        assert!(!gitignore.contains(".zo/memory.local/"));
        let context_md = fs::read_to_string(root.join("context.md")).expect("read context md");
        assert!(context_md.contains("Languages: Rust."));
        assert!(context_md.contains("cargo clippy --workspace --all-targets -- -D warnings"));
        assert!(context_md.contains("Think before coding"));
        assert!(context_md.contains("Simplicity first"));
        assert!(context_md.contains("Surgical changes"));
        assert!(context_md.contains("Goal-driven execution"));
        // The generated context.md routes agents to per-topic docs rather than
        // carrying everything inline, and points at the correct settings paths.
        assert!(context_md.contains(".zo/docs/architecture.md"));
        assert!(context_md.contains(".zo/agents/*.md"));
        assert!(context_md.contains("Keep shared defaults in `.zo/settings.json`"));

        // The generated sub-agent stub parses as a valid custom agent for the
        // loader (frontmatter format mirrors agent_tools/custom.rs).
        let reviewer = fs::read_to_string(root.join(".zo").join("agents").join("reviewer.md"))
            .expect("read reviewer agent");
        assert!(reviewer.starts_with("---\n"));
        assert!(reviewer.contains("permissionMode: read-only"));

        fs::remove_dir_all(root).expect("cleanup temp dir");
    }

    /// Behavioural proof that the recursive `**/.zo/cache/` entry ignores the
    /// whole prompt-cache tree at any nesting depth while leaving persistent
    /// user memory (`.zo/memory/`, a sibling of `cache/`) tracked. Asserting the
    /// literal string is not enough — this exercises the real gitignore matcher
    /// git itself uses, so a future edit that narrows or over-broadens the rule
    /// is caught.
    #[test]
    fn cache_ignore_entry_hides_cache_tree_but_not_user_memory() {
        use ignore::gitignore::GitignoreBuilder;

        let mut builder = GitignoreBuilder::new("/repo");
        for entry in GITIGNORE_ENTRIES {
            builder.add_line(None, entry).expect("valid gitignore line");
        }
        let matcher = builder.build().expect("build gitignore");

        // The rule targets a directory (`**/.zo/cache/`), so git prunes the
        // whole subtree at the `cache` dir. `matched_path_or_any_parents`
        // mirrors that: it reports a file ignored because an ancestor dir is.
        for cache_path in [
            "/repo/.zo/cache/prompt-cache/entry.json",
            "/repo/.zo/cache/other.bin",
            "/repo/nested/deep/.zo/cache/prompt-cache/entry.json",
        ] {
            assert!(
                matcher.matched_path_or_any_parents(cache_path, false).is_ignore(),
                "cache path must be ignored: {cache_path}"
            );
        }

        for memory_path in [
            "/repo/.zo/memory/note.md",
            "/repo/.zo/memory/MEMORY.md",
            "/repo/nested/.zo/memory/note.md",
        ] {
            assert!(
                !matcher.matched_path_or_any_parents(memory_path, false).is_ignore(),
                "user memory must NOT be ignored: {memory_path}"
            );
        }
    }

    #[test]
    fn initialize_repo_is_idempotent_and_preserves_existing_files() {
        let root = temp_dir();
        fs::create_dir_all(&root).expect("create root");
        let legacy_path = root.join(["CLAUDE", ".md"].concat());
        fs::write(&legacy_path, "custom guidance\n").expect("write legacy instructions");
        fs::write(root.join(".gitignore"), ".zo/settings.local.json\n")
            .expect("write gitignore");

        let first = initialize_repo(&root).expect("first init should succeed");
        assert!(first.render().contains("context.md       created"));
        let second = initialize_repo(&root).expect("second init should succeed");
        let second_rendered = second.render();
        assert!(second_rendered.contains(".zo/             skipped (already exists)"));
        assert!(second_rendered.contains(".zo/settings.json skipped (already exists)"));
        assert!(second_rendered.contains(".gitignore       skipped (already exists)"));
        assert!(second_rendered.contains("context.md       skipped (already exists)"));
        assert!(second_rendered.contains(".zo/agents/reviewer.md skipped (already exists)"));
        assert_eq!(
            fs::read_to_string(&legacy_path).expect("read legacy instructions"),
            "custom guidance\n"
        );
        assert!(root.join("context.md").exists());
        let gitignore = fs::read_to_string(root.join(".gitignore")).expect("read gitignore");
        assert_eq!(gitignore.matches("**/.zo/settings.local.json").count(), 1);
        assert_eq!(gitignore.matches("**/.zo/sessions/").count(), 1);
        assert_eq!(gitignore.matches("**/.zo/session-prefs/").count(), 1);
        assert_eq!(gitignore.matches(".zo/memory.local/").count(), 0);

        fs::remove_dir_all(root).expect("cleanup temp dir");
    }

    #[test]
    fn render_init_template_mentions_detected_python_and_nextjs_markers() {
        let root = temp_dir();
        fs::create_dir_all(&root).expect("create root");
        fs::write(root.join("pyproject.toml"), "[project]\nname = \"demo\"\n")
            .expect("write pyproject");
        fs::write(
            root.join("package.json"),
            r#"{"dependencies":{"next":"14.0.0","react":"18.0.0"},"devDependencies":{"typescript":"5.0.0"}}"#,
        )
        .expect("write package json");

        let rendered = render_init_context_md(Path::new(&root));
        assert!(rendered.contains("Languages: Python, TypeScript."));
        assert!(rendered.contains("Frameworks/tooling markers: Next.js, React."));
        assert!(rendered.contains("pyproject.toml"));
        assert!(rendered.contains("Next.js detected"));
        assert!(rendered.contains("Think before coding"));
        assert!(rendered.contains("Simplicity first"));
        assert!(rendered.contains("Surgical changes"));
        assert!(rendered.contains("Goal-driven execution"));

        fs::remove_dir_all(root).expect("cleanup temp dir");
    }
}
