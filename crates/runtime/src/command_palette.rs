use crate::fuzzy_file_picker::fuzzy_find_files;
use core_types::CommandCategory;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteEntry {
    SlashCommand {
        name: String,
        description: String,
        category: CommandCategory,
        aliases: Vec<String>,
        argument_hint: Option<String>,
        has_subcommands: bool,
    },
    RecentSession {
        id: String,
        name: String,
    },
    File {
        path: String,
    },
}

impl PaletteEntry {
    #[must_use]
    pub fn display_text(&self) -> &str {
        match self {
            Self::SlashCommand { name, .. } | Self::RecentSession { name, .. } => name,
            Self::File { path } => path,
        }
    }

    #[must_use]
    pub fn secondary_text(&self) -> &str {
        match self {
            Self::SlashCommand { description, .. } => description,
            Self::RecentSession { id, .. } => id,
            Self::File { .. } => "",
        }
    }

    #[must_use]
    pub fn category(&self) -> Option<CommandCategory> {
        match self {
            Self::SlashCommand { category, .. } => Some(*category),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    pub score: i64,
    pub matched_indices: Vec<usize>,
}

pub struct CommandPalette {
    entries: Vec<PaletteEntry>,
}

impl CommandPalette {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: builtin_slash_commands(),
        }
    }

    #[must_use]
    pub fn entries(&self) -> &[PaletteEntry] {
        &self.entries
    }

    #[must_use]
    pub fn search(&self, query: &str, cwd: &Path, max_results: usize) -> Vec<PaletteEntry> {
        if query.is_empty() {
            return self.entries.iter().take(max_results).cloned().collect();
        }

        let (category_filter, effective_query) = parse_category_filter(query);
        let query_lower = effective_query.to_lowercase();

        let mut results: Vec<(i64, PaletteEntry)> = Vec::new();

        for entry in &self.entries {
            if let Some(cat_filter) = category_filter {
                if entry.category() != Some(cat_filter) {
                    continue;
                }
            }

            if let Some(m) = score_entry(entry, &query_lower) {
                results.push((m.score, entry.clone()));
            }
        }

        if category_filter.is_none() {
            let file_matches = fuzzy_find_files(cwd, effective_query, max_results / 2);
            for fm in file_matches {
                results.push((
                    fm.score,
                    PaletteEntry::File {
                        path: fm.relative_path,
                    },
                ));
            }
        }

        results.sort_by(|a, b| b.0.cmp(&a.0));
        results
            .into_iter()
            .take(max_results)
            .map(|(_, e)| e)
            .collect()
    }

    #[must_use]
    pub fn search_with_matches(
        &self,
        query: &str,
        cwd: &Path,
        max_results: usize,
    ) -> Vec<(PaletteEntry, Vec<usize>)> {
        if query.is_empty() {
            return self
                .entries
                .iter()
                .take(max_results)
                .map(|e| (e.clone(), Vec::new()))
                .collect();
        }

        let (category_filter, effective_query) = parse_category_filter(query);
        let query_lower = effective_query.to_lowercase();

        let mut results: Vec<(i64, PaletteEntry, Vec<usize>)> = Vec::new();

        for entry in &self.entries {
            if let Some(cat_filter) = category_filter {
                if entry.category() != Some(cat_filter) {
                    continue;
                }
            }

            if let Some(m) = score_entry(entry, &query_lower) {
                results.push((m.score, entry.clone(), m.matched_indices));
            }
        }

        if category_filter.is_none() {
            let file_matches = fuzzy_find_files(cwd, effective_query, max_results / 2);
            for fm in file_matches {
                results.push((
                    fm.score,
                    PaletteEntry::File {
                        path: fm.relative_path,
                    },
                    Vec::new(),
                ));
            }
        }

        results.sort_by(|a, b| b.0.cmp(&a.0));
        results
            .into_iter()
            .take(max_results)
            .map(|(_, e, indices)| (e, indices))
            .collect()
    }
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_category_filter(query: &str) -> (Option<CommandCategory>, &str) {
    if let Some(idx) = query.find(':') {
        let prefix = &query[..idx];
        let rest = &query[idx + 1..];
        if let Some(cat) = CommandCategory::from_prefix(prefix) {
            return (Some(cat), rest);
        }
    }
    (None, query)
}

#[allow(clippy::cast_possible_wrap)]
fn score_entry(entry: &PaletteEntry, query: &str) -> Option<FuzzyMatch> {
    let text = entry.display_text().to_lowercase();
    let desc = entry.secondary_text().to_lowercase();
    let combined = format!("{text} {desc}");

    // Tier 1: exact prefix match on display text
    if text.starts_with(query) || text.starts_with(&format!("/{query}")) {
        return Some(FuzzyMatch {
            score: 200 - text.len() as i64,
            matched_indices: prefix_match_indices(&text, query),
        });
    }

    // Tier 2: word-boundary match
    if let Some(indices) = word_boundary_match(&text, query) {
        return Some(FuzzyMatch {
            score: 150 - text.len() as i64,
            matched_indices: indices,
        });
    }

    // Tier 3: exact substring in display text or description
    if text.contains(query) {
        return Some(FuzzyMatch {
            score: 100 - text.len() as i64,
            matched_indices: substring_match_indices(&text, query),
        });
    }

    if desc.contains(query) {
        return Some(FuzzyMatch {
            score: 80 - combined.len() as i64,
            matched_indices: Vec::new(),
        });
    }

    // Tier 4: subsequence with consecutive bonus
    if let Some((indices, consecutive_count)) = subsequence_match(&combined, query) {
        let bonus = consecutive_count as i64 * 10;
        return Some(FuzzyMatch {
            score: 50 + bonus - combined.len() as i64 / 4,
            matched_indices: indices,
        });
    }

    None
}

fn prefix_match_indices(text: &str, query: &str) -> Vec<usize> {
    let offset = usize::from(text.starts_with('/') && !query.starts_with('/'));
    (offset..offset + query.len()).collect()
}

fn substring_match_indices(text: &str, query: &str) -> Vec<usize> {
    if let Some(pos) = text.find(query) {
        (pos..pos + query.len()).collect()
    } else {
        Vec::new()
    }
}

fn word_boundary_match(text: &str, query: &str) -> Option<Vec<usize>> {
    let chars: Vec<char> = text.chars().collect();
    let query_chars: Vec<char> = query.chars().collect();
    if query_chars.is_empty() {
        return None;
    }

    let mut indices = Vec::new();
    let mut qi = 0;

    for (i, &c) in chars.iter().enumerate() {
        if qi >= query_chars.len() {
            break;
        }
        if c == query_chars[qi] {
            let is_boundary =
                i == 0 || matches!(chars.get(i.wrapping_sub(1)), Some('-' | '_' | ' ' | '/'));
            if is_boundary || !indices.is_empty() {
                indices.push(i);
                qi += 1;
            }
        }
    }

    if qi == query_chars.len() {
        Some(indices)
    } else {
        None
    }
}

fn subsequence_match(text: &str, query: &str) -> Option<(Vec<usize>, usize)> {
    let text_chars: Vec<char> = text.chars().collect();
    let query_chars: Vec<char> = query.chars().collect();
    let mut indices = Vec::new();
    let mut qi = 0;
    let mut consecutive = 0;
    let mut max_consecutive = 0;

    for (i, &c) in text_chars.iter().enumerate() {
        if qi < query_chars.len() && c == query_chars[qi] {
            if !indices.is_empty() && indices.last() == Some(&(i - 1)) {
                consecutive += 1;
                max_consecutive = max_consecutive.max(consecutive);
            } else {
                consecutive = 0;
            }
            indices.push(i);
            qi += 1;
        }
    }

    if qi == query_chars.len() {
        Some((indices, max_consecutive))
    } else {
        None
    }
}

fn has_subcommands(hint: Option<&str>) -> bool {
    hint.is_some_and(|h| h.contains('|'))
}

pub type PaletteSpec<'a> = (
    &'a str,
    &'a str,
    CommandCategory,
    &'a [&'a str],
    Option<&'a str>,
);

#[must_use]
pub fn build_entries_from_specs(specs: &[PaletteSpec<'_>]) -> Vec<PaletteEntry> {
    specs
        .iter()
        .map(
            |(name, desc, cat, aliases, hint)| PaletteEntry::SlashCommand {
                name: format!("/{name}"),
                description: desc.to_string(),
                category: *cat,
                aliases: aliases.iter().map(|a| format!("/{a}")).collect(),
                argument_hint: hint.map(String::from),
                has_subcommands: has_subcommands(*hint),
            },
        )
        .collect()
}

use CommandCategory::{Analysis, Appearance, Control, Discovery, Session, Workspace};

#[allow(clippy::too_many_lines)] // a flat slash-command spec table, clearer unsplit
fn builtin_slash_commands() -> Vec<PaletteEntry> {
    let specs: &[PaletteSpec<'_>] = &[
        ("help", "Show available commands", Session, &[], None),
        ("status", "Show session status", Session, &[], None),
        (
            "compact",
            "Compress conversation context",
            Workspace,
            &[],
            None,
        ),
        ("model", "Switch AI model", Session, &[], Some("[model]")),
        ("cost", "Show token usage and cost", Session, &[], None),
        ("diff", "Show git diff", Workspace, &[], None),
        ("commit", "Generate commit message", Workspace, &[], None),
        (
            "pr",
            "Create pull request",
            Workspace,
            &[],
            Some("[context]"),
        ),
        ("undo", "Undo last file changes", Workspace, &[], None),
        (
            "clear",
            "Clear conversation",
            Workspace,
            &[],
            Some("[--confirm]"),
        ),
        (
            "config",
            "Open settings",
            Workspace,
            &[],
            Some("[env|hooks|model|plugins]"),
        ),
        (
            "theme",
            "Change color theme",
            Appearance,
            &[],
            Some("[theme-name]"),
        ),
        ("vim", "Toggle vim mode", Appearance, &[], None),
        (
            "permissions",
            "Manage permissions",
            Session,
            &[],
            Some("[read-only|workspace-write|danger-full-access]"),
        ),
        (
            "mcp",
            "Manage MCP servers",
            Discovery,
            &[],
            Some("[list|show <server>|help]"),
        ),
        ("memory", "Edit context.md", Workspace, &[], None),
        ("init", "Initialize context.md", Workspace, &[], None),
        (
            "review",
            "Review pull request",
            Analysis,
            &[],
            Some("[scope]"),
        ),
        (
            "export",
            "Export conversation",
            Workspace,
            &[],
            Some("[file]"),
        ),
        (
            "session",
            "Session management",
            Session,
            &[],
            Some("[list|switch <session-id>|fork [branch-name]]"),
        ),
        (
            "plugin",
            "Manage plugins",
            Workspace,
            &["plugins", "marketplace"],
            Some("[list|install <path>|enable <name>|disable <name>]"),
        ),
        (
            "agents",
            "Manage agents",
            Discovery,
            &[],
            Some("[list|help]"),
        ),
        (
            "skills",
            "List available skills",
            Discovery,
            &[],
            Some("[list|install <path>|help]"),
        ),
        ("fast", "Toggle fast mode", Appearance, &[], None),
        (
            "effort",
            "Set reasoning effort",
            Appearance,
            &[],
            Some("[low|medium|high]"),
        ),
        ("plan", "Enter plan mode", Analysis, &[], Some("[on|off]")),
        (
            "tasks",
            "List tasks",
            Discovery,
            &[],
            Some("[list|get <id>|stop <id>]"),
        ),
        (
            "branch",
            "Create/switch branch",
            Workspace,
            &[],
            Some("[name]"),
        ),
        (
            "rewind",
            "Rewind conversation turns",
            Control,
            &[],
            Some("[steps]"),
        ),
        (
            "bughunter",
            "Inspect codebase for bugs",
            Analysis,
            &[],
            Some("[scope]"),
        ),
        (
            "ultraplan",
            "Deep planning prompt",
            Analysis,
            &[],
            Some("[task]"),
        ),
        (
            "teleport",
            "Jump to file or symbol",
            Discovery,
            &[],
            Some("<symbol-or-path>"),
        ),
        ("doctor", "Diagnose setup issues", Control, &[], None),
        ("login", "OAuth login", Session, &[], Some("[provider]")),
        ("logout", "Log out", Session, &[], None),
        ("version", "Show CLI version", Session, &[], None),
        ("sandbox", "Show sandbox status", Session, &[], None),
        (
            "resume",
            "Load saved session",
            Session,
            &[],
            Some("<session-path>"),
        ),
        ("exit", "Exit the REPL", Control, &[], None),
    ];
    build_entries_from_specs(specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_finds_slash_commands() {
        let palette = CommandPalette::new();
        let results = palette.search("com", Path::new("."), 10);
        let names: Vec<&str> = results
            .iter()
            .map(super::PaletteEntry::display_text)
            .collect();
        assert!(names
            .iter()
            .any(|n| n.contains("compact") || n.contains("commit")));
    }

    #[test]
    fn empty_query_returns_all() {
        let palette = CommandPalette::new();
        let results = palette.search("", Path::new("."), 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn prefix_match_scores_highest() {
        let palette = CommandPalette::new();
        let results = palette.search("hel", Path::new("."), 5);
        assert_eq!(results[0].display_text(), "/help");
    }

    #[test]
    fn category_filter_works() {
        let palette = CommandPalette::new();
        let results = palette.search("git:", Path::new("."), 50);
        assert!(results.iter().all(|e| matches!(
            e,
            PaletteEntry::SlashCommand {
                category: CommandCategory::Workspace,
                ..
            }
        )));
    }

    #[test]
    fn search_with_matches_returns_indices() {
        let palette = CommandPalette::new();
        let results = palette.search_with_matches("help", Path::new("."), 5);
        assert!(!results.is_empty());
        let (entry, indices) = &results[0];
        assert_eq!(entry.display_text(), "/help");
        assert!(!indices.is_empty());
    }

    #[test]
    fn entries_include_category() {
        let palette = CommandPalette::new();
        let entries = palette.entries();
        let help = entries
            .iter()
            .find(|e| e.display_text() == "/help")
            .unwrap();
        assert_eq!(help.category(), Some(CommandCategory::Session));
    }

    #[test]
    fn description_search_works() {
        let palette = CommandPalette::new();
        let results = palette.search("token", Path::new("."), 10);
        let names: Vec<&str> = results
            .iter()
            .map(super::PaletteEntry::display_text)
            .collect();
        assert!(names.iter().any(|n| n.contains("cost")));
    }
}
