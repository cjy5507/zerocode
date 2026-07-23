//! Scope-based syntax classification.
//!
//! syntect's `HighlightLines` resolves every code token to a base16-ocean.dark
//! RGB color, which the widget layer used to emit as a raw `Color::Rgb`. That
//! broke two invariants at once: the colors had nothing to do with the zo
//! palette (a code card looked like a different app), and RGB never degraded on
//! 256-color / `NO_COLOR` terminals.
//!
//! This module replaces that path. It parses code the same way syntect does
//! (persisting parse + scope state across lines so multi-line strings and block
//! comments stay correct), but instead of asking a syntect theme for a color it
//! classifies each token's scope stack into a small [`SyntaxRole`]. The widget
//! then resolves the role through [`Theme::syntax_style`], so the color comes
//! from the zo [`Palette`] and degrades for free.
//!
//! [`Theme::syntax_style`]: crate::tui::theme::Theme::syntax_style
//! [`Palette`]: crate::tui::theme::Palette

use std::sync::OnceLock;

use syntect::parsing::{ParseState, Scope, ScopeStack, SyntaxReference, SyntaxSet};

use crate::tui::theme::SyntaxRole;

/// Precomputed scope prefixes used for classification. Building a [`Scope`]
/// locks syntect's global scope repository, so they are computed once and then
/// matched with the cheap bitwise [`Scope::is_prefix_of`].
struct ScopePrefixes {
    comment: Scope,
    string: Scope,
    keyword: Scope,
    keyword_operator: Scope,
    storage: Scope,
    entity_name: Scope,
    support: Scope,
}

fn scope_prefixes() -> &'static ScopePrefixes {
    static PREFIXES: OnceLock<ScopePrefixes> = OnceLock::new();
    PREFIXES.get_or_init(|| {
        // These are fixed, well-formed scope atoms, so `Scope::new` cannot fail;
        // a broken build here would surface immediately in tests.
        let mk = |s: &str| Scope::new(s).unwrap_or_else(|_| Scope::new("").expect("empty scope"));
        ScopePrefixes {
            comment: mk("comment"),
            string: mk("string"),
            keyword: mk("keyword"),
            keyword_operator: mk("keyword.operator"),
            storage: mk("storage"),
            entity_name: mk("entity.name"),
            support: mk("support"),
        }
    })
}

/// Map a single syntect [`Scope`] to a zo [`SyntaxRole`], or `None` when this
/// scope carries no color signal (so the caller keeps looking down the stack).
///
/// Operators (`keyword.operator*`) are intentionally demoted to `Plain`: they
/// occur constantly, and painting every `+`/`=`/`=>` violet re-creates exactly
/// the color noise this rework removes. Numbers / constants likewise fall
/// through to `Plain` for restraint.
fn role_for_scope(scope: Scope) -> Option<SyntaxRole> {
    let p = scope_prefixes();
    if p.comment.is_prefix_of(scope) {
        return Some(SyntaxRole::Comment);
    }
    if p.string.is_prefix_of(scope) {
        return Some(SyntaxRole::Str);
    }
    // Check the operator sub-scope before the broad `keyword` prefix so it wins.
    if p.keyword_operator.is_prefix_of(scope) {
        return None;
    }
    if p.keyword.is_prefix_of(scope) || p.storage.is_prefix_of(scope) {
        return Some(SyntaxRole::Keyword);
    }
    if p.entity_name.is_prefix_of(scope) || p.support.is_prefix_of(scope) {
        return Some(SyntaxRole::Name);
    }
    None
}

/// Classify a token's whole scope stack. The most specific scope (top of stack)
/// wins, so a quote glyph scoped `punctuation.definition.string.begin` on top of
/// `string.quoted.double` still reads as a string once punctuation falls
/// through. Anything unrecognized is [`SyntaxRole::Plain`].
fn classify(scopes: &[Scope]) -> SyntaxRole {
    for scope in scopes.iter().rev() {
        if let Some(role) = role_for_scope(*scope) {
            return role;
        }
    }
    SyntaxRole::Plain
}

/// Stateful scope-based highlighter — the palette-driven replacement for
/// `syntect::easy::HighlightLines`. Persists parse + scope state across lines
/// (multi-line strings / block comments) and yields role-tagged token slices.
pub(crate) struct SyntaxHighlighter {
    parse: ParseState,
    stack: ScopeStack,
}

impl SyntaxHighlighter {
    /// Start highlighting `syntax` (from [`SyntaxSet::find_syntax_*`]).
    pub(crate) fn new(syntax: &SyntaxReference) -> Self {
        Self {
            parse: ParseState::new(syntax),
            stack: ScopeStack::new(),
        }
    }

    /// Role-tagged regions for one `line`, which must include its trailing `\n`
    /// (as produced by `syntect::util::LinesWithEndings`). The returned byte
    /// slices concatenate back to `line`, so callers keep their own offset
    /// bookkeeping. Adjacent tokens of the same role are coalesced into one
    /// region — fewer spans, and no visual change since color is the only
    /// per-region attribute.
    ///
    /// Mirrors syntect's `RangedHighlightIterator`: emit `line[pos..end]` under
    /// the *current* scope stack, then apply the op at `end`. A parse error
    /// degrades the whole line to one `Plain` region rather than panicking.
    pub(crate) fn highlight_line<'a>(
        &mut self,
        line: &'a str,
        syntax_set: &SyntaxSet,
    ) -> Vec<(SyntaxRole, &'a str)> {
        let Ok(ops) = self.parse.parse_line(line, syntax_set) else {
            return vec![(SyntaxRole::Plain, line)];
        };

        let mut regions: Vec<(SyntaxRole, usize, usize)> = Vec::new();
        let mut pos = 0usize;
        for (end, op) in &ops {
            push_region(&mut regions, classify(self.stack.as_slice()), pos, *end);
            // A malformed op cannot corrupt the frame; skip it and keep going.
            let _ = self.stack.apply(op);
            pos = *end;
        }
        push_region(&mut regions, classify(self.stack.as_slice()), pos, line.len());

        regions
            .into_iter()
            .map(|(role, start, end)| (role, &line[start..end]))
            .collect()
    }
}

/// Append `[start, end)` as a region of `role`, coalescing with the previous
/// region when it is contiguous and shares the role. Empty ranges are dropped.
fn push_region(regions: &mut Vec<(SyntaxRole, usize, usize)>, role: SyntaxRole, start: usize, end: usize) {
    if start >= end {
        return;
    }
    if let Some(last) = regions.last_mut() {
        if last.0 == role && last.2 == start {
            last.2 = end;
            return;
        }
    }
    regions.push((role, start, end));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assets() -> &'static (SyntaxSet, syntect::highlighting::Theme) {
        crate::tui::markdown::syntect_assets()
    }

    /// Collect `(role, text)` regions for a whole snippet, joining across lines.
    fn roles_for(code: &str, token: &str) -> Vec<SyntaxRole> {
        let (ss, _) = assets();
        let syntax = ss.find_syntax_by_token("rust").expect("rust syntax");
        let mut hl = SyntaxHighlighter::new(syntax);
        let mut hits = Vec::new();
        for line in syntect::util::LinesWithEndings::from(code) {
            for (role, seg) in hl.highlight_line(line, ss) {
                if seg.trim() == token {
                    hits.push(role);
                }
            }
        }
        hits
    }

    #[test]
    fn keywords_strings_names_and_comments_classify() {
        let code = "// note\nfn main() {\n    let s = \"hi\";\n}\n";
        assert!(
            roles_for(code, "fn").contains(&SyntaxRole::Keyword),
            "`fn` is a keyword"
        );
        assert!(
            roles_for(code, "let").contains(&SyntaxRole::Keyword),
            "`let` is a keyword"
        );
        assert!(
            roles_for(code, "main").contains(&SyntaxRole::Name),
            "`main` is a declared function name"
        );
    }

    #[test]
    fn comment_line_is_all_comment_role() {
        let (ss, _) = assets();
        let syntax = ss.find_syntax_by_token("rust").expect("rust syntax");
        let mut hl = SyntaxHighlighter::new(syntax);
        let regions = hl.highlight_line("// a comment here\n", ss);
        assert!(
            regions
                .iter()
                .all(|(role, seg)| seg.trim().is_empty() || *role == SyntaxRole::Comment),
            "every non-blank token on a comment line is a comment: {regions:?}"
        );
    }

    #[test]
    fn string_literal_is_string_role() {
        let (ss, _) = assets();
        let syntax = ss.find_syntax_by_token("rust").expect("rust syntax");
        let mut hl = SyntaxHighlighter::new(syntax);
        let regions = hl.highlight_line("let s = \"hello\";\n", ss);
        // The `"hello"` (quotes included) must classify as a string, not plain.
        assert!(
            regions
                .iter()
                .any(|(role, seg)| *role == SyntaxRole::Str && seg.contains("hello")),
            "the quoted body is a string: {regions:?}"
        );
    }

    #[test]
    fn operators_stay_plain_not_keyword() {
        let (ss, _) = assets();
        let syntax = ss.find_syntax_by_token("rust").expect("rust syntax");
        let mut hl = SyntaxHighlighter::new(syntax);
        let regions = hl.highlight_line("let x = a + b;\n", ss);
        // The `+` operator must not be painted as a keyword (noise-reduction).
        assert!(
            regions
                .iter()
                .all(|(role, seg)| !seg.contains('+') || *role == SyntaxRole::Plain),
            "operators recede to Plain: {regions:?}"
        );
    }

    #[test]
    fn regions_concatenate_back_to_line() {
        let (ss, _) = assets();
        let syntax = ss.find_syntax_by_token("rust").expect("rust syntax");
        let mut hl = SyntaxHighlighter::new(syntax);
        let line = "fn main() { let x = 42; }\n";
        let joined: String = hl
            .highlight_line(line, ss)
            .into_iter()
            .map(|(_, seg)| seg)
            .collect();
        assert_eq!(joined, line, "no bytes lost or reordered");
    }
}
