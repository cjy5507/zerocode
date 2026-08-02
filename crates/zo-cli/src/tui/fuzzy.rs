//! Shared fuzzy subsequence matcher for the slash-command surfaces.
//!
//! The inline `/` hint (`tui/app/slash_hint.rs`) and the command palette
//! (`tui/modals/command_palette.rs`) rank commands with the *same* subsequence
//! rule, so a query that finds a command in one surface finds it in the other.
//! Keeping the matcher here is the single source of truth that stops the two
//! search experiences from drifting apart.

/// Return the indices in `haystack` where each `char` of `needle` matches, in
/// order, or `None` when `needle` is not a subsequence of `haystack`.
///
/// Both inputs are expected to be pre-lowercased by the caller (each command
/// surface lowercases once and caches). An empty needle matches everything
/// with no highlighted indices.
pub(crate) fn subsequence_indices(haystack: &str, needle: &str) -> Option<Vec<usize>> {
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.is_empty() {
        return Some(Vec::new());
    }
    let mut indices = Vec::with_capacity(needle_chars.len());
    let mut next = 0;
    for (i, c) in haystack.chars().enumerate() {
        if next < needle_chars.len() && c == needle_chars[next] {
            indices.push(i);
            next += 1;
        }
    }
    (next == needle_chars.len()).then_some(indices)
}

/// `true` when `needle` is a subsequence of `haystack` (both pre-lowercased).
///
/// A thin predicate wrapper for callers that only need a yes/no answer and not
/// the matched positions.
pub(crate) fn is_subsequence(haystack: &str, needle: &str) -> bool {
    subsequence_indices(haystack, needle).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_contiguous_and_gapped_subsequences() {
        assert_eq!(subsequence_indices("commit", "com"), Some(vec![0, 1, 2]));
        // c-o-m-m-i-t → c@0, m@2 (first m), t@5.
        assert_eq!(subsequence_indices("commit", "cmt"), Some(vec![0, 2, 5]));
    }

    #[test]
    fn rejects_out_of_order_or_too_long() {
        assert_eq!(subsequence_indices("commit", "tim"), None);
        assert_eq!(subsequence_indices("abc", "abcd"), None);
    }

    #[test]
    fn empty_needle_matches_with_no_indices() {
        assert_eq!(subsequence_indices("abc", ""), Some(Vec::new()));
    }

    #[test]
    fn is_subsequence_is_a_predicate_view() {
        assert!(is_subsequence("show git diff for changes", "gitdiff"));
        assert!(!is_subsequence("show git diff", "zzz"));
    }
}
