//! Diff-hygiene scoring for the agent eval harness.
//!
//! A real-agent benchmark run is only "clean" if the agent changed **exactly**
//! the files the task called for — and nothing else. In practice an agent
//! runtime can leak artifacts into the target repo (session transcripts,
//! sandbox scratch dirs, caches), which show up in `git status` and make the
//! diff impossible to review. This module turns a `git status --porcelain`
//! dump into a structured verdict so the harness can *score* hygiene instead
//! of eyeballing it.
//!
//! This is the judgment the native runner ([`crate::runner`]) applies after
//! running an agent against a fixture repo. It is pure (no IO) so the scoring
//! rule itself is unit-tested and can never silently drift.

use serde::Serialize;

/// Path prefixes that indicate runtime pollution — artifacts a non-interactive
/// agent run must never leave inside the target repository. Kept in sync with
/// the locations the Zo runtime is expected to keep *out of tree*:
/// out-of-tree session persistence ([`.zo/`]) and external sandbox scratch
/// dirs (`.sandbox-home`/`.sandbox-tmp`).
pub const POLLUTION_MARKERS: &[&str] = &[
    ".zo/",
    ".zo-todos.json",
    ".sandbox-home",
    ".sandbox-tmp",
];

/// One entry from `git status --porcelain`, classified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatusEntry {
    /// Repo-relative path reported by git.
    pub path: String,
    /// `true` when the path matches the task's intended change set.
    pub intended: bool,
    /// `true` when the path matches a known [`POLLUTION_MARKERS`] prefix.
    pub pollution: bool,
}

/// Structured hygiene verdict for a single agent run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiffHygiene {
    /// Every changed path git reported, classified.
    pub entries: Vec<StatusEntry>,
    /// Count of paths that match the intended change set.
    pub intended_count: usize,
    /// Paths that are neither intended nor recognized pollution — still a
    /// hygiene miss (the agent touched something the task didn't ask for).
    pub unexpected: Vec<String>,
    /// Paths that match a known runtime-pollution marker.
    pub pollution: Vec<String>,
    /// `true` only when every changed path was intended (no pollution, no
    /// surprises). This is the benchmark's "clean diff" success criterion.
    pub clean: bool,
}

/// Outcome of the task's verification command for one agent run, mirroring the
/// shell harness's `test` field (`pass` / `fail` / `skipped`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TestStatus {
    /// No `--test` command was provided, so verification was skipped.
    #[default]
    Skipped,
    /// The test command exited zero.
    Pass,
    /// The test command exited non-zero.
    Fail,
}

/// Score the hygiene of a working tree given a `git status --porcelain` dump
/// and the set of paths the task was expected to change.
///
/// `intended` paths are matched both exactly and as a prefix (so a task that
/// targets `src/` accepts `src/pricing.js`). A run is [`clean`](DiffHygiene::clean)
/// only when there are zero pollution markers and zero unexpected paths.
#[must_use]
pub fn score(porcelain: &str, intended: &[&str]) -> DiffHygiene {
    let mut entries = Vec::new();
    let mut unexpected = Vec::new();
    let mut pollution = Vec::new();
    let mut intended_count = 0;

    for path in porcelain.lines().filter_map(parse_porcelain_path) {
        let is_intended = intended.iter().any(|target| path_matches(&path, target));
        let is_pollution = POLLUTION_MARKERS
            .iter()
            .any(|marker| path_matches(&path, marker));

        if is_intended {
            intended_count += 1;
        } else if is_pollution {
            pollution.push(path.clone());
        } else {
            unexpected.push(path.clone());
        }

        entries.push(StatusEntry {
            path,
            intended: is_intended,
            pollution: is_pollution && !is_intended,
        });
    }

    let clean = pollution.is_empty() && unexpected.is_empty();
    DiffHygiene {
        entries,
        intended_count,
        unexpected,
        pollution,
        clean,
    }
}

/// Decide whether a single agent run *fully* passed.
///
/// Diff hygiene (`DiffHygiene::clean`) is necessary but not sufficient. A run
/// passes only when **all** of the following hold:
/// - the agent command exited zero;
/// - the task's test did not fail (a skipped test — no `--test` — is fine);
/// - the diff is clean ([`DiffHygiene::clean`]);
/// - when the task declared an intended change set (`intended_provided`), at
///   least one intended path actually changed — a clean *no-op* is not a
///   successful coding task;
/// - no permission denial was *blocking* ([`permission_denial_fatal`]).
///
/// A permission denial is **not** an automatic failure. A runner can refuse one
/// tool call (e.g. a `Read` it was not granted) and still complete the task — it
/// found another route, or the refusal was incidental. Such a *recovered* denial
/// must not sink an otherwise-passing run; only a *blocking* one does, and a
/// blocking denial always coincides with a failing test or a no-op, so the two
/// objective clauses above already account for it (the explicit
/// `permission_denial_fatal` term states that intent at the gate). The denial is
/// still surfaced — as a [`warnings`] entry when recovered, as a `fail_reasons`
/// entry when fatal.
///
/// This is the rule the native runner ([`crate::runner`]) applies for each run's
/// `pass`, which the suite aggregates into `all_passed` — kept here, like the
/// `clean` rule above, unit-tested as the single definition of a full pass.
#[must_use]
pub fn run_passed(
    exit_code: i32,
    test: TestStatus,
    hygiene: &DiffHygiene,
    permission_denials: usize,
    intended_provided: bool,
) -> bool {
    exit_code == 0
        && test != TestStatus::Fail
        && hygiene.clean
        && (!intended_provided || hygiene.intended_count > 0)
        && !permission_denial_fatal(permission_denials, test, hygiene, intended_provided)
}

/// Whether a permission denial was *blocking* — fatal to the run — as opposed to
/// one the agent recovered from.
///
/// A runner can refuse a tool call (`permission_denials > 0`) and still finish
/// the task. By itself that refusal is **recovered** and must not sink an
/// otherwise-passing run (it is surfaced via [`warnings`] instead). A denial is
/// **fatal** only when it coincides with objective evidence that the refusal
/// actually blocked the work:
/// - the verification test was left failing (`test == Fail`), or
/// - an intended change set was declared and none of it landed
///   (`intended_provided && intended_count == 0`).
///
/// With no denial (`permission_denials == 0`) this is always `false`. This is
/// the single definition both [`run_passed`] and [`fail_reasons`] consult, so
/// the fatal-vs-recovered split can never disagree between the verdict and its
/// explanation.
#[must_use]
pub fn permission_denial_fatal(
    permission_denials: usize,
    test: TestStatus,
    hygiene: &DiffHygiene,
    intended_provided: bool,
) -> bool {
    permission_denials > 0
        && (test == TestStatus::Fail || (intended_provided && hygiene.intended_count == 0))
}

/// Non-fatal advisories for a run — observations worth surfacing that do **not**
/// change the verdict. Orthogonal to [`fail_reasons`]: a run can pass *with*
/// warnings, and (by construction) the two lists never share an entry.
///
/// The only warning today is `permission_denied_recovered`: the runner refused a
/// tool call but still completed the task, so the denial was not
/// [`permission_denial_fatal`]. It keeps the refusal visible — a passing run that
/// quietly worked around a denial must not hide it — without misreporting the run
/// as failed. The native runner ([`crate::runner`]) surfaces this into each
/// run's `warnings` array.
#[must_use]
pub fn warnings(
    permission_denials: usize,
    test: TestStatus,
    hygiene: &DiffHygiene,
    intended_provided: bool,
) -> Vec<&'static str> {
    let mut warnings = Vec::new();
    if permission_denials > 0
        && !permission_denial_fatal(permission_denials, test, hygiene, intended_provided)
    {
        warnings.push("permission_denied_recovered");
    }
    warnings
}

/// Compact, machine-comparable list of *why* a run did not fully pass — the
/// inverse view of [`run_passed`]. It is empty exactly when the run fully
/// passed, so `fail_reasons(..).is_empty() == run_passed(..)` always holds
/// (asserted in tests).
///
/// The order is canonical and stable so two runs are directly comparable:
/// agent failure, then a *blocking* permission denial, then test failure, then a
/// no-op (`no_intended_changes`), then unexpected edits (`dirty_diff`), then
/// runtime `pollution`. `permission_denied` appears only for a
/// [`permission_denial_fatal`] denial — a *recovered* denial is a [`warnings`]
/// entry, never a failure reason — which is why it always travels alongside the
/// `test_failed` / `no_intended_changes` reason that made it fatal. `dirty_diff`
/// and `pollution` are orthogonal — a diff can carry unexpected source edits,
/// leaked artifacts, or both (`["dirty_diff", "pollution"]`). This is the rule
/// the native runner ([`crate::runner`]) applies for each run's `fail_reasons`
/// array — kept here, unit-tested, as the single definition.
#[must_use]
pub fn fail_reasons(
    exit_code: i32,
    test: TestStatus,
    hygiene: &DiffHygiene,
    permission_denials: usize,
    intended_provided: bool,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if exit_code != 0 {
        reasons.push("agent_failed");
    }
    if permission_denial_fatal(permission_denials, test, hygiene, intended_provided) {
        reasons.push("permission_denied");
    }
    if test == TestStatus::Fail {
        reasons.push("test_failed");
    }
    if intended_provided && hygiene.intended_count == 0 {
        reasons.push("no_intended_changes");
    }
    if !hygiene.unexpected.is_empty() {
        reasons.push("dirty_diff");
    }
    if !hygiene.pollution.is_empty() {
        reasons.push("pollution");
    }
    reasons
}

/// Extract the repo-relative path from one porcelain line.
///
/// Porcelain format is `XY <path>` (e.g. `?? .zo/`, ` M src/cache.js`,
/// `A  docs/design.md`). Rename lines (`R  old -> new`) report the destination.
fn parse_porcelain_path(line: &str) -> Option<String> {
    if line.len() < 4 {
        return None;
    }
    // Columns 0..2 are the status code, column 2 is a space, path starts at 3.
    let rest = line.get(3..)?.trim();
    let path = rest.rsplit(" -> ").next().unwrap_or(rest);
    let path = path.trim_matches('"');
    (!path.is_empty()).then(|| path.to_string())
}

/// A changed `path` matches `target` when it equals it, or sits underneath it
/// (prefix match on a directory-like target).
fn path_matches(path: &str, target: &str) -> bool {
    if path == target {
        return true;
    }
    // Mirror the shell `is_intended` exactly: an exact match, a match against
    // the slash-stripped prefix, or a path sitting *under* that prefix on a `/`
    // boundary. Deliberately NO bare `path.starts_with(target)` — that would
    // accept a partial, non-boundary prefix (`src/pric` matching
    // `src/pricing.js`) that the shell rejects, silently diverging the
    // `intended_count`-driven `no_intended_changes` gate between the two.
    let prefix = target.strip_suffix('/').unwrap_or(target);
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

/// Render a compact summary line for logs / CLI output.
#[must_use]
pub fn summarize(hygiene: &DiffHygiene) -> String {
    if hygiene.clean {
        format!(
            "clean ✓ ({} intended file{} changed, no pollution)",
            hygiene.intended_count,
            if hygiene.intended_count == 1 { "" } else { "s" }
        )
    } else {
        format!(
            "DIRTY ✗ ({} intended, {} polluting, {} unexpected)",
            hygiene.intended_count,
            hygiene.pollution.len(),
            hygiene.unexpected.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        fail_reasons, permission_denial_fatal, run_passed, score, summarize, warnings, TestStatus,
        POLLUTION_MARKERS,
    };

    #[test]
    fn intended_only_change_is_clean() {
        let porcelain = " M src/pricing.js\n";
        let hygiene = score(porcelain, &["src/pricing.js"]);
        assert!(hygiene.clean, "an intended-only diff must score clean");
        assert_eq!(hygiene.intended_count, 1);
        assert!(hygiene.pollution.is_empty());
        assert!(hygiene.unexpected.is_empty());
    }

    #[test]
    fn zo_session_pollution_fails_hygiene() {
        // The exact benchmark finding: agent changed the right file but the
        // runtime also leaked `.zo/` and the sandbox scratch dirs. Written
        // with explicit `\n` so the leading porcelain status space survives
        // (a `\`-continuation would strip it).
        let porcelain = " M src/cache.js\n?? .zo/\n?? .sandbox-home/\n?? .sandbox-tmp/\n";
        let hygiene = score(porcelain, &["src/cache.js"]);
        assert!(!hygiene.clean, "leaked runtime artifacts must fail hygiene");
        assert_eq!(hygiene.intended_count, 1);
        assert_eq!(hygiene.pollution.len(), 3, "{:?}", hygiene.pollution);
        assert!(hygiene.pollution.iter().any(|p| p.starts_with(".zo")));
        assert!(hygiene.unexpected.is_empty(), "all three are known markers");
    }

    #[test]
    fn directory_target_accepts_nested_paths() {
        let porcelain = "A  docs/design.md\n";
        let hygiene = score(porcelain, &["docs/"]);
        assert!(hygiene.clean);
        assert_eq!(hygiene.intended_count, 1);
    }

    #[test]
    fn intended_match_respects_path_boundaries_like_the_shell() {
        // A partial, non-boundary prefix must NOT match — this is exactly where
        // the Rust mirror used to diverge from the shell `is_intended` (which
        // only matches an exact path or a `prefix/` boundary). Since
        // intended_count now gates `no_intended_changes`, a mismatch here would
        // flip the pass verdict between the two implementations.
        let porcelain = " M src/pricing.js\n";
        // `src/pric` is a partial prefix, not a path boundary: not intended.
        let partial = score(porcelain, &["src/pric"]);
        assert_eq!(partial.intended_count, 0, "partial prefix must not match");
        assert_eq!(partial.unexpected, vec!["src/pricing.js".to_string()]);
        // The documented forms still match: exact file and `dir/` prefix.
        assert_eq!(score(porcelain, &["src/pricing.js"]).intended_count, 1);
        assert_eq!(score(porcelain, &["src/"]).intended_count, 1);
        assert_eq!(score(porcelain, &["src"]).intended_count, 1);
    }

    #[test]
    fn unexpected_source_edit_is_flagged_even_without_pollution() {
        let porcelain = " M src/pricing.js\n M src/unrelated.js\n";
        let hygiene = score(porcelain, &["src/pricing.js"]);
        assert!(!hygiene.clean);
        assert_eq!(hygiene.unexpected, vec!["src/unrelated.js".to_string()]);
        assert!(hygiene.pollution.is_empty());
    }

    #[test]
    fn rename_lines_report_destination_path() {
        let porcelain = "R  old.js -> src/pricing.js\n";
        let hygiene = score(porcelain, &["src/pricing.js"]);
        assert!(hygiene.clean, "rename destination should match intended");
    }

    #[test]
    fn summary_distinguishes_clean_from_dirty() {
        let clean = score(" M a.js\n", &["a.js"]);
        assert!(summarize(&clean).contains("clean"));
        let dirty = score("?? .zo/\n", &["a.js"]);
        assert!(summarize(&dirty).contains("DIRTY"));
    }

    #[test]
    fn pollution_markers_cover_the_benchmark_artifacts() {
        for marker in [".zo/", ".sandbox-home", ".sandbox-tmp"] {
            assert!(
                POLLUTION_MARKERS.contains(&marker),
                "{marker} must be a tracked pollution marker"
            );
        }
    }

    #[test]
    fn full_pass_requires_clean_exit_passing_test_and_clean_diff() {
        let clean = score(" M src/pricing.js\n", &["src/pricing.js"]);
        assert!(run_passed(0, TestStatus::Pass, &clean, 0, true));
        // A skipped test (no --test given) is an acceptable pass.
        assert!(run_passed(0, TestStatus::Skipped, &clean, 0, true));
        // With no intended set declared, the no-op rule simply does not apply.
        assert!(run_passed(0, TestStatus::Skipped, &clean, 0, false));
    }

    #[test]
    fn failing_test_sinks_the_run_even_with_a_clean_diff() {
        // The exact harness bug: exit 0 and a clean diff, but the test failed.
        // Diff hygiene alone must never be allowed to report success.
        let clean = score(" M src/pricing.js\n", &["src/pricing.js"]);
        assert!(!run_passed(0, TestStatus::Fail, &clean, 0, true));
    }

    #[test]
    fn nonzero_exit_or_dirty_diff_sinks_the_run() {
        let clean = score(" M src/pricing.js\n", &["src/pricing.js"]);
        let dirty = score(" M src/pricing.js\n?? .zo/\n", &["src/pricing.js"]);
        assert!(
            !run_passed(1, TestStatus::Pass, &clean, 0, true),
            "agent command failed"
        );
        assert!(
            !run_passed(0, TestStatus::Pass, &dirty, 0, true),
            "diff was dirty"
        );
        assert!(
            !run_passed(0, TestStatus::Skipped, &dirty, 0, true),
            "dirty + skipped"
        );
    }

    #[test]
    fn recovered_permission_denial_does_not_sink_a_completed_run() {
        // The false-negative this rule fixes: a runner refused one tool call
        // (e.g. a `Read` it was not granted) yet still made the intended edit,
        // with a clean diff and a passing/skipped test. The refusal was recovered
        // from — it must NOT sink the run. It is surfaced as a non-fatal warning,
        // never a failure reason, so the pass↔fail_reasons invariant still holds.
        let clean = score(" M src/pricing.js\n", &["src/pricing.js"]);
        assert!(run_passed(0, TestStatus::Pass, &clean, 1, true));
        assert!(run_passed(0, TestStatus::Skipped, &clean, 1, true));
        assert!(fail_reasons(0, TestStatus::Pass, &clean, 1, true).is_empty());
        assert!(!permission_denial_fatal(1, TestStatus::Pass, &clean, true));
        assert_eq!(
            warnings(1, TestStatus::Pass, &clean, true),
            vec!["permission_denied_recovered"]
        );
        // No denial ⇒ no warning, no fatality.
        assert!(warnings(0, TestStatus::Pass, &clean, true).is_empty());
        assert!(!permission_denial_fatal(0, TestStatus::Pass, &clean, true));
    }

    #[test]
    fn blocking_permission_denial_sinks_the_run_and_is_reported() {
        // A denial is fatal only when it coincides with objective evidence the
        // work was blocked: a failing test, or a declared intended set that did
        // not change. In both cases `permission_denied` joins the coincident
        // reason (in canonical order) and is NOT downgraded to a warning.
        let clean = score(" M src/pricing.js\n", &["src/pricing.js"]);
        // Denial + failing test: the edit landed but left the test red.
        assert!(permission_denial_fatal(1, TestStatus::Fail, &clean, true));
        assert!(!run_passed(0, TestStatus::Fail, &clean, 1, true));
        assert_eq!(
            fail_reasons(0, TestStatus::Fail, &clean, 1, true),
            vec!["permission_denied", "test_failed"]
        );
        assert!(warnings(1, TestStatus::Fail, &clean, true).is_empty());

        // Denial + no intended change: the refusal blocked the only edit.
        let empty = score("", &["src/pricing.js"]);
        assert!(permission_denial_fatal(
            1,
            TestStatus::Skipped,
            &empty,
            true
        ));
        assert!(!run_passed(0, TestStatus::Skipped, &empty, 1, true));
        assert_eq!(
            fail_reasons(0, TestStatus::Skipped, &empty, 1, true),
            vec!["permission_denied", "no_intended_changes"]
        );
        assert!(warnings(1, TestStatus::Skipped, &empty, true).is_empty());
    }

    #[test]
    fn no_intended_change_is_a_failed_no_op_when_intended_was_declared() {
        // Agent exited clean with a spotless diff but changed none of the
        // declared intended files — a no-op, not a success.
        let empty = score("", &["src/pricing.js"]);
        assert_eq!(empty.intended_count, 0);
        assert!(empty.clean, "an empty diff is trivially clean");
        assert!(!run_passed(0, TestStatus::Skipped, &empty, 0, true));
        assert_eq!(
            fail_reasons(0, TestStatus::Skipped, &empty, 0, true),
            vec!["no_intended_changes"]
        );
        // ...but with no intended set declared, a no-op diff is not penalized.
        assert!(run_passed(0, TestStatus::Skipped, &empty, 0, false));
        assert!(fail_reasons(0, TestStatus::Skipped, &empty, 0, false).is_empty());
    }

    #[test]
    fn fail_reasons_reports_each_distinct_failure() {
        // Each scenario keeps an intended edit present (intended_count == 1) so
        // the no-op rule does not fire and we isolate the one reason under test.
        let clean = score(" M a.js\n", &["a.js"]);
        assert_eq!(
            fail_reasons(1, TestStatus::Pass, &clean, 0, true),
            vec!["agent_failed"]
        );
        // A denial can no longer be isolated as a standalone reason: it is a
        // failure only when blocking, and a blocking denial always travels with
        // the test_failed / no_intended_changes that made it fatal (asserted in
        // blocking_permission_denial_sinks_the_run_and_is_reported). On its own —
        // clean exit, clean diff, intended edit present — it is recovered, so it
        // contributes no failure reason at all.
        assert!(fail_reasons(0, TestStatus::Pass, &clean, 2, true).is_empty());
        assert_eq!(
            fail_reasons(0, TestStatus::Fail, &clean, 0, true),
            vec!["test_failed"]
        );

        let polluted = score(" M a.js\n?? .zo/\n", &["a.js"]);
        assert_eq!(
            fail_reasons(0, TestStatus::Pass, &polluted, 0, true),
            vec!["pollution"]
        );

        let unexpected = score(" M a.js\n M src/other.js\n", &["a.js"]);
        assert_eq!(
            fail_reasons(0, TestStatus::Pass, &unexpected, 0, true),
            vec!["dirty_diff"]
        );
    }

    #[test]
    fn fail_reasons_keeps_dirty_diff_and_pollution_orthogonal_and_ordered() {
        // Both an unexpected source edit and a leaked artifact: the example from
        // the harness contract, in canonical order. The intended edit (a.js) is
        // present so only the orthogonal pair surfaces.
        let both = score(" M a.js\n M src/other.js\n?? .zo/\n", &["a.js"]);
        assert_eq!(
            fail_reasons(0, TestStatus::Pass, &both, 0, true),
            vec!["dirty_diff", "pollution"]
        );
    }

    #[test]
    fn fail_reasons_empty_exactly_when_run_passed() {
        // The load-bearing invariant: the reason list and the boolean gate are
        // two views of one rule, so they can never disagree — across every
        // exit / test / denial / intended-provided / hygiene combination.
        let clean = score(" M a.js\n", &["a.js"]); // intended_count == 1
        let dirty = score("?? .zo/\n", &["a.js"]); // intended_count == 0, polluted
        for exit in [0, 1] {
            for test in [TestStatus::Pass, TestStatus::Fail, TestStatus::Skipped] {
                for denials in [0_usize, 3] {
                    for intended_provided in [false, true] {
                        for hygiene in [&clean, &dirty] {
                            let reasons =
                                fail_reasons(exit, test, hygiene, denials, intended_provided);
                            let passed =
                                run_passed(exit, test, hygiene, denials, intended_provided);
                            assert_eq!(
                                reasons.is_empty(),
                                passed,
                                "exit={exit} test={test:?} denials={denials} \
                                 intended_provided={intended_provided} clean={} reasons={reasons:?}",
                                hygiene.clean
                            );
                            // Warnings are orthogonal to failure reasons: the two
                            // lists never share an entry. A recovered denial shows
                            // up as a warning, never a reason; a warning can still
                            // coexist with a *failed* run when the failure is
                            // unrelated to the (recovered) denial.
                            let warns = warnings(denials, test, hygiene, intended_provided);
                            for w in &warns {
                                assert!(
                                    !reasons.contains(w),
                                    "warning {w:?} must not also be a failure reason \
                                     (exit={exit} test={test:?} denials={denials} \
                                      intended_provided={intended_provided})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
