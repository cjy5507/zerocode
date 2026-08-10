//! Persist oversized inline `python3 - <<EOF` verification scripts to a file.
//!
//! Long sessions grow a habit: every stage the model writes an ad-hoc check
//! script as an inline heredoc, runs it, an `assert` fails, and — because an
//! inline heredoc has no address, nothing to point `edit_file` at — it retypes
//! the *entire* script to change one line. Measured on a deep bench task: 15.6k
//! characters (~4k output tokens) of near-duplicate re-emission
//! (jaccard 0.77–0.98) in a single task.
//!
//! The harness cannot stop the model from writing heredocs, and should not: an
//! inline script is the right shape for a one-shot check. What it can do is
//! give the script an *address*. This module writes the heredoc body to a file
//! next to the session's other machine-generated state and reports the path in
//! that call's tool result, which turns the retry from "retype 2–5k chars" into
//! `edit_file` + `python3 <path>`.
//!
//! Everything here is an amenity: the command has already run (or is about to),
//! and no failure in this module may change its stdout, stderr, exit status, or
//! whether it ran at all. Every fallible step therefore degrades to `None`,
//! which simply omits the note.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

/// Smallest heredoc body that earns a file.
///
/// The affordance is only worth its one line of tool result when retyping the
/// script would actually be expensive. Short heredocs (`print(1+1)`) are
/// cheaper to retype than to open, and every call below this bound stays
/// byte-for-byte identical to what it was before this module existed.
const MIN_BODY_CHARS: usize = 800;

/// Upper bound on saved scripts per session directory. Purely a loop bound for
/// [`write_next_script`], not a policy: a session that legitimately writes a
/// thousand checks simply stops getting the note (and keeps running normally).
const MAX_SCRIPTS_PER_SESSION: u32 = 999;

/// How long an untouched saved script survives before the sweep removes it.
///
/// Same value, same policy and same trigger shape as the REPL kernel's
/// `SNAPSHOT_KEEP` (`tools::repl_kernel`): both are session-scoped, machine
/// generated, disposable state under the zo config home, so they age out the
/// same way rather than by a second invented rule.
const SCRIPT_KEEP: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Root override, mirroring the `ZO_SANDBOX_DIR` / `ZO_ARTIFACT_STORE` family.
///
/// Deliberately *not* added to `bash::STRIPPED_SESSION_SCOPE_ENV`: unlike
/// `ZO_TODO_STORE` this names a root, not one session's file, so a nested `zo`
/// that inherits it lands in its own session subdirectory underneath — the
/// inheritance bug that motivated the strip list cannot occur here.
///
/// `cfg(not(test))` because the in-crate tests reach the root through the
/// thread-local seam instead — a process-global env var would have parallel
/// tests overwriting each other's directory.
#[cfg(not(test))]
const CHECK_SCRIPT_DIR_ENV: &str = "ZO_CHECK_SCRIPT_DIR";

/// The largest saveable `python … <<DELIM` heredoc body in `command`.
///
/// "Saveable" means: a heredoc (not a `<<<` here-string) whose command word is
/// a Python interpreter, whose terminator is present, and whose body clears
/// [`MIN_BODY_CHARS`]. Position in a pipeline is irrelevant — the opener is
/// identified by the word immediately before `<<`, so `cat x | python3 - <<PY`
/// and `python3 - <<PY | tee log` both qualify.
///
/// When several qualify, the largest wins: that is the one whose re-emission
/// costs the most, and saving exactly one keeps both the counter and the note
/// unambiguous.
pub(crate) fn saveable_python_heredoc(command: &str) -> Option<&str> {
    // Fast reject. The scan below is linear, but the overwhelming majority of
    // bash calls contain no heredoc at all and must pay nothing beyond this.
    if !command.contains("<<") {
        return None;
    }

    let mut best: Option<&str> = None;
    let mut cursor = 0usize;
    while let Some(offset) = command[cursor..].find("<<") {
        let operator = cursor + offset;
        // `<<<` is a here-string: no body at all, and its leading `<<` must not
        // be mistaken for a heredoc opener.
        if command[operator + 2..].starts_with('<') {
            cursor = operator + 3;
            continue;
        }
        let Some((body, resume)) = heredoc_body(command, operator + 2) else {
            cursor = operator + 2;
            continue;
        };
        // Resume past the whole heredoc whichever interpreter opened it: a
        // `<<` *inside* another command's body is data, not an opener.
        cursor = resume;
        if !opener_is_python(&command[..operator]) || body.chars().count() < MIN_BODY_CHARS {
            continue;
        }
        if best.is_none_or(|current| body.len() > current.len()) {
            best = Some(body);
        }
    }
    best
}

/// Save `body` for `session_id` and return the one-line note for the tool
/// result, or `None` when anything at all went wrong.
///
/// `None` is the whole error strategy. A read-only home, a full disk, a
/// permission error — none of them may cost the caller its command output, so
/// the failure mode is silence, not a warning the model would have to reason
/// about.
pub(crate) fn save_and_describe(body: &str, session_id: Option<&str>) -> Option<String> {
    let dir = session_scripts_dir(session_id)?;
    std::fs::create_dir_all(&dir).ok()?;
    // Best-effort: the scripts are the model's own scratch, but they can quote
    // repository contents, so they get the same owner-only treatment as the
    // rest of the config home.
    let _ = core_types::paths::restrict_permissions_owner_only(&dir);
    let path = write_next_script(&dir, body)?;
    Some(note_for(&path))
}

/// The one line appended to the tool result.
///
/// Two facts and nothing else: where the body went, and what to do with it
/// next. An auto-firing affordance that narrates itself would spend on every
/// call the tokens it exists to save.
fn note_for(path: &Path) -> String {
    let path = path.display();
    format!(
        "saved this heredoc to {path} — re-run it with `python3 {path}`, and edit_file that path instead of re-typing the script"
    )
}

/// Claim the next free `check-N.py` in `dir`.
///
/// The counter is the file namespace itself: `create_new` fails on a name that
/// already exists, so the first success is the next index. That keeps numbering
/// deterministic and gap-free across process restarts and `/resume` (an
/// in-process counter would restart at 1 and overwrite the earlier session's
/// scripts) without a readdir, and it is race-free against a concurrent agent
/// in the same session because the exclusive create *is* the arbitration.
fn write_next_script(dir: &Path, body: &str) -> Option<PathBuf> {
    for index in 1..=MAX_SCRIPTS_PER_SESSION {
        let path = dir.join(format!("check-{index}.py"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                if file.write_all(body.as_bytes()).is_err() {
                    // Never leave a half-written script wearing a name the note
                    // would advertise as runnable.
                    let _ = std::fs::remove_file(&path);
                    return None;
                }
                return Some(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return None,
        }
    }
    None
}

/// Where saved scripts live: `<config home>/check-scripts/<session>/`.
///
/// A top-level sibling of `~/.zo/kernel-snapshots`, the closest existing
/// precedent (session-keyed, machine-generated, disposable, 30-day sweep). The
/// two alternatives were both rejected on evidence:
///
/// - `~/.zo/projects/<slug>/sessions/` (where `.todos.json` /
///   `.file-reads.json` / `.system-prompt.json` live) is swept by
///   `session_control::cleanup_expired_sessions_under`, which selects on the
///   `json`/`jsonl` *extension* — a `.py` there would never be collected — and
///   counts every file it removes as a removed **session**, so parking foreign
///   files in it corrupts a user-facing number. A new subdirectory under
///   `<slug>/` is worse still: that sweep enumerates only hardcoded child names
///   and then calls `remove_dir` on the slug directory, which a stray sibling
///   would make fail forever.
/// - The OS temp dir (where `sandbox_scratch_dirs` lives) reaps on an idle
///   timer measured in days and is re-pointed by `TMPDIR` inside the sandbox,
///   so the absolute path in the note would mean different things to the model
///   and to the shell it hands the path to.
fn session_scripts_dir(session_id: Option<&str>) -> Option<PathBuf> {
    let root = scripts_root()?;
    prune_stale_scripts(&root);
    Some(root.join(session_scope(session_id)))
}

// The `Option` is the test seam's contract: under `cfg(test)` this is `None`
// until a test opts in, so the production arm alone looks unnecessarily
// wrapped. Same shape and same allow as `repl_kernel::snapshot_dir`.
#[allow(clippy::unnecessary_wraps)]
fn scripts_root() -> Option<PathBuf> {
    #[cfg(test)]
    {
        // Off in tests unless a test opts in with a directory of its own —
        // the production default would have the suite writing into the real
        // config home, the cross-run contamination the kernel snapshot seam
        // exists to avoid.
        ROOT_OVERRIDE.with(|slot| slot.borrow().clone())
    }
    #[cfg(not(test))]
    {
        if let Some(root) = std::env::var_os(CHECK_SCRIPT_DIR_ENV) {
            return Some(PathBuf::from(root));
        }
        Some(core_types::paths::default_config_home().join("check-scripts"))
    }
}

/// Directory name for one session.
///
/// Sanitized rather than hashed, like `repl_kernel::kernel_snapshot_path`, so
/// the mapping stays stable across processes and the directory is readable on
/// disk. A session-less caller (`execute_bash`, workflow checks) still gets
/// scripts, bucketed per process so two of them cannot collide.
fn session_scope(session_id: Option<&str>) -> String {
    match session_id {
        Some(id) => {
            let safe: String = id
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                        c
                    } else {
                        '_'
                    }
                })
                .take(96)
                .collect();
            if safe.is_empty() {
                format!("pid-{}", std::process::id())
            } else {
                safe
            }
        }
        None => format!("pid-{}", std::process::id()),
    }
}

/// Once per process: drop scripts idle past [`SCRIPT_KEEP`].
///
/// Deliberately not a new retention scheme — the same 30-day idle rule, the
/// same lazy once-per-process `OnceLock` trigger and the same "best effort,
/// never fails the caller" shape as `repl_kernel::prune_stale_snapshots`,
/// applied one directory level deeper because scripts are grouped per session.
fn prune_stale_scripts(root: &Path) {
    static SWEPT: OnceLock<()> = OnceLock::new();
    SWEPT.get_or_init(|| sweep_root(root, SCRIPT_KEEP));
}

fn sweep_root(root: &Path, keep: Duration) {
    let Ok(sessions) = std::fs::read_dir(root) else {
        return;
    };
    for session in sessions.flatten() {
        let dir = session.path();
        if let Ok(scripts) = std::fs::read_dir(&dir) {
            for script in scripts.flatten() {
                if is_idle_past(&script, keep) {
                    let _ = std::fs::remove_file(script.path());
                }
            }
        }
        // Non-recursive on purpose: this fails harmlessly while any script
        // remains, so the sweep can never take out a session still in use.
        let _ = std::fs::remove_dir(&dir);
    }
}

fn is_idle_past(entry: &std::fs::DirEntry, keep: Duration) -> bool {
    entry
        .metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age > keep)
}

/// Parse the heredoc opened at `after_operator` (the byte just past `<<`).
///
/// Returns the body and the offset to resume scanning from — the end of the
/// terminator line. An unterminated heredoc yields `None`: the shell rejects it
/// as a syntax error, so there is no program there to save.
fn heredoc_body(command: &str, after_operator: usize) -> Option<(&str, usize)> {
    let rest = &command[after_operator..];
    let mut index = 0usize;
    // `<<-` strips leading tabs from the body *and* from the terminator line.
    let strip_tabs = rest.as_bytes().first() == Some(&b'-');
    if strip_tabs {
        index += 1;
    }
    index += rest[index..].len() - rest[index..].trim_start_matches([' ', '\t']).len();

    let (delimiter, delimiter_len) = parse_delimiter(&rest[index..])?;
    index += delimiter_len;

    // The body starts on the line after the one carrying the operator, so a
    // trailing pipeline (`… <<PY | tee log`) is skipped along with it.
    let body_start = rest[index..].find('\n')? + index + 1;
    let mut line_start = body_start;
    loop {
        let line_end = rest[line_start..]
            .find('\n')
            .map_or(rest.len(), |offset| line_start + offset);
        let mut line = &rest[line_start..line_end];
        if strip_tabs {
            line = line.trim_start_matches('\t');
        }
        if line.trim_end_matches('\r') == delimiter {
            return Some((&rest[body_start..line_start], after_operator + line_end));
        }
        if line_end >= rest.len() {
            return None;
        }
        line_start = line_end + 1;
    }
}

/// Read the heredoc delimiter and return it with its consumed length.
///
/// Quoted (`<<'PY'`, `<<"PY"`) and bare (`<<PY`) forms only. The quotes decide
/// whether the shell expands the body, which changes nothing here: either way
/// the bytes zo saves are the bytes zo was handed.
fn parse_delimiter(text: &str) -> Option<(&str, usize)> {
    let quote = text.chars().next()?;
    if quote == '\'' || quote == '"' {
        let end = text[1..].find(quote)? + 1;
        return (end > 1).then(|| (&text[1..end], end + 1));
    }
    let end = text
        .find(|c: char| c.is_whitespace() || "|&;<>()'\"`".contains(c))
        .unwrap_or(text.len());
    (end > 0).then(|| (&text[..end], end))
}

/// Is the word opening this heredoc a Python interpreter?
///
/// The canonical form is `python3 - <<PY` ("read the program from stdin"), and
/// the leading `-` may be preceded by other flags (`python3 -u - <<PY`), so
/// trailing option words are dropped before the command word is read. That
/// cannot widen the trigger to another tool: whatever remains still has to be
/// `python`/`python3`, path-qualified or not.
fn opener_is_python(prefix: &str) -> bool {
    const WORD_BREAKS: [char; 8] = [' ', '\t', '\n', '|', '&', ';', '(', '`'];

    let mut head = prefix.trim_end_matches([' ', '\t']);
    while let Some(token) = head.rsplit([' ', '\t']).next() {
        if !token.starts_with('-') {
            break;
        }
        head = head[..head.len() - token.len()].trim_end_matches([' ', '\t']);
    }
    let word = head.rsplit(WORD_BREAKS).next().unwrap_or_default();
    matches!(
        word.rsplit('/').next().unwrap_or_default(),
        "python" | "python3"
    )
}

#[cfg(test)]
thread_local! {
    static ROOT_OVERRIDE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Test seam: point this thread's saved scripts at `root` (see [`scripts_root`]).
#[cfg(test)]
pub(crate) fn set_root_for_this_thread(root: Option<PathBuf>) {
    ROOT_OVERRIDE.with(|slot| *slot.borrow_mut() = root);
}

#[cfg(test)]
mod tests {
    use super::{
        heredoc_body, opener_is_python, saveable_python_heredoc, session_scope, sweep_root,
        MIN_BODY_CHARS,
    };
    use std::time::Duration;

    /// A python body comfortably over [`MIN_BODY_CHARS`], shaped like the real
    /// thing (imports, asserts) rather than one repeated character.
    fn big_body(marker: &str) -> String {
        let mut body = format!("import sys\nsys.path.insert(0, \".\")\n# {marker}\n");
        while body.chars().count() < MIN_BODY_CHARS + 40 {
            body.push_str("assert 1 + 1 == 2, \"arithmetic still works in this build\"\n");
        }
        body
    }

    #[test]
    fn quoted_heredoc_body_is_extracted_without_the_delimiter_lines() {
        let body = big_body("quoted");
        let command = format!("python3 - <<'EOF'\n{body}EOF\n");
        assert_eq!(saveable_python_heredoc(&command), Some(body.as_str()));
    }

    #[test]
    fn bare_and_double_quoted_delimiters_are_recognized() {
        for opener in ["<<EOF", "<<\"EOF\"", "<< 'EOF'"] {
            let body = big_body("delim");
            let command = format!("python3 - {opener}\n{body}EOF\n");
            assert_eq!(
                saveable_python_heredoc(&command),
                Some(body.as_str()),
                "opener {opener} must parse"
            );
        }
    }

    #[test]
    fn pipelines_on_either_side_do_not_hide_the_opener() {
        let body = big_body("pipeline");
        let piped_in = format!("cat data.json | python3 - <<'PY'\n{body}PY\n");
        let piped_out = format!("python3 - <<'PY' | tee /dev/null\n{body}PY\n");
        assert_eq!(saveable_python_heredoc(&piped_in), Some(body.as_str()));
        assert_eq!(saveable_python_heredoc(&piped_out), Some(body.as_str()));
    }

    #[test]
    fn tab_stripping_heredocs_drop_leading_tabs_from_the_terminator() {
        let body = big_body("dash");
        let command = format!("python3 - <<-'PY'\n{body}\tPY\n");
        assert_eq!(saveable_python_heredoc(&command), Some(body.as_str()));
    }

    #[test]
    fn non_python_interpreters_and_here_strings_never_qualify() {
        let body = big_body("negative");
        for command in [
            format!("node - <<'EOF'\n{body}EOF\n"),
            format!("bash <<'EOF'\n{body}EOF\n"),
            format!("cat <<'EOF' > out.py\n{body}EOF\n"),
            format!("python3 - <<<'{body}'"),
            format!("python3 -c '{body}'"),
        ] {
            assert_eq!(
                saveable_python_heredoc(&command),
                None,
                "must not fire for: {}",
                &command[..command.len().min(24)]
            );
        }
    }

    #[test]
    fn a_short_heredoc_is_cheaper_to_retype_than_to_save() {
        let command = "python3 - <<'EOF'\nprint(1 + 1)\nEOF\n";
        assert_eq!(saveable_python_heredoc(command), None);
    }

    #[test]
    fn an_unterminated_heredoc_has_no_program_to_save() {
        let body = big_body("unterminated");
        let command = format!("python3 - <<'EOF'\n{body}");
        assert_eq!(saveable_python_heredoc(&command), None);
    }

    #[test]
    fn the_largest_of_several_heredocs_wins() {
        let small = big_body("small");
        let mut large = big_body("large");
        large.push_str("# tail padding to make this one the biggest by a clear margin\n");
        let command = format!(
            "python3 - <<'A'\n{small}A\npython3 - <<'B'\n{large}B\npython3 - <<'C'\n{small}C\n"
        );
        assert_eq!(saveable_python_heredoc(&command), Some(large.as_str()));
    }

    #[test]
    fn a_heredoc_nested_in_another_commands_body_is_data_not_an_opener() {
        // The inner `python3 - <<PY` lives inside a `cat` heredoc, so the shell
        // never opens it; the scanner must resume past the outer terminator
        // rather than re-reading the body as shell text.
        let inner = big_body("inner");
        let command = format!("cat <<'OUTER' > note.txt\npython3 - <<'PY'\n{inner}PY\nOUTER\n");
        assert_eq!(saveable_python_heredoc(&command), None);
    }

    #[test]
    fn interpreter_word_matching_tolerates_flags_and_paths_only() {
        assert!(opener_is_python("python3 - "));
        assert!(opener_is_python("python -"));
        assert!(opener_is_python("python3 -u - "));
        assert!(opener_is_python("/usr/bin/python3 - "));
        assert!(opener_is_python("cat x | python3 - "));
        assert!(opener_is_python("python3"));
        assert!(!opener_is_python("node - "));
        assert!(!opener_is_python("mypython3 - "));
        assert!(!opener_is_python("./run-python3-checks - "));
        assert!(!opener_is_python(""));
    }

    #[test]
    fn heredoc_body_reports_where_to_resume_scanning() {
        let command = "python3 - <<'PY'\nbody\nPY\ntrailing";
        let (body, resume) = heredoc_body(command, command.find("<<").unwrap() + 2).unwrap();
        assert_eq!(body, "body\n");
        assert_eq!(&command[resume..], "\ntrailing");
    }

    #[test]
    fn session_scope_sanitizes_and_never_yields_an_empty_directory_name() {
        assert_eq!(session_scope(Some("session-123.abc_X")), "session-123.abc_X");
        // Path separators are neutralized, so no session id can escape the root.
        assert_eq!(session_scope(Some("a/../b")), "a_.._b");
        assert_eq!(session_scope(Some("///")), "___");
        assert!(session_scope(Some("")).starts_with("pid-"));
        assert!(session_scope(None).starts_with("pid-"));
        assert_eq!(session_scope(Some(&"z".repeat(300))).len(), 96);
    }

    #[test]
    fn the_sweep_removes_idle_scripts_and_their_emptied_session_directory() {
        let root = tempfile::tempdir().expect("tempdir");
        let session = root.path().join("session-a");
        std::fs::create_dir_all(&session).expect("session dir");
        std::fs::write(session.join("check-1.py"), "print(1)").expect("script");

        // A keep window far longer than the file has existed: nothing goes.
        sweep_root(root.path(), Duration::from_secs(3600));
        assert!(session.join("check-1.py").exists(), "fresh script must stay");

        // A zero keep window makes every file idle past its bound.
        sweep_root(root.path(), Duration::ZERO);
        assert!(!session.exists(), "emptied session directory must be removed");
    }

    #[test]
    fn the_sweep_survives_a_missing_root_and_stray_files() {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("stray"), "not a session dir").expect("stray");
        sweep_root(&root.path().join("does-not-exist"), Duration::ZERO);
        // A plain file where a session directory was expected must not panic
        // and must not be deleted by `remove_dir`.
        sweep_root(root.path(), Duration::ZERO);
        assert!(root.path().join("stray").exists());
    }
}
