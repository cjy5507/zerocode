//! The running build's identity, shown on the startup splash and in the sidebar.
//!
//! `CARGO_PKG_VERSION` alone cannot answer "is my fix in this binary?": a
//! release and a local build off a later commit both print `0.1.9`. That
//! ambiguity is how a stale copy earlier on `PATH` silently reverted a shipped
//! fix, with the only reliable signal buried in `zo --version`. Both surfaces
//! render the string from here so they can never disagree about it.

/// Version plus the short commit, e.g. `0.1.9 · 3b3014b1`.
///
/// `GIT_SHA` is stamped by this package's build script, which applies to the
/// library target as well, so the sidebar can resolve it without the binary
/// passing it down.
#[must_use]
pub fn label() -> String {
    from_parts(env!("CARGO_PKG_VERSION"), option_env!("GIT_SHA"))
}

/// Pure core of [`label`], split out so the formatting is testable without
/// depending on this build's compile-time `GIT_SHA`.
#[must_use]
pub fn from_parts(version: &str, sha: Option<&str>) -> String {
    let Some(sha) = sha else {
        return version.to_string();
    };
    // `GIT_SHA` carries its own dirty marker, so peel it before truncating —
    // otherwise an 8-char cut silently drops the one bit that says the build
    // matches no commit at all.
    let (commit, dirty) = match sha.strip_suffix("-dirty") {
        Some(commit) => (commit, "-dirty"),
        None => (sha, ""),
    };
    let short = commit.chars().take(8).collect::<String>();
    if short.is_empty() || short == "unknown" {
        return version.to_string();
    }
    format!("{version} \u{00b7} {short}{dirty}")
}

#[cfg(test)]
mod tests {
    use super::{from_parts, label};

    #[test]
    fn shows_the_short_commit_so_a_build_is_identifiable() {
        assert_eq!(from_parts("0.1.9", Some("42e8a70dc262")), "0.1.9 · 42e8a70d");
    }

    #[test]
    fn keeps_the_dirty_marker_through_truncation() {
        // The marker is the whole point: a dirty build matches no commit, so
        // truncating it away would claim a provenance the binary lacks.
        assert_eq!(
            from_parts("0.1.9", Some("42e8a70dc262-dirty")),
            "0.1.9 · 42e8a70d-dirty"
        );
    }

    /// Guards the wiring [`label`] depends on: `GIT_SHA` is emitted by this
    /// package's build script, and `cargo:rustc-env` has to reach the *library*
    /// target for the sidebar to resolve it without the binary passing it down.
    /// If that ever stops holding, the sidebar silently degrades to a bare
    /// version — the exact ambiguity this module exists to remove — so assert it
    /// rather than trusting the assumption.
    #[test]
    fn label_carries_the_commit_whenever_the_build_stamped_one() {
        match option_env!("GIT_SHA") {
            Some(sha) if !sha.is_empty() && sha != "unknown" => assert!(
                label().contains('\u{00b7}'),
                "GIT_SHA was stamped but label dropped it: {}",
                label()
            ),
            // A build with no git metadata (release tarball) legitimately has
            // no commit to show, and must print the version alone.
            _ => assert_eq!(label(), env!("CARGO_PKG_VERSION")),
        }
    }

    #[test]
    fn falls_back_to_the_bare_version_without_provenance() {
        // Release tarballs build without git metadata; `unknown` is what the
        // build script emits then, and neither case should print noise.
        assert_eq!(from_parts("0.1.9", None), "0.1.9");
        assert_eq!(from_parts("0.1.9", Some("unknown")), "0.1.9");
    }
}
