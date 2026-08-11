#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

fail() {
  printf 'release.sh: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -eq 1 ]] || fail "usage: scripts/release.sh MAJOR.MINOR.PATCH"
version="$1"
[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || \
  fail "version must be strict MAJOR.MINOR.PATCH"
tag="v${version}"

# Everything below shells out to cargo (directly and via ensure-cargo-space/just),
# and a missing cargo used to surface as a mid-script death that an outer
# `| tail` could mask into apparent success. Fail here, by name, before any
# state is touched.
command -v cargo >/dev/null 2>&1 || \
  fail "cargo is not on PATH (try PATH=\"\$HOME/.cargo/bin:\$PATH\")"

[[ "$(git branch --show-current)" == "main" ]] || fail "release must run from main"
[[ -z "$(git status --porcelain)" ]] || fail "working tree must be clean"
git remote get-url origin >/dev/null 2>&1 || fail "origin remote is missing"
git fetch --quiet origin main --tags
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || \
  fail "local main must exactly match origin/main"
if git rev-parse --verify --quiet "refs/tags/${tag}" >/dev/null; then
  fail "local tag ${tag} already exists"
fi
if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
  fail "remote tag ${tag} already exists"
fi

current="$(awk '
  /^\[workspace.package\]$/ { in_workspace_package=1; next }
  /^\[/ { in_workspace_package=0 }
  in_workspace_package && /^version = / {
    gsub(/^version = "/, ""); gsub(/"$/, ""); print; exit
  }
' Cargo.toml)"
[[ -n "$current" ]] || fail "could not read workspace version"
python3 - "$current" "$version" <<'PY'
import sys
current = tuple(map(int, sys.argv[1].split(".")))
requested = tuple(map(int, sys.argv[2].split(".")))
if requested <= current:
    raise SystemExit(f"release.sh: version must increase from {sys.argv[1]}")
PY

rollback=1
cleanup() {
  if [[ "$rollback" -eq 1 ]]; then
    git restore --worktree --staged Cargo.toml Cargo.lock 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python3 - "$current" "$version" <<'PY'
from pathlib import Path
import sys
path = Path("Cargo.toml")
text = path.read_text()
old = f'[workspace.package]\nversion = "{sys.argv[1]}"'
new = f'[workspace.package]\nversion = "{sys.argv[2]}"'
if text.count(old) != 1:
    raise SystemExit("release.sh: workspace version declaration was not unique")
path.write_text(text.replace(old, new))
PY

# Refresh workspace package versions in Cargo.lock before the locked release gate.
scripts/ensure-cargo-space.sh -- cargo check --workspace
just release-verify

# Opt-in paid smoke (ZO_RELEASE_SMOKE=1): one headless `-p` turn against the
# real provider, proving the auth + one-shot path end to end before the tag.
# Opt-in, not default: a network flake must not be able to hold a release
# hostage, and the free gates above already cover everything deterministic.
# perf-gate (inside release-verify) just built target/release/zo, so the
# binary under test is fresh.
if [[ "${ZO_RELEASE_SMOKE:-0}" == "1" ]]; then
  smoke_reply="$(target/release/zo --output-format json --permission-mode read-only \
    -p 'Reply with exactly this single word and nothing else: SMOKE-OK' 2>&1 | tail -1)" || \
    fail "paid smoke: headless zo -p run failed"
  grep -q "SMOKE-OK" <<<"$smoke_reply" || \
    fail "paid smoke: reply did not contain SMOKE-OK: ${smoke_reply:0:200}"
  printf 'Paid smoke passed.\n'
fi

git diff --check
git add Cargo.toml Cargo.lock
git commit -m "release: ${tag}"
git tag -a "$tag" -m "$tag"
rollback=0
trap - EXIT INT TERM
git push --atomic origin main "$tag"

# Post-push verification — success is judged by what the REMOTE holds, never by
# this script having reached its last line (a masked mid-script death once read
# as a finished release until the missing remote tag exposed it). Two facts:
# the tag must be on origin now, and the release workflow must publish assets.
git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null || \
  fail "pushed, but origin does not show ${tag} — the release did NOT land"
printf 'Remote tag %s verified.\n' "$tag"

if command -v gh >/dev/null 2>&1; then
  # Bounded wait for the workflow's release assets — 40 minutes, calibrated to
  # the measured ~23-minute workflow (the first 10-minute bound cried wolf on a
  # healthy in-progress run). A timeout here is a warning, not a failure: the
  # tag landed, so the release exists; assets can lag. What must never happen
  # is this script CLAIMING assets it never saw.
  assets=""
  for _ in $(seq 1 120); do
    assets="$(gh release view "$tag" --json assets \
      --jq '.assets | length' 2>/dev/null || true)"
    [[ -n "$assets" && "$assets" != "0" ]] && break
    sleep 20
  done
  if [[ -n "$assets" && "$assets" != "0" ]]; then
    printf 'Release %s verified: %s asset(s) published.\n' "$tag" "$assets"
  else
    printf 'WARNING: %s tag landed but no assets after 40min — workflow state: %s\n' \
      "$tag" "$(gh run list --limit 1 --json status,conclusion \
        --jq '.[0] | .status + "/" + (.conclusion // "-")' 2>/dev/null || echo unknown)" >&2
  fi
else
  printf 'gh CLI not found; asset publication NOT verified (tag is confirmed).\n' >&2
fi
