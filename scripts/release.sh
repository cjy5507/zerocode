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
git diff --check
git add Cargo.toml Cargo.lock
git commit -m "release: ${tag}"
git tag -a "$tag" -m "$tag"
rollback=0
trap - EXIT INT TERM
git push --atomic origin main "$tag"
printf 'Pushed %s; the release workflow will publish its assets.\n' "$tag"
