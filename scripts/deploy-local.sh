#!/bin/sh
# Local deploy for the daily-driver zo binary, encoding every landmine the
# manual procedure kept tripping:
#
# - Builds WITH symbols: CARGO_PROFILE_RELEASE_STRIP=none keeps symbol tables
#   and line-tables-only debug info in an otherwise codegen-identical release
#   binary (fat LTO, codegen-units=1, opt3 — strip only ever REMOVED data).
#   A freeze `sample` against this binary symbolizes immediately with atos,
#   instead of requiring a same-commit retro rebuild first.
# - rm-then-cp, never in-place: overwriting a running arm64 Mach-O in place
#   corrupts code signing for live processes.
# - Verifies the DEPLOYED binary's baked Git SHA against HEAD, and refuses a
#   dirty tree unless --allow-dirty: a binary stamped with a SHA whose tree it
#   was not built from is the "which build am I even running" incident.
set -eu

cd "$(dirname "$0")/.."
DEST="${ZO_DEPLOY_DEST:-$HOME/.local/bin/zo}"

if [ "${1:-}" != "--allow-dirty" ] && ! git diff --quiet HEAD -- 2>/dev/null; then
    echo "deploy-local: working tree is dirty — commit first, or pass --allow-dirty" >&2
    exit 1
fi

echo "building release with symbols (codegen-identical; strip=none) …"
CARGO_PROFILE_RELEASE_STRIP=none \
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
cargo build --release -p zo-cli

expected=$(git rev-parse --short=12 HEAD)
built=$(./target/release/zo --version | sed -n 's/.*Git SHA[[:space:]]*\([0-9a-f]*\).*/\1/p')
if [ "$built" != "$expected" ]; then
    echo "deploy-local: built SHA '$built' != HEAD '$expected' — refusing to deploy" >&2
    exit 1
fi

rm -f "$DEST"
cp target/release/zo "$DEST"

deployed=$("$DEST" --version | sed -n 's/.*Git SHA[[:space:]]*\([0-9a-f]*\).*/\1/p')
if [ "$deployed" != "$expected" ]; then
    echo "deploy-local: DEPLOYED binary reports '$deployed', expected '$expected'" >&2
    exit 1
fi
size=$(du -h "$DEST" | cut -f1)
echo "deployed $expected -> $DEST ($size, symbol-bearing)"
