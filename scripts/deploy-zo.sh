#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/target/release/zo"

cd "$repo_root"
"$HOME/.cargo/bin/cargo" build --release -p zo-cli

if [[ ! -f "$binary" ]]; then
    echo "deploy aborted: release binary not found at $binary" >&2
    exit 1
fi

binary_type="$(file -b "$binary")"
if [[ "$binary_type" != *"Mach-O"* || "$binary_type" != *"arm64"* ]]; then
    echo "deploy aborted: $binary is not a Mach-O arm64 binary ($binary_type)" >&2
    exit 1
fi

if [[ -z "$(find "$binary" -mmin -2 -print -quit)" ]]; then
    echo "deploy aborted: $binary is older than 2 minutes; refusing to deploy a stale build" >&2
    exit 1
fi

targets=("/opt/homebrew/bin/zo" "$HOME/.local/bin/zo")
for target in "${targets[@]}"; do
    parent="$(dirname "$target")"
    if [[ ! -d "$parent" ]]; then
        echo "Skipping $target: parent directory $parent does not exist"
        continue
    fi

    rm -f "$target"
    cp -p "$binary" "$target"
    echo "Deployed $(stat -f '%z bytes, modified %Sm' -t '%Y-%m-%d %H:%M:%S %z' "$target") to $target"
done

zo --version | head -2
echo "Running zo sessions keep the old inode until /restart."
