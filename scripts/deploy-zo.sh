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

targets=("$HOME/.local/bin/zo")
# 과거 배포 위치에 남은 옛 zo 사본(예: /opt/homebrew/bin/zo)이 PATH에서 새
# 배포본보다 먼저 잡혀 stale 빌드가 계속 실행되는 사고를 막는다: PATH에 이미
# 존재하는 다른 zo 사본도 같은 빌드로 함께 갱신한다.
# 격리 테스트(test-deploy.sh)는 ZO_DEPLOY_NO_SHADOW_REFRESH=1 로 이 스캔을 끈다.
if [[ -z "${ZO_DEPLOY_NO_SHADOW_REFRESH:-}" ]]; then
    while IFS= read -r existing; do
        [[ -z "$existing" ]] && continue
        duplicate=0
        for known in "${targets[@]}"; do
            if [[ "$existing" == "$known" || "$existing" -ef "$known" ]]; then
                duplicate=1
                break
            fi
        done
        [[ "$duplicate" -eq 0 ]] && targets+=("$existing")
    done < <(which -a zo 2>/dev/null || true)
fi
deployed=0
for target in "${targets[@]}"; do
    parent="$(dirname "$target")"
    if [[ ! -d "$parent" ]]; then
        echo "Creating parent directory $parent..."
        mkdir -p "$parent"
    fi

    rm -f "$target"
    cp -p "$binary" "$target"
    echo "Deployed $(stat -f '%z bytes, modified %Sm' -t '%Y-%m-%d %H:%M:%S %z' "$target") to $target"
    
    # Validate the newly deployed binary directly
    "$target" --version | head -2
    deployed=1
done

if [[ "$deployed" -ne 1 ]]; then
    echo "deploy failed: no targets were deployed" >&2
    exit 1
fi

echo "Running zo sessions keep the old inode until /restart."
