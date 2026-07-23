#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

desired_free_gib=${ZO_CARGO_DESIRED_FREE_GIB:-40}
lock_wait_seconds=${ZO_CARGO_LOCK_WAIT_SECONDS:-900}
for setting in "$desired_free_gib" "$lock_wait_seconds"; do
    case "$setting" in
        ''|*[!0-9]*)
            echo "error: space-guard settings must be non-negative integers" >&2
            exit 2
            ;;
    esac
done

force_clean=false
if [[ ${1:-} == "--force-clean" ]]; then
    force_clean=true
    shift
fi
if [[ ${1:-} == "--" ]]; then
    shift
elif (( $# > 0 )); then
    echo "usage: $0 [--force-clean] [-- command [args...]]" >&2
    exit 2
fi

lock_dir="$root/.zo/cargo-space.lock"
mkdir -p "$root/.zo"
acquire_lock() {
    local waited=0 missing_owner=0 owner
    while ! mkdir "$lock_dir" 2>/dev/null; do
        owner=""
        [[ -r "$lock_dir/pid" ]] && read -r owner <"$lock_dir/pid"
        if [[ $owner =~ ^[0-9]+$ ]] && ps -p "$owner" >/dev/null 2>&1; then
            missing_owner=0
        else
            missing_owner=$((missing_owner + 1))
            if (( missing_owner >= 3 )); then
                rm -rf "$lock_dir"
                missing_owner=0
                continue
            fi
        fi
        if (( waited >= lock_wait_seconds )); then
            echo "error: timed out waiting for Cargo space lock held by pid ${owner:-unknown}" >&2
            exit 3
        fi
        sleep 1
        waited=$((waited + 1))
    done
    printf '%s\n' "$$" >"$lock_dir/pid"
}
release_lock() {
    local owner=""
    [[ -r "$lock_dir/pid" ]] && read -r owner <"$lock_dir/pid"
    [[ $owner == "$$" ]] && rm -rf "$lock_dir"
}
acquire_lock
trap release_lock EXIT
trap 'exit 130' INT TERM HUP

disk_kib() {
    df -Pk "$root" | awk 'END { print $2, $4 }'
}
read -r total_kib before_kib < <(disk_kib)
configured_kib=$((desired_free_gib * 1024 * 1024))
capacity_target_kib=$((total_kib / 4))
if (( configured_kib < capacity_target_kib )); then
    cleanup_target_kib=$configured_kib
else
    cleanup_target_kib=$capacity_target_kib
fi
hard_floor_kib=$((total_kib / 10))
eight_gib_kib=$((8 * 1024 * 1024))
if (( hard_floor_kib > eight_gib_kib )); then
    hard_floor_kib=$eight_gib_kib
fi

if $force_clean || (( before_kib < cleanup_target_kib )); then
    if command -v pgrep >/dev/null 2>&1 \
        && { pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; }; then
        echo "error: an unmanaged cargo/rustc process is active; wait and retry" >&2
        exit 4
    fi
    printf '⚠ Cargo space guard: %d GiB available; cleaning regenerable target artifacts\n' \
        "$((before_kib / 1024 / 1024))"
    cargo clean --manifest-path "$root/Cargo.toml"
    read -r _ after_kib < <(disk_kib)
    printf '✓ Cargo space guard: %d GiB available after cleanup\n' \
        "$((after_kib / 1024 / 1024))"
    if (( after_kib < hard_floor_kib )); then
        printf 'error: only %d GiB remains after cleanup; this volume needs at least %d GiB free\n' \
            "$((after_kib / 1024 / 1024))" "$((hard_floor_kib / 1024 / 1024))" >&2
        exit 5
    fi
elif (( $# == 0 )); then
    printf '✓ Cargo space guard: %d GiB available (cleanup target %d GiB)\n' \
        "$((before_kib / 1024 / 1024))" "$((cleanup_target_kib / 1024 / 1024))"
fi

if (( $# > 0 )); then
    "$@"
fi
