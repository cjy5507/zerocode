# zo 개발 레시피 — `just <recipe>` (전체 목록: `just`)
# just 미설치 시: brew install just

# just 는 non-login 셸로 레시피를 실행하므로 rustup 의 cargo 가 PATH 에 없을 수 있다.
export PATH := env_var('HOME') / '.cargo' / 'bin' + ':' + env_var('PATH')

# 설치 위치. 재정의: `ZO_INSTALL_DIR=/some/bin just install`
# 기본값은 PATH 1순위인 ~/.local/bin (bare `zo` 명령이 가리키는 곳)
install_dir := env_var_or_default('ZO_INSTALL_DIR', env_var('HOME') / '.local' / 'bin')

# 실배포 위치. `which -a zo` 1순위이자 이 머신이 실제로 실행하는 경로.
# 재정의: `ZO_DEPLOY_DIR=/some/bin just deploy`
deploy_dir := env_var_or_default('ZO_DEPLOY_DIR', '/opt/homebrew/bin')

# 레시피 목록 (기본 타깃)
default:
    @just --list

# Cargo 장기 작업 전에 최소 여유 공간을 보장한다.
space-guard:
    @scripts/ensure-cargo-space.sh

# 수동 강제 정리. 소스는 건드리지 않고 target/만 재생성 대상으로 제거한다.
space-clean:
    @scripts/ensure-cargo-space.sh --force-clean

# CI-equivalent 로컬 게이트. 하나의 lock 아래 정리와 Cargo 작업을 직렬화한다.
verify:
    @scripts/ensure-cargo-space.sh -- bash -c 'set -euo pipefail; cargo check --workspace --all-targets --locked; cargo clippy --workspace --all-targets --locked -- -D warnings; cargo test --workspace --locked'

# 릴리스 체계 게이트: 전체 Rust 검증 + 셸 구문 + 격리 installer 테스트.
release-verify: verify
    @bash -n install.sh scripts/release.sh scripts/test-install.sh
    @scripts/test-install.sh

# zo 릴리스 바이너리 빌드 → target/release/zo
build:
    @scripts/ensure-cargo-space.sh -- cargo build --release --bin zo

# 빌드 후 install_dir 에 zo 설치 (빌드만 하고 stale 바이너리 실행하는 사고 방지)
install: build
    @mkdir -p "{{install_dir}}"
    install -m 755 target/release/zo "{{install_dir}}/zo"
    # 리네임 전 `forge` 바이너리가 PATH에 남지 않도록 설치 경로와 옛
    # `cargo install` 경로를 함께 정리한다. 없으면 무시(-f).
    @rm -f "{{install_dir}}/forge"
    @rm -f "{{env_var('HOME')}}/.cargo/bin/forge"
    # 옛 `cargo install` 바이너리(~/.cargo/bin/zo)가 PATH에서 먼저 잡혀 stale
    # 버전이 실행되는 사고를 막는다. 없으면 무시(-f).
    @rm -f "{{env_var('HOME')}}/.cargo/bin/zo"
    @echo "✓ 설치 완료: {{install_dir}}/zo"
    @echo "  (셸 명령 캐시 때문에 기존 터미널에서는 'hash -r' 또는 새 터미널 필요)"

# 빌드 후 실배포 경로(deploy_dir)에 설치하고, 옛 바이너리를 물고 있는 실행
# 프로세스를 색출해 재시작 필요를 알린다. install -m 755 는 unlink+create 라
# arm64 in-place 덮어쓰기 함정(실행 중 매핑 손상)을 피한다.
deploy: build
    #!/usr/bin/env bash
    set -euo pipefail
    install -m 755 target/release/zo "{{deploy_dir}}/zo"
    rm -f "{{deploy_dir}}/forge"
    echo "✓ 배포 완료: {{deploy_dir}}/zo ($({{deploy_dir}}/zo version 2>/dev/null | awk '/Git SHA/{print $3}'))"
    # 새로 배포한 바이너리의 inode 와 실행 중 프로세스가 매핑한 inode 를 대조.
    disk_inode=$(stat -f %i "{{deploy_dir}}/zo")
    stale=0
    for pid in $(pgrep -x zo 2>/dev/null || true); do
        exe_inode=$(lsof -p "$pid" 2>/dev/null | awk '$4=="txt" && /zo$/ {print $(NF-1); exit}' || true)
        if [ -n "${exe_inode:-}" ] && [ "$exe_inode" != "$disk_inode" ]; then
            echo "  ⚠ pid $pid 은 옛 바이너리 실행 중 — 재시작해야 새 빌드 반영"
            stale=$((stale + 1))
        fi
    done
    for pid in $(pgrep -x forge 2>/dev/null || true); do
        echo "  ⚠ pid $pid 은 legacy forge 바이너리 실행 중 — 재시작해야 zo 전환 반영"
        stale=$((stale + 1))
    done
    [ "$stale" -eq 0 ] && echo "  실행 중 stale 프로세스 없음" || echo "  → $stale 개 세션 재시작 필요 (/restart 또는 새 터미널)"
