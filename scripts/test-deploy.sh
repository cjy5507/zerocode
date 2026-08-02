#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work="$(mktemp -d)"

cleanup() {
  rm -rf "$work"
}
trap cleanup EXIT INT TERM

fail() {
  printf 'test-deploy.sh: %s\n' "$*" >&2
  exit 1
}

# 1. Setup mock workspace directories
mkdir -p "${work}/scripts"
mkdir -p "${work}/target/release"
mkdir -p "${work}/bin"
mkdir -p "${work}/home/.cargo/bin"

# Copy the real deploy-zo.sh to the mock workspace scripts folder
cp "${repo_root}/scripts/deploy-zo.sh" "${work}/scripts/deploy-zo.sh"

# 2. Write mock file command
cat > "${work}/bin/file" <<'EOF'
#!/usr/bin/env bash
# Mock file to always report Mach-O arm64
echo "Mach-O 64-bit executable arm64"
EOF
chmod +x "${work}/bin/file"

# 3. Write mock cargo command (supporting both --version and version)
cat > "${work}/home/.cargo/bin/cargo" <<'EOF'
#!/usr/bin/env bash
# Mock cargo build: create dummy executable target/release/zo
binary_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../target/release" && pwd)"
cat > "${binary_dir}/zo" <<'ZO'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" || "${1:-}" == "version" ]]; then
  echo "zo"
  echo "  Version          0.1.7"
  echo "  Git SHA          test-git-sha"
  exit 0
fi
echo "fake zo"
exit 0
ZO
chmod +x "${binary_dir}/zo"
exit 0
EOF
chmod +x "${work}/home/.cargo/bin/cargo"

# Prepend our mock bin to PATH and export mock HOME
export PATH="${work}/bin:$PATH"
export HOME="${work}/home"
# Shadow refresh would rewrite real zo copies on the host PATH; keep the
# sandbox hermetic.
export ZO_DEPLOY_NO_SHADOW_REFRESH=1

# Test Case 1: Parent directory is absent, deploy-zo.sh should create it and deploy successfully
if [[ -d "${HOME}/.local/bin" ]]; then
  fail "Test setup error: local bin directory already exists"
fi

# Run deploy-zo.sh
bash "${work}/scripts/deploy-zo.sh" > /dev/null

# Assertions
[[ -f "${HOME}/.local/bin/zo" ]] || fail "deploy-zo.sh did not create the target binary"
[[ -x "${HOME}/.local/bin/zo" ]] || fail "deploy-zo.sh target binary is not executable"
[[ "$("${HOME}/.local/bin/zo" --version | head -2)" == *"Version          0.1.7"* ]] || fail "deploy-zo.sh target binary does not report expected version"

# Test Case 2: Silent-success regression (wrong PATH binary, validation failure)
# Clean up target
rm -rf "${HOME}/.local"

# Place an old/different binary on PATH (in work/bin/zo) that returns a different version
cat > "${work}/bin/zo" <<'OLDZO'
#!/usr/bin/env bash
echo "zo"
echo "  Version          0.1.0-stale"
exit 0
OLDZO
chmod +x "${work}/bin/zo"

# Mock cargo to write a broken binary that fails validation (non-zero exit code on --version)
cat > "${work}/home/.cargo/bin/cargo" <<'EOF'
#!/usr/bin/env bash
binary_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../target/release" && pwd)"
cat > "${binary_dir}/zo" <<'ZO'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" || "${1:-}" == "version" ]]; then
  exit 1
fi
exit 0
ZO
chmod +x "${binary_dir}/zo"
exit 0
EOF

# Run deploy-zo.sh and expect it to FAIL due to validation failure
if bash "${work}/scripts/deploy-zo.sh" > /dev/null 2>&1; then
  fail "deploy succeeded silently despite validation failure"
fi

# Test Case 3: just deploy with parent directory absent
# Ensure just is installed
if ! command -v just >/dev/null 2>&1; then
  fail "just command not found, cannot run just deploy test"
fi

# Clean up target local bin and PATH mock zo to avoid interference
rm -rf "${HOME}/.local"
rm -f "${work}/bin/zo"

# Copy the real justfile to mock workspace
cp "${repo_root}/justfile" "${work}/justfile"

# Create a mock ensure-cargo-space.sh script
mkdir -p "${work}/scripts"
cat > "${work}/scripts/ensure-cargo-space.sh" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
chmod +x "${work}/scripts/ensure-cargo-space.sh"

# Precreate release directory so the mock cargo writes to it
mkdir -p "${work}/target/release"

# Reset mock cargo to write a valid binary supporting both --version and version
cat > "${work}/home/.cargo/bin/cargo" <<'EOF'
#!/usr/bin/env bash
binary_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../target/release" && pwd)"
cat > "${binary_dir}/zo" <<'ZO'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" || "${1:-}" == "version" ]]; then
  echo "zo"
  echo "  Version          0.1.7"
  echo "  Git SHA          test-git-sha"
  exit 0
fi
echo "fake zo"
exit 0
ZO
chmod +x "${binary_dir}/zo"
exit 0
EOF
chmod +x "${work}/home/.cargo/bin/cargo"

# Verify that local bin does not exist
if [[ -d "${HOME}/.local/bin" ]]; then
  fail "Test setup error for Test Case 3: local bin directory already exists"
fi

# Run just deploy from the mock workspace directory
(cd "${work}" && just deploy >/dev/null)

# Assertions
[[ -f "${HOME}/.local/bin/zo" ]] || fail "just deploy did not create the target binary"
[[ -x "${HOME}/.local/bin/zo" ]] || fail "just deploy target binary is not executable"
[[ "$("${HOME}/.local/bin/zo" --version | head -2)" == *"Version          0.1.7"* ]] || fail "just deploy target binary does not report expected version"

printf 'deploy tests passed\n'
