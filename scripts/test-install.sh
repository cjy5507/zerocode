#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work="$(mktemp -d)"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT INT TERM

fail() {
  printf 'test-install.sh: %s\n' "$*" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

release_dir="${work}/release"
home="${work}/home"
config_home="${work}/config-home"
mkdir -p "$release_dir" "$home" "$config_home"
printf 'preserve-me\n' > "${config_home}/settings.json"

version="$(awk '
  /^\[workspace.package\]$/ { in_workspace_package=1; next }
  /^\[/ { in_workspace_package=0 }
  in_workspace_package && /^version = / {
    gsub(/^version = "/, ""); gsub(/"$/, ""); print; exit
  }
' "${repo_root}/Cargo.toml")"
[[ -n "$version" ]] || fail "could not read workspace version"
cat > "${work}/fake-zo" <<BINARY
#!/usr/bin/env bash
if [[ "\${1:-}" == "--version" ]]; then
  printf 'zo\n  Version          ${version}\n  Git SHA          test\n  Target           test\n  Build date       test\n'
  exit 0
fi
printf 'fake zo\n'
BINARY
chmod 0755 "${work}/fake-zo"

targets=(
  "aarch64-apple-darwin"
  "x86_64-apple-darwin"
  "x86_64-unknown-linux-gnu"
)
for target in "${targets[@]}"; do
  cp "${work}/fake-zo" "${release_dir}/zo-v${version}-${target}"
done

cat > "${work}/server.py" <<'PY'
import http.server
import os
import socketserver
import sys

os.chdir(sys.argv[1])
with socketserver.TCPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler) as server:
    with open(sys.argv[2], "w", encoding="utf-8") as port_file:
        port_file.write(str(server.server_address[1]))
    server.serve_forever()
PY
python3 "${work}/server.py" "$release_dir" "${work}/port" >"${work}/server.log" 2>&1 &
server_pid=$!
for _ in {1..100}; do
  [[ -s "${work}/port" ]] && break
  sleep 0.05
done
[[ -s "${work}/port" ]] || fail "local HTTP server did not start"
base="http://127.0.0.1:$(cat "${work}/port")"

write_manifest() {
  {
    printf 'schema=1\n'
    printf 'version=%s\n' "$version"
    printf 'base=%s\n' "$base"
    for target in "${targets[@]}"; do
      asset="${release_dir}/zo-v${version}-${target}"
      printf 'asset=%s|zo-v%s-%s|%s|%s\n' \
        "$target" "$version" "$target" "$(sha256_file "$asset")" "$(wc -c < "$asset" | tr -d ' ')"
    done
  } > "${release_dir}/manifest.txt"
}
write_manifest

run_installer() {
  HOME="$home" \
  ZO_CONFIG_HOME="$config_home" \
  ZO_INSTALLER_TEST_ONLY=1 \
  ZO_INSTALLER_TEST_BASE="$base" \
    bash "${repo_root}/install.sh" >/dev/null
}

# The local HTTP override must be inert unless the explicit test-only guard is set.
if HOME="$home" ZO_CONFIG_HOME="$config_home" ZO_INSTALLER_TEST_BASE="$base" \
  bash "${repo_root}/install.sh" >/dev/null 2>&1; then
  fail "unguarded local HTTP override was accepted"
fi

# First install writes the binary and marker without modifying existing config.
run_installer
installed="${home}/.local/bin/zo"
[[ -x "$installed" ]] || fail "first install did not create an executable"
[[ "$(cat "${config_home}/settings.json")" == "preserve-me" ]] || \
  fail "first install changed existing settings"
canonical_installed="$(cd "$(dirname "$installed")" && pwd -P)/zo"
expected_marker="$(printf 'schema=1\npath=%s\n' "$canonical_installed")"
[[ "$(cat "${config_home}/managed-install")" == "$expected_marker" ]] || \
  fail "managed-install marker is not exact"

# Reinstalling the same release is idempotent and keeps config intact.
first_hash="$(sha256_file "$installed")"
run_installer
[[ "$(sha256_file "$installed")" == "$first_hash" ]] || fail "idempotent install changed bytes"
[[ "$(cat "${config_home}/settings.json")" == "preserve-me" ]] || \
  fail "idempotent install changed existing settings"

# A validly shaped but incorrect checksum must leave the installed binary unchanged.
python3 - "${release_dir}/manifest.txt" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
lines = path.read_text().splitlines()
for index in range(3, 6):
    parts = lines[index].split("|")
    parts[2] = "0" * 64
    lines[index] = "|".join(parts)
path.write_text("\n".join(lines) + "\n")
PY
if run_installer >/dev/null 2>&1; then
  fail "checksum mismatch install unexpectedly succeeded"
fi
[[ "$(sha256_file "$installed")" == "$first_hash" ]] || \
  fail "checksum failure changed the installed binary"
write_manifest

# A destination symlink must be refused without modifying its target.
rm -f "$installed"
victim="${work}/victim"
printf 'do-not-touch\n' > "$victim"
ln -s "$victim" "$installed"
if run_installer >/dev/null 2>&1; then
  fail "symlink destination install unexpectedly succeeded"
fi
[[ "$(cat "$victim")" == "do-not-touch" ]] || fail "symlink target was modified"

printf 'installer tests passed\n'
