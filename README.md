# ZeroCode

ZeroCode is a Rust-native coding agent CLI for interactive terminal work, one-shot automation, resumable sessions, MCP integrations, and parallel agent workflows. The installed command is `zo`.

## Install

```bash
curl -fsSL https://github.com/cjy5507/zerocode/releases/latest/download/install.sh | bash
```

The installer selects the correct release for macOS Apple Silicon, macOS Intel, or Linux x86_64, verifies its SHA-256 digest, and installs it to `~/.local/bin/zo` without `sudo`. Existing `~/.zo` settings and credentials are preserved.

Ensure `~/.local/bin` is on `PATH`, then start ZeroCode:

```bash
zo
```

Useful commands:

```bash
zo --version
zo doctor --check
zo update --check
zo update
```

Installations created by the release installer check for stable updates in the background. Cargo, Homebrew, copied, and development builds are never overwritten by the automatic updater.

## Build from source

```bash
cargo install \
  --git https://github.com/cjy5507/zerocode.git \
  --locked \
  --bin zo \
  --root "$HOME/.local" \
  zo-cli
```

ZeroCode is an independent project and is not affiliated with Anthropic, OpenAI, Google, xAI, or other model providers. See `LICENSE`, `NOTICE`, and `PRIVACY.md` for distribution and privacy terms.
