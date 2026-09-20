# ZeroCode

ZeroCode is a macOS app for working with coding agents: a terminal window with an orchestration ledger, a built-in browser, a mobile emulator and desktop Computer Use, plus the `zo` coding agent CLI it ships with. This repository carries the releases; the app updates itself from them.

## Install (macOS, Apple Silicon)

```bash
curl -fsSL https://github.com/cjy5507/zerocode/releases/latest/download/install.sh | bash
```

The installer reads the release feed (`latest.json`), downloads that version's `ZeroCode_<version>_aarch64.app.tar.gz`, verifies the feed's minisign signature when `minisign` is installed (`brew install minisign`), installs `ZeroCode.app` into `/Applications` by one rename, clears the quarantine attribute so the first launch is not refused, and links the bundled `zo` to `~/.local/bin/zo`. Knobs: `ZEROCODE_VERSION=v1.1.4` pins a release, `ZEROCODE_INSTALL_DIR` and `ZEROCODE_BIN_DIR` move the app or the link.

Manual install: download `ZeroCode_<version>_aarch64.dmg` from the [latest release](https://github.com/cjy5507/zerocode/releases/latest), drag `ZeroCode.app` to Applications, and open it once with right-click → Open (the app is signed but not notarised, so Gatekeeper asks on the first launch). `zo` sits at `/Applications/ZeroCode.app/Contents/Resources/bin/zo`; link or add it to `PATH` yourself.

Ensure `~/.local/bin` is on `PATH`, then:

```bash
zo --version
zo doctor --check
```

## Updates

The app checks this repository's release feed and installs updates itself; the install script can also be rerun at any time. `zo` is updated with the app.

Releases are published for macOS on Apple Silicon only.

ZeroCode is an independent project and is not affiliated with Anthropic, OpenAI, Google, xAI, or other model providers. See `LICENSE`, `NOTICE`, and `PRIVACY.md` for distribution and privacy terms.
