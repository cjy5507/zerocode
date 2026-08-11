#!/usr/bin/env bash
# Model-free startup-performance gate: measure the RELEASE `zo` binary's
# deterministic startup scenarios (perf-harness suite: --version/--help/
# system-prompt/bootstrap-plan) against the checked-in baseline and fail the
# release on a regression. No network, no model calls — this is the cheap,
# honest gate; the paired model benches stay a scheduled (paid) lane.
#
#   scripts/perf-gate.sh            # gate against bench/perf-baseline.json
#   scripts/perf-gate.sh --bless    # measure and (re)write the baseline
#
# Blessing is an explicit act reserved for when a startup-cost change is
# intentional; the diff to bench/perf-baseline.json then documents it.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

command -v cargo >/dev/null 2>&1 || PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null 2>&1 || {
  echo "perf-gate.sh: cargo is not on PATH" >&2
  exit 1
}

baseline="bench/perf-baseline.json"
mode_flag=""
if [[ "${1:-}" == "--bless" ]]; then
  mode_flag="--bless"
elif [[ ! -f "$baseline" ]]; then
  echo "perf-gate.sh: no baseline at $baseline — run scripts/perf-gate.sh --bless once" >&2
  exit 1
fi

# 20GiB, the release.sh convention — the default 40GiB threshold cleans the
# target dir on this box every call and turns each gate into a full rebuild.
ZO_CARGO_DESIRED_FREE_GIB="${ZO_CARGO_DESIRED_FREE_GIB:-20}" \
  scripts/ensure-cargo-space.sh -- cargo build --release --bin zo
cargo build --release -p compat-harness --bin perf-harness

# 25% time tolerance: the suite runs on a developer box, not a quiet CI
# runner, and each scenario medians 5 runs after 2 warmups — wide enough to
# ignore load noise, tight enough to catch a real startup regression class
# (the M5 startup-hang family this gate exists for).
exec target/release/perf-harness suite \
  --bin target/release/zo \
  --baseline "$baseline" \
  --tolerance 0.25 \
  $mode_flag
