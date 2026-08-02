#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVALUATOR="$ROOT/scripts/zo-opus-max-analysis-evaluator.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/zo-opus-eval-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail=0

write_summary() {
  local path="$1" zo_wall="$2" zo_tokens="$3" zo_cost="$4" zo_tools="$5" \
    claude_wall="$6" score="$7" false_claims="$8" valid="${9:-true}"
  cat >"$path" <<JSON
{
  "fairness_contract": { "valid": $valid },
  "zo": {
    "median": {
      "wall_seconds": $zo_wall,
      "total_tokens": $zo_tokens,
      "total_cost_usd": $zo_cost,
      "tool_uses": $zo_tools
    }
  },
  "claude": {
    "median": {
      "wall_seconds": $claude_wall,
      "total_tokens": 384392,
      "total_cost_usd": 1.0268
    }
  },
  "judges": {
    "average_score": $score,
    "false_claims_total": $false_claims
  }
}
JSON
}

expect_exit() {
  local label="$1" expected="$2" summary="$3" rc=0
  local output="$WORK/$label.out"
  "$EVALUATOR" "$summary" >"$output" 2>&1 || rc=$?
  if [ "$rc" -ne "$expected" ]; then
    echo "FAIL: $label expected exit $expected got $rc"
    cat "$output"
    fail=1
  else
    echo "ok: $label"
  fi
}

pass="$WORK/pass.json"
write_summary "$pass" 120 240000 0.72 12 180 94 0 true
expect_exit "passing-summary" 0 "$pass"

slow="$WORK/slow.json"
write_summary "$slow" 170 240000 0.72 12 180 94 0 true
expect_exit "slow-summary" 1 "$slow"

bad_quality="$WORK/bad-quality.json"
write_summary "$bad_quality" 120 240000 0.72 12 180 91.9 0 true
expect_exit "bad-quality-summary" 1 "$bad_quality"

bad_fairness="$WORK/bad-fairness.json"
write_summary "$bad_fairness" 120 240000 0.72 12 180 94 0 false
expect_exit "bad-fairness-summary" 1 "$bad_fairness"

exit "$fail"
