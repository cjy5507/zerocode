#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/zo-opus-max-analysis-evaluator.sh SUMMARY.json

Validates a fair Opus 4.8 max-effort Zo-vs-Claude analysis benchmark
summary. The summary is produced by the benchmark runner, not by this checker.

Required shape:
{
  "fairness_contract": { "valid": true },
  "zo": {
    "median": {
      "wall_seconds": 120,
      "total_tokens": 240000,
      "total_cost_usd": 0.72,
      "tool_uses": 12
    }
  },
  "claude": { "median": { "wall_seconds": 180 } },
  "judges": { "average_score": 94, "false_claims_total": 0 }
}
EOF
}

summary="${1:-${ZO_OPUS_MAX_EVAL_SUMMARY:-}}"
if [ -z "$summary" ]; then
  usage
  exit 2
fi
if [ ! -f "$summary" ]; then
  echo "zo-opus-max-analysis-evaluator: summary not found: $summary" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "zo-opus-max-analysis-evaluator: jq is required" >&2
  exit 2
fi

result="$(
  jq '
    {
      fairness_valid: (.fairness_contract.valid == true),
      zo_wall_seconds: (.zo.median.wall_seconds // null),
      zo_total_tokens: (.zo.median.total_tokens // null),
      zo_total_cost_usd: (.zo.median.total_cost_usd // null),
      zo_tool_uses: (.zo.median.tool_uses // null),
      claude_wall_seconds: (.claude.median.wall_seconds // null),
      judge_average: (.judges.average_score // null),
      false_claims_total: (.judges.false_claims_total // null)
    } as $m
    | [
        if $m.fairness_valid then empty else "fairness_contract_invalid" end,
        if (($m.judge_average // -1) >= 92) then empty else "judge_average_below_92" end,
        if (($m.false_claims_total // 999999) == 0) then empty else "false_claims_present" end,
        if (
          (($m.zo_wall_seconds // 1000000000) <= 160)
          or (
            (($m.claude_wall_seconds // 0) > 0)
            and (($m.zo_wall_seconds // 1000000000) <= (($m.claude_wall_seconds // 0) * 0.85))
          )
        ) then empty else "zo_not_fast_enough" end,
        if (($m.zo_total_tokens // 1000000000) <= 300000) then empty else "zo_total_tokens_above_300k" end,
        if (($m.zo_total_cost_usd // 1000000000) <= 0.90) then empty else "zo_cost_above_0_90" end,
        if (($m.zo_tool_uses // 1000000000) <= 15) then empty else "zo_tool_uses_above_15" end
      ] as $reasons
    | {
        pass: (($reasons | length) == 0),
        reasons: $reasons,
        metrics: $m,
        thresholds: {
          judge_average_min: 92,
          false_claims_total_max: 0,
          zo_wall_seconds_max: 160,
          zo_vs_claude_speedup_min: 0.15,
          zo_total_tokens_max: 300000,
          zo_total_cost_usd_max: 0.90,
          zo_tool_uses_max: 15
        }
      }
  ' "$summary"
)"

printf '%s\n' "$result"
[ "$(printf '%s' "$result" | jq -r '.pass')" = true ]
