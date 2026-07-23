#!/usr/bin/env bash
# usage_bench.sh — headless multi-turn prompt-cache benchmark for zo.
#
# Purpose
#   Measure per-request usage (input / cache_read / cache_creation) across N
#   consecutive user turns in ONE zo session, to verify that the prompt
#   cache stays warm turn-over-turn. Written to validate the reasoning-replay
#   fix (commit 7ef148b0: session-scoped replay cache + history persistence),
#   whose failure mode was: replay-cache churn re-serializes old assistant
#   thinking blocks differently on later turns -> the Anthropic prompt cache
#   diverges at the first old message -> every request re-creates the whole
#   prefix (cache_read collapses, cache_creation ~= full context, repeatedly).
#
# What it does
#   1. Builds an NDJSON stdin of N user turns; each turn asks the model to
#      read one repo file with the read_file tool and summarize it (so every
#      turn contains a tool round-trip => >= 2 API requests per turn).
#   2. Runs the deployed zo binary headlessly in a single process/session:
#        zo -p --input-format stream-json --output-format stream-json
#      ZO_EFFORT (default: high) keeps extended thinking ON so assistant
#      reasoning blocks are actually present in history (the replay path).
#      --strict-mcp-config (no --mcp-config) disables all MCP servers.
#      CAVEAT: the --settings smart-off overlay below reaches only the runtime
#      ConfigLoader chain — the smart-router readers load
#      <config_home>/settings.json directly and never see it. The main-turn
#      model still stays pinned via -m (smart routing only redirects subagent
#      spawns / deep-verify legs, and the bench prompts spawn no agents), but
#      do NOT rely on this overlay to force smart off globally.
#   3. Parses the stream-json usage events (session-cumulative counters),
#      derives per-request deltas, prints a time-series table and a verdict:
#        PASS         all turn>=2 requests: cache_read/(input+cache_read) >= 0.80
#        FAIL         >=2 turn>=2 requests where the ratio < 0.50 while
#                     cache_creation recreates >50% of the context (churn)
#        INCONCLUSIVE anything in between
#
# Usage
#   scripts/usage_bench.sh [-m MODEL] [-e EFFORT] [-b ZO_BIN] [-o OUTDIR] [FILE ...]
#     -m MODEL   model id            (default: claude-fable-5; Anthropic only —
#                                     reasoning replay is an Anthropic path)
#     -e EFFORT  ZO_EFFORT value  (default: high; keeps thinking ON)
#     -b BIN     zo binary        (default: /opt/homebrew/bin/zo)
#     -o OUTDIR  artifacts directory (default: mktemp -d under $TMPDIR)
#     FILE ...   repo files to read, one per turn (default: 6 stable files)
#
# Examples
#   scripts/usage_bench.sh
#   scripts/usage_bench.sh -m claude-sonnet-4-5 -o /tmp/bench README.md Cargo.toml
#
# Notes
#   * Run from the repo root (read_file targets must be inside the workspace).
#   * Uses the real ~/.zo home (credentials). It does NOT modify settings.
#     The --settings overlay is best-effort only (see CAVEAT above); the
#     measured requests stay on -m MODEL because no subagents are spawned.
#     Headless -p sessions are SessionScope::Ephemeral (stored out-of-tree
#     under ~/.zo/projects/<slug>/...; subject to normal retention).
#   * Exit code: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE, >2 = harness error.

set -euo pipefail

MODEL="claude-fable-5"
EFFORT="high"
ZO_BIN="/opt/homebrew/bin/zo"
OUTDIR=""

while getopts "m:e:b:o:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    e) EFFORT="$OPTARG" ;;
    b) ZO_BIN="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    h) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "usage: $0 [-m MODEL] [-e EFFORT] [-b BIN] [-o OUTDIR] [FILE ...]" >&2; exit 3 ;;
  esac
done
shift $((OPTIND - 1))

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(
    "README.md"
    "Cargo.toml"
    "crates/runtime/src/jsonl_log.rs"
    "crates/zo-cli/src/sinks/text.rs"
    "crates/runtime/src/green_contract.rs"
    "crates/runtime/src/image_guard.rs"
  )
fi

if [ -z "$OUTDIR" ]; then
  OUTDIR="$(mktemp -d "${TMPDIR:-/tmp}/zo-usage-bench.XXXXXX")"
fi
mkdir -p "$OUTDIR"

if [ ! -x "$ZO_BIN" ]; then
  echo "error: zo binary not found/executable: $ZO_BIN" >&2
  exit 3
fi

"$ZO_BIN" --version | sed 's/^/[bench] binary: /' >&2 || true
echo "[bench] model=$MODEL effort=$EFFORT turns=${#FILES[@]} outdir=$OUTDIR" >&2

# --- 1. settings overlay: pin the model by disabling smart routing ----------
SETTINGS_OVERLAY="$OUTDIR/bench_settings.json"
printf '%s' '{"smart": {"enabled": false}}' > "$SETTINGS_OVERLAY"

# --- 2. build the NDJSON turn script ----------------------------------------
TURNS_FILE="$OUTDIR/turns.ndjson"
: > "$TURNS_FILE"
for f in "${FILES[@]}"; do
  abs="$f"
  case "$abs" in
    /*) : ;;
    *) abs="$REPO_ROOT/$f" ;;
  esac
  if [ ! -f "$abs" ]; then
    echo "error: turn target file not found: $abs" >&2
    exit 3
  fi
  python3 - "$abs" >> "$TURNS_FILE" <<'PY'
import json, sys
path = sys.argv[1]
prompt = (
    f"Use the read_file tool to read {path} "
    "and then summarize that file in exactly one sentence."
)
print(json.dumps({"role": "user", "content": prompt}))
PY
done

N_TURNS="${#FILES[@]}"
# --max-turns caps BOTH the number of stdin lines processed AND the agentic
# loop iterations per turn; each bench turn needs ~2-3 iterations, so the cap
# must be max(N_TURNS, per-turn-iterations) with headroom.
MAX_TURNS=$(( N_TURNS > 6 ? N_TURNS + 2 : 8 ))

# --- 3. run one zo process = one session, N user turns --------------------
RAW="$OUTDIR/stream.ndjson"
ERR="$OUTDIR/stream.stderr"
echo "[bench] running $N_TURNS turns in one session ..." >&2
set +e
( cd "$REPO_ROOT" && \
  ZO_EFFORT="$EFFORT" "$ZO_BIN" -p \
    --model "$MODEL" \
    --output-format stream-json \
    --input-format stream-json \
    --max-turns "$MAX_TURNS" \
    --max-tool-calls 4 \
    --permission-mode read-only \
    --strict-mcp-config \
    --settings "$SETTINGS_OVERLAY" \
    < "$TURNS_FILE" > "$RAW" 2> "$ERR" )
ZO_EXIT=$?
set -e
if [ "$ZO_EXIT" -ne 0 ]; then
  echo "[bench] zo exited non-zero ($ZO_EXIT); stderr tail:" >&2
  tail -5 "$ERR" >&2 || true
  # keep going if we captured at least some usage events; the analyzer
  # will report on whatever arrived.
fi

# --- 4. analyze: per-request deltas from session-cumulative usage events -----
set +e
python3 - "$RAW" "$N_TURNS" <<'PY'
import json, sys

raw_path, n_turns = sys.argv[1], int(sys.argv[2])

events = []          # (turn_idx, ctx_tokens, cumulative dict)
sessions = set()
turn = 1
with open(raw_path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "usage":
            cum = {k: obj.get(k, 0) for k in
                   ("input_tokens", "output_tokens",
                    "cache_read_tokens", "cache_creation_tokens")}
            if all(v == 0 for v in cum.values()):
                continue  # ctx-only snapshot at message_start
            events.append((turn, obj.get("ctx_tokens", 0), cum))
        elif t == "result":
            sessions.add(obj.get("session_id"))
            turn += 1

if not events:
    print("no usage events captured — run failed before the first response")
    sys.exit(3)

# dedupe identical consecutive cumulative snapshots
deduped = []
for ev in events:
    if deduped and deduped[-1][2] == ev[2]:
        continue
    deduped.append(ev)

print(f"session ids observed: {len(sessions)} {sorted(s for s in sessions if s)}")
print(f"user turns completed: {turn - 1} / {n_turns}")
print()
hdr = (f"{'req':>3} {'turn':>4} {'ctx_tok':>8} {'d_input':>8} "
       f"{'d_cache_rd':>10} {'d_cache_cr':>10} {'d_output':>8} "
       f"{'rd/(in+rd)':>10} {'rd/total_in':>11}")
print(hdr)
print("-" * len(hdr))

prev = {"input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_creation_tokens": 0}
rows = []
for i, (tn, ctx, cum) in enumerate(deduped, 1):
    d = {k: cum[k] - prev[k] for k in cum}
    prev = cum
    din, drd, dcr = d["input_tokens"], d["cache_read_tokens"], d["cache_creation_tokens"]
    ratio_task = drd / (din + drd) if (din + drd) > 0 else 0.0
    total_in = din + drd + dcr
    ratio_total = drd / total_in if total_in > 0 else 0.0
    rows.append((i, tn, ctx, din, drd, dcr, d["output_tokens"], ratio_task, ratio_total))
    print(f"{i:>3} {tn:>4} {ctx:>8} {din:>8} {drd:>10} {dcr:>10} "
          f"{d['output_tokens']:>8} {ratio_task:>10.1%} {ratio_total:>11.1%}")

# --- verdict ---------------------------------------------------------------
# Judged on requests belonging to user turn >= 2 (turn 1 is legitimately cold).
judged = [r for r in rows if r[1] >= 2]
print()
if not judged:
    print("VERDICT: INCONCLUSIVE — no turn>=2 requests captured")
    sys.exit(2)

low = [r for r in judged if r[7] < 0.80]
# churn signature: ratio collapses while cache_creation rebuilds >50% of ctx
churn = [r for r in judged if r[7] < 0.50 and r[2] > 0 and r[5] > 0.5 * r[2]]

mean_task = sum(r[7] for r in judged) / len(judged)
min_task = min(r[7] for r in judged)
print(f"turn>=2 requests: {len(judged)} | ratio cache_read/(input+cache_read): "
      f"mean {mean_task:.1%}, min {min_task:.1%}")

if len(churn) >= 2:
    print("VERDICT: FAIL — repeated full-prefix cache re-creation (cache churn signature)")
    sys.exit(1)
if not low:
    print("VERDICT: PASS — warm cache sustained on every turn>=2 request (>=80%)")
    sys.exit(0)
print(f"VERDICT: INCONCLUSIVE — {len(low)}/{len(judged)} turn>=2 requests below 80%")
sys.exit(2)
PY
ANALYZE_EXIT=$?
set -e

echo "[bench] artifacts: $OUTDIR (turns.ndjson, stream.ndjson, stream.stderr)" >&2
exit "$ANALYZE_EXIT"
