#!/usr/bin/env bash
#
# Build the self-contained Zo-vs-OpenCode analysis benchmark prompt.
#
# The older prompt told the agent to read the evidence pack as its first tool
# call. That preserved accuracy, but it still encouraged an avoidable discovery
# loop. This builder inlines the verified current-code pack into the prompt so
# the model starts with the anchors already in context and should spend tools
# only on disputed or missing facts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="$ROOT/bench/evidence/opencode-gap-current-anchors.md"

die() {
  echo "build-opencode-analysis-prompt: $*" >&2
  exit 2
}

[ -f "$PACK" ] || die "missing evidence pack: $PACK"

QUESTION="${*:-}"
if [ -z "$QUESTION" ] && [ ! -t 0 ]; then
  QUESTION="$(cat)"
fi
[ -n "${QUESTION//[[:space:]]/}" ] || die "question argument or stdin is required"

cat <<'PROMPT_HEADER'
# Zo <-> OpenCode analysis benchmark

You are answering an analysis benchmark. Accuracy, evidence discipline, and
minimal tool use matter more than breadth.

## Embedded current-code evidence pack

The evidence pack below is already in this prompt. Treat it as Phase-1 ground
truth. Do NOT call tools merely to read the pack again. Use tools only for
targeted verification of disputed or missing current-code facts.

Every P0/P1 Zo claim in the final answer must be one of:
- `verified-current-code`: backed by an embedded anchor or a freshly read line.
- `verified-benchmark-artifact`: backed by a benchmark artifact.
- `inference`: clearly labelled as inference.
- `not-verified`: explicitly marked as not verified.

If you read a current-code line that is not already in the embedded pack, the
final answer must say `verified-current-code (fresh read: path:line)` for that
claim. Do not group a verified claim and a not-verified claim in the same bullet
without tagging each part separately. When in doubt, say `not-verified` instead
of inventing a line anchor.

In the final answer, use the tag itself; do not narrate provenance mechanics such
as "embedded anchor", "no fresh reads", "the pack says", or "do not call it a
stub". Those are benchmark instructions, not answer content. The final answer
must not contain the word `stub` unless the user explicitly asks about stubs.
Use `path:line` or `path:line-range` citation syntax with a colon, never
space-separated `path line` citations.
Embedded current-code anchors are `verified-current-code`, not
`verified-benchmark-artifact`. Use `verified-benchmark-artifact` only for claims
about benchmark result files, not source-code anchors.

Before the final answer, make sure the top recommendations are anchored. Prefer
one targeted file read over repo-wide grep. Do not grep `docs/` for current
state; stale docs lose to current code.

PROMPT_HEADER

cat "$PACK"

cat <<'PROMPT_FOOTER'

---

## Answer Contract

Compare Zo against OpenCode using the official OpenCode behavior categories
already referenced by the benchmark: provider breadth, agents, permissions, MCP,
share, LSP/editor surfaces, install/onboarding, and serve/attach style product
surfaces.

For each high-value item, state:
- Zo current state with file:line or embedded anchor label.
- OpenCode behavior or expected parity target.
- Verdict: real gap, closed gap, intentional difference, or not verified.
- Clean-code improvement needed, if any.

Mandatory coverage checklist:
- Built-in provider catalog breadth/data-driven discovery.
- Hosted always-on share URL versus Zo local/gist share.
- Permission-ordering documentation/consistency across global ordered rules and
  per-agent category buckets. Treat the behavior split as intentional, but the
  documentation/consistency work as a real improvement gap.
- LSP/editor: the embedded pack verifies client/tool infrastructure; editor-grade
  UX depth, auto-detection, and long-run e2e quality remain separate verification
  targets unless freshly anchored.
- Free-model/install/onboarding: the embedded pack verifies keyless self-hosted
  OpenAI-compatible providers; hosted free-model onboarding remains separate and
  `not-verified` unless freshly anchored.
- Local `/desktop` handoff versus richer desktop/IDE surfaces.

Lead with the real, still-open gaps. Keep the answer concise enough that output
tokens stay below the benchmark target. Do not re-explain every embedded anchor.
Do not turn unverified editor/desktop/LSP depth into a verified current-code
claim; use the embedded `/desktop` anchor only for the local session-file handoff.
If a ranked item is not freshly anchored as absent, its `Verdict` must be
`not verified`, not `real gap`. This applies especially to hosted free-model
onboarding, richer desktop/IDE product depth, and editor-grade LSP depth.
Prefer ranked gap numbers over P0/P1 severity labels unless the evidence pack
explicitly gives a severity.

Answer shape:
- Start the answer exactly with `## Ranked open gaps`; no preamble before that
  heading.
- Use these ranked headings exactly, in this order:
  1. `Built-in provider catalog breadth / data-driven discovery`
  2. `Hosted always-on share URL service`
  3. `Permission-ordering documentation / consistency`
  4. `LSP/editor UX depth and auto-detection`
  5. `Hosted free-model onboarding`
  6. `Desktop / IDE product surfaces`
- Format each ranked heading as a bold numbered line, e.g.
  `**1. Built-in provider catalog breadth / data-driven discovery**`. Do not use
  Markdown subheadings (`###`) below `## Ranked open gaps`.
- For each ranked open gap, use exactly four compact fields: `Zo`, `OpenCode`,
  `Verdict`, `Action`.
- The `Verdict` field must be exactly one short category sentence with no
  parentheses, no semicolon, and no provenance words. Use these exact verdicts:
  1. `Verdict: real gap.`
  2. `Verdict: real gap.`
  3. `Verdict: real gap.`
  4. `Verdict: not verified.`
  5. `Verdict: not verified.`
  6. `Verdict: not verified.`
  Put intentional-behavior and verified-infrastructure nuance in the `Zo`
  field, not in `Verdict`.
- Keep closed gaps to one compact `Do not regress` list, at most four bullets,
  with one short sentence per bullet. Do not add long-run caveats there.
- Start the closed-gap section exactly with `## Do not regress`; do not use
  `### Do not regress`.
- In `Do not regress`, include `verified-current-code` on each bullet, but keep
  each bullet terse. Do not enumerate serve/attach subfeatures or internal details
  such as hard-deny timeout or second-connection cancel; just say the F1-F4 path is
  implemented and tested with citations.
- Do not add a separate cross-cutting caveat if the same caveat was already stated
  in a ranked item or closed-gap bullet.
- Do not mention tool use, reads, hooks, "embedded anchors", "no fresh reads", or
  benchmark do-not-misjudge wording in the final answer. In particular, do not
  write "not a stub", "do not call it", or "not a ... gap"; state the missing
  target positively instead. Avoid contrast caveats such as "distinct from X" in
  `Verdict`; put only the positive verdict category there.
- Do not use the words `anchor`, `anchored`, or `freshly` in the final answer.
- In the share gap, cite both the local-artifact helper anchor
  (`live_cli_commands.rs:145–183`) and the share/gist command anchor
  (`live_cli_commands.rs:1235–1267` or `toggles.rs:297–361`).
- Make actions implementation-specific: provider action should mention decoupling
  the catalog list from first-class provider behavior/`ProviderKind`; permission
  action should mention a grammar-to-semantics doc/table and a warning or lint for
  order-dependent per-agent rules.
- For LSP, mention an e2e harness around the existing `LspRegistry::dispatch`
  path plus auto-detection/editor-surface work; for hosted free-model onboarding,
  mention a first-run hosted profile that reuses the existing keyless
  OpenAI-compatible client path; for desktop/IDE, mention reusing session-path and
  attach/session plumbing beyond the local file open.
- For share, the `Action` must explicitly name `ShareArtifact` and the redaction
  path.
- Do not include `file_ops` edit-uniqueness in the closed-gap list unless the
  question asks about editing semantics.
- Do not call a `not-verified` item "likely"; either mark it verified,
  inference, or not-verified.

## Question

PROMPT_FOOTER

printf '%s\n' "$QUESTION"
