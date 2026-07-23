//! Registered `ToolSpec` table for every tool in this crate.
//!
//! Each entry declares the wire-level schema (`input_schema`) plus the
//! permission level the tool runs under. The dispatcher in `super`
//! looks the spec up by name when a model invokes the tool.
//!
//! Kept separate from the dispatchers (`run_*` and `execute_*`) so
//! reading the catalogue is independent from reading the executor
//! plumbing.

use runtime::PermissionMode;
use serde_json::json;

use super::{
    ToolSpec, MAX_COUNCIL_CANDIDATES, MAX_COUNCIL_CANDIDATE_CHARS, MAX_SEND_TO_USER_CHARS,
    MAX_SPAWN_MULTI_AGENT_AGENTS,
};

// --- tool_specs and dispatch ---

#[allow(clippy::too_many_lines)] // a flat spec table, clearer unsplit
pub(crate) fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "Skill",
            description: "Load a Zo skill from this repo's `.zo/skills/<name>/SKILL.md` or Zo's global skill directory (`ZO_CONFIG_HOME/skills`, `ZO_HOME/skills`, or `~/.zo/skills`) and run its procedure. Use only when the user names a discovered skill or the task clearly calls for a Zo skill; do not load non-Zo global skill stores.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "skill": { "type": "string" },
                    "args": { "type": "string" }
                },
                "required": ["skill"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "SkillDistill",
            description: "Write a reusable skill draft as `.zo/skills/<slug>/SKILL.md` with `state: proposed`. Use only after a task reveals repeatable procedure knowledge worth saving; proposed drafts are not auto-activated until approved. To evolve an existing draft (or augment one a duplicate check pointed you to), pass `update: true` to bump its version and rewrite the body.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": "Lowercase kebab-case skill slug, used as the directory name under .zo/skills."
                    },
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "body": { "type": "string" },
                    "update": {
                        "type": "boolean",
                        "description": "Re-distill an existing same-slug draft: bump its version and rewrite the body (stays proposed). Defaults to false, which refuses to overwrite."
                    }
                },
                "required": ["slug", "description", "body"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "SkillReview",
            description: "Approve or discard a proposed skill draft created under `.zo/skills/<slug>/SKILL.md`. Approval changes `state: proposed` to `state: active`; discard removes the draft cleanly.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": "Lowercase kebab-case skill slug under .zo/skills."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["approve", "discard"]
                    }
                },
                "required": ["slug", "action"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "Agent",
            description: "Launch one specialized sub-agent to handle an explicitly delegated task in its own context. Use `Agent` for a single focused specialist; do not treat several blocking `Agent` calls as a parallel swarm. Delegate to preserve context or to run genuinely independent work in parallel — for a simple question, a single-fact lookup, or a small bounded change you can make in a few tool calls, work directly instead of spawning anything. For real parallel fan-out across independent subtasks, use `SpawnMultiAgent`; for dependent plan→implement→verify work, use `Workflow`. `subagent_type` picks the harness (tool allowlist + system prompt): `general-purpose` (default, full tools), `Explore` (read-only codebase search), `Plan` (architecture/design, no edits), `Verification` (run tests/builds), `deep-research` (multi-pass code+web investigation, no edits), `code-reviewer` (adversarial review, no edits), `debugger` (reproduce→root-cause→fix), `data-analyst` (data/log/metric analysis), `refactor` (behavior-preserving cleanup), `zo-guide`, `statusline-setup`, or any custom type defined in `.zo/agents/<name>.md`. Sub-agents inherit the active parent/session model by default; an explicit `model` is honored inside the same provider family, and crossing provider families additionally requires `allow_cross_provider: true` — set it ONLY when the user explicitly asked for that model, and never substitute a different model when the user named one. The user-level ZO_AGENT_MODEL override still forces one model for every sub-agent. Task difficulty tunes the inherited model's reasoning budget/effort rather than silently switching to a different model. In the interactive main session this call is DETACHED by default (`background` omitted = background): it returns immediately with `status: \"running\"` and you are notified when the agent completes. Keep making progress on OTHER independent work meanwhile — if the agent finishes while your turn is still running, its result is delivered to you MID-TURN at your next tool boundary as a task notification; if your turn has already ended, it arrives as a follow-up message. Do NOT poll the output file and do NOT idle-wait for the result; when nothing is left but waiting, end your turn. Set `background: false` to block until the sub-agent finishes and get its result inline (status, result, error) — use that only when you cannot take a single further step without the result. In sub-agent and headless contexts an omitted `background` defaults to blocking instead, because no host is present to deliver a detached result. The sub-agent's final message comes back to YOU as the tool result — the user never sees it, so relay what matters in your own reply. Once you delegate a search or investigation, do not also run it yourself — wait for the result.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "description": { "type": "string" },
                    "prompt": { "type": "string" },
                    "subagent_type": { "type": "string" },
                    "name": { "type": "string" },
                    "model": { "type": "string", "description": "Explicit model for this sub-agent. Same provider family as the session by default; crossing families requires allow_cross_provider." },
                    "allow_cross_provider": { "type": "boolean", "description": "Set true ONLY when the user explicitly asked for a model outside the session's provider family." },
                    "background": { "type": "boolean" }
                },
                "required": ["description", "prompt"],
                "additionalProperties": false
            }),
            // Spawning is not itself a privileged act: the child's enforcer is
            // clamped to the parent's active mode (`clamped_spawn_mode`), so a
            // read-only session can delegate read-only research/analysis.
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "ToolSearch",
            description: "Load deferred tools (builtin orchestration families, MCP-server and plugin tools) by exact name (query \"select:Name1,Name2\") or keywords. Returns each match's full schema and adds it to your tool list for subsequent turns. Batch the load: request every tool you expect the task to need in ONE call — the select query accepts a comma-separated list — instead of one call per tool, which wastes a full round-trip each.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_results": { "type": "integer", "minimum": 1 }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "Audit",
            description: "Summarize this session's tool-invocation ledger: how many tools ran, how many the policy allowed / denied / failed, per-family counts, and the reason for every denial. Read-only; takes no arguments.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "NotebookEdit",
            description: "Replace, insert, or delete a cell in a Jupyter notebook.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "notebook_path": { "type": "string" },
                    "cell_id": { "type": "string" },
                    "new_source": { "type": "string" },
                    "cell_type": { "type": "string", "enum": ["code", "markdown"] },
                    "edit_mode": { "type": "string", "enum": ["replace", "insert", "delete"] }
                },
                "required": ["notebook_path"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "Sleep",
            description: "Wait for a specified duration without holding a shell process.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "duration_ms": { "type": "integer", "minimum": 0 }
                },
                "required": ["duration_ms"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "send_to_user",
            description: "During a long autonomous run, push important verbatim content straight to the user without ending the turn — a finding, a diff, a URL, a config block, the exact text they must see now. This is NOT a status report or progress ping: reserve it for content that is worth showing verbatim, and do not narrate every step with it (spamming the user each turn defeats the purpose). In headless / sub-agent runs with no interactive surface the message is returned inline instead.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "maxLength": MAX_SEND_TO_USER_CHARS
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "Brief",
            description: "Legacy alias for `send_to_user`: push a message to the user mid-run. Prefer `send_to_user`.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "SyntheticOutput",
            description: "Inject synthetic tool output for testing or scripted flows.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tool_name": { "type": "string" },
                    "output": { "type": "string" }
                },
                "required": ["tool_name", "output"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
        ToolSpec {
            name: "SpawnMultiAgent",
            description: "Spawn multiple sub-agents in one flat parallel fan-out and collect their results once all finish. Use this for independent research, review, verification, or implementation slices that should run as a real swarm; use `Workflow` instead when phases depend on earlier results. Size the fan-out to the ask, not the session mode: never spawn a swarm or a multi-perspective verification panel for a simple question, a lookup, or a bounded single-file fix — those take zero agents, or at most one. Each entry's `subagent_type` selects that agent's harness (see Agent). Provider requests flow through an adaptive per-provider rate governor that opens with genuine parallelism when headroom is healthy, stays serial during recent rate-limit pressure, ramps while quota has headroom, and backs off on rate limits; optional `concurrency` can only tighten this. Sub-agents inherit the active parent/session model by default; a per-agent `model` is honored inside the same provider family, and crossing provider families additionally requires that member's `allow_cross_provider: true` — set it ONLY when the user explicitly asked for that model, and never substitute a different model when the user named one. The user-level ZO_AGENT_MODEL override still forces one model for every sub-agent. Task difficulty tunes the inherited model's reasoning budget/effort rather than silently switching to a different model.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "maxItems": MAX_SPAWN_MULTI_AGENT_AGENTS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "subagent_type": { "type": "string" },
                                "prompt": { "type": "string" },
                                "description": { "type": "string" },
                                "name": { "type": "string" },
                                "model": { "type": "string" },
                                "allow_cross_provider": { "type": "boolean", "description": "Set true ONLY when the user explicitly asked for a model outside the session's provider family." }
                            },
                            "required": ["prompt"]
                        }
                    },
                    "concurrency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SPAWN_MULTI_AGENT_AGENTS,
                        "description": "Live sub-agent window; unset defaults to min(16, cores-2) and later members queue for freed slots. Provider requests still pass through the adaptive per-provider rate governor, so widening this never raises real API concurrency."
                    }
                },
                "required": ["agents"],
                "additionalProperties": false
            }),
            // Like `Agent`: spawning is unprivileged, the members' enforcers
            // are clamped to the parent's active mode (`clamped_spawn_mode`).
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "Council",
            description: "Compare multiple candidate answers for the same task without exposing candidate source/model identity. Use after fan-out (for example SpawnMultiAgent) to select a self-consistent majority when one exists, or return an honest tie when candidates fail or disagree.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "maxItems": MAX_COUNCIL_CANDIDATES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "maxLength": MAX_COUNCIL_CANDIDATE_CHARS
                                },
                                "status": { "type": "string" }
                            },
                            "required": ["text"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["candidates"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "SendUserMessage",
            description: "Legacy alias for `send_to_user`: push a message to the user mid-run. Prefer `send_to_user`.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "Config",
            description: "Get or set Zo settings.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "setting": { "type": "string" },
                    "value": {
                        "type": ["string", "boolean", "number"]
                    }
                },
                "required": ["setting"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "EnterPlanMode",
            description: "Enable a worktree-local planning mode override and remember the previous local setting for ExitPlanMode.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "ExitPlanMode",
            description: "Clear the worktree-local settings override created by EnterPlanMode (settings-file management; needs write access). NOT the plan-submission tool: while in read-only plan mode, present your plan with ExitPlanModeV2 instead — plan mode is lifted only by the user.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "StructuredOutput",
            description: "Return structured output in the requested format.",
            input_schema: json!({
                "type": "object",
                "additionalProperties": true
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "AskUserQuestion",
            description: "Ask the user a question and wait for their response. Reserve this for decisions where the answer changes what you do next — a missing target, success criterion, or safety boundary with more than one viable path. Do not use it for choices with a conventional default or facts you can verify in the codebase yourself: pick the obvious option, say so, and proceed. Offer 2-4 options with one-line descriptions of their tradeoffs; the user can always type a free-form answer instead, so the returned answer may not match any option. Options are mutually exclusive by default; set `multiSelect: true` when the user may pick several, in which case `answer` comes back as an array.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete question to ask, ending with a question mark."
                    },
                    "header": {
                        "type": "string",
                        "description": "Very short topic chip shown beside the prompt title (max ~12 chars), e.g. \"Auth method\"."
                    },
                    "options": {
                        "type": "array",
                        "description": "Fixed choices in display order. Plain strings or {label, description} objects; the description explains the tradeoff in one line.",
                        "items": {
                            "oneOf": [
                                { "type": "string" },
                                {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string" },
                                        "description": { "type": "string" }
                                    },
                                    "required": ["label"],
                                    "additionalProperties": false
                                }
                            ]
                        }
                    },
                    "multiSelect": {
                        "type": "boolean",
                        "description": "When true, the user may check several options and the `answer` is returned as an array of the selected values. Defaults to false (a single mutually-exclusive choice)."
                    }
                },
                "required": ["question"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "MemoryWrite",
            description: "Atomically record a persistent project memory entry: writes `<global-project-memory>/<slug>.md` and upserts its pointer line in `<global-project-memory>/MEMORY.md` in one tool call. The global project memory lives under Zo's config home (`~/.zo/projects/<project-slug>/memory` by default), not in the repository. Set `local: true` to write the machine-local overlay (`memory.local`) instead of the durable store; recall merges both.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "slug": { "type": "string" },
                    "summary": { "type": "string" },
                    "body": { "type": "string" },
                    "local": { "type": "boolean", "description": "Write the global per-project `memory.local` overlay instead of the durable `memory` store. Defaults to false." }
                },
                "required": ["slug", "summary", "body"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "RemoteTrigger",
            description: "Trigger a remote action or webhook endpoint.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string" },
                    "method": { "type": "string", "enum": ["GET", "POST", "PUT", "DELETE"] },
                    "headers": { "type": "object" },
                    "body": { "type": "string" }
                },
                "required": ["url"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
        ToolSpec {
            name: "TestingPermission",
            description: "Test-only tool for verifying permission enforcement behavior.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string" }
                },
                "required": ["action"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
        ToolSpec {
            name: "Monitor",
            description: "Stream stdout lines from a background process as notifications.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "process_id": { "type": "string" },
                    "command": { "type": "string" },
                    "lines": { "type": "integer", "minimum": 1 }
                },
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "SendMessage",
            description: "Send a message to a previously spawned agent by name or id. A RUNNING \
                          agent receives it mid-turn (injected at its next tool boundary) and \
                          keeps working with the new guidance. A COMPLETED/failed/stopped agent \
                          is RESUMED in the background with its full prior context intact and \
                          your message as its next turn — its reply is delivered to you in a \
                          later message, so do NOT poll. Use this to steer, follow up, or ask \
                          an agent to go deeper instead of spawning a fresh agent and \
                          re-explaining the task.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "to": { "type": "string", "description": "Agent name or id (prefix match allowed)" },
                    "message": { "type": "string" }
                },
                "required": ["to", "message"],
                "additionalProperties": false
            }),
            // Same grade as `Agent`/`SpawnMultiAgent`: the resume path re-spawns
            // a worker with its original permission envelope, so a lower grade
            // here would let a read-only context relaunch a full-access agent.
            required_permission: PermissionMode::DangerFullAccess,
        },
        ToolSpec {
            name: "ScheduleWakeup",
            description: "Schedule a delayed re-invocation for dynamic loop mode.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "delaySeconds": { "type": "number", "minimum": 0 },
                    "reason": { "type": "string" },
                    "prompt": { "type": "string" }
                },
                "required": ["delaySeconds", "reason", "prompt"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "retrieve_tool_output",
            description: "Recover the FULL original of a tool output that was truncated. When a result ends with a notice naming a sha256 artifact id, call this with that id to read the untruncated output from the local artifact store; narrow large outputs with `offset`/`limit` (a 0-based line window, same semantics as read_file).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "sha256": { "type": "string", "description": "64-char hex id from the truncation notice" },
                    "offset": { "type": "integer", "minimum": 0, "description": "0-based first line of the window" },
                    "limit": { "type": "integer", "minimum": 1, "description": "maximum lines to return" }
                },
                "required": ["sha256"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
        ToolSpec {
            name: "session_recall",
            description: "Selectively read a PAST conversation (read-only) without resuming it. Two modes: (1) RECALL — pass `session_ref` (a session id, \"latest\"/\"last\"/\"recent\", or \"current\" for THIS session) to read that session, optionally narrowed by a substring `query`, a `role`, a `last_n` tail, and/or a `seq_from`/`seq_to` window; (2) SEARCH — omit `session_ref` and pass `query` to find which prior sessions discussed something, with per-session match counts, optionally restricted to a `since_days`/`before_days` time window. Use \"current\" with a `seq_from`/`seq_to` range to pull back the EXACT raw originals a compaction round sealed to this session's vault (the continuation message names the spans). Returns only the matched excerpt as text and never touches the live transcript.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_ref": { "type": "string", "description": "Session id, \"latest\"/\"last\"/\"recent\", or \"current\" (THIS session, e.g. to recover its just-compacted originals). Omit (or \"all\") + query to search across all sessions. Paths are not accepted; discover ids via search mode." },
                    "query": { "type": "string", "description": "Case-insensitive substring to match in message text. Required in search mode." },
                    "role": { "type": "string", "enum": ["user", "assistant", "tool", "system"] },
                    "last_n": { "type": "integer", "minimum": 1, "description": "Return only the last N matching messages." },
                    "seq_from": { "type": "integer", "minimum": 0, "description": "Inclusive lower bound of the seq window. Seqs are one monotonic domain per session: an evicted vault record's vault_seq and a live message's absolute index share it, so this window addresses both. Combined (AND) with query/role." },
                    "seq_to": { "type": "integer", "minimum": 0, "description": "Inclusive upper bound of the seq window. Must be >= seq_from." },
                    "include_tool_results": { "type": "boolean", "description": "Include tool-result blocks in the output (default true). Set false to omit them; a message with no other content is then skipped." },
                    "since_days": { "type": "number", "minimum": 0, "description": "SEARCH only: keep sessions modified within the last N days (recent edge; fractional ok). Rejected in recall mode." },
                    "before_days": { "type": "number", "minimum": 0, "description": "SEARCH only: keep sessions modified more than N days ago (older edge). Pair with since_days to bracket a span (since_days >= before_days); since_days < before_days is an empty window and is rejected." }
                },
                "additionalProperties": false
            }),
            required_permission: PermissionMode::ReadOnly,
        },
    ]
}
