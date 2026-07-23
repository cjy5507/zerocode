# Privacy

zo is a local command-line tool. It runs on your machine and stores its
state on your disk. This document explains what data leaves your machine, when,
and how to control it.

## What is sent to model providers

zo is an AI coding agent, so by design it sends your input to the model
provider you have selected (Anthropic by default; optionally OpenAI, Google, or
xAI). To answer your prompts, the following may be transmitted to that
provider's API:

- your typed prompts and instructions;
- the contents of files zo reads on your behalf, and tool/command output it
  feeds back to the model;
- conversation history for the current session (so the model has context);
- system prompt and tool definitions.

This is inherent to how an AI coding agent works. Only the provider you
authenticate against receives this data. Review your provider's privacy policy
and terms of service for how they handle it.

## Telemetry

Telemetry is **off by default**. zo emits no metrics, traces, or logs to any
network endpoint unless you explicitly opt in.

- Enable with `ZO_ENABLE_TELEMETRY=1` (or the compatibility variable
  `CLAUDE_CODE_ENABLE_TELEMETRY=1`).
- When enabled, zo uses standard OpenTelemetry (OTLP). Configure the
  destination and behavior with the usual `OTEL_*` environment variables
  (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`,
  `OTEL_TRACES_EXPORTER`, `OTEL_METRICS_EXPORTER`, `OTEL_LOGS_EXPORTER`,
  `OTEL_SERVICE_NAME`, …). With no endpoint configured, nothing is exported.

## Local storage

zo keeps state on your local filesystem:

- **Global user state** under `~/.zo/` (overridable with `ZO_CONFIG_HOME`
  or `ZO_HOME`):
  - `credentials.json` — OAuth tokens / API credentials;
  - `registered_clients.json` — MCP OAuth client registrations;
  - `settings.json` — user configuration;
  - `sessions/*.jsonl` — saved session transcripts.
- **Per-project state** under `<project>/.zo/` (relocatable with
  `ZO_STATE_DIR`): session files, turn traces, todos, and share/export
  artifacts.

On Unix, zo restricts its credential, session, and turn-trace files and
directories to owner-only permissions (`0o600` files / `0o700` directories) so
other local users cannot read your prompts, transcripts, or tokens. These files
are never transmitted anywhere except as needed to call your chosen provider.

## Sharing (`/share`)

The `/share` command is opt-in and you invoke it explicitly. It:

- always writes a local copy of the transcript to `.zo/share/<id>.txt`; and
- uploads a redacted copy to a GitHub gist (created **secret/unlisted**,
  attributed to your active `gh` account).

A secret gist is unlisted, **not private** — anyone with the link can read it.
Redaction masks common secret patterns (API-key prefixes, `KEY=`/`TOKEN=`/
`SECRET=`/`PASSWORD=` assignments, `Authorization` headers) on a best-effort
basis and **cannot guarantee** that every secret is removed. Review content
before sharing. Revoke a shared gist any time with `/unshare <id>`.

## Network access

Beyond model-provider API calls and an explicit `/share`, zo makes network
requests only for features you invoke — for example web fetch/search tools, MCP
servers you configure, and OAuth login flows. It does not phone home.
