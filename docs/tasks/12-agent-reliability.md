# Task 12: 12-Factor Agent Reliability Upgrade

## Overview

Upgrade the natural-language CLI Agent from a process-local tool loop to a
durable, resumable execution system while retaining TubeTalk's deterministic
service boundary. The implementation is CLI-first: SQLite persists runs under
`DATA_DIR`, and a reusable application boundary leaves HTTP, webhook, and batch
adapters for later work.

## Decisions

- Each run has an append-only event log and a stable `run_id`. The reducer
  reconstructs its state and model context solely from persisted events.
- Event data is retained until the user explicitly deletes a run. Events never
  store API keys, full rendered prompts, or transcript source text.
- Read-only tools run immediately. New external analysis/generation calls and
  cache regeneration pause for an explicit user approval.
- The CLI remains the only delivered trigger in this task. Its implementation
  calls a reusable application service so later interface adapters do not own
  Agent control flow.
- Long conversations and large tool results are compacted to configurable
  context budgets. User-visible evidence and citations remain intact.
- Every input and output at a tool boundary is a validated Pydantic model;
  model-facing results are compact, structured, and safe to persist.

## Checklist

- [x] **Check 12.1**: Add this Task 12 specification, the Phase 6 roadmap entry,
  and the durable-run architecture to the design document.
- [x] **Check 12.2**: Add immutable Agent run/event/state contracts and a
  SQLite-backed append-only repository with unit tests for atomic persistence,
  listing, loading, and explicit deletion.
- [x] **Check 12.3**: Turn `AgentSession` into an event-driven reducer that
  rebuilds state and model context from persisted events; attach run IDs to
  structured diagnostics and add reconstruction tests.
- [x] **Check 12.4**: Add a reusable `AgentRunService` with launch, status,
  approve, reject, resume, list, and delete operations; enforce state
  transitions and never re-run completed calls on resume.
- [x] **Check 12.5**: Add configurable context budgets and deterministic compact
  views for conversation history and large tool results, with preservation tests
  for current-video context and citations.
- [x] **Check 12.6**: Replace generic tool-result dictionaries with per-tool
  validated input/output contracts, call IDs, compact error codes, and safe
  recommended next actions.
- [x] **Check 12.7**: Add an Agent-visible approval request tool and policy:
  read-only operations run immediately, while new paid generation/analysis and
  cache regeneration wait for approval.
- [x] **Check 12.8**: Extend the Typer CLI with run inspection, approval,
  rejection, resumption, and deletion commands; Ctrl-C must leave a resumable
  paused run.
- [x] **Check 12.9**: Define the trigger port used by the CLI and document how a
  future HTTP, webhook, or batch adapter invokes the same run service without
  duplicating Agent flow.
- [x] **Check 12.10**: Update user-facing documentation; cover persistence,
  state transitions, resume idempotency, approvals, compaction, tool contracts,
  and redaction with mocked unit tests; run the quality gate.

## Completion Rules

Implement exactly one unchecked item at a time. After each implementation item,
run `poetry run poe check`, report the result for review, receive approval, and
commit before continuing to the next item.
