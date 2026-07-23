# Task 03: Transcript summary and timestamp chapters

## Overview

Generate a Korean, transcript-grounded summary and chronological table of
contents with Gemini Flash-Lite. Cache the result locally and regenerate it
when its transcript or generation settings change.

## Decisions

- The default model is `gemini-3.5-flash-lite`; `summary_model` may override it.
- `summary_language` defaults to `ko` and may be configured.
- `summary.json` stores summary text, `{start_sec, title}` chapters, transcript
  digest, model, prompt version, language, and generation time.
- `process` refreshes and renders missing or stale summaries. `summary` reads a
  current cache, while `summary --generate` explicitly generates a missing or
  stale cache.
- `status` always exposes summary freshness and its available generation
  metadata alongside the other cached video artefacts.

## Checklist

- [x] **Check 3.1**: Add summary domain models, cache manifest persistence,
  freshness validation, and unit tests.
- [x] **Check 3.2**: Add the Gemini Flash-Lite summary port and adapter,
  production wiring, and mocked unit tests.
- [x] **Check 3.3**: Synchronize summaries from the video service and preserve
  cached resources when generation fails.
- [ ] **Check 3.4**: Add the `summary` CLI command and process rendering with
  unit tests.
- [ ] **Check 3.5**: Update documentation and roadmap after the feature is
  complete.
