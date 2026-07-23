# Task 03: Interface-independent video service refactor

## Overview

Move video processing and cache-status use cases out of the CLI so a future
FastAPI interface can reuse their application logic without duplicating it.

## Checklist

- [x] **Check 3.1**: Add the video application service for `process` and
  `status`, migrate the CLI to it, and add unit tests.
- [x] **Check 3.2**: Run `poetry run poe check` and confirm the quality gate.
- [x] **Check 3.3**: Decouple `VideoService` from Gemini credentials through an
  argument-free `EmbeddingProvider` factory and update unit tests.
- [x] **Check 3.4**: Run `poetry run poe check` and confirm the quality gate.
- [x] **Check 3.5**: Extract embedding and transcript-index repository ports,
  move Gemini/Chroma adapters to infrastructure, and select them from settings.
- [x] **Check 3.6**: Run `poetry run poe check` and confirm the quality gate.
- [x] **Check 3.7**: Move `VideoStatus` to the domain layer and make cache
  status APIs return the typed model instead of dictionaries.
- [x] **Check 3.8**: Run `poetry run poe check` and confirm the quality gate.
- [x] **Check 3.9**: Remove the obsolete `storage` compatibility package and
  update direct imports to domain, ports, and infrastructure modules.
- [x] **Check 3.10**: Run `poetry run poe check` and confirm the quality gate.
- [x] **Check 3.11**: Replace broad exception handling with adapter-specific
  infrastructure errors and preserve unexpected errors for diagnosis.
- [x] **Check 3.12**: Run `poetry run poe check` and confirm the quality gate.
