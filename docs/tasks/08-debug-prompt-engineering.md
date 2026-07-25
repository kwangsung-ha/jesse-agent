# Task 08: Debug Trace and Prompt Engineering

## Overview

Expose the command-to-result path for learning and diagnosis, and move all
Gemini prompt text into versioned package templates that can be selected by
environment configuration.

## Checklist

- [x] **Check 8.1**: Add global `--debug` and `--verbose`, configure Loguru at
  the CLI entry point, and trace cache, loader, Chroma, service, retrieval, and
  Gemini request/response boundaries without logging secrets.
- [x] **Check 8.2**: Add versioned summary, vision, and chat templates with
  function-specific environment-variable selection.
- [x] **Check 8.3**: Run `poetry run poe check`.
- [x] **Check 8.4**: Review and commit the task.
