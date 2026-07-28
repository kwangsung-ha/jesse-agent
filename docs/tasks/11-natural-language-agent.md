# Task 11: Natural-language Tool-calling Agent CLI

## Overview

Replace explicit user-facing CLI commands with a Gemini native-function-calling
Agent. The model selects a bounded typed tool; deterministic application code
validates and executes it through `VideoService`.

## Checklist

- [x] **Check 11.1**: Add the Agent loop, typed service tools, natural-language
  CLI, versioned prompt, tests, and documentation.
- [x] **Check 11.2**: Run `poetry run poe check`.
- [x] **Check 11.3**: Review and commit the task.
