# Task 04: Gemini Direct Video Scene Analysis

## Overview

Analyze public YouTube URLs directly with Gemini rather than downloading videos
or extracting frames locally. Persist timestamped visual scene descriptions with
their source and model provenance. The provider boundary must permit a future
local-frame or local-model implementation without changing application services.

## Decisions

- Default model: `gemini-3.5-flash`, configurable through `GEMINI_VISION_MODEL`.
- Supported input in this task: public YouTube URLs only. Private, unlisted, and
  inaccessible URLs will be surfaced as provider errors.
- `vision_index.json` contains chronological scenes plus source URL, model,
  prompt version, schema version, and generation time.
- OpenCV and local video downloads are intentionally out of scope. They may be
  added later behind the vision-source/provider interface.

## Checklist

- [x] **Check 4.1**: Add vision domain models, Gemini URL-video analyzer,
  cache freshness support, default model configuration, and mocked unit tests.
- [ ] **Check 4.2**: Wire vision generation into `process`, preserve cached
  resources on failure, and expose freshness through `status`.
- [ ] **Check 4.3**: Run `poetry run poe check`, review, and commit the
  completed task.
