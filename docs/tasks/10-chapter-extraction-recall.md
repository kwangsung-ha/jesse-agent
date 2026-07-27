# Task 10: Chapter extraction recall upgrade

## Overview

Improve timestamp chapter recall by extracting local semantic-transition
candidates from overlapping transcript windows, then consolidating candidates
without allowing any to be omitted. Keep the user-facing summary format stable.

Automated chapter-quality evaluation is intentionally deferred until the prompt
and extraction strategy are stable. This task uses qualitative review of real
video outputs instead.

## Decisions

- Prioritize recall: retain every meaningful topic transition; remove only true
  duplicates during consolidation.
- Windows are bounded to 8 minutes or 12,000 transcript characters with 30
  seconds of overlap.
- Candidate prompts merge adjacent caption fragments into readable blocks. The
  original transcript remains the source of truth for storage and final chapter
  timestamp snapping.
- Block duration, character limit, and permitted caption gap are configurable
  through `CHAPTER_BLOCK_*` environment variables and invalidate stale summaries
  when changed.
- Candidate extraction returns a listed prompt-block index rather than a free-form
  timestamp; the application maps it to the block's source start time. Invalid
  block indexes get one window-scoped corrective retry.
- v2 prompt-version freshness invalidates v1 summary caches.
- Chapter quality is reviewed qualitatively against real videos while prompt and
  extraction behavior are still evolving.

## Checklist

- [x] **Check 10.1**: Add immutable chapter-candidate and overlapping transcript
  window domain primitives with unit tests.
- [x] **Check 10.2**: Add Gemini local candidate extraction and coverage-safe
  consolidation with mocked adapter tests.
- [x] **Check 10.3**: Wire v2 generation, prompts, configuration, cache
  freshness, and service tests.
- [x] **Check 10.5**: Update documentation, run the quality gate, review, and
  commit the completed task after approval.
