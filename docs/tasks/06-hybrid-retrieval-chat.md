# Task 06: Hybrid Retrieval and Interactive Chat

## Overview

Fuse transcript and visual-scene retrieval with Reciprocal Rank Fusion, then
use the fused evidence for citation-validated Gemini Q&A in a process-local CLI
session. This task also standardizes immutable application value models on
Pydantic `BaseModel` while preserving the existing cache and manifest schemas.

## Checklist

- [x] **Check 6.1**: Add dual Chroma retrieval, RRF, grounded Gemini answers,
  `tubetalk chat [video_id]`, and Pydantic model migration with mocked tests.
- [x] **Check 6.2**: Run `poetry run poe check`.
- [x] **Check 6.3**: Review and commit the task.
