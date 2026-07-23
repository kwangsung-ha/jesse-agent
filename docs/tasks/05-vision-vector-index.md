# Task 05: Vision Scene Vector Index

## Overview

Embed cached Gemini visual-scene descriptions into a video-scoped ChromaDB
`vision_collection`. This is text-only for now; the record metadata reserves
`vector_type="image"` for a later local frame/image-embedding provider.

## Checklist

- [x] **Check 5.1**: Add the explicit Chroma vision repository, source-scene
  manifest validation, Gemini Embedding 2 retrieval formatting, and mocked tests.
- [ ] **Check 5.2**: Wire vision-vector synchronization into `process` and
  expose vector-index freshness through `status`.
- [ ] **Check 5.3**: Run `poetry run poe check`, review, and commit the task.
