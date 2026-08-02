# Maintenance: Package Architecture Refactor

기능 동작을 유지하면서 application workflow, source/sink 경계, Agent tool,
infrastructure adapter의 패키지 소유권을 명확하게 정리한다. 각 항목은 하나씩
구현하고 `poetry run poe check` 통과, 검토, 사용자 승인, 커밋 순서로 진행한다.

## Checklist

- [x] **Check A.1**: Move video, knowledge, and Agent-run use cases and their
  required contracts into a feature-local `application` layer.
- [x] **Check A.2**: Separate Source and Sink contracts and move pure knowledge
  chunking out of concrete adapters.
- [x] **Check A.3**: Consolidate concrete adapters under technology-oriented
  `infrastructure` packages and remove the obsolete `pipeline` package.
- [x] **Check A.4**: Split Agent tool handlers from the central executor while
  preserving tool names, schemas, approval behavior, and result payloads.
- [ ] **Check A.5**: Reorganize prompts by Agent and video application use case,
  then update architecture documentation and remove obsolete packages.
