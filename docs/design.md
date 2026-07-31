# System Design Specification: JesseAgent

## 1. 목표 아키텍처

JesseAgent는 Source·검색·Agent 작업·Sink를 분리한다. 현재 YouTube는 첫 Source이며,
Obsidian을 추가해도 기존 영상 처리와 durable Agent run의 책임은 변하지 않는다.

```mermaid
graph TD
    User[User] --> CLI[Typer CLI]
    CLI --> RunService[AgentRunService]
    RunService --> Agent[AgentSession]
    Agent --> Tasks[Task Registry]
    Tasks --> Sources[Source Connectors]
    Sources --> Sync[Knowledge Sync Service]
    Sync --> Catalog[SQLite Catalog + FTS5]
    Sync --> Vectors[Chroma Knowledge Index]
    Tasks --> Retrieve[Hybrid Retriever]
    Retrieve --> Catalog
    Retrieve --> Vectors
    Tasks --> Sinks[Sink Connectors]
    Sinks --> Approval[Preview + explicit approval]
```

## 2. 경계와 계약

### Knowledge model

`KnowledgeDocument`는 source-neutral 원문 레코드다. `source_id`, `document_id`, URI,
title, content, SHA-256, 변경 시각, JSON-safe metadata를 가진다. `KnowledgeChunk`는
문서 ID와 순번·본문·메타데이터를 가진 검색 단위다.

`SourceConnector`는 `source_id`와 안정적으로 정렬된 `list_documents()`를 제공한다.
sync 서비스가 connector, hash, 저장소, 재시도 정책을 조합하며 connector는 임베딩·DB·Agent
상태를 직접 다루지 않는다.

`SinkConnector`는 후속 항목에서 `plan()`과 `apply(approved_plan)` 계약을 갖는다. `plan()`은
사용자 미리보기에 충분한 변경 요약을 반환하고 `apply()`는 durable run의 승인 후에만 호출된다.
connector는 원문 읽기나 외부 반영만 담당하며 작업의 프롬프트·비즈니스 흐름·승인 판단을
소유하지 않는다.

`SinkPlan`은 plan ID, Sink ID, 작업명, preview와 JSON-safe payload를 가진 불변 계약이다.
승인 요청 이벤트는 이 계획과 preview를 함께 저장한다. 승인 후 `resume`은 이벤트에서 동일
계획을 복원해 재계획 없이 한 번만 `apply()`하며, 거절되거나 이미 완료된 계획은 적용하지
않는다. 현재는 이 계약과 상태 전이만 제공하고 production Sink connector는 등록하지 않는다.

### Source 구현 상태

- `YouTubeSourceConnector`: 현재 로컬 캐시의 완전한 자막을 timestamp-preserving
  `KnowledgeDocument`로 투영한다. 비전 장면 공통화는 공통 인덱스 작업과 함께 추가한다.
- `ObsidianSourceConnector`: 다음 구현 항목에서 Markdown을 읽고 frontmatter·heading·tags·Wiki
  links를 metadata로 변환한다. 읽기 전용이며 watcher는 포함하지 않는다.

## 3. 색인과 검색 흐름

Obsidian sync의 목표 흐름은 아래와 같다.

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Source as ObsidianSourceConnector
    participant Sync as KnowledgeSyncService
    participant SQL as SQLite Catalog/FTS5
    participant Chroma
    Sync->>SQL: compare manifest and update links/FTS
    alt new or changed document
        Sync->>Sync: heading-first chunking
        Sync->>Chroma: replace document chunks and embeddings
    end
    Sync->>SQL: remove deleted documents
    Sync->>Chroma: remove deleted document chunks
```

질문은 Gemini query embedding을 통한 Chroma 검색과 SQLite FTS5 검색을 병렬 수행하고 RRF로
결합한다. 답변 모델에는 상위의 제한된 evidence만 보낸다. 모델이 인용한 document/chunk ID는
검색 결과에 존재해야 하며, 인터페이스는 이를 Obsidian URI 또는 YouTube timestamp로 렌더링한다.

## 4. Agent 실행과 승인

CLI root는 명령 도움말만 제공한다. `jesseagent run`은 REPL이며, 같은 명령군의
`list/status/approve/reject/resume/delete`가 durable run lifecycle을 관리한다.
`jesseagent sources sync obsidian`은 Agent 대화와 분리된 명시적 Source 동기화 명령이다.

`AgentRunService`와 SQLite append-only event log는 계속 lifecycle source of truth다. 읽기
작업(검색, 원문 조회, 상태 조회)은 즉시 실행한다. 유료 생성, 새 Source 수집, Sink 적용은
`approval_requested` 이벤트를 기록하고 `pending_approval`에서 중지한다. 재개는 완료된
도구 호출을 다시 실행하지 않는다.

`TaskDefinition`은 Agent가 호출할 하나의 등록형 작업이다. 이름·설명, Pydantic 입력/출력
schema, side-effect policy, 실행기, 필요한 Source/Sink, 프롬프트 kind/version을 가진다.
Gemini tool declaration은 Task Registry에서 파생하며, 모델은 등록된 작업과 schema-valid
인자만 선택할 수 있다.

`TaskExecutor`는 TaskDefinition을 실행해 입력 검증, 제한된 context 조립, 프롬프트 렌더링,
결정론적 검증·재시도와 오류 요약을 담당한다. 사람이 조정하는 프롬프트는 저장소의 버전된
파일로 관리하고, 생성 산출물과 durable run에는 사용 버전을 기록한다. 따라서 새 작업은
Agent loop를 수정하지 않고 작업 등록·prompt 파일·bootstrap wiring으로 추가하며, Source와
Sink는 그 작업이 의존하는 I/O adapter로만 조합한다.

## 5. 저장소와 호환성

- 기존 `data/<video_id>/` JSON/영상별 Chroma 캐시는 유지한다.
- `knowledge.sqlite3`와 `knowledge_chromadb/`는 새 공통 검색 영역이며 기존 캐시와
  충돌하지 않는다.
- `agent_runs.sqlite3`의 event schema는 유지한다. Sink 계획에는 API 키, 전체 원문,
  렌더링 프롬프트를 넣지 않는다.
- Python 패키지와 CLI는 `jesseagent`로 통합됐으며, durable run 관리는
  `jesseagent runs` 하위 명령으로 제공한다. `tubetalk` 호환 별칭은 제공하지 않는다.

## 6. 구현 순서

1. 제품·설계 문서를 개인 Agent 경계로 갱신한다.
2. Obsidian parser와 heading chunker를 단위 테스트로 구현한다.
3. SQLite/FTS5·Chroma 공통 카탈로그와 명시적 sync CLI를 구현한다.
4. 하이브리드 검색과 등록형 Agent 조회 작업을 기존 run lifecycle에 연결한다.
5. Sink 계획·미리보기·승인 계약을 추가한다.
