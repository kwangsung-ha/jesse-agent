# PRD: JesseAgent — Personal Knowledge & Task Agent

## 1. 제품 개요

JesseAgent는 Jesse의 개인 지식과 작업을 한 곳에서 다루는 로컬 우선 CLI Agent다.
YouTube와 Obsidian 같은 **Source connector**에서 정보를 수집·색인하고, 근거가 있는
답변을 제공한다. **Sink connector**는 계획·미리보기·명시 승인 계약과 durable lifecycle만
구현되어 있으며, 실제 외부 시스템 반영 adapter는 아직 제공하지 않는다.

CLI는 `jesseagent run`으로 멀티턴 REPL을 시작한다. durable run의 조회·승인·거절·재개·삭제는
`jesseagent run <관리 명령>`으로 수행하며, 단발 자연어 요청 명령은 제공하지 않는다. Source
동기화는 별도 명령군인 `jesseagent sources sync obsidian`으로 명시적으로 실행한다.

### 핵심 가치

- 개인 자료를 원본 위치에 유지하고, 필요한 근거만 검색해 Gemini에 전달한다.
- 소스별 형식 차이를 공통 지식 모델로 흡수해 한 번의 검색과 대화에서 활용한다.
- 모든 쓰기·외부 전송은 미리보기와 명시적 승인 뒤에만 실행한다.
- 답변에는 원문으로 돌아갈 수 있는 출처를 제공한다.

### 현재 상태

- JesseAgent CLI와 durable Agent run/승인 기반은 구현됐다.
- YouTube 수집·자막/비전 색인·영상 Q&A는 계속 지원한다.
- `KnowledgeDocument`, `KnowledgeChunk`, Source connector 계약과 YouTube transcript
  adapter가 구현됐다.
- Obsidian parser·heading chunker·명시적 증분 sync가 구현됐다.
- SQLite FTS5·Chroma 공통 인덱스와 RRF 기반 `search_knowledge` Agent 도구가 구현됐다.
- Sink plan의 preview·승인·거절·승인 후 단일 적용 상태 전이가 구현됐다. 실제 Sink는 없다.

## 2. 기능 요구사항

### F-1. Source connector와 수집

- 모든 Source connector는 안정적인 source/document ID, URI, 제목, 원문, 메타데이터,
  콘텐츠 SHA-256, 변경 시각을 가진 `KnowledgeDocument`를 반환한다.
- YouTube는 정식 Source로 유지한다. 기존 영상 캐시는 삭제하거나 마이그레이션하지 않는다.
- Obsidian은 `OBSIDIAN_VAULT_PATH` 아래의 Markdown만 읽는다. 첫 릴리스에서는
  `jesseagent sources sync obsidian`의 명시적 동기화만 제공한다.
- Obsidian 문서에서 frontmatter, 태그, 경로, 제목, heading, Wiki link, 파일 수정 시각을
  추출한다. 첨부 파일과 비 Markdown 파일은 제외한다.

### F-2. 공통 지식 색인

- 문서를 heading 우선으로 청킹하고, 긴 섹션만 크기 제한과 중첩을 적용해 추가 분할한다.
- SQLite는 문서 manifest, 콘텐츠 해시, 메타데이터, Wiki link, FTS5 키워드 인덱스를 가진다.
- Chroma는 모든 Source의 검색용 청크와 Gemini 임베딩을 저장한다.
- sync는 해시가 같은 문서를 재임베딩하지 않고, 수정 문서는 교체하며 삭제된 문서는 두
  저장소에서 제거한다.

### F-3. 검색과 근거형 답변

- 벡터 검색과 FTS5 검색을 독립적으로 수행하고 RRF로 결합한다.
- 검색 결과는 Source·문서 URI·경로·heading·본문 발췌를 유지한다.
- Obsidian 근거는 `obsidian://open` URI로, YouTube 근거는 원본 URL과 타임스탬프로
  원문을 열 수 있어야 한다.
- Agent는 근거만으로 답할 수 없으면 그 한계를 분명히 말하고 사실을 추정하지 않는다.

### F-4. 확장 가능한 Agent 작업과 Sink

- 목표 Agent 작업은 `TaskDefinition`으로 등록하는 확장 단위다. 작업은 이름·설명, Pydantic
  입력/출력 계약, side-effect·승인 정책, 실행기, 사용하는 Source/Sink와 버전된 프롬프트를
  명시한다.
- Agent는 자연어 요청을 등록된 작업의 검증 가능한 tool call로 변환할 뿐이다. 작업 실행기는
  검색·생성·검증·오류 처리를 결정론적으로 조합하며, 임의의 tool이나 인자를 실행하지 않는다.
- 작업별 프롬프트는 저장소의 버전 파일로 관리한다. 생성 산출물과 실행 기록에는 사용한
  프롬프트 버전을 남겨 사람이 품질을 조정하고 결과를 재현할 수 있어야 한다.
- 읽기 작업은 즉시 실행하고, Source 수집 비용·외부 생성·Sink 적용은 durable run의
  `pending_approval` 상태에서 멈춘다.
- Source connector는 원문 읽기만, Sink connector는 변경 계획·사용자용 미리보기와 승인된
  계획의 적용만 담당한다. 작업의 비즈니스 흐름과 프롬프트는 connector에 두지 않는다.
- 첫 릴리스는 Sink 계약과 승인 흐름만 제공하며, Obsidian 쓰기·파일 이동·외부 서비스
  전송은 제공하지 않는다.

현재 구현은 중앙 `ToolExecutor`가 video·knowledge handler의 Pydantic 입력 schema에서 Gemini
tool declaration을 만든다. `search_knowledge`가 첫 공통 지식 조회 작업이며, 독립
`TaskDefinition`/`TaskExecutor` 타입으로의 일반화는 후속 범위다.

## 3. 기술 및 운영 제약

| 구분 | 선택 | 역할 |
| --- | --- | --- |
| CLI/Agent | Python, Typer, Rich, Gemini function calling | 자연어 작업과 durable run |
| Source | YouTube, Obsidian Markdown | 개인 지식 원본 |
| 생성·임베딩 | Google Gemini | 검색 질의·제한된 근거 기반 답변 |
| 검색 저장소 | SQLite FTS5, Chroma PersistentClient | 키워드·벡터 하이브리드 검색 |
| 영속성 | JSON, SQLite | 기존 영상 캐시, 지식 카탈로그, Agent event log |
| 품질 | pytest, Ruff, mypy, Poe | 90% 이상 커버리지 품질 게이트 |

Vault 원문과 색인 저장소는 로컬에 남는다. 질문과 최종 검색된 일부 청크만 Gemini API로
전송될 수 있다. API 키, 전체 렌더링 프롬프트, 원문 전체는 Agent event log에 저장하지
않는다.

## 4. 목표 로컬 데이터 구조

```text
data/
├── <video_id>/                   # 보존되는 기존 YouTube 캐시
│   ├── metadata.json
│   ├── transcript.json
│   ├── summary.json
│   ├── vision_index.json
│   └── chromadb/
├── knowledge.sqlite3             # 문서 manifest, links, FTS5
├── knowledge_chromadb/           # 모든 Source의 공통 청크 벡터 인덱스
└── agent_runs.sqlite3            # 재개 가능한 실행 이벤트
```

## 5. 성공 기준과 비범위

- 수정되지 않은 Obsidian 문서는 재임베딩하지 않으며, 수정·삭제 사항은 다음 sync에서
  검색 결과에 반영된다.
- 개인 지식 검색 결과는 가능한 경우 원문을 여는 Obsidian URI와 함께 제공한다.
- 승인 없는 Sink 적용이나 외부 전송은 절대 실행하지 않는다.
- 첫 릴리스에는 파일 watcher, Obsidian 쓰기, 웹/API UI, 다중 사용자/권한 관리, 로컬 LLM
  운영을 포함하지 않는다.
