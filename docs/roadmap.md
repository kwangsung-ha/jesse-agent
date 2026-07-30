# JesseAgent Master Task Roadmap

본 문서는 JesseAgent 프로젝트의 상위 레벨 로드맵 및 기능별 작업 진행 현황을 관리하는 마스터 트래커입니다.

* 상태 설명: `[ ]` 미진행 | `[i]` 진행 중 | `[x]` 완료

---

## Phase 0: 프로젝트 기반 및 Test Harness 구축
- [x] **Task 00: 프로젝트 환경 설정 및 Pytest/Coverage 구축** `docs/tasks/00-project-setup.md`
  - Python 패키지 구조 (`jesseagent/`, `tests/`), pyproject.toml 설정
  - Pytest & Coverage 90%+ 검증 자동화 환경 구축
  - Pydantic Settings 기반 설정 및 로컬 데이터 경로 구조화 (`./data/`)

## Phase 1: 로컬 영속성 & 텍스트 자막 Ingestion 파이프라인
- [x] **Task 01: 자막 파싱 & 캐시 수집기 및 CLI status 커맨드 (`pipeline/loader.py`, `cli/`)** `docs/tasks/01-loader-pipeline.md`
  - `youtube-transcript-api` 자막 추출기 및 `yt-dlp` 메타데이터 수집
  - JSON 캐시 관리자 (`core/cache.py`) 연동 및 로컬 캐싱 검증 (2초 이내 로딩)
  - `jesseagent status [video_id]` CLI 커맨드로 로컬 캐시 현황 조회 기능 구현

- [x] **Task 02: ChromaDB 텍스트 자막 Vector Store 구축 (`infrastructure/repositories/chroma_transcript.py`)** `docs/tasks/02-vector-store.md`
  - ChromaDB Local Client 연동 및 `transcript_collection` 스키마 정의
  - 타임스탬프 메타데이터 포함 자막 청크 임베딩 저작

- [x] **Task 03: 자막 기반 종합 요약 & 타임스탬프 목차 파이프라인** `docs/tasks/03-summary-chaptering.md`
  - Gemini Flash-Lite 기반 구조화된 요약·목차 생성 및 자막 근거 타임스탬프 검증

## Phase 2: Vision Scene Indexing 파이프라인
- [x] **Task 04: Gemini 직접 비디오 씬 분석 기반 (`infrastructure/visions/`)** `docs/tasks/04-gemini-video-analysis.md`
  - 공개 YouTube URL 직접 분석 및 timestamped scene description 캐시
  - 로컬 프레임/모델 provider로 확장 가능한 vision interface

- [x] **Task 05: Gemini Vision Scene Vector Index (`infrastructure/repositories/`)** `docs/tasks/05-vision-vector-index.md`
  - `vision_index.json` 설명 텍스트의 Gemini Embedding 2 저작
  - ChromaDB `vision_collection` Vector 인덱싱

## Phase 3: Hybrid Retrieval & Interactive CLI Agent
- [x] **Task 06: Hybrid Dual Retriever & RRF Fusion Engine (`agent/retriever.py`)** `docs/tasks/06-hybrid-retrieval-chat.md`
  - 텍스트 자막 + 비전 씬 디스크립터 검색결과 Reciprocal Rank Fusion
  - Timestamp Citation 검증 후처리

- [x] **Task 07: Typer & Rich CLI 인터페이스 완성 (`cli/`)**
  - [x] `jesseagent process <url>`
  - [x] `jesseagent status [video_id]`
  - [x] `jesseagent summary [video_id]` 및 `--generate`
  - [x] `jesseagent chat [video_id]` (멀티턴 대화형 Q&A, ID 생략 시 목록 선택)

## Phase 4: 관찰 가능성 및 프롬프트 엔지니어링
- [x] **Task 08: Debug Trace & Versioned Prompt Catalog** `docs/tasks/08-debug-prompt-engineering.md`
  - 전역 `--debug`, `--verbose`로 실행 단계·검색 근거·Gemini 요청/응답을 관찰
  - 요약·비전·채팅 프롬프트의 버전 파일 관리 및 환경 변수 기반 선택

- [x] **Task 09: 문서-구현 정합성 및 Mypy 품질 게이트** `docs/tasks/09-documentation-quality-alignment.md`

- [x] **Task 10: 챕터 추출 재현율 고도화** `docs/tasks/10-chapter-extraction-recall.md`
  - PRD·설계서·로드맵의 구현 기준 정합성 유지
  - `poetry run poe check`에 Mypy strict 정적 타입 검사 포함

## Phase 5: Natural-language Agent CLI
- [x] **Task 11: Native Tool-calling Agent CLI** `docs/tasks/11-natural-language-agent.md`
  - 자연어 요청을 Gemini function call과 결정론적 `VideoService` 도구로 연결
  - 멀티턴 REPL 및 단발성 CLI 요청 지원

## Phase 6: Durable 12-Factor Agent Reliability
- [x] **Task 12: 12-Factor Agent Reliability Upgrade** `docs/tasks/12-agent-reliability.md`
  - SQLite append-only run/event log, reducer 기반 상태 재구성, pause/resume
  - 컨텍스트 예산, 타입화된 도구 계약, 승인 흐름, CLI run 관리

## Phase 7: Personal Knowledge Agent Foundation
- [i] **Task 13: JesseAgent 개명 및 확장 가능한 개인 Agent 기반** `docs/tasks/13-personal-agent-foundation.md`
  - [x] JesseAgent 패키지·CLI·문서 정식 개명 (기존 이름 호환 없음)
  - [ ] 공통 KnowledgeDocument/KnowledgeChunk와 Source connector 계약
  - [ ] Obsidian 증분 색인과 SQLite FTS5·Chroma 공통 검색 저장소
  - [ ] 등록형 Agent 작업과 Source 검색 도구
  - [ ] Sink connector 계약과 승인 미리보기 흐름
