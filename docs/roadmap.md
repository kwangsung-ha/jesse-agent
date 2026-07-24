# TubeTalk Master Task Roadmap

본 문서는 TubeTalk 프로젝트의 상위 레벨 로드맵 및 기능별 작업 진행 현황을 관리하는 마스터 트래커입니다.

* 상태 설명: `[ ]` 미진행 | `[i]` 진행 중 | `[x]` 완료

---

## Phase 0: 프로젝트 기반 및 Test Harness 구축
- [x] **Task 00: 프로젝트 환경 설정 및 Pytest/Coverage 구축** `docs/tasks/00-project-setup.md`
  - Python 패키지 구조 (`tubetalk/`, `tests/`), pyproject.toml 설정
  - Pytest & Coverage 90%+ 검증 자동화 환경 구축
  - Pydantic Settings 기반 설정 및 로컬 데이터 경로 구조화 (`./data/`)

## Phase 1: 로컬 영속성 & 텍스트 자막 Ingestion 파이프라인
- [x] **Task 01: 자막 파싱 & 캐시 수집기 및 CLI status 커맨드 (`pipeline/loader.py`, `cli/`)** `docs/tasks/01-loader-pipeline.md`
  - `youtube-transcript-api` 자막 추출기 및 `yt-dlp` 메타데이터 수집
  - JSON 캐시 관리자 (`core/cache.py`) 연동 및 로컬 캐싱 검증 (2초 이내 로딩)
  - `tubetalk status [video_id]` CLI 커맨드로 로컬 캐시 현황 조회 기능 구현

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
  - [x] `tubetalk process <url>`
  - [x] `tubetalk status [video_id]`
  - [x] `tubetalk summary [video_id]` 및 `--generate`
  - [x] `tubetalk chat [video_id]` (멀티턴 대화형 Q&A, ID 생략 시 목록 선택)
