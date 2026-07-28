# PRD: TubeTalk — YouTube Video Intelligence Agent

## 1. 프로젝트 개요

- **프로젝트명**: TubeTalk
- **목적**: YouTube URL의 자막과 시각적 장면 정보를 로컬에 축적하고, 요약·타임스탬프 목차·향후 하이브리드 검색과 Q&A를 제공하는 CLI AI 에이전트
- **핵심 가치**: 자막만으로 찾기 어려운 화면 속 인물·객체·차트·행동을 비전 장면 설명으로 함께 인덱싱한다.

### 현재 제공 범위

현재 자연어 CLI Agent는 다음 기능을 제공한다.

- YouTube 메타데이터와 한국어/영어 자막 수집 및 로컬 캐시
- Gemini Embedding 2 기반 자막 벡터 인덱스
- Gemini Flash-Lite 기반 자막 요약과 타임스탬프 목차
- Gemini의 공개 YouTube URL 직접 분석 기반 시각 장면 설명 및 비전 벡터 인덱스
- 캐시와 각 인덱스의 최신 상태 조회

자막·비전 검색 결과를 융합한 근거 인용형 멀티턴 `chat` Q&A를 제공한다.

## 2. 기능 요구사항

### F-1. 로컬 영속성 및 수집

- YouTube URL에서 `video_id`를 추출하고 `./data/{video_id}/` 캐시를 먼저 확인한다.
- 캐시가 없으면 `yt-dlp`로 메타데이터를, `youtube-transcript-api`로 한국어/영어 자막을 수집한다.
- 메타데이터·자막·요약·비전 장면·벡터 인덱스 manifest를 영상별 로컬 디렉터리에 저장한다.
- 캐시가 있으면 원본 수집을 생략하고, 원본 데이터·모델·프롬프트·임베딩 설정이 변경된 인덱스만 갱신한다.

현재 자막이 없는 영상에 대한 Whisper 또는 오디오 기반 fallback은 지원하지 않는다.

### F-2. 자막 요약 및 타임스탬프 목차

- 자막에서 근거를 찾을 수 있는 3~5문장 요약과 시간순 목차를 Gemini에 요청한다. 구현은 비어 있지 않은 요약과 시간순 목차를 검증하지만 문장 수는 강제하지 않는다.
- 목차의 `start_sec`는 실제 자막 시간 범위 안에 있는지 검증한다.
- 생성 결과와 자막 해시·모델·프롬프트·언어 정보를 `summary.json`에 저장해 최신성을 판단한다.

현재 요약 입력은 자막이며, 비전 장면 설명을 결합한 멀티모달 요약은 후속 범위다.

### F-3. 비전 씬 인덱싱

- 공개 YouTube URL을 Gemini에 직접 제공해 영상 전체를 덮는 시간순 시각 장면 설명을 Gemini에 요청한다. 구현은 장면 시간을 영상 길이 안으로 보정하고 시간순으로 정렬하지만, 장면 간 공백 없는 전체 커버리지는 강제하지 않는다.
- 장면마다 시작/종료 시간, 시각적 설명, 감지 객체를 저장한다.
- 장면 설명을 Gemini Embedding 2로 임베딩하고 영상별 ChromaDB `vision_collection`에 저장한다.
- 비전 분석이나 인덱싱 실패는 이미 수집된 메타데이터·자막·요약 캐시를 삭제하지 않고 경고로 보고한다.

로컬 영상 다운로드, OpenCV 키프레임 추출, 프레임 이미지 저장은 현재 범위에 포함하지 않는다.

### F-4. 하이브리드 검색 및 Q&A

- 자막과 비전 장면 컬렉션을 각각 검색한 뒤 Reciprocal Rank Fusion(RRF)으로 결과를 결합한다.
- 답변에 포함한 타임스탬프가 실제 자막 또는 장면 구간인지 검증한다.
- CLI 세션에서 멀티턴 대화 맥락을 유지한다.

## 3. 기술 스택

| 구분 | 사용 기술 | 역할 |
| --- | --- | --- |
| CLI | Python, Typer, Rich | 자연어 요청을 서비스 도구 호출로 변환하는 멀티턴 Agent |
| 설정 | Pydantic Settings | `.env`와 환경 변수 기반 설정 |
| 수집 | `yt-dlp`, `youtube-transcript-api` | 메타데이터와 자막 수집 |
| 생성 모델 | Google Gemini (`google-genai`) | 자막 요약 및 공개 URL 비전 장면 분석 |
| 임베딩 | Gemini Embedding 2 | 자막 청크와 비전 설명 임베딩 |
| 저장소 | ChromaDB PersistentClient + JSON | 영상별 벡터 컬렉션·캐시·manifest 영속화 |
| 품질 | pytest, pytest-cov, Ruff, Poe the Poet | 테스트·90% 커버리지 기준·정적 검사 |

## 4. 로컬 데이터 구조

```text
data/
└── {video_id}/
    ├── metadata.json                # 제목, 채널, 길이, 원본 URL 등
    ├── transcript.json              # 타임스탬프 자막
    ├── summary.json                 # 자막 요약, 목차, 생성 manifest
    ├── vision_index.json            # 시각 장면, 생성 manifest
    ├── index_manifest.json          # 자막 벡터 인덱스 manifest
    ├── vision_vector_manifest.json  # 비전 벡터 인덱스 manifest
    └── chromadb/                    # transcript_collection, vision_collection
```

## 5. 단계별 로드맵

### Phase 1: 로컬 캐시·자막 인덱스·요약 — 완료

- 수집, 영상별 JSON 캐시, 캐시 상태 CLI
- 자막 청크의 명시적 Gemini 임베딩과 ChromaDB 인덱스
- 자막 기반 요약 및 타임스탬프 목차

### Phase 2: Gemini 비전 장면 인덱싱 — 완료

- 공개 URL 직접 분석 기반 시각 장면 설명 캐시
- 장면 설명의 ChromaDB 벡터 인덱스

### Phase 3: 하이브리드 검색 및 대화형 CLI — 완료

- Dual Retriever, RRF, 타임스탬프 인용 검증
- 멀티턴 자연어 `tubetalk` Agent와 단발성 `tubetalk "요청"`

## 6. 성공 지표

- 동일 영상 재처리에서 유효한 캐시와 인덱스를 재사용한다.
- 각 캐시·인덱스는 원본 해시와 모델·프롬프트·임베딩 설정 변경을 감지해 stale 상태를 표시한다.
- 하이브리드 Q&A 완성 후 시각 이벤트 검색의 타임스탬프 오차를 10초 이하, 답변 인용 일치율을 90% 이상으로 측정한다.
