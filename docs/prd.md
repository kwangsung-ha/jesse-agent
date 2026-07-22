# PRD: YouTube Video Intelligence Agent (CLI & Multimodal)

## 1. 프로젝트 개요 (Overview)
* **프로젝트명**: YouTube Video Intelligence Agent (CLI)
* **목적**: 유튜브 URL을 입력받아 자막(Text)과 비전(Vision) 데이터를 복합적으로 분석하여 정교한 요약, 타임스탬프 목차 생성, 비전 기반 장면 탐색, 그리고 질의응답(Q&A)을 제공하는 CLI 기반 AI Agent 개발.
* **개발 배경 및 핵심 가치**:
  * 기존 텍스트 자막 중심 요약 서비스는 시각적 이벤트(예: 스포츠 골 장면, 화면 내 그래프/코드 시각화, 인물 등장 등)를 포착하지 못하는 한계 존재.
  * 멀티모달 API(Gemini 1.5 Pro / GPT-4o) 및 비전 키프레임 인덱싱을 통해 텍스트+시각 하이브리드 RAG 파이프라인 구축.
  * 백엔드/ML 플랫폼 엔지니어 관점에서 LLM Orchestration, Vector DB, Multimodal Scene Indexing, Caching Architecture 실무 역량 확보.

---

## 2. 주요 기능 요구사항 (Key Features)

### F-1. 로컬 영속성 및 전처리 (Ingestion & Local Caching)
* **URL 파싱 및 캐시 검사**: 입력된 유튜브 `video_id`를 기반으로 로컬 캐시(`./data/{video_id}/`) 존재 여부 우선 확인.
* **텍스트 자막 파싱**: 1순위로 공식/자동 자막 추출 (`youtube-transcript-api`). 자막 부재 시 Whisper API fallback.
* **키프레임 추출 및 비전 씬 인덱싱 (Vision Scene Indexing)**:
  * 영상 다운로드/스트리밍 후 주기적 샘플링(예: 3~5초 간격) 또는 Scene Change Detection으로 주요 키프레임 추출.
  * 전처리 단계에서 비전 LLM API(Gemini 1.5 Pro)를 호출해 각 프레임/구간별 시각적 설명(Visual Description) 데이터 구축.
* **로컬 저장**: 추출된 자막, 비전 씬 인덱스 JSON, Vector DB(ChromaDB) 데이터를 로컬 디렉토리에 영속화.

### F-2. 정교한 요약 및 하이브리드 목차 생성 (Multimodal Chaptering)
* **종합 요약**: 자막 텍스트와 비전 인덱스를 결합하여 영상 전체의 핵심 내용을 3~5줄로 정리.
* **타임스탬프 목차**: `[MM:SS] 주제 및 시각적 요약` 형태의 구조화된 목차 생성.

### F-3. 비전 연동 대화형 CLI Q&A (Hybrid Search & Q&A)
* **하이브리드 RAG**: 사용자 질문 수신 시 텍스트 자막 임베딩 검색과 비전 씬 인덱스 검색을 함께 수행.
* **시각적 질의 대응**: "골 장면 어디야?", "빨간 옷 입은 사람이 등장하는 구간 알려줘" 등 자막에 없는 시각적 이벤트 탐색 및 타임스탬프 반환.
* **대화 맥락 유지**: CLI 세션 동안 히스토리를 유지하여 연속적 Multi-turn 질의응답 지원.

---

## 3. 시스템 아키텍처 및 기술 스택 (Technical Architecture)

| 구분 | 기술 스택 / 도구 | 역할 및 상세 설명 |
| :--- | :--- | :--- |
| **Interface** | Python CLI (Typer / Click / Rich) | 터미널 기반 상호작용 UI, 가독성 높은 텍스트/타임스탬프 출력 |
| **Orchestration** | LangChain / LlamaIndex | RAG 파이프라인 구성, 메모리 관리 및 에이전트 루프 제어 |
| **Multimodal LLM** | Gemini 1.5 Pro / GPT-4o API | 비전 프레임 분석, 롱컨텍스트 종합 요약 및 최종 Q&A 생성 |
| **Vector & Data Store** | ChromaDB (Local) + JSON File Store | 텍스트/비전 인덱스 임베딩 저장 및 `./data/` 로컬 캐싱 |
| **Media Processing** | yt-dlp, OpenCV (cv2), ffmpeg | 영상 다운로드, 키프레임 프레임 샘플링 및 오디오 추출 |

---

## 4. 로컬 데이터 구조 (Local Persistence Schema)
```text
./data/
└── {video_id}/
    ├── metadata.json           # 영상 제목, 길이, 파싱 일시 등
    ├── transcript.json         # 타임스탬프별 자막 데이터
    ├── vision_index.json       # [MM:SS] 별 비전 LLM 씬 설명 데이터
    ├── frames/                 # 추출된 키프레임 이미지 (선택적)
    └── chromadb/               # ChromaDB 임베딩 데이터베이스
```

---

## 5. 단계별 개발 로드맵 (Milestones)

### Phase 1: CLI & Text RAG 기반 구축
* CLI 인터페이스 구현 (URL 입력 받기).
* 자막 추출 파이프라인 및 로컬 파일 캐싱(JSON) 구조 설계.
* 자막 기반 전체 요약 및 타임스탬프 목차 출력 기능 개발.

### Phase 2: Vision Scene Indexing 파이프라인 추가
* `yt-dlp` 및 OpenCV를 활용한 N초 간격 키프레임 샘플링 모듈 개발.
* Gemini 1.5 Pro 비전 API 호출을 통해 프레임별 시각적 설명 생성 및 `vision_index.json` 구축.
* 텍스트 자막과 비전 씬 데이터를 병합하는 인덱서 구현.

### Phase 3: Hybrid Retrieval & Q&A Agent 완성
* 텍스트+비전 하이브리드 Vector DB Retrieval 파이프라인 완성.
* 시각적 질문 대응 프롬프트 엔지니어링 및 타임스탬프 출처(Citation) 검증.
* CLI 대화형 멀티턴 질의응답 루프 연결.

---

## 6. 성공 지표 (Success Metrics)
* **비전 탐색 정확도**: 시각적 이벤트 질문("골 장면", "그래프 제시 구간" 등) 검색 시 실제 타임스탬프 오차 $\le$ 10초 이내.
* **캐시 효율성**: 동일 영상 재검색 시 전처리 없이 로컬 캐시를 통한 CLI 세션 로딩 시간 $\le$ 2초.
* **Q&A 신뢰도**: 답변 내 명시된 타임스탬프 구간의 실제 내용 일치율 90% 이상.
