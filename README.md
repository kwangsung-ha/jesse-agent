# TubeTalk

YouTube 영상의 자막과 시각적 장면을 로컬에 저장하고, Gemini와 ChromaDB로
요약·타임스탬프 목차·검색용 인덱스를 만드는 Python CLI입니다.

> 영상 처리, 캐시 상태 확인, 요약 조회, 근거 인용을 포함한 대화형 Q&A를 사용할 수 있습니다.

## 주요 기능

- YouTube URL에서 메타데이터와 한국어/영어 자막 수집
- 영상별 JSON 캐시 및 ChromaDB 텍스트 벡터 인덱스 생성
- Gemini 기반 3~5문장 요약과 타임스탬프 목차 생성
- 공개 YouTube URL을 Gemini로 분석해 시각적 장면 설명과 비전 벡터 인덱스 생성
- 캐시·텍스트 인덱스·요약·비전 인덱스의 최신 상태 확인

## 요구 사항

- Python 3.12
- [Poetry](https://python-poetry.org/)
- Gemini API 키

자막을 제공하지 않거나 접근할 수 없는 영상, 비공개·제한된 영상은 처리하지 못할 수
있습니다. 비전 분석은 공개 YouTube URL만 지원합니다.

## 설치

```bash
git clone <repository-url>
cd tubetalk
poetry install
```

프로젝트 루트에 `.env` 파일을 만들고 Gemini API 키를 설정합니다.

```dotenv
GEMINI_API_KEY=your_api_key
```

또는 현재 셸 환경 변수로 전달할 수도 있습니다.

```bash
export GEMINI_API_KEY=your_api_key
```

## 사용법

`tubetalk`은 자연어 요청을 안전한 내부 도구 호출로 바꾸는 멀티턴 Agent입니다.

```bash
# 대화 세션
poetry run tubetalk

# 한 번 요청하고 종료
poetry run tubetalk 'https://www.youtube.com/watch?v=VIDEO_ID 를 처리하고 요약해줘'

# 단계, 프롬프트, 검색 근거를 stderr로 확인
poetry run tubetalk --debug '캐시된 영상 목록을 보여줘'
```

예를 들어 “이 URL을 처리해줘”, “캐시된 영상 목록을 보여줘”, “video_id의 요약을
새로 만들어줘”, “방금 영상에서 그래프가 나오는 부분은 언제야?”라고 요청할 수 있습니다.
처리 결과는 `data/<video_id>/`에 캐시되며, 답변에는 검증된 타임스탬프 인용이 표시됩니다.

전체 명령 목록은 다음으로 확인할 수 있습니다.

```bash
poetry run tubetalk --help
```

## 설정

`.env` 또는 환경 변수로 아래 값을 조정할 수 있습니다.

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `GEMINI_API_KEY` | 없음 | Gemini API 키 |
| `DATA_DIR` | `./data` | 영상 캐시와 ChromaDB 저장 경로 |
| `SUMMARY_MODEL` | `gemini-3.5-flash-lite` | 자막 요약 모델 |
| `VISION_MODEL` | `gemini-3.5-flash` | 영상 장면 분석 모델 |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | 텍스트/비전 설명 임베딩 모델 |
| `SUMMARY_LANGUAGE` | `ko` | 요약 및 목차 언어 |
| `SUMMARY_PROMPT_VERSION` | `summary-chapters-v2` | 자막 요약 템플릿 버전 |
| `CHAPTER_WINDOW_MAX_SECONDS` | `480` | 챕터 후보 추출 자막 윈도우의 최대 시간(초) |
| `CHAPTER_WINDOW_MAX_CHARACTERS` | `12000` | 챕터 후보 추출 자막 윈도우의 최대 문자 수 |
| `CHAPTER_WINDOW_OVERLAP_SECONDS` | `30` | 인접 챕터 윈도우의 중첩 시간(초) |
| `CHAPTER_BLOCK_MAX_SECONDS` | `20` | 후보 추출용 병합 자막 블록의 최대 시간(초) |
| `CHAPTER_BLOCK_MAX_CHARACTERS` | `400` | 후보 추출용 병합 자막 블록의 최대 문자 수 |
| `CHAPTER_BLOCK_MAX_GAP_SECONDS` | `1.5` | 같은 병합 블록으로 허용하는 자막 간 최대 공백(초) |
| `VISION_PROMPT_VERSION` | `vision-scenes-v2-30s` | 비전 장면 템플릿 버전 |
| `CHAT_PROMPT_VERSION` | `grounded-chat-v1` | 대화 템플릿 버전 |
| `AGENT_PROMPT_VERSION` | `tool-agent-v1` | 자연어 도구 호출 Agent 템플릿 버전 |
| `AGENT_MAX_STEPS` | `8` | 요청당 최대 도구 호출 단계 수 |

전체 설정값은 [`tubetalk/core/config.py`](tubetalk/core/config.py)에서 확인할 수 있습니다.
프롬프트는 `tubetalk/prompts/`의 버전 파일로 관리합니다. 템플릿을 변경할 때는 새 버전을
추가하고 해당 환경 변수를 바꾸세요. 디버그 로그에는 자막·질문·근거가 포함될 수 있습니다.

## 로컬 데이터 구조

```text
data/
└── <video_id>/
    ├── metadata.json        # 제목, 채널, 길이, 원본 URL 등
    ├── transcript.json      # 타임스탬프 자막
    ├── summary.json         # 요약과 타임스탬프 목차
    ├── vision_index.json    # Gemini가 생성한 시각적 장면 설명
    └── chromadb/            # 자막·비전 벡터 인덱스
```

`data/`와 `.env`는 Git에서 제외됩니다.

## 개발 및 검증

```bash
# 코드 포맷 적용
poetry run poe format

# 린트
poetry run poe lint

# 테스트 및 커버리지 검사
poetry run poe test

# 전체 품질 게이트
poetry run poe check
```

품질 게이트는 Ruff 포맷 검사, Ruff 린트, 테스트와 `tubetalk` 기준 90% 이상
커버리지를 실행합니다.

## 문서

- [제품 요구사항](docs/prd.md)
- [시스템 설계](docs/design.md)
- [로드맵](docs/roadmap.md)
- [개발 프로세스](docs/README.md)
