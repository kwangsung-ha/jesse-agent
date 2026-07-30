# Task 00: Poetry 프로젝트 환경 설정, PoeTask Runner 및 Test/Quality Harness 구축

## 1. 개요 (Overview)
JesseAgent 프로젝트의 Python 환경을 **Poetry** 기반으로 관리하고, **PoeThePoet (poe)**를 적용하여 단일 명령어로 포맷팅, 린팅, 정적 분석 및 90% 커버리지 유닛 테스트를 수행하는 품질 관리 체계를 구축합니다.

---

## 2. 세부 스펙 (Detailed Specifications)

### 2.1 디렉토리 및 파일 구조
```text
jesseagent/
├── __init__.py
├── cli/
│   └── __init__.py
├── core/
│   ├── __init__.py
│   └── config.py          # Settings 클래스 (Pydantic Settings)
├── pipeline/
│   └── __init__.py
├── storage/
│   └── __init__.py
└── agent/
    └── __init__.py

tests/
├── __init__.py
├── conftest.py            # Pytest Fixtures
└── test_config.py         # Config & Setup Unit Tests

pyproject.toml             # Poetry 의존성 및 PoeThePoet 스크립트 정의
AGENTS.md                  # 루트 에이전트 품질 규칙 및 명령 가이드
```

### 2.2 핵심 구현 요구사항

#### 1) Poetry & poethepoet (`pyproject.toml`) 설정
* **Dependencies**: `python = "^3.10"`, `typer`, `rich`, `pydantic`, `pydantic-settings`, `google-genai`, `chromadb`, `yt-dlp`, `youtube-transcript-api`, `opencv-python`
* **Dev Dependencies**: `pytest`, `pytest-cov`, `pytest-mock`, `ruff`, `mypy`, `poethepoet`
* **Poe Tasks (`[tool.poe.tasks]`)**:
  - `test`: `pytest --cov=jesseagent --cov-report=term-missing --cov-fail-under=90`
  - `format`: `ruff format .`
  - `lint`: `ruff check . && mypy jesseagent`
  - `check`: `["format-check", "lint", "test"]` (포맷 검사, 린트/정적분석, 유닛테스트 일괄 수행)
  - `format-check`: `ruff format --check .`

#### 2) Pydantic Settings (`jesseagent/core/config.py`)
* `Settings` 클래스:
  - `GEMINI_API_KEY: str = ""`
  - `DATA_DIR: Path = Path("./data")`
  - `DEFAULT_SAMPLE_INTERVAL_SEC: float = 5.0`
  - `VISION_MODEL: str = "gemini-2.5-flash"`
  - `LLM_MODEL: str = "gemini-2.5-pro"`
* 초기화 시 `DATA_DIR` 경로가 없으면 자동으로 생성하는 메서드 및 디렉토리 관리 기능.

#### 3) 검증 및 품질 게이트
* 모든 세부 항목 개발 완료 시 `poetry run poe check` 실행 및 100% 통과 필수.

---

## 3. 작업 체크리스트 (Checklist)

- [x] **Check 0.1**: Poetry 환경 초기화 및 `pyproject.toml` (의존성 + poethepoet task 설정) 작성
- [x] **Check 0.2**: `jesseagent/core/config.py` 구현 (Settings & 데이터 경로 자동 생성)
- [x] **Check 0.3**: `tests/test_config.py` 작성 및 `poetry run poe check` (포맷, 린트, 정적분석, 커버리지 90%+ 유닛테스트) 일괄 성공 검증
