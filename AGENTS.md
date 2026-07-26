# AGENTS.md - TubeTalk Coding Agent Rules & Quality Gate

본 문서는 TubeTalk 프로젝트에 참여하는 모든 코딩 에이전트(AI Agent)가 반드시 준수해야 하는 개발 규칙 및 품질 검증 가이드라인입니다.

---

## 1. 품질 검증 캔버스 (Quality Gate Rule)

모든 코드 구현, 리팩토링, 버그 수정 작업이 완료될 때마다 반드시 아래 커맨드를 실행하여 **100% 통과**를 확인한 후 사용자에게 검토 및 커밋을 요청해야 합니다.

```bash
poetry run poe check
```

`poetry run poe check` 명령어는 아래 작업들을 일괄 수행합니다:
1. **`poetry run poe format`** (또는 `format-check`): 코드 포맷팅 (Ruff)
2. **`poetry run poe lint`**: Ruff 린팅
3. **`poetry run poe static`**: Mypy strict 정적 타입 분석
4. **`poetry run poe test`**: 유닛 테스트 실행 및 **커버리지 90% 이상** 통과 (`pytest --cov=tubetalk --cov-fail-under=90`)

---

## 2. 개발 및 협업 가이드라인

1. **Poetry 기반 패키지 관리**:
   - 패키지 의존성 추가 시 `poetry add` 또는 `poetry add --group dev`를 사용합니다.
   - 단일 스크립트 실행 시 `poetry run ...` 또는 `poetry run poe <task>`를 활용합니다.

2. **단위 구현 & 유닛 테스트 1:1 작성**:
   - 새로운 모듈/클래스/함수를 작성할 때 반드시 이에 대응하는 유닛 테스트(`tests/test_*.py`)를 함께 작성합니다.
   - 외부 API (Gemini API, YouTube API, ChromaDB 등) 호출부는 Pytest Mocking(`pytest-mock`) 처리하여 테스트 속도 및 독립성을 확보합니다.

3. **체크리스트 기반 1개 항목 점진적 진행**:
   - `docs/tasks/XX-feature.md` 내 체크리스트 항목을 **한 번에 1개씩만** 구현합니다.
   - 구현 -> `poetry run poe check` 실행 -> 검토 보고 -> 사용자 승인 -> Git 커밋 순서로 진행합니다.


---

## 3. 주요 명령어 요약

| 명령 | 설명 |
| :--- | :--- |
| `poetry run poe check` | **[필수]** 포맷 체크, 린트/정적 분석 및 90%+ 유닛 테스트 검증 |
| `poetry run poe test` | 유닛 테스트 구동 및 커버리지 리포트 출력 |
| `poetry run poe format` | Ruff 코드 포맷팅 자동 적용 |
| `poetry run poe lint` | Ruff 린트 검사 |
| `poetry run poe static` | Mypy strict 정적 타입 검사 |
