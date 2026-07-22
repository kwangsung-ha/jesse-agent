# Task 01: 로컬 캐시 관리자 및 유튜브 자막 Ingestion 파이프라인

## 1. 개요 (Overview)
유튜브 URL에서 `video_id`를 파싱하고, `youtube-transcript-api`를 활용하여 타임코드별 텍스트 자막 및 메타데이터를 수집합니다. 수집된 데이터는 `./data/{video_id}/` 경로에 로컬 JSON 파일로 영속화되며, 재요청 시 로컬 캐시(2초 이내)를 우선 검사합니다.

---

## 2. 세부 스펙 (Detailed Specifications)

### 2.1 주요 모듈 사양
1. **`tubetalk/core/cache.py` (LocalCacheManager)**:
   - `get_video_dir(video_id: str) -> Path`: `./data/{video_id}/` 디렉토리 경로 반환 및 생성
   - `has_cache(video_id: str) -> bool`: `metadata.json` 및 `transcript.json` 존재 여부 확인
   - `save_json(video_id: str, filename: str, data: Any)` / `load_json(video_id: str, filename: str) -> Any`

2. **`tubetalk/pipeline/loader.py` (YouTubeLoader)**:
   - `extract_video_id(url: str) -> str`: 표준 유튜브 URL / shorts / youtu.be 링크에서 `video_id` 정규식 추출
   - `fetch_transcript(video_id: str) -> List[Dict[str, Any]]`: `youtube-transcript-api` 사용해 `[{start_sec, duration_sec, text}]` 변환
   - `fetch_metadata(url: str) -> Dict[str, Any]`: `yt-dlp` 사용해 영상 제목, 길이, 채널명 등 수집

---

## 3. 작업 체크리스트 (Checklist)

- [ ] **Check 1.1**: `tubetalk/core/cache.py` 구현 및 로컬 JSON 캐시 유닛 테스트 작성
- [ ] **Check 1.2**: `tubetalk/pipeline/loader.py` (URL video_id 파싱 & transcript 수집) 구현 및 Mock 유닛 테스트 작성
- [ ] **Check 1.3**: `poetry run poe check` (포맷팅, 린팅, 커버리지 90%+ 검증) 일괄 통과 확인
