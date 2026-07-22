# Task 01: 로컬 캐시 관리자, 유튜브 자막 Ingestion 파이프라인 및 CLI 커맨드

## 1. 개요 (Overview)
유튜브 URL에서 `video_id`를 파싱하고, `youtube-transcript-api`를 활용하여 타임코드별 텍스트 자막 및 메타데이터를 수집합니다. 수집된 데이터는 `./data/{video_id}/` 경로에 로컬 JSON 파일로 영속화되며, 재요청 시 로컬 캐시(2초 이내)를 우선 검사합니다. 또한 로컬 캐시 상태를 확인할 수 있는 CLI `status` 커맨드를 제공합니다.

---

## 2. 세부 스펙 (Detailed Specifications)

### 2.1 주요 모듈 사양
1. **`tubetalk/core/cache.py` (LocalCacheManager)**:
   - `get_video_dir(video_id: str) -> Path`: `./data/{video_id}/` 디렉토리 경로 반환 및 생성
   - `has_cache(video_id: str) -> bool`: `metadata.json` 및 `transcript.json` 존재 여부 확인
   - `save_json(video_id: str, filename: str, data: Any)` / `load_json(video_id: str, filename: str) -> Any`
   - `list_cached_videos() -> List[Dict[str, Any]]`: 현재 `./data/` 하위의 모든 캐시 비디오 목록 및 상태 수집
   - `get_video_status(video_id: str) -> Optional[Dict[str, Any]]`: 특정 `video_id` 캐시 상세 정보 (제목, 자막 수, 비전인덱스 유무 등) 반환

2. **`tubetalk/pipeline/loader.py` (YouTubeLoader)**:
   - `extract_video_id(url: str) -> str`: 표준 유튜브 URL / shorts / youtu.be 링크에서 `video_id` 정규식 추출
   - `fetch_transcript(video_id: str) -> List[Dict[str, Any]]`: `youtube-transcript-api` 사용해 `[{start_sec, duration_sec, text}]` 변환
   - `fetch_metadata(url: str) -> Dict[str, Any]`: `yt-dlp` 사용해 영상 제목, 길이, 채널명 등 수집

3. **`tubetalk/cli/main.py` (CLI status 커맨드)**:
   - `tubetalk status [VIDEO_ID]`: 
     - `VIDEO_ID` 미입력 시: 전체 로컬 캐시 영상 목록 및 상태 요약 표(Rich Table) 출력
     - `VIDEO_ID` 입력 시: 해당 비디오의 상세 메타데이터, 자막 세그먼트 개수, 캐시 저장 시각 등 출력

4. **`tubetalk/cli/main.py` (초기 CLI process 커맨드)**:
   - `tubetalk process <YOUTUBE_URL>`:
     - URL에서 `video_id`를 파싱하고 완전한 로컬 캐시가 있으면 원격 호출 없이 재사용
     - 캐시 미스이면 영상 메타데이터와 자막을 수집하여 `metadata.json`, `transcript.json`에 저장
     - 이 단계에서는 비전 인덱싱 및 요약을 수행하지 않음

---

## 3. 작업 체크리스트 (Checklist)

- [ ] **Check 1.1**: `tubetalk/core/cache.py` 구현 (캐시 저장/로드, `list_cached_videos`, `get_video_status`) 및 유닛 테스트 작성
- [ ] **Check 1.2**: `tubetalk/pipeline/loader.py` (URL video_id 파싱 & transcript/metadata 수집) 구현 및 Mock 유닛 테스트 작성
- [ ] **Check 1.3**: Typer & Rich 기반 `tubetalk status [VIDEO_ID]` CLI 커맨드 구현 및 CLI 유닛 테스트 작성
- [ ] **Check 1.4**: Typer & Rich 기반 `tubetalk process <YOUTUBE_URL>` 초기 ingestion 커맨드 구현 및 Mock 유닛 테스트 작성
- [ ] **Check 1.5**: `poetry run poe check` (포맷팅, 린팅, 커버리지 90%+ 검증) 일괄 통과 확인
