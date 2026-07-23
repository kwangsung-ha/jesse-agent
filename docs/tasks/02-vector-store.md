# Task 02: ChromaDB 텍스트 자막 Vector Store

## 1. 개요

Task 01에서 로컬 JSON으로 저장한 자막을 영상별 ChromaDB 컬렉션에 영속화한다. 이 작업은 이후 요약, 검색, Q&A에서 사용할 자막 검색 기반을 제공한다. ChromaDB 기본 임베딩 함수 대신 Gemini API가 생성한 벡터를 명시적으로 저장한다.

## 2. 첫 증분 범위

### 2.1 임베딩과 청킹 정책

- **모델**: Gemini API `gemini-embedding-2`
- **차원**: 768 (`output_dimensionality=768`)
- **거리 함수**: ChromaDB cosine 거리
- **문서 포맷**: `title: {video_title} | text: {chunk_text}`
- **향후 Q&A 질의 포맷**: `task: question answering | query: {query}`
- **청크 경계**: 시간 순서의 인접 자막을 최대 45초 또는 1,200자까지 병합하며, 청크 간 텍스트 중복은 두지 않는다.
- **타임코드**: 각 청크에 시작·종료 시각과 원본 첫·마지막 세그먼트 인덱스를 보존한다.

### 2.2 영속화와 재색인 정책

- 영상별 `./data/{video_id}/chromadb/`에 `transcript_collection`을 저장한다.
- 컬렉션에는 명시적 벡터, 원본 청크 텍스트, 시간 범위, 모델·차원 메타데이터를 upsert한다.
- `./data/{video_id}/index_manifest.json`에 자막 SHA-256, 청크 정책 버전, 모델, 차원, 청크 수, 인덱싱 시각을 저장한다.
- `process`는 캐시 히트라도 manifest·컬렉션 수·자막 해시가 다르면 자동 재색인한다.
- Gemini API 키 또는 네트워크 오류 시 메타데이터·자막 JSON 캐시는 저장한 채 경고를 출력한다. manifest를 갱신하지 않아 이후 `process`에서 재시도한다.
- `status`는 manifest를 사용해 자막 인덱스의 현재·stale·invalid·missing 상태, 청크 수, 모델, 차원, 인덱싱 시각을 표시한다.
- Batch API, 검색 API, `chat` 연결은 후속 작업 범위다.

## 3. 작업 체크리스트

- [x] **Check 2.1**: Chroma transcript index repository의 로컬 컬렉션 생성·자막 upsert·개수 조회 구현 및 Mock 유닛 테스트 작성
- [x] **Check 2.2**: Gemini Embedding 2 기반 문맥 청크·manifest 재색인과 `process` 자동 동기화 구현 및 Mock 유닛 테스트 작성
- [x] **Check 2.3**: `poetry run poe check` 일괄 통과 확인
- [x] **Check 2.4**: `status`에 자막 인덱스 상태 및 manifest 메타데이터 표시와 유닛 테스트 추가
- [x] **Check 2.5**: `poetry run poe check` 일괄 통과 확인
