# Maintenance: Domain · Architecture · Persistence Refactor

기능 로드맵과 분리된 구조·안정성 리팩토링 작업이다. 각 구현 항목은 하나씩만
진행하며, 매 항목 뒤에는 `poetry run poe check` 통과, 검토, 사용자 승인, 커밋을
수행한다.

## Checklist

- [x] **Check M.1**: 불변 Pydantic `TranscriptSegment`, `Transcript`,
  `VideoMetadata`와 typed video cache API를 도입하고, 자막·메타데이터가 dict로
  전달되지 않도록 loader, service, summary, transcript-index 경로를 전환한다.
  Cache는 Pydantic JSON 직렬화·역직렬화를 사용하고, `VideoStatus`에 중첩 상태
  모델을 제공한다.
- [x] **Check M.2**: 상태값과 시각 표현을 공통 타입으로 강화한다.
- [x] **Check M.3**: loader의 외부 라이브러리 오류를 adapter 전용 오류 경계로
  변환한다.
- [x] **Check M.4**: 전역 settings 의존을 불변 Pydantic 정책 객체와 bootstrap
  주입으로 교체한다.
- [x] **Check M.5**: 텍스트·비전 Chroma 저장소의 공통 lifecycle과 manifest 검증을
  내부 기반으로 추출한다.
- [x] **Check M.6**: `VideoService`의 수집·인덱싱·요약·비전 orchestration을
  독립 협력자로 분리한다.
- [x] **Check M.7**: JSON atomic write와 세대별 Chroma collection 교체로 저장
  실패 시 기존 캐시·인덱스를 보존한다.
