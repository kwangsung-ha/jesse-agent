# Task 13: JesseAgent Personal Knowledge Agent Foundation

## Overview

JesseAgent를 YouTube 전용 도구에서 개인 지식과 작업을 위한 확장 가능한 Agent로
전환한다. 첫 릴리스는 기존 YouTube를 정식 Source로 보존하면서 Obsidian Vault를 읽고
증분 색인해 근거 기반 질의응답을 제공한다. 모든 Sink 작업은 미리보기와 명시적 승인을
기본으로 하며, 이 작업에서는 Sink 계약과 승인 흐름만 제공한다.

## Decisions

- 패키지와 CLI 이름은 `jesseagent`, `jesseagent-runs`로 즉시 전환하며 `tubetalk` 호환
  별칭은 제공하지 않는다.
- Obsidian은 `OBSIDIAN_VAULT_PATH`에서 명시적 sync 명령으로만 읽는다. 파일 watcher와
  실제 파일 쓰기는 후속 범위다.
- Gemini 임베딩과 답변 모델을 계속 사용하되, 검색된 제한된 근거만 모델로 보낸다.
- YouTube 자막·비전 데이터와 Obsidian Markdown은 공통 지식 모델과 검색층을 사용한다.

## Checklist

- [x] **Check 13.1**: 프로젝트·Python 패키지·CLI·테스트·문서 참조를 JesseAgent로
  일괄 개명하고 품질 게이트를 통과한다.
- [x] **Check 13.2**: 공통 지식 문서·청크 모델과 Source connector Protocol을 정의하고,
  기존 YouTube 데이터를 정식 Source로 투영하는 단위 테스트를 작성한다.
- [x] **Check 13.3**: PRD와 설계 문서를 개인 Agent의 Source·검색·작업·Sink 경계로
  갱신하고 구현 순서와 비범위를 검토한다.
- [x] **Check 13.4**: Obsidian Markdown 파서와 heading 기반 chunker를 구현하고,
  frontmatter·태그·Wiki link·변경 감지 테스트를 작성한다.
- [ ] **Check 13.5**: SQLite 카탈로그/FTS5와 Chroma 공통 인덱스를 구현하고,
  `jesseagent sources sync obsidian` 증분 동기화를 제공한다.
- [ ] **Check 13.6**: 벡터·키워드 RRF 검색, Obsidian URI 근거, 등록형 Agent 작업과
  조회 도구를 구현한다.
- [ ] **Check 13.7**: Sink connector의 계획·미리보기·명시 승인 계약을 durable run
  lifecycle에 연결하고 실제 Sink 없이 상태 전이 테스트를 작성한다.

## Completion Rules

한 번에 하나의 미완료 체크리스트만 구현한다. 각 항목 뒤에는 `poetry run poe check`를
통과시키고, 검토·승인·커밋 후 다음 항목으로 진행한다.
