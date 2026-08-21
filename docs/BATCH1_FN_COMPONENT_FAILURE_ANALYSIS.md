# Batch 1 FN 및 Component Failure 원인 분석

분석 대상은 기존 `evaluation_v1` benchmark artifact와 수동 검수 배치 1뿐이다. 모델 재실행, 품질 튜닝, 판단 규칙 변경 없이 기록된 trace를 해석했다.

## 요약

- FN 3건은 모두 수동으로는 수정이 필요했지만 Agent가 초기 후보를 `passed=true`로 수락한 `AGENT_FALSE_PASS`이다.
- 세 건 모두 Agent가 summary에서 적어도 일부 오류를 언급하면서도 `error_types=[]`, `next_action=accept`로 종료했다.
- Component failure 1건은 Agent가 허용 taxonomy 밖의 `pronoun`을 반환해 구조화 judgment 검증에 실패한 `PARSE_OR_RUNTIME_ERROR`이다.

## 사례별 분석

### evaluation-v1-009::agent_rag

- 후보에는 `Road balance`와 `single-point disorders`가 남아 있었다.
- Agent summary도 `single-point disorders`가 `single-point failure` 대신 쓰였다고 인식했다.
- 그러나 최종 judgment는 `passed=true`, `error_types=[]`, `accept`였고 retry 없이 종료했다.
- RAG 검색은 `서버 → server`만 제공했으며 candidate가 이미 `servers`를 포함해 검색 용어 제약은 충족했다.
- 같은 초기 번역을 Agent-only 모드는 실패로 판정했다. 이 한 쌍만으로 RAG가 오판을 유발했다고 단정할 수는 없지만, 동일 후보에 대한 판정 일관성 부족은 확인된다.

### evaluation-v1-019::agent_rag

- RAG는 `배포 → deployment`를 적용했다.
- 후보에는 주어 `Website development`와 “여러 단계” 구조의 누락, 원문에 없는 `all sorts of things`가 남았다.
- Agent summary는 추가 표현을 인식했지만 `passed=true`로 종료했다.
- 검색된 glossary term은 충족했으나, 검색 범위 밖의 누락·추가·유창성 품질을 과소평가한 사례다.

### evaluation-v1-027::agent

- source analysis에는 `컨설`과 `구성`이 key term으로 잡혔다.
- 후보는 `Consul`을 누락하고 `configure`를 `organize`로 옮겼다.
- Agent summary는 `organize`와 `configure`의 차이를 언급했지만 `passed=true`로 결정했고, 고유명사 누락은 지적하지 않았다.
- Agent-only 경로에는 retrieval/term guard가 없으므로 source-analysis key term이 후보에 보존됐는지를 결정적으로 검사하지 못했다.

### evaluation-v1-024::agent_rag

- translation과 retrieval은 완료됐지만 첫 judgment가 `null`이다.
- warning은 `error_types`에 허용되지 않은 `pronoun`이 포함되어 Pydantic 검증이 실패했다고 기록한다.
- 파이프라인은 `stop_reason=component_failure`, `selection_reason=only_candidate`, retry 0으로 안전 종료했다.
- RAG는 `데이터베이스 → database`만 검색했고, 핵심 `널 → null`은 source analysis 및 retrieval evidence에 포함되지 않았다.
- artifact에는 raw Agent JSON 전체가 보존되지 않아 `pronoun` 외 payload와 생성 이유는 확정할 수 없다.

## 공통 원인 경계

1. summary의 오류 언급과 최종 `passed` 결정이 일치하지 않아도 기존 artifact에서는 `passed`가 우선됐다.
2. deterministic term guard는 검색된 glossary term만 검사하므로 검색되지 않은 고유명사·핵심 용어와 일반 의미·누락 오류는 보호 범위 밖이다.
3. component failure는 번역 실패가 아니라 Agent judgment의 schema/taxonomy 불일치다.
4. component failure 행은 첫 judgment가 없으므로 판단 정확도 분모에서 제외하는 것이 맞고, 번역의 initial→final 수동 outcome은 별도로 유지할 수 있다.

이 문서는 원인 기록만 제공하며 코드 또는 Agent 판단 규칙 변경을 포함하지 않는다.
