# 한계와 개선 계획

## 1. 현재 시스템이 증명한 것

- 로컬 open-source 모델만으로 Baseline → RAG → NMT → Judge → 조건부 retry →
  trace의 end-to-end 흐름이 정상 실행된다.
- 같은 NMT를 사용하는 4-way ablation으로 RAG와 Agent의 기여를 분리했다.
- RAG는 동결셋의 표시 용어를 40/44에서 44/44로 개선했다.
- Agent는 모든 문장을 무조건 수정하지 않고 필요한 경우에만 최대 2회 재시도한다.
- 실패, retry regression, component failure와 latency 비용을 artifact에 남겼다.

이 결과는 완벽한 번역 품질을 증명한 것이 아니라, 비교·관찰·개선 가능한 MVP
워크플로우를 만들었다는 증거다.

## 2. 품질 한계

### 2.1 Judge recall

진단 표본에서 수정 필요 recall은 46.2%이고 MAJOR false PASS가 7건이었다. 가장
큰 병목은 정상 번역을 불필요하게 고치는 것보다, 중요한 오류를 보지 못하고 PASS
하는 것이다.

### 2.2 Source coverage

standalone 주어, 제품명과 핵심 기술어가 `It`으로 사라져도 유창성 때문에 PASS했다.
Glossary target 존재만 검사해서는 원문 entity와 정보 구조가 보존됐는지 알 수 없다.

### 2.3 용어 지표의 좁은 범위

자동 용어 정확도 100%는 44개 marked occurrence에 대한 결과다. 문장 전체 의미,
누락·추가, 문법과 register는 별도 사람 평가가 필요하다.

### 2.4 Revision regression

Reviser가 `Consul`을 `Consult`로 바꾸는 새 오류를 만들 수 있다. Judge와 Reviser가
같은 모델이면 잘못된 수정을 다시 승인하는 자기확증 위험도 있다.

### 2.5 구조화 출력 안정성

소형 LLM은 JSON schema 밖 error label을 만들 수 있다. 한 malformed payload가
Agent 판단 전체를 component failure로 만들 수 있어 fail-safe와 감사 로그가
필요하다.

## 3. 성능·운영 한계

- Agent 평균 latency가 약 11초로 Baseline보다 약 25~27배 느리다.
- 이 latency는 단일 warm run이며 조건별 3회 반복 SLA가 아니다.
- 16GB 로컬 환경에서는 NMT, embedding, Ollama 모델의 동시 상주가 제한된다.
- in-memory vector search는 작은 glossary용이며 대규모 corpus나 다중 worker에는
  적합하지 않다.
- 인증, queue, rate limit, distributed tracing, deployment manifest는 범위 밖이다.

## 4. 데이터·평가 한계

- 평가셋은 software 단일 도메인 40문장이다.
- 수동 검수는 10 source의 20 paired mode-case 행뿐이고 무작위가 아니다.
- 수정 필요 사례가 모두 MAJOR여서 MINOR 판정 성능을 알 수 없다.
- 용어 빈도가 불균형하고 일부 용어는 1회뿐이다.
- 동결 결과를 본 뒤 개선한 규칙을 같은 데이터에 반복 적용하면 과적합 위험이
  생긴다.
- 사람 reference도 가능한 번역 중 하나이며 완전한 의미 gold가 아니다.

## 5. 우선 개선 항목

### 우선순위 1 — 증거 기반 source coverage

원문에서 명시적 standalone 주어, 고유명사, 독립 provenance가 있는 기술어를
추출하고 매 candidate에서 보존 여부를 검사한다.

- source에 실제 term이 존재할 때만 적용
- accepted alias 근거가 없으면 `UNVERIFIABLE`
- 명백한 누락만 blocking error
- 모든 retry 뒤 재검사
- 임시 must-preserve는 Reviser에만 전달

성공 기준은 고정 FN/TN replay에서 기존 FN을 더 잡으면서 TN false positive를 만들지
않는 것이다.

### 우선순위 2 — 판정 일관성과 regression guard

- 구조화 `error_types`에서 최종 PASS/REVISE를 코드가 파생
- LLM의 중복 passed/action claim은 감사용으로만 보존
- summary와 structured decision 모순 기록
- revision 전후 source entity·숫자·용어 coverage 비교
- 새 오류가 생기면 이전 후보 우선 또는 안전 rollback

## 6. 단계적 최소 실험 계획

| 단계 | 입력 | 변경 범위 | Gate |
|---|---|---|---|
| A. Detector replay | 저장 FN 7 + TN 6 | 번역 호출 없음 | FN 추가 탐지, TN FP 0 |
| B. Judge-only replay | 같은 저장 후보 | Judge 계약만 | recall 상승, component failure 0 |
| C. Revision micro-run | 잡힌 누락 사례만 | Reviser만 | improved > worse, entity regression 0 |
| D. Reserve review | authoring에 쓰지 않은 사례 | 사람 검수 | 자동 gold 생성 금지 |
| E. 새 동결 benchmark | 새 또는 reserve set | 전체 파이프라인 | 품질·latency 동시 보고 |

각 단계가 실패하면 같은 표본에 규칙을 계속 덧붙이지 않고 원인과 한계를 기록한다.
다음 전체 benchmark는 설정과 hash를 다시 고정한 뒤 별도 iteration으로 실행한다.

## 7. 모델·인프라 후속 대안

- Agent LLM을 3B급과 비교하되 동일 Judge-only 입력과 latency를 함께 평가한다.
- 대규모 glossary가 필요하면 FAISS/Chroma와 영속 index를 도입한다.
- Judge와 Reviser 모델을 분리하거나 rule-based verifier를 독립 gate로 둔다.
- production에서는 async worker, model pool, request timeout, auth와 observability를
  추가한다.
- 충분한 독립 도메인 데이터가 확보된 뒤에만 NMT fine-tuning을 검토한다.

## 8. 제출 시 정직하게 유지할 주장

- “RAG가 동결셋의 marked term accuracy를 개선했다”는 주장 가능
- “Agent conditional workflow가 실행됐다”는 주장 가능
- “Agent가 전체 번역 품질을 개선했다”는 일반 주장 불가
- “Agent+RAG가 Agent보다 우수하다”는 일반 주장 불가
- “현재 guard 개선이 전체 benchmark를 개선했다”는 주장 불가
- 결과는 해당 hardware, dataset, config와 선택된 수동 표본의 관찰로 제한

