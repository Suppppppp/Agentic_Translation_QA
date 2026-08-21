# 1차 시스템 워크플로우 기준선 및 현재 상태

## 1. 문서 목적

이 문서는 품질 개선 전 **최초 정상 동작 시스템(S1)**의 상태를 고정한다.
이후 Agent 판단 규칙, RAG, reviser 또는 번역 로직을 개선했을 때 무엇이
달라졌는지 같은 기준으로 비교하기 위한 출발점이다.

- 기준일: 2026-08-21
- 단계: S1 — 1차 MVP 워크플로우 구현 및 최초 수동 검수 완료
- 번역 방향: 한국어 → 영어
- 평가셋: `evaluation_v1`, 40문장, 8개 glossary 용어
- 동결 benchmark run: `429d6c4a-a1b6-4514-bfd3-dab6966c4101`
- 본 문서의 성격: 개선 전 기준선이며 최종 성능 보고가 아님

## 2. 현재 시스템 워크플로우

현재 시스템은 동일한 기본 번역기를 중심으로 Baseline, RAG, Agent의 기여를
분리해 볼 수 있는 4개 실행 조건을 제공한다.

```text
한국어 source
├─ Baseline
│  └─ Marian NMT → 초기 번역 → 종료
├─ RAG only
│  └─ exact 우선 safe-hybrid glossary 검색
│     → 용어 constraint → Marian NMT → deterministic post-edit → 종료
├─ Agent only
│  └─ Baseline 초기 번역 → Qwen Judge
│     → PASS면 종료 / NEEDS_REVISION이면 Qwen Reviser
│     → 최대 2회 추가 시도 → 최선 후보 선택·rollback
└─ Agent + RAG
   └─ RAG 초기 번역과 검색 근거 → Qwen Judge
      → 조건부 revision → 최대 2회 추가 시도 → 최선 후보 선택·rollback
```

### 구성 요소

| 영역 | S1에서 채택한 구성 | 역할 |
| --- | --- | --- |
| 기본 번역 | `Helsinki-NLP/opus-mt-ko-en` | 네 조건이 공유하는 한국어→영어 NMT |
| 용어 검색 | `ExactFirstHybridGlossaryRetriever` | exact 결과를 우선하고 필요할 때 vector 후보를 보조로 사용 |
| 용어 반영 | glossary post-edit | 검색된 용어를 결정적으로 반영 |
| Judge/Reviser | Ollama `qwen3:1.7b` | 번역 판단, 오류 설명, 조건부 수정 |
| 재시도 | 초기 번역 후 최대 2회 | 무조건 재번역하지 않고 판단 결과에 따라 실행 |
| 서비스 | FastAPI | baseline, agent-rag, benchmark API 제공 |
| 추적성 | attempt trace | 검색 결과, 번역, 판단, retry, stop reason, latency 기록 |

### 구현 선택의 이유

Marian에 glossary target을 직접 주입하는 `force_words_ids`와 source augmentation은 파일럿에서 반복·무관 토큰 또는 용어 미반영을 보여 현재 runtime에 채택하지 않았다. 대신 동일 Marian의 일반 번역 뒤 검색된 명시적 replacement rule만 적용하는 deterministic post-edit를 사용해 퇴행 위험과 조건 간 모델 차이를 줄였다. 품질 미달 재시도도 모든 실패에서 NMT를 다시 호출하지 않고, Agent가 `retry_with_rag`를 요청할 때만 검색 근거를 갱신한 뒤 로컬 Ollama reviser가 최대 2회 수정하도록 해 불필요한 검색과 제약 디코딩 퇴행을 피했다.

Baseline도 같은 translator wrapper를 사용하지만 검색 constraint가 없으므로
glossary 기반 변경은 적용되지 않는다. Reference 번역은 runtime 입력에 노출하지
않고 오프라인 평가에만 사용한다.

## 3. 1차 구현 완료 상태

| 항목 | 현재 상태 |
| --- | --- |
| 정상 실행 가능한 end-to-end 흐름 | 완료 |
| Baseline / RAG / Agent / Agent+RAG 분리 실행 | 완료 |
| 최대 2회 조건부 retry와 조기 종료 | 완료 |
| component trace와 fallback | 완료 |
| 40문장 공개 데이터 평가셋 동결 | 완료 |
| 동일 설정의 4-way benchmark | 완료 |
| Agent 출력 대표 표본 수동 검수 | 10개 source, 20개 mode-case 행 완료 |
| 전체 40문장의 사람 품질 판정 | 미실시 |
| 판단 규칙·번역 품질 튜닝 | 미실시 — 다음 개선 단계의 대상 |

즉, S1은 “완벽한 번역기”가 아니라 **정상적으로 실행되고 비교 가능한 실험
워크플로우**를 만든 단계다.

## 4. 전체 4-way benchmark 기준선

### 4.1 실행 범위

- 40문장 × 4조건 = 160개 결과
- 빈 번역 0건
- 동일 평가셋과 기본 번역 모델 사용
- warm-up 활성화
- 아래 latency는 동결 run에서 관찰한 값이며 반복 측정의 일반적 SLA가 아님
- 용어 정확도는 glossary occurrence 기준의 자동 지표이며 문장 전체 의미 품질을
  뜻하지 않음

### 4.2 결과

| 조건 | 용어 정확도 | 평균 / 중앙 / p95 latency | 문장 변경 | retry |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 40/44 = **90.9%** | 442.65 / 412.81 / 628.21 ms | N/A | N/A |
| RAG only | 44/44 = **100.0%** | 458.20 / 425.49 / 641.16 ms | N/A | N/A |
| Agent only | 41/44 = **93.2%** | 11,833.72 / 9,392.60 / 21,943.63 ms | 9/40 = 22.5% | 9/40 |
| Agent + RAG | 44/44 = **100.0%** | 11,187.07 / 9,285.75 / 22,367.25 ms | 6/40 = 15.0% | 6/40 |

### 4.3 현재 해석

1. RAG는 이 평가셋의 glossary occurrence를 40/44에서 44/44로 높였고 평균
   latency 증가는 약 15.6ms였다.
2. Agent 경로는 조건부 수정 흐름이 실제로 작동했지만 평균 latency가 약
   11.2~11.8초로, Baseline 약 0.44초보다 크게 증가했다.
3. Agent가 문장을 변경했다는 사실은 품질 개선을 뜻하지 않는다. 변경의 실제
   효과는 아래의 수동 검수로 따로 판단한다.
4. 자동 용어 정확도 100%도 주어 누락, 의미 왜곡, 고유명사 오류가 없다는 뜻은
   아니다. 실제 수동 검수에서 이 차이가 확인됐다.

## 5. Agent 수동 검수 기준선

### 5.1 표본 구성

수동 검수는 전체 40문장을 무작위로 뽑은 것이 아니라 서로 다른 실패 지점을
보기 위해 행동 기준으로 선정했다.

| 배치 | 구성 | 선정 목적 | 사람 판정 |
| --- | --- | --- | --- |
| 배치 1 | 5개 source × Agent/Agent+RAG = 10행 | retry, 모드 간 판단 불일치, component failure 확인 | 수정 필요 10, 모두 MAJOR |
| 배치 2 | 별도 5개 source × Agent/Agent+RAG = 10행 | 두 모드가 모두 첫 시도 PASS한 결과 감사 | 수정 필요 4(MAJOR), 실제 PASS 6 |
| 합계 | 10개 source, 20개 mode-case 행 | 디버깅용 대표 표본 | 수정 필요 14, 실제 PASS 6 |

따라서 아래 수치는 전체 benchmark 품질 추정치가 아니라 **선정된 진단 표본에서의
Agent 판단 성능**이다. 같은 source의 두 모드를 각각 한 행으로 집계했으므로
20개의 독립 source로 해석하지 않는다.

### 5.2 혼동행렬과 품질 지표

Positive class는 사람이 `NEEDS_REVISION`으로 판정한 경우다. Component failure
1건은 사용 가능한 Agent judgment가 없어 정확도와 recall 분모에서 제외한다.
다만 해당 행에는 사람이 확정한 initial→final outcome이 있으므로 현재 scorer
계약에 따라 successful correction 분모에는 포함한다.

| 범위 | TP / TN / FP / FN | 판단 정확도 | 수정 필요 recall | MAJOR false PASS | Successful correction | Component failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 | 6 / 6 / 0 / 7 | 12/19 = **63.2%** | 6/13 = **46.2%** | **7건** | 5/14 = **35.7%** | 1/20 |
| Agent only | 4 / 3 / 0 / 3 | 7/10 = **70.0%** | 4/7 = **57.1%** | 3건 | 4/7 = **57.1%** | 0/10 |
| Agent + RAG | 2 / 3 / 0 / 4 | 5/9 = **55.6%** | 2/6 = **33.3%** | 4건 | 1/7 = **14.3%** | 1/10 |

### 5.3 지표 해석

- 가장 큰 병목은 불필요한 수정이 아니라, 수정이 필요한 MAJOR 오류를 PASS한
  7개의 FN이다.
- `FP=0`이지만 사람 PASS가 6행뿐이고 모두 PASS 사례를 감사한 배치 2에서
  나왔으므로, 일반적인 불필요 수정률이 0이라고 주장할 수 없다.
- 14개 수정 필요 행이 모두 MAJOR이고 MINOR 표본은 없다. Severity 분포를
  일반화할 수 없다.
- Successful correction 5/14는 탐지부터 최종 개선까지 본 end-to-end 지표다.
  오류를 PASS해 retry하지 않은 경우도 개선 실패로 포함되므로 reviser 자체의
  조건부 성공률과는 다르다.
- 이 표본에서 Agent only가 Agent+RAG보다 좋았지만, 작은 행동 선정 표본과
  Agent+RAG의 component failure 1건이 포함돼 있다. RAG가 일반적으로 품질을
  악화시킨다는 인과 결론은 내리지 않는다.

## 6. 반복 오류와 대표 사례

### 6.1 오류 분포

14개 수정 필요 mode-case 행의 primary error는 다음과 같다.

| Primary error | 행 수 | 관찰 |
| --- | ---: | --- |
| `omission_addition` | 10 | 주어·고유명사·핵심 개념 누락 또는 원문에 없는 정보 추가가 지배적 |
| `term` | 4 | glossary 밖 핵심 용어 또는 잘못된 표현이 남음 |

여러 오류를 동시에 허용한 전체 태그는 `omission_addition` 12회, `term` 8회,
`meaning` 4회, `fluency_grammar` 4회였다. 태그는 중복되며 같은 source의 두
모드가 포함돼 있으므로 고유 문장 수와 같지 않다.

특히 FN 7건은 모두 MAJOR였으며, 그중 6건의 primary error가
`omission_addition`이었다.

### 6.2 반복 원인

#### A. Standalone 주어·고유명사 보존 실패

가장 반복적인 실패는 원문의 명시적 기술 주어를 문맥 없는 `It`으로 바꾸는
것이었다. Docker, Load balancing, Website development, Consul 등이 사라졌지만
Agent는 문장의 표면적 유창성을 보고 PASS했다. Source analysis에 key term이
있어도 Judge의 blocking 조건으로 연결되지 않았다.

#### B. 진단 내용과 최종 PASS 결정의 모순

일부 judgment는 summary에서 실제 오류를 언급하면서도 `passed=true`,
`error_types=[]`, `accept`를 반환했다. 현재 S1은 구조화 필드의 값은 검증하지만
summary의 의미와 최종 결론이 일치하는지는 강제하지 않는다.

#### C. RAG의 범위와 과신

RAG는 `deployment` 같은 등록 용어는 고쳤지만 Docker 같은 주어 누락까지
복원하지 못했다. 검색된 glossary term을 만족했다는 사실이 문장 전체 의미의
정확성을 보장하지 않는데도 Judge가 이를 충분히 구분하지 못한 정황이 있다.

#### D. Retry regression과 최종 후보 안전성

오류를 탐지해도 revision이 새로운 고유명사 오류를 만들 수 있었다. 이후
Judge가 새 후보를 승인하면 처음보다 악화된 결과가 최종 선택될 수 있다.

#### E. 구조화 출력의 component failure

Agent가 계약에 없는 error type을 반환한 한 사례에서 judgment parsing이 실패했다.
번역 결과는 남았지만 Agent 판단 정확도에는 사용할 수 없었다.

### 6.3 대표 사례

| 사례 | 관찰된 결과 | 원인 분류 | 이후 비교 포인트 |
| --- | --- | --- | --- |
| `evaluation-v1-012::{agent,agent_rag}` | `Docker`가 `It`으로 누락. RAG는 `distribution`을 `deployment`로 고쳤지만 주어 누락은 그대로 PASS | 주어 보존 실패, RAG 범위 한계 | entity coverage 검사가 두 모드의 FN을 잡는지 |
| `evaluation-v1-015::{agent,agent_rag}` | `Load balancing`이 `It`으로 바뀌었지만 두 모드 모두 PASS | 주어 보존 실패 | standalone source의 핵심 주어를 blocking error로 잡는지 |
| `evaluation-v1-009::agent_rag` | `Road balance`, `single-point disorders`가 남았고 summary도 오류를 언급했지만 PASS | 진단-결정 모순 | summary와 structured decision의 일관성 확보 여부 |
| `evaluation-v1-019::agent_rag` | `Website development` 누락과 원문에 없는 `all sorts of things` 추가를 summary에서 지적하고도 PASS | 누락·추가, 진단-결정 모순 | omission/addition recall 변화 |
| `evaluation-v1-027::agent_rag` | retry가 `Consul` 누락을 `Consult`라는 새 오류로 바꾸고 최종 승인, 사람 판정 `WORSE` | retry regression | 새 entity 생성 차단 및 이전 후보 rollback 여부 |
| `evaluation-v1-024::agent_rag` | 비계약 error type `pronoun`으로 judgment parse 실패 | component contract failure | malformed judgment가 전체 흐름을 중단하지 않는지 |

## 7. S1의 강점과 한계

### 강점

- 네 조건을 같은 평가셋과 기본 번역기로 비교할 수 있다.
- RAG가 glossary occurrence를 실제로 개선했다.
- Agent의 conditional retry, trace, stop reason과 fallback이 실제 실행된다.
- 수동 라벨과 자동 benchmark artifact가 분리돼 reference 누수를 막는다.
- 실패 사례와 component failure도 결과에서 제거하지 않고 보존했다.

### 한계

- Judge의 수정 필요 recall이 6/13, 46.2%로 낮다.
- 주어·고유명사·핵심 의미 보존을 deterministic하게 확인하지 않는다.
- summary와 `passed`의 의미적 모순을 차단하지 않는다.
- reviser가 만든 새 entity 오류와 regression을 충분히 보호하지 못한다.
- Agent 경로 latency가 약 11초로 Baseline보다 매우 크다.
- 사람 검수는 10개 source에 한정돼 전체 40문장의 사람 품질 지표가 아니다.

## 8. 다음 개선 문서에서 비교할 기준

| 비교 축 | S1 기준값 | 다음 단계에서 확인할 것 |
| --- | ---: | --- |
| Agent 판단 정확도 | 12/19 = 63.2% | 동일 수동 표본 또는 사전 고정 replay에서 변화 |
| 수정 필요 recall | 6/13 = 46.2% | MAJOR FN 7건을 얼마나 추가 탐지하는지 |
| MAJOR false PASS | 7건 | 주어·entity 검사 후 감소 여부 |
| 실제 PASS 오탐 | FP 0/6 | recall 향상 때문에 정상 번역을 과도하게 수정하지 않는지 |
| Successful correction | 5/14 = 35.7% | 탐지 이후 final이 실제로 개선되는지 |
| Component failure | 1/20 | 구조화 출력 복구 후 0건인지 |
| Retry regression | 대표 1건 | `Consul → Consult` 유형을 차단하는지 |
| RAG 용어 정확도 | 44/44 = 100% | 의미 품질 개선 중에도 용어 정확도를 유지하는지 |
| Agent 평균 latency | 11,833.72ms | 품질 향상 비용과 함께 보고 |
| Agent+RAG 평균 latency | 11,187.07ms | 품질 향상 비용과 함께 보고 |

가장 먼저 비교할 개선 후보는 다음 두 가지다.

1. Source의 standalone 주어·고유명사·핵심 기술어 보존과 진단-결정 일관성을
   PASS 전 필수 검사로 두는 것
2. malformed judgment 복구와 revision 전·후 entity 비교를 통해 component
   failure 및 retry regression을 막는 것

최소 실험은 전체 benchmark 재실행보다 먼저, 저장된 후보 중 현재 FN 7행과
실제 PASS 6행을 고정 replay set으로 사용한다. 새 판단 방식이 FN을 더 잡으면서
기존 6개 TN에 FP를 만들지 않는지 확인한 뒤에만 더 큰 실험으로 확장한다.

## 9. 재현 근거

### 동결 benchmark

- [4-way benchmark artifact](../../artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json)
- artifact SHA-256: `e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf`
- dataset SHA-256: `cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650`
- config SHA-256: `2a2935ab0c366522aeae3cbbbb7c7e35c1fe96c654b3131bdcca164daaee16b1`

### 수동 검수

- [배치 1 severity 점수](../../data/manual_reviews/evaluation_v1_batch1_severity/scores.json)
  - SHA-256: `789391c7dde790d4c5ccc94cb56fe1c4389e40579bead49f6766f904ece3cf8b`
- [배치 2 점수](../../data/manual_reviews/evaluation_v1_batch2/scores.json)
  - SHA-256: `48f53f4c25bf353b10065f4a13ec9dc208ccc03307cbef261e4c737f27f7b0bc`
- [평가 계약](../EVALUATION_CONTRACT.md)
- [배치 1 FN 및 component failure 분석](../BATCH1_FN_COMPONENT_FAILURE_ANALYSIS.md)

## 10. 보고서용 요약

1차 단계에서는 Marian 기반 한영 번역, glossary RAG, 로컬 Qwen Judge/Reviser,
조건부 retry와 trace를 통합한 정상 동작 워크플로우를 구현했다. 40문장 4-way
benchmark에서 RAG는 용어 정확도를 90.9%에서 100%로 높였으나, Agent 경로는
평균 약 11초의 지연을 보였다. 대표 10개 source의 수동 검수에서는 Agent 판단
정확도 12/19(63.2%), 수정 필요 recall 6/13(46.2%), MAJOR false PASS 7건,
successful correction 5/14(35.7%)가 관찰됐다. 반복 오류는 glossary 용어보다
주어·고유명사·핵심 정보 누락과 판단 summary-결정 불일치에 집중됐다. 따라서
다음 개선 단계는 번역 모델 자체를 먼저 바꾸기보다 source-aligned Judge 검사와
revision regression 방어를 작은 고정 replay 실험으로 검증하는 방향이 적절하다.
