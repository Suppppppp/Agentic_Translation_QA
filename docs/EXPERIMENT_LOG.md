# Experiment Log

> 상태: 기능 MVP 기준선 완료 — **`evaluation_v1` 동결 및 첫 4-way 실행 완료**

이 문서는 파일럿과 공개 데이터 기준선에서 시도한 선택과 채택·폐기 근거를 기록한다. EXP-014에서 40문장·8용어 `evaluation_v1`을 동결했고 EXP-015에서 첫 실제 4-way 실행을 완료했다. 이 수치는 품질 튜닝 전의 기능 기준선이며 최종 성능으로 해석하지 않는다.

## 기록 원칙

- **확인된 사실**: 저장된 artifact나 정확히 보존된 입력·출력으로 재확인할 수 있는 내용
- **정성 관찰**: 실행 중 확인했지만 전체 출력이나 측정값이 보존되지 않은 내용
- **가설**: 아직 비교 실험으로 검증하지 않은 다음 후보

최종 평가셋을 실행하기 전까지는 구현 방식과 설정을 파일럿에서 바꿀 수 있다. 최종 평가셋, glossary, prompt, retrieval 설정과 임계값을 동결한 뒤에는 결과를 보고 재조정하지 않는다.

## EXP-001. MarianMT Baseline 실행 확인

- 날짜: 2026-08-20
- 증거 수준: 확인된 사실
- 환경: Apple M1 Pro, CPU, Python 3.12
- 번역 모델: `Helsinki-NLP/opus-mt-ko-en`
- Transformers: 4.56.2
- 입력: `오늘 회의는 오후 세 시에 시작한다.`
- 출력: `This meeting begins at three o'clock in the afternoon.`
- 최초 모델 로딩: 약 58.8초(다운로드 포함)
- 결과: Baseline 후보로 채택

모델 로딩 이후 API를 통한 warm 번역도 정상 동작했다. 네 비교 조건에서 같은 기본 모델과 디코딩 설정을 사용한다.

## EXP-002. Lexically constrained decoding 파일럿

- 날짜: 2026-08-20
- 데이터: `pilot-001` 한 문장
- 검색: software glossary exact match
- 검색 용어: `배포 → deployment`와 허용형 `deploy`, `deployed`, `deploying`
- 주입 방식: MarianMT `force_words_ids` 기반 constrained beam search
- 입력: `개발팀은 금요일에 새 버전을 운영 환경에 배포한다.`

### 확인된 사실

- Baseline 최종 출력은 `Development team distributes a new version of the operating environment on Friday.`였다.
- Retriever는 `sw-deployment`를 exact match, score 1.0으로 반환했다.
- guard와 fallback을 포함한 저장 artifact에서 Baseline과 RAG 최종 출력의 용어 정확도는 모두 `0/1 (0%)`였다.
- 같은 artifact의 1회 warm 측정은 Baseline `282.64ms`, RAG `1,349.82ms`였다.
- 근거 artifact: `artifacts/benchmark-e227b798-6886-4902-ade7-034ff9472869.json`

### 정성 관찰

강제 제약 후보에는 target term이 들어갔지만 괄호형 반복 문구와 무관한 토큰이 길게 생성되어 원문 의미와 유창성이 크게 퇴행했다. 퇴행 후보의 전체 문자열은 artifact에 보존되지 않았으므로 이를 정량 사례로 사용하지 않는다.

### 결정

현재 모델·용어·디코딩 조합의 `force_words_ids`를 RAG 주입 방식으로 채택하지 않는다. 한 문장만 확인했으므로 constrained decoding 일반의 실패로 확대 해석하지 않는다.

## EXP-003. Source augmentation 세 방식 비교

- 날짜: 2026-08-20
- 데이터: EXP-002와 같은 `pilot-001` 한 문장
- 목적: decoder 강제 제약 없이 source에 target term을 노출하면 MarianMT가 용어를 보존하는지 확인

### 확인된 사실

혼합 영문 입력 `개발팀은 금요일에 새 버전을 운영 환경에 deployment한다.`에서 다음 출력이 보존됐다.

```text
Development team is running a new version on Friday, defaulting on the operating environment.
```

target term을 원하는 의미로 보존하지 못했고 `defaulting`이라는 의미 오류가 생겼다.

### 정성 관찰

- `배포(deployment)한다.`처럼 괄호로 target을 추가한 방식은 Baseline과 같은 `distributes` 계열로 번역되어 용어를 반영하지 못했다. 정확한 전체 출력은 저장되지 않았다.
- 용어를 placeholder로 치환한 방식은 출력에서 placeholder가 `_BAR_0___`와 같이 손상됐다. 정확한 전체 출력과 timing은 저장되지 않았다.

### 결정

세 방식 모두 이 한 문장에서 target term을 안정적으로 보존하지 못했으므로 현재 형태의 source augmentation은 채택하지 않는다. 이는 source augmentation 전체가 효과 없다는 결론이 아니며, 정확한 재현 artifact가 없는 두 방식은 정성 관찰로만 남긴다.

## EXP-004. Degeneracy guard와 fallback 확인

- 날짜: 2026-08-20
- 증거 수준: 확인된 사실
- 근거 artifact: `artifacts/benchmark-e227b798-6886-4902-ade7-034ff9472869.json`

constrained decoding 후보가 퇴행하자 guard가 이를 감지했고, 파이프라인은 같은 NMT를 제약 없이 다시 호출했다.

- 최종 RAG 출력은 Baseline 출력과 같았다.
- trace의 종료 사유는 `constraint_fallback`이었다.
- warning에 constraint 퇴행과 unconstrained fallback 필요성이 기록됐다.
- fallback 뒤 `applied_constraints`는 빈 목록으로 기록됐다.
- RAG latency는 이 1회 측정에서 Baseline의 약 4.78배였지만 용어 정확도 개선은 없었다.

### 결정

fallback은 명백한 퇴행 후보가 사용자에게 반환되는 것을 막는 **안전장치**로 유지한다. 다만 fallback 성공은 RAG 품질 향상이 아니라 Baseline으로의 안전한 복귀다. 따라서 이 결과를 RAG 채택 근거로 사용하지 않는다.

## EXP-005. 12문장 Baseline 파일럿

- 날짜: 2026-08-20
- 증거 수준: 확인된 사실
- 환경: CPU, `HF_HUB_OFFLINE=1`
- 데이터: 합성 `pilot_v1` 12문장, 필수 용어 occurrence 12개
- 조건: Baseline only, 모델 로딩 후 warm 실행
- 근거 artifact: `artifacts/benchmark-8f27b86f-1fe7-447b-826b-2a494f1b6f28.json`

### 확인된 결과

- 용어 정확도: `3/12 (25.0%)`
- mean latency: `302.28ms`
- median latency: `296.04ms`
- nearest-rank p95 latency: `398.79ms`

### 해석 범위

이 실행은 12개 파일럿 문장의 Baseline 오류와 대략적인 warm latency를 확인하기 위한 것이다. 공개 데이터 기반 최종셋이 아니며, 조건별 3회 반복 측정도 아직 수행하지 않았으므로 평가 계약의 최종 latency 결과가 아니다. 다만 용어 9/12 occurrence를 놓친 결과는 파일럿에서 RAG 주입 방식을 계속 비교할 필요가 있음을 보여준다.

## EXP-006. Multilingual MiniLM 검색 비교

- 날짜: 2026-08-20
- 증거 수준: 실제 CLI 실행 및 기대 term 대비 수동 집계
- 환경: CPU, `HF_HUB_OFFLINE=1`
- 데이터: 합성 `pilot_v1` 12문장과 pilot glossary
- 임베딩 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- vector minimum score: `0.35`
- 실행 명령:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/compare_retrieval.py --embedder sentence-transformer --min-score 0.35
```

### 확인된 결과

| 검색 방식 | 기대 hit | 잘못된 vector-only hit | 수동 precision | 수동 recall |
| --- | ---: | ---: | ---: | ---: |
| exact | 12/12 | 해당 없음 | 별도 집계 안 함 | 100% |
| vector | 8/12 | 8 | 50.0% | 66.7% |
| hybrid raw union | 12/12 | 8 | 60.0% | 100% |

- exact 검색은 source에 literal로 존재하는 기대 용어 12/12를 회수했다.
- 용어가 없는 no-match 문장에서는 exact hit가 0개였다.
- vector 검색은 기대 hit 8개와 잘못된 vector-only hit 8개를 반환했다.
- 단순 hybrid union은 exact의 기대 hit 12개를 모두 유지했지만 vector의 오검색 8개도 함께 포함했다.

### 결정

vector 결과를 무조건 합치는 naive union은 현재 설정에서 noise가 커 채택하지 않았다. 이후 safe `ExactFirstHybridGlossaryRetriever`를 구현했다. literal exact hit가 있으면 해당 term만 유지하고 vector score는 같은 term의 보조 점수로만 사용하며, exact hit가 없을 때에만 제한된 vector fallback을 허용한다. 이 정책은 literal pilot에서 raw union이 추가했던 잘못된 vector-only hit 8개를 결과에서 제거했다.

따라서 exact-first 정책은 더 이상 미구현 가설이 아니라 현재 RAG 검색 경로다. 다만 이번 확인은 literal term 중심 합성 파일럿이므로, exact가 실패하는 실제 paraphrase와 진짜 no-match에서 vector fallback이 정확한지는 아직 별도 검증이 필요하다. 검색 비교 자체도 최종 검색 성능이 아니다.

## EXP-007. 실제 모델 5-arm 용어 주입 비교

- 날짜: 2026-08-20
- 증거 수준: 실제 CLI 실행, 명령 재현 가능, JSON 출력 artifact는 미보존
- 데이터: `pilot-001` 한 문장
- 비교 조건: Baseline, lexical constraints, mixed English, parenthetical, quoted marker
- 재현 명령:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/compare_injection.py --real-model --device cpu --case-id pilot-001 --format json
```

### 확인된 결과

| 주입 방식 | 필수 용어 반영 | 실행 상태 |
| --- | --- | --- |
| Baseline | miss | 정상 완료 |
| lexical constraints | miss | degeneracy failure |
| mixed English | miss | 정상 완료 |
| parenthetical | miss | 정상 완료 |
| quoted marker | miss | 정상 완료 |

lexical constraints는 backend degeneracy guard가 거부했고 나머지 source augmentation도 `deployment` 또는 허용형을 최종 후보에 보존하지 못했다. 실제 JSON 전체는 artifact로 저장하지 않았으므로 이 표는 실행 중 확인한 term hit와 상태만 기록하며, 후보별 정확한 문장이나 latency는 주장하지 않는다.

### 결정

EXP-002~003의 수동 관찰을 실행 가능한 동일 스크립트에서 다시 확인했다. 이 5개 방식은 현재 모델과 `pilot-001` 조합의 주입 방식으로 채택하지 않는다. 한 문장 결과이므로 다른 모델이나 모든 제약 디코딩 방식의 실패로 일반화하지 않는다.

## EXP-008. Pilot-only deterministic post-edit

- 날짜: 2026-08-20
- 증거 수준: 확인된 artifact
- 데이터: 합성 `pilot_v1` 12문장, 필수 용어 occurrence 12개
- 조건: 동일 MarianMT의 Baseline과 exact retrieval + deterministic replacement rule RAG
- 근거 artifact: `artifacts/benchmark-4671df40-a473-44b8-837e-5c4afd2d7917.json`

### 확인된 결과

| 조건 | 용어 정확도 | mean | median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `4/12 (33.33%)` | `289.44ms` | `289.77ms` | `355.58ms` |
| RAG post-edit | `12/12 (100%)` | `289.74ms` | `288.89ms` | `365.31ms` |

post-edit는 retrieved glossary에 명시된 pilot 전용 replacement rule만 사용하고, 이미 허용형이 있는 번역은 그대로 둔다. 이 실행에서는 용어 occurrence를 모두 맞혔지만 문장 전체 품질을 보장하지 않았다.

EXP-005와 이 artifact의 12개 Baseline 번역 문자열은 같지만 용어 점수는 `25.0%`에서 `33.33%`로 달라졌다. 이는 파일럿 사이에 평가 annotation 또는 metric 구성이 아직 동결되지 않았음을 보여준다. 따라서 artifact 사이의 점수를 추세처럼 비교하지 않고, 여기서는 같은 EXP-008 artifact 안의 paired Baseline/RAG 결과만 비교한다.

- `pilot-003`: `When the main server is suspended, automatic failover activate the waiting server.`
  - `failover`는 맞지만 `failover activate`의 주어·동사 일치 오류가 남았다.
- `pilot-011`: `Cache and his load balancing have been used to reduce response delays.`
  - `cache`, `load balancing`은 모두 맞지만 `his`가 들어간 문맥 오류가 남았다.

### 결정과 한계

deterministic post-edit는 현재 RAG-only 파일럿 후보로 유지하되, `100%`는 **용어 정확도만의 결과**다. replacement rule은 이 합성 파일럿의 Baseline 오역을 보고 작성된 pilot-only 규칙이므로 과적합 위험이 매우 크다. 최종셋의 출력이나 reference를 보고 새 치환 규칙을 추가해서는 안 되며, 최종 평가 전에 독립 glossary와 파일럿에서 규칙을 확정해야 한다.

## EXP-009. 실제 로컬 Agent 판단과 retry

- 날짜: 2026-08-20
- 증거 수준: 전체 run 전에 수행한 실제 선택 사례 3건
- 입력 후보: EXP-008의 RAG post-edit 결과

### 확인된 성공 사례

- `pilot-003`
  - 초기 후보: `When the main server is suspended, automatic failover activate the waiting server.`
  - 1회 retry 결과: `When the main server is suspended, automatic failover activates the waiting server.`
  - 종료: `passed`
  - total latency: `45,806.78ms`
- `pilot-012`
  - 구조화 응답 정규화 후 재시도 없이 fast path 종료
  - retry: `0`
  - total latency: `18,697.16ms`
- `pilot-011`
  - 초기 후보: `Cache and his load balancing have been used to reduce response delays.`
  - 1회 retry 결과: `Cache and load balancing have been used to reduce response delays.`
  - total latency: `32,978.67ms`

세 사례에서 불필요한 전체 재번역 대신 Agent revision이 표면적인 문법·문맥 오류를 고쳤거나 정상 후보를 재시도 없이 통과시켰다. 이후 전체 12문장 실행은 EXP-010과 EXP-011에서 수행했다. 이 초기 세 사례에는 수동 gold가 없으므로 Agent 정확도나 successful correction rate로 환산하지 않는다. 또한 Agent 경로는 이 실행에서 문장당 약 18.7~45.8초가 걸려 NMT-only 경로보다 비용이 매우 컸다.

### 발견한 구조화 판단 문제와 보정

- Agent가 source domain을 `IT`로, glossary entry는 `software`로 표기하는 불일치가 관찰됐다. literal Korean term이 source에 실제 존재하면 Agent domain label보다 exact literal match를 우선하도록 검색 규칙을 보정했다.
- raw Agent JSON에서 `passed=true`이면서 error와 revision action을 함께 내는 모순이 관찰됐다. 명시적인 `passed`를 우선하고 종속 필드인 `error_types`와 `next_action`을 일관된 값으로 canonicalize한 뒤 schema validation하도록 보정했다.
- LLM이 용어 누락 후보를 통과시키는 경우를 막기 위해, source에 적용 가능한 retrieved term의 target 존재 여부를 deterministic term guard로 다시 확인하도록 보정했다.

이 보정에도 한계가 있다. exact literal priority는 중의어에서 잘못된 domain term을 선택할 수 있고, term guard는 용어의 존재만 확인할 뿐 의미·문법을 보장하지 않는다. canonicalization도 명시적인 `passed` 자체가 잘못된 경우를 고치지 못한다. 따라서 이 세 장치는 Agent의 정확성을 입증하는 결과가 아니라 작은 로컬 모델의 불안정한 출력을 안전하게 다루기 위한 방어선이다.

## EXP-010. 모든 Agent 호출에 `think=false`

- 날짜: 2026-08-20
- 증거 수준: 확인된 artifact
- 데이터: 합성 `pilot_v1` 12문장
- 조건: Agent analyze, judge, revise 전부 `think=false`
- 근거 artifact: `artifacts/benchmark-f3904560-64de-4524-b671-c94b43cab047.json`

### 확인된 결과

- 용어 정확도: `12/12 (100%)`
- mean latency: `2,633.04ms`
- median latency: `2,641.72ms`
- p95 latency: `2,890.42ms`
- retry 분포: `{0: 12}`
- 문장 단위 수정률: `0/12 (0%)`

속도는 크게 줄었지만, EXP-008에서 확인한 명백한 오류도 모두 통과시켰다.

- `pilot-003`: `When the main server is suspended, automatic failover activate the waiting server.`를 `passed`로 판정했다.
- `pilot-011`: `Cache and his load balancing have been used to reduce response delays.`를 `passed`로 판정했다.

두 judgment summary도 각각 grammar 또는 `his` 문제를 언급하면서 최종 action은 `accept`였다. 즉 빠른 실행이 품질 판정 능력을 잃은 결과로 이어졌다.

### 결정

global `think=false`는 명백한 오류를 수정하지 못하고 12/12 모두 retry 없이 통과시켰으므로 폐기한다. 용어 정확도 100%는 앞단 deterministic post-edit의 결과이며 Agent 판단 성공의 근거가 아니다. 수동 gold label은 `0`건이므로 artifact의 Agent 정확도와 confusion counts도 미산출이다.

## EXP-011. Analyze만 `think=false`인 절충안

- 날짜: 2026-08-20
- 증거 수준: 확인된 artifact
- 데이터: 합성 `pilot_v1` 12문장
- 조건: analyze는 `think=false`, 품질에 직접 관여하는 judge와 revise는 `think=true`
- 근거 artifact: `artifacts/benchmark-ad03a99d-cdee-4458-acee-e89767eb231e.json`

### 확인된 결과

- 용어 정확도: `12/12 (100%)`
- mean latency: `11,487.50ms`
- median latency: `7,727.83ms`
- p95 latency: `29,398.47ms`
- retry 분포: `{0: 10, 1: 2}`
- 문장 단위 수정률: `2/12 (16.67%)`

전체 실행에서도 EXP-009에서 관찰한 두 수정 동작을 유지했다.

- `pilot-003`: `...automatic failover activate...`를 1회 retry해 `...automatic failover activates...`로 수정하고 `passed`로 종료했다.
- `pilot-011`: `Cache and his load balancing...`를 1회 retry해 `Cache and load balancing...`로 수정하고 `passed`로 종료했다.

### 결정과 한계

global `think=false`보다 느리지만 전체 파일럿에서 기존의 두 명백한 오류 수정 동작을 보존했으므로 이 절충안을 현재 Agent 설정으로 잠정 채택한다. 그러나 `changed 16.67%`는 단순 변경률이고 성공 수정률이 아니다. 수동 gold label과 manual outcome이 모두 `0`건이므로 Agent 정확도, confusion counts와 successful correction rate는 artifact에서 모두 미산출이다. 따라서 10개 무수정 통과가 실제 true negative인지도 아직 주장할 수 없다.

## 현재 수동 평가 지표 상태

benchmark 코드는 confirmed 수동 `PASS/NEEDS_REVISION` label이 있을 때만 첫 Agent 판단의 정확도와 confusion counts를 계산한다. confirmed initial label과 `IMPROVED/SAME/WORSE` outcome이 함께 있을 때만 successful correction rate도 계산한다. 현재 `pilot_v1`에는 두 종류의 confirmed label이 모두 `0`건이다.

- Agent 판단 정확도·revision recall·unnecessary revision rate·confusion counts: 미산출
- successful correction rate와 `IMPROVED` count: 미산출
- 문장 단위 수정률: 자동 계산 가능하지만 성공 여부를 대신하지 않음

artifact의 `unavailable_metrics`에도 이 사유가 기록되어 있다. 라벨이 없는 결과에서 어느 지표도 추정하지 않는다.

## 아직 검증하지 않은 가설과 다음 실험

다음 내용은 현재 사실이 아니라 이후 파일럿에서 검증할 가설이다.

- placeholder를 NMT가 직접 생성하게 하지 않고 번역 전후의 결정적 치환으로 관리하면 손상을 피할 수 있다.
- 한 문장에서 실패한 constrained decoding도 제약 후보와 beam 설정을 바꾸면 일부 용어에는 작동할 수 있다.
- safe exact-first hybrid의 vector fallback이 실제 paraphrase를 회수하면서 no-match 오검색을 통제할 수 있다.
- pilot-only replacement rule의 효과가 규칙 작성에 사용하지 않은 문장에서도 유지될 수 있다.
- EXP-011의 12개 첫 판단이 confirmed 수동 `PASS/NEEDS_REVISION` label과 충분히 일치하고, 두 retry가 실제 `IMPROVED`로 판정될 수 있다.

이 시점의 다음 후보는 EXP-011의 12개 initial/final 결과에 수동 오류 label과 `IMPROVED/SAME/WORSE` outcome을 붙여 confusion matrix와 successful correction rate를 계산하고, pilot-only replacement rule과 vector fallback을 독립 문장으로 검증하는 것이었다. 이후 공개 데이터셋 동결과 첫 4-way 실행은 EXP-014~015에 기록한다. EXP-011의 100% 용어 정확도와 두 Agent 수정 사례는 여전히 최종 성능으로 해석하지 않는다.

## EXP-012. 공개 데이터 source-only 후보 선정과 40문장 검수 초안

- 날짜: 2026-08-20
- 증거 수준: 재현 가능한 후보 pool과 AI 보조 이중언어 검수 초안
- 원천 데이터: `lemon-mint/korean_parallel_sentences_v1.1` 고정 revision
- 후보 artifact: `artifacts/eda/evaluation_candidates.jsonl`
- 후보 요약: `artifacts/eda/evaluation_candidates_summary.json`
- 선택 manifest: `data/evaluation_selection_v1_draft.json`
- 검수표: `artifacts/reviews/evaluation_v1_draft_review.xlsx`
- materialized draft: `data/evaluation_v1_draft.jsonl`
- draft 요약: `artifacts/eda/evaluation_v1_draft_summary.json`

### 누수 방지 절차

첫 단계에서는 한국어 source와 독립적으로 고정한 한국어 glossary lexicon만 읽어 후보 ID를 선정했다. English reference는 ID 고정 후 이중언어 정렬 검수용으로만 join했으며, 번역 모델이나 benchmark 출력은 후보 추출·용어 확장·초안 선택에 사용하지 않았다. 후보와 glossary hash를 manifest에 기록해 이후 파일 변경도 감지한다.

### 확인된 결과

- source-only 조건을 통과한 고유 행: `269`
- 검수 pool: `154`
- AI 보조 초안: `40`문장, 중복 ID `0`
- 서로 다른 glossary 용어: `8`
- 길이: `31~99`자, median `56.5`자
- multi-term 문장: `4`
- 용어 occurrence: 데이터베이스 `9`, 소프트웨어 `9`, 서버 `6`, 운영 체제 `5`, 클라우드 컴퓨팅 `5`, 오픈 소스 `5`, 배포 `4`, 캐시 `1`

세 분할 검수에서 다른 적용 검수가 `REJECT` 또는 `AMBIGUOUS`로 판정한 행은 40문장 초안에서 제외했다. 유일한 롤백 후보는 용어 의미와 한영 정렬은 맞았지만 고유명사 `Muchwhat`이 불명확하다는 의견 충돌 때문에 제외했다. 접근 제어 후보는 software가 아닌 물리 NFC 문맥이어서 제외했다.

### 상태와 다음 gate

이 결과는 `human_confirmed=false`, `benchmark_allowed=false`인 **AI 보조 초안**이다. materializer의 30~50문장, 5개 이상 용어, 중복·hash 검증만 통과했으며, 사람의 번역 품질 판단을 통과했다는 뜻은 아니다. 사람이 XLSX의 40개 source/reference 쌍과 용어 의미를 확인한 후에만 별도 confirmed manifest와 `evaluation_v1.jsonl`을 만들고 4-way 최종 benchmark를 실행한다. 따라서 이 draft에는 최종 성능 수치를 산출하지 않았다.

## EXP-013. 외부 번역 의견의 독립 검토와 reference provenance

- 날짜: 2026-08-20
- 증거 수준: 외부 의견을 독립 검토한 AI 보조 초안
- 결정 기록: `docs/REFERENCE_REVIEW_DECISIONS.md`
- reference overlay: `data/reference_reviews/evaluation_v1_draft.json`
- 최신 검수표: `artifacts/reviews/evaluation_v1_draft_review.xlsx`

외부 의견을 지시로 실행하지 않고 의미 충실도·문법·평가 케이스 명확성을 다시 판단했다. Case 8·9·11·16·30은 교정을 채택했고, 20·21·24는 문법 오류가 아닌 스타일 선호라서 유지했다. Case 5는 번역 문구만 고치지 않고 원문의 기술적 사실성 위험을 피할 Apache 문장으로, Case 35는 의미 대상이 모호한 원문을 명확한 클라우드 컴퓨팅 문장으로 교체했다. 교체 후 Case 35의 reference 교정까지 합쳐 effective reference 변경은 6건이다.

공개 dataset reference는 후보 pool에서 수정하지 않았다. overlay는 source ID·원 reference hash·교정문·근거를 보존하며, 명시적 오프라인 materialize에서만 effective reference로 적용된다. 최신 초안은 40문장·8용어, 교정 6건·미검수 34건이고 `human_confirmed=false`, `benchmark_allowed=false`다. 두 reference 모두 runtime에 노출되지 않으며 이 단계에서는 benchmark를 실행하지 않았다.

## EXP-014. 대화 승인 근거로 `evaluation_v1` 동결

- 날짜: 2026-08-20
- 증거: 명시적 대화 승인, workbook hash, finalizer·materializer 검증
- 최종 파일: `data/evaluation_selection_v1.json`, `data/reference_reviews/evaluation_v1.json`, `data/evaluation_v1.jsonl`

초안 XLSX의 행별 결정 셀은 저장된 입력 없이 비어 있었다. 사용자가 이후 대화에서 `검수끝냈어`라고 명시적으로 확인했으므로, 이를 현재 40개 선정과 effective reference의 일괄 승인으로 기록했다. 사용자가 XLSX 행별 셀을 채웠다고 주장하지 않으며, 승인 문구·reviewer·UTC 시각·workbook hash를 확정 이력에 보존했다.

결과는 `HUMAN_CONFIRMED_FROZEN`, 40문장·8용어, 원 reference 유지 34건·교정 6건, `human_confirmed=true`, `benchmark_allowed=true`다. 이는 MVP 평가셋을 고정하기 위한 절차적 gate이며 reference의 완전무결함을 주장하지 않는다.

- final manifest SHA-256: `19cb3f8c52ab9296f636ae667e26a8c42b4ca3b18a855970b5bc824723256b0a`
- final reference overlay SHA-256: `9e5eefc70976104e4e77ee79a1401ea87a511d1f0d25b83ec9a2e6acbeace2ba`
- materialized dataset SHA-256: `cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650`
- confirmation workbook SHA-256: `ffb526f7fb278582a456fe24fa21e21b50207cc2317d69aa25a1c3d04580471a`

## EXP-015. `evaluation_v1` 실제 4-way 기능 기준선

- 날짜: 2026-08-20
- run ID: `429d6c4a-a1b6-4514-bfd3-dab6966c4101`
- artifact: `artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json`
- 조건: Baseline, RAG-only, Agent-only, Agent+RAG; 각 40문장

| 조건 | 용어 정확도 | mean latency | 문장 변경률 | retry 문장 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `90.909%` | `442.65ms` | - | - |
| RAG | `100%` | `458.20ms` | - | - |
| Agent | `93.182%` | `11,833.72ms` | `22.5%` | `9/40` |
| Agent+RAG | `100%` | `11,187.07ms` | `15.0%` | `6/40` |

총 `160` 결과가 생성됐고 빈 번역은 없었다. RAG의 100%는 용어 occurrence 지표일 뿐 문장 전체 품질을 의미하지 않는다. Agent+RAG 1건의 비표준 `pronoun` label은 safe fallback으로 빈 번역 없이 보존됐고, 이후 알 수 없는 label을 `OTHER`로 정규화하도록 방어를 보강했다. case 24 실제 재실행(request ID `aa888ddb-e070-46fb-b08b-d349347b30e3`)에서 `pronoun`→`other` 흔적, 1회 revision, `stop_reason=passed`를 확인했다.

FastAPI smoke의 `GET /health`, `POST /translate/baseline`, `POST /translate/agent-rag`, 최종셋 baseline 1건 `POST /benchmark`는 모두 HTTP 200을 반환했다. 전체 테스트는 `136 passed`이고 외부 deprecation warning 1건이 남았다. 이 보강은 품질 튜닝이 아니라 기준선 후 실행 안정성 검증이며 prompt·glossary·규칙은 재조정하지 않았다.

수동 Agent 출력 label이 없으므로 Agent 판단 정확도·confusion counts·성공적 수정률은 `0`이 아니라 **미산출**이다. 변경률은 개선 여부를 말하지 않는다. 다음 반복은 Agent/Agent+RAG 출력에 수동 판정과 `IMPROVED/SAME/WORSE`를 붙여 이 수치를 계산하는 작업이다.
