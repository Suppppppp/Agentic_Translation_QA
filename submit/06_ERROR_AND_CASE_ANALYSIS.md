# 오류 유형과 대표 사례 분석

## 1. 분석 범위

이 분석은 Sup이 confirmed한 10개 source, 20개 mode-case 행을 대상으로 한다.
Batch 1은 retry·불일치·component failure, Batch 2는 초기 PASS를 중심으로 골랐다.
따라서 오류의 메커니즘을 보는 진단 자료이며 전체 benchmark의 오류율 추정이 아니다.

## 2. 오류 빈도

사람이 수정 필요로 판정한 14행의 primary error는 다음과 같다.

| Primary error | 행 수 | 비고 |
|---|---:|---|
| `omission_addition` | 10 | 주어·고유명사·핵심 정보 누락 또는 정보 추가 |
| `term` | 4 | 도메인 용어와 핵심 기술 표현 오류 |

한 행에 여러 tag를 허용한 전체 오류 태그는 다음과 같다.

| Error tag | 빈도 |
|---|---:|
| `omission_addition` | 12 |
| `term` | 8 |
| `meaning` | 4 |
| `fluency_grammar` | 4 |

태그는 중복되며 같은 source의 Agent와 Agent+RAG 결과도 별도 행이다. FN 7건은
모두 MAJOR이고, 그중 6건의 primary error가 `omission_addition`이었다.

## 3. 반복되는 근본 원인

### 3.1 Standalone 주어·고유명사 누락

문두의 명시적 기술 주어가 문맥 없는 `It`으로 바뀌는 패턴이 가장 반복적이었다.
`Docker`, `Load balancing`, `Website development`, `Consul`이 사라져도 영어
문장은 표면적으로 유창해 Judge가 PASS했다.

### 3.2 진단과 최종 decision의 모순

`009::agent_rag`, `019::agent_rag`, `027::agent`는 summary가 각각 용어 오류,
원문에 없는 추가, 모호한 술어를 언급했지만 구조화 결과는 PASS였다. 자유 서술과
`passed/error_types/next_action`에 중복 권위를 주면 서로 모순될 수 있음을 보여준다.

### 3.3 Glossary 정확도와 문장 정확도의 혼동

RAG는 `distribution`을 `deployment`로 고쳤지만, `Docker`나 `Website development`
주어 누락은 해결하지 못했다. 검색 hit 충족은 국소 용어 증거이지 문장 전체 의미
보존 증거가 아니다.

### 3.4 Reviser regression

초기 후보의 문제를 인지해도 Reviser가 `Consul`을 `Consult`로 바꾸는 새 entity
오류를 만들었다. 최종 Judge가 이를 승인하면 retry가 오히려 품질을 낮춘다.

### 3.5 구조화 출력 오류

한 Agent+RAG 사례에서 계약에 없는 `pronoun` error type 때문에 judgment parsing이
실패했다. 후보 번역은 존재했지만 judgment 기반 metric에 사용할 수 없었다.

## 4. 개선 성공 사례

### 성공 1 — `evaluation-v1-009::agent`

- Source: `로드 밸런싱은 여러 서버에 걸쳐 트래픽을 분산시켜 ... 기술입니다.`
- Initial: `Road balance ... single-point disorders.`
- Final: `Load balancing ... single-point failures.`
- 사람 판정: `IMPROVED`

Agent가 재시도를 요청해 핵심 기술어와 장애 표현을 교정했다. 다만 이 사례의
Agent+RAG 쌍은 같은 오류를 PASS했으므로, 성공 원인을 RAG로 돌릴 수 없다.

### 성공 2 — `evaluation-v1-019::agent`

- Source: `웹사이트 개발은 복잡한 과정으로 ... 배포 등 여러 단계가 포함됩니다.`
- Initial: `It's a complex process ... distribution, all sorts of things.`
- Final: `Website development is a complex process ... deployment.`
- 사람 판정: `IMPROVED`

주어 누락, `distribution`, 구어적 추가 표현과 run-on 구조를 한 번의 revision에서
고쳤다. 같은 source의 Agent+RAG는 `deployment`만 맞고 나머지 오류를 PASS했다.

### 성공 3 — `evaluation-v1-004::agent`

- Source: `쿠버네티스는 애플리케이션의 배포, 스케일링, 관리를 자동화...`
- Initial: `It can be used to automaticize the distribution, sketching...`
- Final: `Kubernetes can be used to automate the distribution, scaling...`
- 사람 판정: `IMPROVED`

Kubernetes 주어, `automaticize`, `sketching`을 개선했다. 다만 `배포`가 여전히
`distribution`이어서 완전한 정답은 아니다. Successful correction이 곧 완벽한
번역을 의미하지 않는 대표적인 부분 성공이다.

### 보조 성공 — `evaluation-v1-024::agent`

Initial의 잘못된 `you`를 `the field`로 고쳐 `IMPROVED`였지만 핵심 `null`은 최종에도
누락됐다. 이 역시 binary success보다 잔여 오류 추적이 필요함을 보여준다.

## 5. 실패 사례

### 실패 1 — `evaluation-v1-009::agent_rag`: MAJOR false PASS

- Initial/Final: `Road balance ... single-point disorders.`
- Agent summary: `single-point disorders` 오류를 언급
- 실제 action: PASS, retry 0
- 사람 판정: `SAME`, 수정 필요 MAJOR

Judge가 오류를 말하면서도 `error_types=[]`로 끝낸 판단 일관성 실패다. 검색 hit는
`server`였고 `로드 밸런싱` 표준 alias 근거는 없었다.

### 실패 2 — `evaluation-v1-027::agent_rag`: retry regression

- Initial: `It's an open source tool ... organize services...`
- Final: `Consult is an open source tool ... organize services...`
- 사람 판정: `WORSE`

초기에는 `Consul`이 누락됐고 configure도 `organize`로 번역됐다. Agent는 주어
누락을 잡았지만 Reviser가 `Consult`라는 잘못된 entity를 만들었고 final Judge가
승인했다.

### 실패 3 — `evaluation-v1-024::agent_rag`: component failure

- Initial/Final: `The database shows that you have no value in the field.`
- Stop reason: `component_failure`
- 사람 판정: `SAME`, 핵심 `null` 누락

Agent가 비계약 error type `pronoun`을 반환해 구조화 schema 검증이 실패했다. 이
행은 confusion matrix에서 제외했지만 failure 자체를 결과에서 숨기지 않았다.

### 실패 4 — `evaluation-v1-012::{agent,agent_rag}`

두 모드 모두 `Docker`를 `It`으로 바꾸고 PASS했다. RAG는 `distribution`을
`deployment`로 고쳤지만 standalone 주어 누락은 남았다. 국소 용어 개선과 전체
문장 수용 가능성을 분리해야 하는 사례다.

### 실패 5 — `evaluation-v1-015::{agent,agent_rag}`

`Load balancing`이 `It`으로 사라졌지만 두 모드 모두 정확하고 명확하다고 판단했다.
source의 명시적 주어 coverage를 별도 blocking check로 둘 필요가 있다.

## 6. Agent와 Agent+RAG 비교

| 진단 표본 지표 | Agent | Agent+RAG |
|---|---:|---:|
| 판단 정확도 | 70.0% | 55.6% |
| 수정 필요 recall | 57.1% | 33.3% |
| Successful correction | 57.1% | 14.3% |
| Component failure | 0 | 1 |

이 표본에서는 Agent가 더 좋았지만 다음 이유로 RAG의 인과적 악화를 주장하지 않는다.

- 표본이 작고 무작위가 아니다.
- 각 모드의 initial candidate가 다를 수 있다.
- Agent+RAG에만 component failure가 1건 있다.
- RAG는 자동 용어 정확도에서는 100%를 달성했다.

확인된 결론은 “검색 근거가 있어도 Judge가 문장 전체 alignment를 별도로 확인해야
한다”는 것이다.

## 7. 개선 우선순위

1. source의 standalone 주어, 고유명사, 핵심 기술어 coverage를 각 attempt마다
   확인하고 누락 시 PASS를 막는다.
2. 구조화 error list에서 최종 PASS/REVISE를 코드가 파생해 summary와 decision의
   모순을 감사한다.
3. revision 전후 entity set을 비교하고 새 entity hallucination이 있으면 이전
   후보로 rollback한다.
4. malformed Agent JSON은 제한된 repair/fallback으로 격리한다.

이 개선은 먼저 저장된 FN 7행과 TN 6행의 detector/Judge-only replay로 검증하고,
정상 6건에 false positive를 만들지 않을 때만 전체 benchmark로 확장해야 한다.

## 8. 근거 파일

- Batch 1 reviewed labels: [`../data/manual_reviews/evaluation_v1_batch1_severity/review_labels_with_severity.csv`](../data/manual_reviews/evaluation_v1_batch1_severity/review_labels_with_severity.csv)
- Batch 2 reviewed labels: [`../data/manual_reviews/evaluation_v1_batch2/review_labels_reviewed.csv`](../data/manual_reviews/evaluation_v1_batch2/review_labels_reviewed.csv)
- 초기 실패 분석: [`../docs/BATCH1_FN_COMPONENT_FAILURE_ANALYSIS.md`](../docs/BATCH1_FN_COMPONENT_FAILURE_ANALYSIS.md)

