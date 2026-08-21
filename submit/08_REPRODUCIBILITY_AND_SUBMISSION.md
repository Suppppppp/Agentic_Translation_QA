# 재현 방법과 제출 체크리스트

## 1. 환경

- Python 3.11 이상
- 검증 환경: macOS, Apple M1 Pro, RAM 16GB
- 유료 API: 사용하지 않음
- NMT/embedding: Hugging Face 공개 모델
- Agent: 로컬 Ollama

모델 cache와 버전 차이로 결과가 달라질 수 있다. 제출 결과의 dataset, config,
artifact hash를 아래에 고정한다.

## 2. 설치

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

실제 모델 실행이 필요할 때만 ML extra를 설치한다.

```bash
.venv/bin/python -m pip install -e '.[ml]'
ollama serve
ollama pull qwen3:1.7b
```

최초 Hugging Face 모델 load는 네트워크와 저장 공간이 필요하다. 이후 cache가 있으면
`HF_HUB_OFFLINE=1`로 오프라인 테스트할 수 있다.

## 3. 테스트

문서 작성 시 필수 API와 benchmark 계약 테스트는 다음 명령으로 재검증했다.

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_api_benchmark.py
```

결과: **18 passed**, 기존 Starlette/httpx deprecation warning 1건.

전체 오프라인 suite를 실행하려면 다음을 사용한다.

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider
```

## 4. API 실행과 스모크 테스트

```bash
.venv/bin/uvicorn translation_qa.main:app --reload
```

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/translate/baseline \
  -H 'Content-Type: application/json' \
  -d '{"text":"도커는 컨테이너 플랫폼입니다."}'
curl -sS -X POST http://127.0.0.1:8000/translate/agent-rag \
  -H 'Content-Type: application/json' \
  -d '{"text":"도커는 컨테이너 플랫폼입니다."}'
```

실제 번역 endpoint는 Marian model cache가 필요하고 Agent+RAG는 Ollama server와
`qwen3:1.7b`가 필요하다.

## 5. Benchmark 재현

API 호출:

```bash
curl -sS -X POST http://127.0.0.1:8000/benchmark \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"evaluation_v1","modes":["baseline","rag","agent","agent_rag"],"limit":40,"warmup":true}'
```

동결 artifact는 이미 저장돼 있으며 제출 문서 수치는 이를 재실행하지 않고 읽었다.

| 항목 | 고정 값 |
|---|---|
| Run ID | `429d6c4a-a1b6-4514-bfd3-dab6966c4101` |
| Dataset SHA-256 | `cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650` |
| Config SHA-256 | `2a2935ab0c366522aeae3cbbbb7c7e35c1fe96c654b3131bdcca164daaee16b1` |
| Artifact SHA-256 | `e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf` |

Frozen run 뒤의 코드 개선은 이 artifact에 반영되지 않는다. 같은 파일의 수치를
현재 guard 성능으로 재해석하지 않는다.

## 6. 평가 데이터와 수동 검수 provenance

- final dataset: [`../data/evaluation_v1.jsonl`](../data/evaluation_v1.jsonl)
- selection manifest: [`../data/evaluation_selection_v1.json`](../data/evaluation_selection_v1.json)
- reference review: [`../data/reference_reviews/evaluation_v1.json`](../data/reference_reviews/evaluation_v1.json)
- manual review batch 1: [`../data/manual_reviews/evaluation_v1_batch1_severity/`](../data/manual_reviews/evaluation_v1_batch1_severity/)
- manual review batch 2: [`../data/manual_reviews/evaluation_v1_batch2/`](../data/manual_reviews/evaluation_v1_batch2/)

Runtime에는 `source_text`만 전달한다. Reference와 수동 label은 모델 판정이 끝난
뒤 offline scoring에서만 결합한다. Component failure는 판단 정확도 분모에서
제외하지만 검수 결과와 correction eligibility에서 숨기지 않는다.

## 7. 제출 문서 구성

| 파일 | 목적 |
|---|---|
| `README.md` | 제출 문서 인덱스 |
| `01_SYSTEM_ARCHITECTURE_AND_WORKFLOW.md` | 시스템 구조와 상태 전이 |
| `02_MODEL_AND_FRAMEWORK_SELECTION.md` | 선택 이유, 대안과 한계 |
| `03_DATASET_AND_EDA.md` | 데이터 선정, provenance, EDA |
| `04_API_SPECIFICATION.md` | 필수 API 3개 명세 |
| `05_BENCHMARK_RESULTS.md` | 4-way 자동 지표와 수동 평가 |
| `06_ERROR_AND_CASE_ANALYSIS.md` | 오류 빈도와 성공·실패 심화 분석 |
| `07_LIMITATIONS_AND_IMPROVEMENTS.md` | 한계와 최소 실험 계획 |
| `08_REPRODUCIBILITY_AND_SUBMISSION.md` | 설치, 재현, 제출 점검 |
| `FINAL_REPORT.md` / `FINAL_REPORT.pdf` | 통합 최종 보고서 |
| `assets/` | 보고서 시각화 |

## 8. 요구사항 체크리스트

- [x] Python 3.11+와 FastAPI
- [x] 외부 유료 API 미사용, 공개 로컬 모델
- [x] vector search를 포함한 RAG
- [x] Agent 분석·검색·번역·QA·조건부 retry 최대 2회
- [x] 필수 endpoint 3개
- [x] 공개 데이터 기반 40문장 단일 도메인 평가셋
- [x] 서로 다른 핵심 용어 8개와 영어 reference
- [x] EDA, 구조·길이·용어 분포와 시각화
- [x] Baseline/RAG/Agent/Agent+RAG 4-way 비교
- [x] 용어 정확도, 문장 수정률, retry, latency
- [x] 수동 검수 대비 Agent 판단 정확도
- [x] 오류 유형 빈도와 성공·실패 각 3건 이상 분석
- [x] 모델·프레임워크 선정 이유와 한계
- [x] 현재 한계와 개선 우선순위
- [x] trace, hash, 실행·테스트 명령

## 9. 제출 전 마지막 확인

1. `.venv`, model cache, weight, local absolute cache path가 제출물에 포함되지 않았는지
   확인한다.
2. `evaluation_v1.jsonl`, glossary, benchmark artifact와 manual score가 누락되지
   않았는지 확인한다.
3. 문서의 hash와 실제 파일 hash가 같은지 확인한다.
4. fresh venv에서 README 명령과 API 테스트를 실행한다.
5. EDA image와 `FINAL_REPORT.pdf`를 전 페이지 렌더링해 잘림·깨짐을 확인한다.
6. 데이터·모델 license와 attribution을 README 또는 최종 보고서에 유지한다.
7. 개선 후 benchmark를 재실행한다면 기존 artifact를 덮어쓰지 않고 새 run ID로
   저장한다.

