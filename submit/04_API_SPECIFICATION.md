# FastAPI 명세

## 1. 상태

과제의 필수 API 3개는 모두 구현돼 있다. 새 API 코드를 추가할 필요는 없다.

| Method | Path | 상태 | 역할 |
|---|---|---|---|
| POST | `/translate/baseline` | 구현·테스트 완료 | Marian 기반 기준 번역 |
| POST | `/translate/agent-rag` | 구현·테스트 완료 | RAG + Agent 조건부 검수 번역 |
| POST | `/benchmark` | 구현·테스트 완료 | 동결 데이터셋 평가 실행 |

보조 상태 확인용 `GET /health`도 제공한다. FastAPI 앱 version은 `0.1.0`이며
모델은 요청 시 lazy-load된다.

## 2. 실행

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/uvicorn translation_qa.main:app --reload
```

실제 모델 실행에는 ML extra와 모델 cache가 필요하다.

```bash
.venv/bin/python -m pip install -e '.[ml]'
ollama serve
ollama pull qwen3:1.7b
```

OpenAPI UI는 실행 후 `http://127.0.0.1:8000/docs`에서 확인할 수 있다.

## 3. 공통 번역 요청

```json
{
  "text": "쿠버네티스는 애플리케이션 배포를 자동화합니다."
}
```

- `text`: 앞뒤 공백을 제거한 1~5000자 문자열
- reference, domain gold, expected term, manual label은 받지 않는다.
- 빈 문자열이나 5000자 초과 입력은 FastAPI/Pydantic `422` 응답이다.

## 4. `POST /translate/baseline`

### 호출 예

```bash
curl -sS -X POST http://127.0.0.1:8000/translate/baseline \
  -H 'Content-Type: application/json' \
  -d '{"text":"쿠버네티스는 애플리케이션 배포를 자동화합니다."}'
```

Baseline은 RAG와 Agent 없이 동일 번역기를 한 번 호출한다. `retry_count`는 0이고
trace에는 번역 attempt와 latency, 모델 ID, 종료 사유가 남는다.

## 5. `POST /translate/agent-rag`

### 호출 예

```bash
curl -sS -X POST http://127.0.0.1:8000/translate/agent-rag \
  -H 'Content-Type: application/json' \
  -d '{"text":"쿠버네티스는 애플리케이션 배포를 자동화합니다."}'
```

### 응답 구조

아래는 필드 구조를 보여 주는 축약 예시다. 실제 번역과 score는 모델 실행에 따라
달라질 수 있다.

```json
{
  "request_id": "uuid",
  "mode": "agent_rag",
  "source_text": "쿠버네티스는 애플리케이션 배포를 자동화합니다.",
  "translation": "Kubernetes automates application deployment.",
  "retry_count": 1,
  "final_judgment": {
    "passed": true,
    "quality_score": 0.91,
    "error_types": [],
    "summary": "Required subject and terminology are preserved.",
    "confidence": 0.87,
    "next_action": "accept"
  },
  "trace": {
    "source_analysis": {},
    "coverage_requirements": [],
    "attempts": [],
    "final_attempt_index": 1,
    "stop_reason": "passed",
    "selection_reason": "latest_passed",
    "total_latency_ms": 0,
    "component_call_counts": {},
    "warnings": [],
    "model_versions": {}
  }
}
```

실제 `attempts`에는 retrieval query/hit, candidate, judgment, coverage finding,
적용 action과 단계별 timing이 포함된다. `retry_count`는 0~2이며 trace의 후보 수보다
항상 1 작다.

## 6. `POST /benchmark`

### 요청

```json
{
  "dataset_id": "evaluation_v1",
  "modes": ["baseline", "rag", "agent", "agent_rag"],
  "limit": 40,
  "warmup": true
}
```

| 필드 | 기본값 | 제약 |
|---|---|---|
| `dataset_id` | `evaluation_v1` | 영숫자, `_`, `.`, `-` |
| `modes` | `baseline`, `agent_rag` | 중복 없는 1개 이상 |
| `limit` | 전체 | 1~50 또는 생략 |
| `warmup` | `true` | warm latency 측정 여부 |

### 호출 예

```bash
curl -sS -X POST http://127.0.0.1:8000/benchmark \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"evaluation_v1","modes":["baseline","rag","agent","agent_rag"],"limit":40,"warmup":true}'
```

### 응답 핵심 필드

- `run_id`, `dataset_id`, `artifact_path`
- `metrics_by_mode`
  - sample count
  - terminology accuracy
  - sentence modification rate
  - successful correction rate와 분자·분모
  - mean/median/p95 latency
  - retry distribution
  - Agent confusion counts, accuracy, revision recall
  - 자동·수동 오류 유형 count
- `unavailable_metrics`: gold label이 없어 계산하지 못한 지표와 이유
- `metadata`: dataset/config hash, warmup, reference leakage guard 등

## 7. 오류 응답

| 상태 | 조건 |
|---:|---|
| 422 | 입력 스키마 위반 |
| 404 | benchmark dataset 없음 |
| 503 | 번역·Agent·benchmark 구성요소 사용 불가 |
| 500 | 그 밖의 Translation QA 도메인 오류 |

구성요소 오류가 번역 attempt 도중 발생하면 가능한 경우 이미 생성한 후보를
보존하고 trace warning과 stop reason으로 남긴다.

## 8. 검증 결과

다음 API·benchmark 테스트를 오프라인 모드로 실행했다.

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_api_benchmark.py
```

결과: **18 passed**, 기존 Starlette/httpx deprecation warning 1건. 테스트는 실제
모델을 다운로드하지 않는 fake service로 API 계약, 오류 응답, metric 계산과 gold
누수 방지를 검증한다.

## 9. 구현 근거

- Routes: [`../src/translation_qa/main.py`](../src/translation_qa/main.py)
- Schemas: [`../src/translation_qa/schemas.py`](../src/translation_qa/schemas.py)
- Benchmark: [`../src/translation_qa/benchmark.py`](../src/translation_qa/benchmark.py)
- API tests: [`../tests/test_api_benchmark.py`](../tests/test_api_benchmark.py)

