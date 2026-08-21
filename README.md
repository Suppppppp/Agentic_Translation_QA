# Agentic Translation QA

한국어 문장을 영어로 번역하고, 도메인 용어 검색(RAG)과 품질 판정 Agent가 번역을 검토하는 과제용 프로젝트다. 한 번에 완성형 시스템을 만들기보다 Baseline, RAG, Agent를 작은 실험으로 검증한 뒤 효과가 확인된 구성을 통합한다.

최종 결과 보고서는 [`submit/FINAL_REPORT.pdf`](submit/FINAL_REPORT.pdf), 항목별 상세 문서는 [`submit/`](submit/) 폴더에 있다.

## 현재 상태

정상 동작하는 MVP 워크플로우와 첫 공개 데이터 기준선은 완성했다. 현재 결과는 품질 튜닝을 끝낸 최종 성능이 아니라, 이후 개선 실험의 출발점이다.

| 영역 | 현재 상태 |
| --- | --- |
| 평가 계약·API 스키마 | 구현 및 테스트됨 |
| Baseline | Marian 한영 번역 경로가 지연 로딩으로 구현됨 |
| RAG | exact·vector·safe hybrid 검색과 deterministic glossary post-edit 파일럿을 구현함 |
| Agent | Ollama 판정·수정, 최대 2회 retry, 조기 종료·rollback을 구현함 |
| FastAPI·benchmark runner | 필수 API 3개와 4-way 실행 경로를 구현함 |
| 공개 데이터 평가셋 | 40문장·8용어 `evaluation_v1` 동결, 원 reference 34건 유지·6건 교정, `benchmark_allowed=true` |
| 공개 데이터 기준선 | 실제 모델 4-way 160결과 생성 완료; 빈 번역 없음 |
| 남은 평가 | Agent 출력 수동 라벨, Agent 판단 정확도·성공적 수정률, 개선 실험과 최종 보고서 |

현재 공개 데이터 수치도 첫 기능 기준선일 뿐 최종 성능으로 해석하지 않는다. 기준선 실행 뒤 prompt·glossary·규칙을 결과에 맞춰 재조정하지 않았으며, 실험 사실과 폐기한 방식은 [실험 로그](docs/EXPERIMENT_LOG.md), 지표와 누수 방지 규칙은 [평가 계약](docs/EVALUATION_CONTRACT.md)을 기준으로 한다.

## 실행 환경

- Python 3.11 이상
- macOS, Linux 또는 Windows
- 모델을 실행할 수 있는 CPU 환경. Apple Silicon에서는 추후 PyTorch MPS를 별도로 비교할 수 있다.

PyTorch와 번역·임베딩·Agent 모델을 한꺼번에 설치하면 디스크가 빠르게 부족해질 수 있다. ML extra와 모델을 받기 전에 15~20 GiB 이상의 여유 공간을 권장하며, 기본 API/테스트 의존성과 ML 의존성을 분리한다.

macOS의 Homebrew `python3` 링크는 업그레이드 시 Python 3.14 등으로 바뀔 수 있다. 이 프로젝트는 재현성을 위해 Python 3.12를 권장하며, 현재 머신의 실행 파일은 `/opt/homebrew/opt/python@3.12/bin/python3.12`다.

## 기본 개발 환경

먼저 API와 모델 비의존 테스트에 필요한 core/dev 의존성만 설치한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`python3.12`가 PATH에 없다면 첫 명령만 다음처럼 실행한다.

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
```

Windows PowerShell에서는 활성화 명령으로 `.venv\Scripts\Activate.ps1`을 사용한다.

전체 단위·API·스크립트 테스트는 모델을 새로 받지 않고 실행할 수 있다. `python -m pytest`를 사용해 프로젝트 root와 `src` 둘 다 import path에 포함한다.

```bash
HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider
```

## ML 의존성

Baseline 번역 단계에서는 필요한 패키지만 먼저 설치한다.

```bash
python -m pip install -e ".[baseline]"
```

RAG 임베딩과 EDA 도구는 해당 단계에서 따로 추가한다.

```bash
python -m pip install -e ".[baseline,rag]"
python -m pip install -e ".[eda]"
```

모든 ML 의존성이 한꺼번에 필요한 경우에만 통합 extra를 사용한다.

```bash
python -m pip install -e ".[ml]"
```

새 환경에 개발 도구와 ML 패키지를 동시에 설치해야 할 때는 다음 명령을 사용할 수 있지만, 디스크 여유를 먼저 확인한다.

```bash
python -m pip install -e ".[dev,ml]"
```

`ml` extra에는 PyTorch, constrained beam search가 내장된 Transformers 4.41~4.56, SentencePiece, Sentence Transformers, Datasets, Pandas, Matplotlib이 포함된다.

## Ollama Agent

기본 Agent 모델은 메모리 사용량을 줄인 `qwen3:1.7b`다. macOS에서는 다음 순서로 준비한다.

```bash
brew install ollama
ollama serve
```

`ollama serve`를 실행한 터미널은 그대로 두고, 다른 터미널에서 모델을 한 번 내려받는다.

```bash
ollama pull qwen3:1.7b
```

Ollama 없이 API 구조만 확인할 때는 `TRANSLATION_QA_AGENT_BACKEND=rule`을 사용할 수 있다.

Qwen3의 reasoning을 모든 호출에서 끄면 12문장 파일럿의 평균 지연은 약 2.6초로 줄었지만, 명백한 문법 오류도 전부 통과시켰다. 현재 구성은 단순 source domain/키워드 분석만 `think=false`로 실행하고, 품질 판정과 revision은 reasoning을 유지한다. 해당 결정과 수치는 `docs/EXPERIMENT_LOG.md`에 보존했다.

## 모델 다운로드 정책

패키지 설치만으로 모델 가중치를 내려받지 않는다. 애플리케이션 import, 서버 시작, 기본 테스트 수집 단계에서도 모델을 생성하거나 다운로드하지 않고, 실제로 해당 모델이 필요한 첫 작업에서 지연 로딩한다. 따라서 최초 ML 요청은 모델 다운로드와 초기화 때문에 오래 걸릴 수 있으며 인터넷 연결과 추가 디스크 공간이 필요하다.

여러 대형 모델을 동시에 메모리에 올리지 않는다. 초기 실험에서는 번역 모델과 임베딩 모델을 하나씩 고정하고, Agent 단계에서도 소형 양자화 모델 한 개만 검토한다.

필요한 Hugging Face 모델이 한 번 캐시에 저장된 뒤에는 다음 설정으로 우발적인 네트워크 접근을 막을 수 있다. 최초 다운로드 전에는 이 값을 설정하면 모델 로딩이 실패한다.

```bash
export HF_HUB_OFFLINE=1
```

## 현재 채택한 실행 구성

RAG runtime은 동일 Marian 번역기 위에 **glossary post-edit**를 적용하고, 검색은 exact 결과를 우선하는 **safe hybrid**를 사용한다. API 환경 변수에서는 이 검색기를 `hybrid`로 지정한다. 원시 가중합 `hybrid`는 비교 Spike에만 남겨 두었다.

```bash
export HF_HUB_OFFLINE=1
export TRANSLATION_QA_RETRIEVER=hybrid
export TRANSLATION_QA_AGENT_BACKEND=ollama
export TRANSLATION_QA_AGENT_MODEL=qwen3:1.7b
uvicorn translation_qa.main:app --reload
```

파일럿에서 `force_words_ids`는 퇴행 출력을 만들었고, 혼합 영문·괄호·quoted marker source augmentation도 용어를 안정적으로 보존하지 못했다. 따라서 두 방식은 현재 runtime에 채택하지 않고 재현용 비교 스크립트에만 유지한다.

## API

- `POST /translate/baseline`: 동일한 번역 모델을 사용하는 기준 경로
- `POST /translate/agent-rag`: 검색, 품질 판정, 조건부 재시도를 포함하는 경로
- `POST /benchmark`: 고정 평가셋에서 비교 조건을 실행하는 경로

`GET /health`는 모델을 로딩하지 않으며, 실제 모델은 첫 번역 요청에서 준비된다.

간단한 smoke 요청은 다음과 같다.

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/translate/baseline \
  -H 'Content-Type: application/json' \
  -d '{"text":"새 버전을 배포한다."}'
curl -sS -X POST http://127.0.0.1:8000/translate/agent-rag \
  -H 'Content-Type: application/json' \
  -d '{"text":"새 버전을 배포한다."}'
```

## 재현 가능한 비교 Spike

검색 비교는 exact, vector, 원시 가중합 hybrid, exact 우선 safe_hybrid를 같은 입력에서 함께 출력한다. 기본 결정적 임베더는 다운로드가 필요 없다.

```bash
python scripts/compare_retrieval.py --case-id pilot-001 --format json
python scripts/compare_retrieval.py --embedder sentence-transformer --case-id pilot-001 --format json
```

주입 비교는 Baseline, lexical constraints와 세 source augmentation을 비교한다. 기본은 fake backend이며 `--real-model`에서만 Marian을 지연 로딩한다.

```bash
python scripts/compare_injection.py --case-id pilot-001 --format json
python scripts/compare_injection.py --case-id pilot-001 --real-model --device cpu --format json
```

## 데이터 EDA와 선정 gate

두 공개 데이터셋의 이미 받은 Hugging Face snapshot을 오프라인으로 비교한다. JSON 요약과 길이·용어 분포 PNG는 `artifacts/eda/`에 생성된다.

```bash
HF_HUB_OFFLINE=1 python scripts/compare_datasets.py
```

현재 잠정 선택은 `lemon-mint/korean_parallel_sentences_v1.1`이다. 중복과 기초 이상치가 더 적고 software source 후보가 더 많지만, 데이터 카드의 출처·사실 정확성 한계를 그대로 유지한다.

모델 출력과 영어 reference를 보지 않고 한국어 source만으로 검수 pool을 만든다. 선정 ID를 고정한 다음에만 English를 join하며, `배포판`을 `배포`로 오인하지 않는 한국어 경계 matcher를 공유한다.

```bash
HF_HUB_OFFLINE=1 python scripts/build_evaluation_candidates.py \
  --glossary data/glossary_evaluation_v2.csv
```

이 출력은 `unreviewed` pool이며 최종 평가셋이 아니다. 선택 manifest가 사람에게 확정되지 않으면 materializer는 기본적으로 중단한다. AI 검수 초안을 만들 때만 명시적으로 `--allow-unconfirmed-draft`를 사용하고, 결과 metadata의 `benchmark_allowed`는 `false`로 남는다.

현재 `artifacts/eda/evaluation_candidates.jsonl`에 154개 후보가 있고, `artifacts/reviews/evaluation_candidates_review_{a,b,c}.json`에 영어 reference 정렬·software 의미·glossary target 적합성을 본 AI 보조 검수 초안이 있다. 이 파일은 모두 `human_confirmed=false`이며, gold label이나 최종 선정 결과로 사용할 수 없다. 사람이 KEEP/AMBIGUOUS 후보의 한영 정렬, software 의미, 용어 커버리지, 중복을 직접 확인해야 한다.

이 검수 초안에서 40문장·8용어를 균형 선정한 `data/evaluation_selection_v1_draft.json`과 `data/evaluation_v1_draft.jsonl`이 생성되었다. 외부 번역 의견은 독립 검토해 채택한 6개 reference 교정만 `data/reference_reviews/evaluation_v1_draft.json` overlay로 적용했고, 공개 reference는 provenance에 그대로 보존했다. 검증 summary는 `artifacts/eda/evaluation_v1_draft_summary.json`, 검수 초안은 `artifacts/reviews/evaluation_v1_draft_review.xlsx`다.

초안 XLSX에는 행별 결정 셀이 저장되어 있지 않았다. 이후 사용자가 이 대화에서 검수 완료를 명시적으로 확인했으므로, 이를 현재 40개 선정과 effective reference 전체의 일괄 승인으로 기록했다. 행별 입력이 있었다고 소급하지 않고 승인 근거와 workbook hash를 별도 확정 기록에 남겼다. 최종 `data/evaluation_selection_v1.json`, `data/reference_reviews/evaluation_v1.json`, `data/evaluation_v1.jsonl`은 원 reference 34건 유지·6건 교정 상태로 동결되었고 `benchmark_allowed=true`다.

### 선택 manifest와 materialize

선택 manifest는 후보·glossary 파일의 SHA-256과 30~50개의 고정된 `source_record_id`를 보존한다. 최소 형태는 다음과 같다.

```json
{
  "human_confirmed": false,
  "candidate_sha256": "<evaluation_candidates.jsonl SHA-256>",
  "glossary_sha256": "<glossary_evaluation_v2.csv SHA-256>",
  "reference_review_file": "reference_reviews/evaluation_v1_draft.json",
  "reference_review_sha256": "<reference overlay SHA-256>",
  "selected": [
    {
      "source_record_id": "<fixed public source ID>",
      "selection_note": "<alignment/domain/coverage reason>"
    }
  ]
}
```

AI 보조 선정 초안을 스키마·개수·용어 커버리지 검사용으로 재생성하려면 다음과 같이 실행한다.

```bash
python scripts/materialize_evaluation_dataset.py \
  --candidates artifacts/eda/evaluation_candidates.jsonl \
  --manifest data/evaluation_selection_v1_draft.json \
  --glossary data/glossary_evaluation_v2.csv \
  --reference-review data/reference_reviews/evaluation_v1_draft.json \
  --output data/evaluation_v1_draft.jsonl \
  --summary-output artifacts/eda/evaluation_v1_draft_summary.json \
  --allow-unconfirmed-draft
```

이 초안은 `benchmark_allowed=false`이므로 benchmark에 사용하지 않는다. 이번 확정은 boolean만 바꾼 것이 아니라 workbook·hash·reviewer·UTC 시각·대화 승인 근거를 검증해 별도 최종 manifest와 overlay를 생성하는 `scripts/finalize_evaluation_review.py`로 수행했다.

```bash
python scripts/finalize_evaluation_review.py \
  --candidates artifacts/eda/evaluation_candidates.jsonl \
  --draft-manifest data/evaluation_selection_v1_draft.json \
  --draft-reference-review data/reference_reviews/evaluation_v1_draft.json \
  --review-workbook outputs/<run-id>/evaluation_v1_review_confirmed.xlsx \
  --output-manifest data/evaluation_selection_v1.json \
  --output-reference-review data/reference_reviews/evaluation_v1.json \
  --reviewer 'User' --reviewed-at-utc '2026-08-20T11:47:56Z' \
  --confirmation-basis 'The user explicitly confirmed review completion'
```

```bash
python scripts/materialize_evaluation_dataset.py \
  --candidates artifacts/eda/evaluation_candidates.jsonl \
  --manifest data/evaluation_selection_v1.json \
  --glossary data/glossary_evaluation_v2.csv \
  --reference-review data/reference_reviews/evaluation_v1.json \
  --output data/evaluation_v1.jsonl \
  --summary-output artifacts/eda/evaluation_v1_summary.json
```

materializer는 manifest·candidate·glossary·reference overlay hash, 선정 순서, 원 reference hash, 중복 ID, 30~50문장, 서로 다른 glossary 용어 5개 이상을 검사한다. 최종 overlay에는 모든 행의 결정과 reviewer·검수 시각이 필요하다. 이 검사를 통과해도 번역 품질 판정 자체가 자동으로 확정되는 것은 아니다.

## 벤치마크와 수동 검수

`evaluation_v1.jsonl`은 동결되었고 summary의 `benchmark_allowed=true`를 확인했다. `--mode`를 생략하면 Baseline, RAG-only, Agent-only, Agent+RAG 네 조건을 같은 셋에서 실행한다.

```bash
HF_HUB_OFFLINE=1 TRANSLATION_QA_GLOSSARY_PATH=data/glossary_evaluation_v2.csv \
TRANSLATION_QA_RETRIEVER=hybrid TRANSLATION_QA_AGENT_BACKEND=ollama \
python scripts/run_benchmark.py --dataset-id evaluation_v1
```

파일럿을 빠르게 확인하려면 조건을 제한한다.

```bash
python scripts/run_benchmark.py --dataset-id pilot_v1 \
  --mode baseline --mode rag --no-warmup
```

runtime artifact에는 reference가 없다. 실행 후 수동 판정을 위해서만 별도 offline sheet를 만든다.

```bash
python scripts/export_manual_review.py \
  --dataset data/pilot_v1.jsonl \
  --artifact artifacts/benchmark-<run-id>.json \
  --output artifacts/reviews/pilot_review.csv
```

confirmed `PASS/NEEDS_REVISION`이 있을 때만 Agent 정확도·confusion matrix를, confirmed `IMPROVED/SAME/WORSE`까지 있을 때만 성공적 수정률을 계산한다. 라벨이 없으면 수치를 추정하지 않고 `unavailable_metrics`에 사유를 기록한다.

첫 실제 4-way 실행은 run ID `429d6c4a-a1b6-4514-bfd3-dab6966c4101`로 완료했다. 40문장 × 4조건의 160결과에 빈 번역은 없었다.

| 조건 | 용어 정확도 | 평균 지연 | 변경률 | retry |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `90.909%` | `442.65ms` | - | - |
| RAG | `100%` | `458.20ms` | - | - |
| Agent | `93.182%` | `11,833.72ms` | `22.5%` | `9/40` |
| Agent+RAG | `100%` | `11,187.07ms` | `15.0%` | `6/40` |

artifact은 `artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json`이다. Agent가 비표준 `pronoun` 오류 label을 반환한 1건은 safe fallback으로 빈 결과 없이 종료했고, 이후 알 수 없는 label을 `OTHER`로 정규화하도록 방어를 보강했다. 실제 재실행에서도 `pronoun`이 `other`로 정규화된 흔적과 1회 revision 후 `passed`를 확인했다. 이는 번역 품질 튜닝이 아니라 실행 안정성 수정이며, 실행 결과를 보고 prompt·glossary·규칙을 재조정하지 않았다. Agent 판단 정확도와 성공적 수정률은 출력 수동 라벨이 없어 아직 **미산출**이다. 최종 평가셋으로 `/health`, `/translate/baseline`, `/translate/agent-rag`, `/benchmark` FastAPI smoke도 모두 HTTP 200을 반환했다.

## 주요 환경 변수

| 변수 | 기본값·용도 |
| --- | --- |
| `TRANSLATION_QA_TRANSLATOR_MODEL` | `Helsinki-NLP/opus-mt-ko-en` |
| `TRANSLATION_QA_AGENT_MODEL` | `qwen3:1.7b` |
| `TRANSLATION_QA_DEVICE` | `auto`; `cpu`, `mps`, `cuda` 선택 가능 |
| `TRANSLATION_QA_GLOSSARY_PATH` | `data/glossary_pilot.csv` |
| `TRANSLATION_QA_RETRIEVER` | `exact`, `vector`, `hybrid`(safe hybrid) |
| `TRANSLATION_QA_AGENT_BACKEND` | `ollama` 또는 테스트용 `rule` |
| `TRANSLATION_QA_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `HF_HUB_OFFLINE` | 캐시 사용 시 `1`로 설정해 Hugging Face 네트워크 접근 차단 |

## 프로젝트 구조

```text
src/translation_qa/   애플리케이션 패키지
tests/                단위·API 계약 테스트
data/                 버전 관리할 평가 자료와 용어집
docs/                 실험 기준과 분석 문서
scripts/              EDA·후보 선정·materialize·benchmark·검색/주입 Spike
```

다운로드한 원본 데이터, 모델 가중치, 벡터 인덱스와 실행 로그는 Git에 포함하지 않는다.
