from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from translation_qa.agent import RuleBasedTermAgent
from translation_qa.benchmark import BenchmarkRunner, JsonlDatasetRepository
from translation_qa.config import Settings
from translation_qa.errors import ComponentUnavailableError, DatasetNotFoundError
from translation_qa.main import create_app
from translation_qa.pipeline import TranslationPipeline
from translation_qa.retrieval import ExactGlossaryRetriever, GlossaryEntry
from translation_qa.schemas import (
    AttemptTrace,
    BenchmarkRequest,
    CandidateOrigin,
    ExecutionMode,
    NextAction,
    QualityJudgment,
    StageTimings,
    StopReason,
    TermConstraint,
    TranslationCandidate,
    TranslationRequest,
    TranslationResponse,
    TranslationTrace,
)


class ConstraintAwareTranslator:
    model_id = "fake-translator"

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        text = "Start the deployment." if constraints else "Release the new version."
        return TranslationCandidate(text=text, model_id=self.model_id)


def pipeline_for_api() -> TranslationPipeline:
    retriever = ExactGlossaryRetriever(
        [
            GlossaryEntry(
                term_id="t1",
                source_term="배포",
                target_term="deployment",
                domain="software",
            )
        ]
    )
    return TranslationPipeline(
        Settings(),
        ConstraintAwareTranslator(),
        retriever=retriever,
        agent=RuleBasedTermAgent(),
    )


class RaisingService:
    def translate(
        self,
        request: TranslationRequest,
        mode: ExecutionMode,
    ) -> TranslationResponse:
        raise ComponentUnavailableError("translation model is not installed")


class StaticBenchmarkService:
    def __init__(self, response: object) -> None:
        self.response = response

    def run(self, request: BenchmarkRequest) -> object:
        return self.response


def test_translation_api_contract_and_validation() -> None:
    app = create_app(translation_service=pipeline_for_api())
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok", "models": "lazy"}

    baseline = client.post(
        "/translate/baseline",
        json={"text": "새 버전을 배포한다."},
    )
    assert baseline.status_code == 200
    assert baseline.json()["mode"] == "baseline"
    assert baseline.json()["retry_count"] == 0

    proposed = client.post(
        "/translate/agent-rag",
        json={"text": "새 버전을 배포한다."},
    )
    assert proposed.status_code == 200
    body = proposed.json()
    assert body["mode"] == "agent_rag"
    assert body["translation"] == "Start the deployment."
    assert body["final_judgment"]["passed"] is True
    assert body["trace"]["attempts"][0]["retrieval_hits"][0]["term_id"] == "t1"

    invalid = client.post("/translate/baseline", json={"text": "   "})
    assert invalid.status_code == 422


def test_translation_api_maps_missing_model_to_503() -> None:
    client = TestClient(create_app(translation_service=RaisingService()))

    response = client.post("/translate/baseline", json={"text": "원문"})

    assert response.status_code == 503
    assert "not installed" in response.json()["detail"]


def _response(
    request: TranslationRequest,
    mode: ExecutionMode,
) -> TranslationResponse:
    initial = TranslationCandidate(text="Use temporary storage.", model_id="fake")
    attempts = [
        AttemptTrace(
            attempt_index=0,
            candidate=initial,
            timings=StageTimings(total_ms=2.0),
        )
    ]
    stop_reason = StopReason.BASELINE_COMPLETE
    final_index = 0
    final_judgment = None
    if mode in {ExecutionMode.AGENT, ExecutionMode.AGENT_RAG}:
        improved = TranslationCandidate(text="Use the cache.", model_id="fake")
        final_judgment = QualityJudgment(
            passed=True,
            quality_score=0.9,
            error_types=[],
            summary="The required term is present.",
            confidence=0.9,
            next_action=NextAction.ACCEPT,
        )
        attempts.append(
            AttemptTrace(
                attempt_index=1,
                candidate_origin=CandidateOrigin.AGENT_REVISION,
                parent_attempt_index=0,
                requested_action=NextAction.REVISE,
                applied_action=NextAction.REVISE,
                candidate=improved,
                judgment=final_judgment,
                timings=StageTimings(total_ms=3.0),
            )
        )
        stop_reason = StopReason.PASSED
        final_index = 1

    trace = TranslationTrace(
        attempts=attempts,
        final_attempt_index=final_index,
        stop_reason=stop_reason,
        total_latency_ms=5.0,
    )
    return TranslationResponse(
        mode=mode,
        source_text=request.text,
        translation=attempts[final_index].candidate.text,
        retry_count=len(attempts) - 1,
        final_judgment=final_judgment,
        trace=trace,
    )


class RecordingTranslationService:
    def __init__(self) -> None:
        self.requests: list[TranslationRequest] = []

    def translate(
        self,
        request: TranslationRequest,
        mode: ExecutionMode,
    ) -> TranslationResponse:
        self.requests.append(request)
        return _response(request, mode)


class GoldJudgmentService:
    """Return inspectable first judgments; a requested revision later passes."""

    def __init__(self, predicted_revision_by_source: dict[str, bool]) -> None:
        self.predicted_revision_by_source = predicted_revision_by_source
        self.requests: list[TranslationRequest] = []

    def translate(
        self,
        request: TranslationRequest,
        mode: ExecutionMode,
    ) -> TranslationResponse:
        assert mode is ExecutionMode.AGENT_RAG
        self.requests.append(request)
        predicts_revision = self.predicted_revision_by_source[request.text]
        first_judgment = QualityJudgment(
            passed=not predicts_revision,
            quality_score=0.4 if predicts_revision else 0.9,
            error_types=[] if not predicts_revision else ["term"],
            summary=(
                "The candidate needs revision."
                if predicts_revision
                else "The candidate is acceptable."
            ),
            confidence=0.9,
            next_action=NextAction.REVISE if predicts_revision else NextAction.ACCEPT,
        )
        initial = TranslationCandidate(text="Initial candidate.", model_id="fake")
        attempts = [
            AttemptTrace(
                attempt_index=0,
                candidate=initial,
                judgment=first_judgment,
                timings=StageTimings(total_ms=2.0),
            )
        ]
        final_index = 0
        if predicts_revision:
            final_judgment = QualityJudgment(
                passed=True,
                quality_score=0.9,
                error_types=[],
                summary="The revision is acceptable.",
                confidence=0.9,
                next_action=NextAction.ACCEPT,
            )
            attempts.append(
                AttemptTrace(
                    attempt_index=1,
                    candidate_origin=CandidateOrigin.AGENT_REVISION,
                    parent_attempt_index=0,
                    requested_action=NextAction.REVISE,
                    applied_action=NextAction.REVISE,
                    candidate=TranslationCandidate(
                        text="Revised candidate.",
                        model_id="fake",
                    ),
                    judgment=final_judgment,
                    timings=StageTimings(total_ms=3.0),
                )
            )
            final_index = 1

        final_judgment = attempts[final_index].judgment
        assert final_judgment is not None
        trace = TranslationTrace(
            attempts=attempts,
            final_attempt_index=final_index,
            stop_reason=StopReason.PASSED,
            total_latency_ms=5.0,
        )
        return TranslationResponse(
            mode=mode,
            source_text=request.text,
            translation=attempts[final_index].candidate.text,
            retry_count=len(attempts) - 1,
            final_judgment=final_judgment,
            trace=trace,
        )


def write_evaluation_dataset(root: Path) -> None:
    original_reference = "SECRET_ORIGINAL_PUBLIC_REFERENCE"
    case = {
        "case_id": "case-1",
        "source_record_id": "source-1",
        "source_text": "캐시를 사용한다.",
        "reference_text": "SECRET_CORRECTED_EFFECTIVE_REFERENCE",
        "reference_provenance": {
            "original_reference_text": original_reference,
            "original_reference_sha256": hashlib.sha256(
                original_reference.encode("utf-8")
            ).hexdigest(),
            "effective_origin": "reviewer_correction",
            "review_status": "human_confirmed",
            "decision": "corrected",
            "reviewer_type": "bilingual_human",
            "reviewer": "reviewer-1",
            "reviewed_at_utc": "2026-08-20T00:00:00Z",
            "rationale": "The public reference omitted source information.",
            "reference_review_sha256": "a" * 64,
            "source_feedback_sha256": "b" * 64,
        },
        "domain": "software",
        "expected_terms": [
            {"source_term": "캐시", "accepted_targets": ["cache"]}
        ],
    }
    (root / "evaluation_v1.jsonl").write_text(
        json.dumps(case, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_benchmark_runner_calculates_metrics_without_reference_leak(
    tmp_path: Path,
) -> None:
    write_evaluation_dataset(tmp_path)
    service = RecordingTranslationService()
    runner = BenchmarkRunner(
        service,
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )

    result = runner.run(
        BenchmarkRequest(
            dataset_id="evaluation_v1",
            modes=[ExecutionMode.BASELINE, ExecutionMode.AGENT_RAG],
            warmup=False,
        )
    )

    assert result.metrics_by_mode[ExecutionMode.BASELINE].terminology_accuracy_pct == 0.0
    proposed = result.metrics_by_mode[ExecutionMode.AGENT_RAG]
    assert proposed.terminology_accuracy_pct == 100.0
    assert proposed.changed_sentence_rate_pct == 100.0
    assert proposed.mean_retry_count == 1.0
    assert proposed.agent_judgment_accuracy_pct is None
    assert proposed.agent_confusion_counts is None
    assert proposed.agent_judgment_labeled_count == 0
    assert proposed.successful_correction_rate_pct is None
    assert proposed.successful_correction_improved_count == 0
    assert proposed.successful_correction_labeled_count == 0
    assert (
        "agent_rag.agent_judgment_accuracy_pct" in result.unavailable_metrics
    )
    assert (
        "agent_rag.successful_correction_rate_pct" in result.unavailable_metrics
    )
    assert all(request.model_dump() == {"text": "캐시를 사용한다."} for request in service.requests)
    assert result.metadata["reference_exposed_to_runtime"] is False
    assert result.metadata["manual_judgment_gold_exposed_to_runtime"] is False
    assert result.metadata["manual_outcome_gold_exposed_to_runtime"] is False
    assert len(result.metadata["dataset_sha256"]) == 64
    assert len(result.metadata["config_sha256"]) == 64
    assert set(result.metadata["model_versions_by_mode"]) == {
        "baseline",
        "agent_rag",
    }
    assert result.artifact_path is not None
    assert Path(result.artifact_path).is_file()
    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    assert (
        "agent_rag.agent_judgment_accuracy_pct"
        in artifact["unavailable_metrics"]
    )
    assert artifact["dataset_sha256"] == result.metadata["dataset_sha256"]
    assert artifact["config_sha256"] == result.metadata["config_sha256"]
    assert artifact["run_config"]["warmup"] is False
    assert "reference" not in json.dumps(artifact["run_config"]).casefold()
    serialized_artifact = json.dumps(artifact, ensure_ascii=False)
    assert "SECRET_ORIGINAL_PUBLIC_REFERENCE" not in serialized_artifact
    assert "SECRET_CORRECTED_EFFECTIVE_REFERENCE" not in serialized_artifact


@pytest.mark.parametrize(
    "reference_provenance",
    [
        None,
        {
            "original_reference_text": "Public reference.",
            "original_reference_sha256": hashlib.sha256(
                b"Public reference."
            ).hexdigest(),
            "effective_origin": "public_dataset",
            "review_status": "ai_assisted_draft",
        },
    ],
)
def test_evaluation_v1_requires_human_confirmed_reference_provenance(
    tmp_path: Path,
    reference_provenance: dict[str, object] | None,
) -> None:
    row = {
        "case_id": "unconfirmed",
        "source_text": "원문",
        "reference_text": "Public reference.",
        "reference_provenance": reference_provenance,
    }
    (tmp_path / "evaluation_v1.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    service = RecordingTranslationService()
    runner = BenchmarkRunner(
        service,
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )

    with pytest.raises(ValueError, match="human-confirmed reference provenance"):
        runner.run(
            BenchmarkRequest(
                dataset_id="evaluation_v1",
                modes=[ExecutionMode.BASELINE],
                warmup=False,
            )
        )

    assert service.requests == []
    assert not (tmp_path / "artifacts").exists()


def test_pilot_v1_remains_runnable_without_reference_provenance(
    tmp_path: Path,
) -> None:
    row = {
        "case_id": "pilot",
        "source_text": "원문",
        "reference_text": "Reference.",
    }
    (tmp_path / "pilot_v1.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    service = RecordingTranslationService()
    runner = BenchmarkRunner(
        service,
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )

    result = runner.run(
        BenchmarkRequest(
            dataset_id="pilot_v1",
            modes=[ExecutionMode.BASELINE],
            warmup=False,
        )
    )

    assert result.metrics_by_mode[ExecutionMode.BASELINE].sample_count == 1
    assert len(service.requests) == 1


def test_benchmark_scores_only_confirmed_manual_first_judgment_labels(
    tmp_path: Path,
) -> None:
    rows = []
    expectations = [
        ("tp", True, True),
        ("fn", True, False),
        ("fp", False, True),
        ("tn", False, False),
    ]
    predictions: dict[str, bool] = {}
    for case_id, manual_needs_revision, agent_needs_revision in expectations:
        source = f"source-{case_id}"
        predictions[source] = agent_needs_revision
        label: dict[str, object] = {
            "needs_revision": manual_needs_revision,
            "review_status": "confirmed",
        }
        if manual_needs_revision:
            label.update(
                {
                    "primary_error": "term",
                    "error_types": ["term"],
                }
            )
        rows.append(
            {
                "case_id": case_id,
                "source_text": source,
                "reference_text": f"reference-{case_id}",
                "manual_judgments": {"agent_rag": label},
            }
        )
    (tmp_path / "manual_gold.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    service = GoldJudgmentService(predictions)
    runner = BenchmarkRunner(
        service,
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )
    result = runner.run(
        BenchmarkRequest(
            dataset_id="manual_gold",
            modes=[ExecutionMode.AGENT_RAG],
            warmup=False,
        )
    )

    metrics = result.metrics_by_mode[ExecutionMode.AGENT_RAG]
    assert metrics.agent_judgment_labeled_count == 4
    assert metrics.agent_judgment_accuracy_pct == 50.0
    assert metrics.agent_revision_recall_pct == 50.0
    assert metrics.agent_unnecessary_revision_rate_pct == 50.0
    assert metrics.agent_confusion_counts is not None
    assert metrics.agent_confusion_counts.model_dump() == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    # Two initial judgments report TERM and then pass after revision. Counting the
    # selected final judgment would incorrectly erase both errors.
    assert metrics.error_counts == {"term": 2}
    assert metrics.manual_primary_error_counts == {"term": 2}
    assert metrics.manual_error_type_counts == {"term": 2}
    assert "agent_rag.agent_judgment_accuracy_pct" not in result.unavailable_metrics
    assert all(
        request.model_dump() == {"text": source}
        for request, source in zip(
            service.requests,
            predictions,
            strict=True,
        )
    )

    assert result.artifact_path is not None
    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    artifact_metrics = artifact["metrics_by_mode"]["agent_rag"]
    assert artifact_metrics["agent_judgment_labeled_count"] == 4
    assert artifact_metrics["agent_confusion_counts"]["false_negative"] == 1
    assert artifact_metrics["error_counts"] == {"term": 2}


def test_successful_correction_uses_only_confirmed_initial_revision_outcomes(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "case_id": "improved",
            "source_text": "source-improved",
            "reference_text": "private-reference-improved",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": True,
                    "review_status": "confirmed",
                    "primary_error": "term",
                    "error_types": ["term", "meaning"],
                }
            },
            "manual_outcomes": {
                "agent_rag": {
                    "outcome": "improved",
                    "review_status": "confirmed",
                }
            },
        },
        {
            "case_id": "same",
            "source_text": "source-same",
            "reference_text": "private-reference-same",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": True,
                    "review_status": "confirmed",
                    "primary_error": "meaning",
                    "error_types": ["meaning"],
                }
            },
            "manual_outcomes": {
                "agent_rag": {
                    "outcome": "same",
                    "review_status": "confirmed",
                }
            },
        },
        {
            "case_id": "worse",
            "source_text": "source-worse",
            "reference_text": "private-reference-worse",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": True,
                    "review_status": "confirmed",
                    "primary_error": "fluency_grammar",
                    "error_types": ["fluency_grammar"],
                }
            },
            "manual_outcomes": {
                "agent_rag": {
                    "outcome": "worse",
                    "review_status": "confirmed",
                }
            },
        },
        {
            "case_id": "initial-pass",
            "source_text": "source-initial-pass",
            "reference_text": "private-reference-initial-pass",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": False,
                    "review_status": "confirmed",
                }
            },
            "manual_outcomes": {
                "agent_rag": {
                    "outcome": "improved",
                    "review_status": "confirmed",
                }
            },
        },
        {
            "case_id": "ambiguous-outcome",
            "source_text": "source-ambiguous-outcome",
            "reference_text": "private-reference-ambiguous-outcome",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": True,
                    "review_status": "confirmed",
                    "error_types": ["term"],
                }
            },
            "manual_outcomes": {
                "agent_rag": {
                    "outcome": "improved",
                    "review_status": "ambiguous",
                }
            },
        },
        {
            "case_id": "ambiguous-initial",
            "source_text": "source-ambiguous-initial",
            "reference_text": "private-reference-ambiguous-initial",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": True,
                    "review_status": "ambiguous",
                    "error_types": ["term"],
                }
            },
            "manual_outcomes": {
                "agent_rag": {
                    "outcome": "improved",
                    "review_status": "confirmed",
                }
            },
        },
        {
            "case_id": "missing-outcome",
            "source_text": "source-missing-outcome",
            "reference_text": "private-reference-missing-outcome",
            "manual_judgments": {
                "agent_rag": {
                    "needs_revision": True,
                    "review_status": "confirmed",
                    "error_types": ["term"],
                }
            },
        },
    ]
    (tmp_path / "manual_outcomes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    sources = {row["source_text"]: True for row in rows}
    service = GoldJudgmentService(sources)
    runner = BenchmarkRunner(
        service,
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )

    result = runner.run(
        BenchmarkRequest(
            dataset_id="manual_outcomes",
            modes=[ExecutionMode.AGENT_RAG],
            warmup=False,
        )
    )

    metrics = result.metrics_by_mode[ExecutionMode.AGENT_RAG]
    assert metrics.successful_correction_improved_count == 1
    assert metrics.successful_correction_labeled_count == 3
    assert metrics.successful_correction_rate_pct == pytest.approx(100 / 3)
    assert (
        "agent_rag.successful_correction_rate_pct"
        not in result.unavailable_metrics
    )
    assert result.metadata["reference_exposed_to_runtime"] is False
    assert result.metadata["manual_judgment_gold_exposed_to_runtime"] is False
    assert result.metadata["manual_outcome_gold_exposed_to_runtime"] is False
    assert metrics.error_counts == {"term": 7}
    assert metrics.manual_primary_error_counts == {
        "fluency_grammar": 1,
        "meaning": 1,
        "term": 1,
    }
    assert metrics.manual_error_type_counts == {
        "fluency_grammar": 1,
        "meaning": 2,
        "term": 3,
    }

    assert all(
        request.model_dump() == {"text": row["source_text"]}
        for request, row in zip(service.requests, rows, strict=True)
    )
    assert result.artifact_path is not None
    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    serialized_results = json.dumps(artifact["results"], ensure_ascii=False)
    assert "private-reference" not in serialized_results
    assert "manual_judgments" not in serialized_results
    assert "manual_outcomes" not in serialized_results


def test_ambiguous_manual_label_is_not_scored_or_inferred(tmp_path: Path) -> None:
    source = "ambiguous-source"
    row = {
        "case_id": "ambiguous",
        "source_text": source,
        "reference_text": "reference",
        "manual_judgments": {
            "agent_rag": {
                "needs_revision": True,
                "review_status": "ambiguous",
                "error_types": ["meaning"],
            }
        },
    }
    (tmp_path / "ambiguous.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    runner = BenchmarkRunner(
        GoldJudgmentService({source: True}),
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )

    result = runner.run(
        BenchmarkRequest(
            dataset_id="ambiguous",
            modes=[ExecutionMode.AGENT_RAG],
            warmup=False,
        )
    )

    metrics = result.metrics_by_mode[ExecutionMode.AGENT_RAG]
    assert metrics.agent_judgment_labeled_count == 0
    assert metrics.agent_judgment_accuracy_pct is None
    assert metrics.agent_confusion_counts is None
    assert metrics.successful_correction_rate_pct is None
    assert metrics.successful_correction_labeled_count == 0
    assert "confirmed manual" in result.unavailable_metrics[
        "agent_rag.agent_judgment_accuracy_pct"
    ]
    assert "confirmed manual" in result.unavailable_metrics[
        "agent_rag.successful_correction_rate_pct"
    ]


def test_agent_prediction_without_manual_gold_remains_unavailable(
    tmp_path: Path,
) -> None:
    source = "unlabeled-source"
    row = {
        "case_id": "unlabeled",
        "source_text": source,
        "reference_text": "reference",
    }
    (tmp_path / "unlabeled.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    runner = BenchmarkRunner(
        GoldJudgmentService({source: True}),
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )

    result = runner.run(
        BenchmarkRequest(
            dataset_id="unlabeled",
            modes=[ExecutionMode.AGENT_RAG],
            warmup=False,
        )
    )

    metrics = result.metrics_by_mode[ExecutionMode.AGENT_RAG]
    assert metrics.agent_judgment_accuracy_pct is None
    assert metrics.agent_confusion_counts is None
    assert metrics.agent_judgment_labeled_count == 0
    assert metrics.successful_correction_rate_pct is None
    assert metrics.successful_correction_labeled_count == 0
    assert "confirmed manual" in result.unavailable_metrics[
        "agent_rag.agent_judgment_accuracy_pct"
    ]
    assert "confirmed manual" in result.unavailable_metrics[
        "agent_rag.successful_correction_rate_pct"
    ]


def test_benchmark_endpoint_and_missing_dataset(tmp_path: Path) -> None:
    write_evaluation_dataset(tmp_path)
    service = RecordingTranslationService()
    runner = BenchmarkRunner(
        service,
        JsonlDatasetRepository(tmp_path),
        artifact_directory=tmp_path / "artifacts",
    )
    client = TestClient(
        create_app(
            translation_service=service,
            benchmark_service=runner,
        )
    )

    ok = client.post(
        "/benchmark",
        json={
            "dataset_id": "evaluation_v1",
            "modes": ["baseline"],
            "warmup": False,
        },
    )
    assert ok.status_code == 200
    assert ok.json()["metrics_by_mode"]["baseline"]["sample_count"] == 1

    missing = client.post(
        "/benchmark",
        json={"dataset_id": "missing", "modes": ["baseline"]},
    )
    assert missing.status_code == 404


def test_repository_rejects_missing_or_duplicate_cases(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    with pytest.raises(DatasetNotFoundError):
        repository.load("missing")

    case = {
        "case_id": "duplicate",
        "source_text": "원문",
        "reference_text": "Reference",
    }
    (tmp_path / "duplicates.jsonl").write_text(
        "\n".join([json.dumps(case, ensure_ascii=False)] * 2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        repository.load("duplicates")


@pytest.mark.parametrize(
    "manual_judgments",
    [
        {
            "agent_rag": {
                "needs_revision": True,
                "error_types": ["style_not_in_contract"],
            }
        },
        {"baseline": {"needs_revision": False}},
        {
            "agent": {
                "needs_revision": False,
                "error_types": ["meaning"],
            }
        },
    ],
)
def test_repository_rejects_invalid_manual_gold_labels(
    tmp_path: Path,
    manual_judgments: dict[str, object],
) -> None:
    row = {
        "case_id": "invalid-gold",
        "source_text": "원문",
        "reference_text": "Reference",
        "manual_judgments": manual_judgments,
    }
    (tmp_path / "invalid_gold.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid case"):
        JsonlDatasetRepository(tmp_path).load("invalid_gold")


@pytest.mark.parametrize(
    "manual_outcomes",
    [
        {"baseline": {"outcome": "improved"}},
        {"agent": {"outcome": "not-a-valid-outcome"}},
        {
            "agent_rag": {
                "outcome": "same",
                "review_status": "pending",
            }
        },
    ],
)
def test_repository_rejects_invalid_manual_outcome_labels(
    tmp_path: Path,
    manual_outcomes: dict[str, object],
) -> None:
    row = {
        "case_id": "invalid-outcome",
        "source_text": "원문",
        "reference_text": "Reference",
        "manual_outcomes": manual_outcomes,
    }
    (tmp_path / "invalid_outcome.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid case"):
        JsonlDatasetRepository(tmp_path).load("invalid_outcome")
