"""Small synchronous benchmark runner for frozen JSONL datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from uuid import uuid4

from pydantic import ValidationError

from translation_qa.contracts import TranslationService
from translation_qa.errors import DatasetNotFoundError
from translation_qa.metrics import (
    agent_judgment_metrics,
    latency_statistics,
    sentence_modification_rate,
    terminology_accuracy,
)
from translation_qa.schemas import (
    AgentConfusionCounts,
    BenchmarkRequest,
    BenchmarkResponse,
    EvaluationCase,
    ExecutionMode,
    ManualOutcome,
    ManualReviewStatus,
    ModeMetrics,
    ReferenceReviewStatus,
)


class JsonlDatasetRepository:
    """Load only validated dataset identifiers from one configured directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def path_for(self, dataset_id: str) -> Path:
        path = (self.root / f"{dataset_id}.jsonl").resolve()
        if path.parent != self.root:
            raise DatasetNotFoundError(f"invalid dataset identifier: {dataset_id}")
        if not path.is_file():
            raise DatasetNotFoundError(f"benchmark dataset not found: {dataset_id}")
        return path

    def load(self, dataset_id: str) -> list[EvaluationCase]:
        path = self.path_for(dataset_id)

        cases: list[EvaluationCase] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    cases.append(EvaluationCase.model_validate_json(line))
                except ValidationError as exc:
                    raise ValueError(
                        f"invalid case on line {line_number} of {path.name}: {exc}"
                    ) from exc
        if not cases:
            raise ValueError(f"benchmark dataset is empty: {dataset_id}")
        identifiers = [case.case_id for case in cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"benchmark dataset contains duplicate case IDs: {dataset_id}")
        return cases


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BenchmarkRunner:
    """Run paired conditions and persist traceable machine-readable results."""

    def __init__(
        self,
        translation_service: TranslationService,
        repository: JsonlDatasetRepository,
        *,
        artifact_directory: str | Path,
    ) -> None:
        self.translation_service = translation_service
        self.repository = repository
        self.artifact_directory = Path(artifact_directory)

    def run(self, request: BenchmarkRequest) -> BenchmarkResponse:
        dataset_path = self.repository.path_for(request.dataset_id)
        dataset_sha256 = _file_sha256(dataset_path)
        cases = self.repository.load(request.dataset_id)
        if request.dataset_id == "evaluation_v1":
            unconfirmed_case_ids = [
                case.case_id
                for case in cases
                if case.reference_provenance is None
                or case.reference_provenance.review_status
                is not ReferenceReviewStatus.HUMAN_CONFIRMED
            ]
            if unconfirmed_case_ids:
                preview = ", ".join(unconfirmed_case_ids[:3])
                if len(unconfirmed_case_ids) > 3:
                    preview += ", ..."
                raise ValueError(
                    "evaluation_v1 requires human-confirmed reference provenance "
                    f"for every case; unconfirmed cases: {preview}"
                )
        if request.limit is not None:
            cases = cases[: request.limit]

        run_id = uuid4()
        metrics_by_mode: dict[ExecutionMode, ModeMetrics] = {}
        result_records: list[dict[str, object]] = []
        model_versions_by_mode: dict[str, dict[str, str]] = {}
        unavailable_metrics: dict[str, str] = {}

        if request.warmup:
            warmup_request = cases[0].to_translation_request()
            for mode in request.modes:
                self.translation_service.translate(warmup_request, mode)

        for mode in request.modes:
            responses = []
            for case in cases:
                response = self.translation_service.translate(
                    case.to_translation_request(),
                    mode,
                )
                responses.append((case, response))
                result_records.append(
                    {
                        "case_id": case.case_id,
                        "mode": mode.value,
                        "response": response.model_dump(mode="json"),
                    }
                )

            model_versions_by_mode[mode.value] = dict(
                responses[0][1].trace.model_versions
            )

            term_occurrences = [
                (response.translation, expected.accepted_targets)
                for case, response in responses
                for expected in case.expected_terms
            ]
            term_metric = terminology_accuracy(term_occurrences)
            latencies = [response.trace.total_latency_ms for _, response in responses]
            latency = latency_statistics(latencies)
            assert latency.mean_ms is not None
            assert latency.median_ms is not None
            assert latency.p95_ms is not None
            retry_counts = [response.retry_count for _, response in responses]
            retry_distribution = dict(sorted(Counter(retry_counts).items()))
            error_counts: Counter[str] = Counter()
            for _, response in responses:
                first_judgment = response.trace.attempts[0].judgment
                if first_judgment is None:
                    continue
                error_counts.update(
                    error.value for error in first_judgment.error_types
                )

            uses_agent = mode in {ExecutionMode.AGENT, ExecutionMode.AGENT_RAG}
            changed_rate = None
            successful_correction_rate = None
            successful_correction_improved_count = 0
            successful_correction_labeled_count = 0
            agent_labeled_count = 0
            agent_accuracy = None
            agent_revision_recall = None
            agent_unnecessary_revision_rate = None
            agent_confusion = None
            manual_primary_error_counts: Counter[str] = Counter()
            manual_error_type_counts: Counter[str] = Counter()
            if uses_agent:
                modification = sentence_modification_rate(
                    (
                        response.trace.attempts[0].candidate.text,
                        response.translation,
                    )
                    for _, response in responses
                )
                changed_rate = modification.percentage

                manual_labels: list[bool] = []
                agent_labels: list[bool] = []
                confirmed_gold_count = 0
                for case, response in responses:
                    gold = case.manual_judgments.get(mode)
                    if (
                        gold is not None
                        and gold.review_status is ManualReviewStatus.CONFIRMED
                    ):
                        confirmed_gold_count += 1
                        if gold.primary_error is not None:
                            manual_primary_error_counts.update(
                                [gold.primary_error.value]
                            )
                        manual_error_type_counts.update(
                            error.value for error in gold.error_types
                        )

                        first_judgment = response.trace.attempts[0].judgment
                        if first_judgment is not None:
                            manual_labels.append(gold.needs_revision)
                            agent_labels.append(not first_judgment.passed)

                    outcome = case.manual_outcomes.get(mode)
                    if (
                        gold is not None
                        and gold.review_status is ManualReviewStatus.CONFIRMED
                        and gold.needs_revision
                        and outcome is not None
                        and outcome.review_status is ManualReviewStatus.CONFIRMED
                    ):
                        successful_correction_labeled_count += 1
                        if outcome.outcome is ManualOutcome.IMPROVED:
                            successful_correction_improved_count += 1

                if successful_correction_labeled_count:
                    successful_correction_rate = (
                        successful_correction_improved_count
                        / successful_correction_labeled_count
                        * 100.0
                    )
                else:
                    unavailable_metrics[
                        f"{mode.value}.successful_correction_rate_pct"
                    ] = (
                        "requires confirmed manual initial-to-final outcomes paired "
                        "with a confirmed initial NEEDS_REVISION label"
                    )

                if manual_labels:
                    judgment_metrics = agent_judgment_metrics(
                        manual_labels,
                        agent_labels,
                    )
                    matrix = judgment_metrics.confusion
                    agent_labeled_count = len(manual_labels)
                    agent_accuracy = judgment_metrics.accuracy.percentage
                    agent_revision_recall = (
                        judgment_metrics.revision_recall.percentage
                    )
                    agent_unnecessary_revision_rate = (
                        judgment_metrics.unnecessary_revision_rate.percentage
                    )
                    agent_confusion = AgentConfusionCounts(
                        true_positive=matrix.true_positive,
                        true_negative=matrix.true_negative,
                        false_positive=matrix.false_positive,
                        false_negative=matrix.false_negative,
                    )
                else:
                    reason = (
                        "confirmed manual labels exist, but no first-attempt Agent "
                        "judgments were available"
                        if confirmed_gold_count
                        else "requires confirmed manual PASS/NEEDS_REVISION labels"
                    )
                    unavailable_metrics[
                        f"{mode.value}.agent_judgment_accuracy_pct"
                    ] = reason

            metrics_by_mode[mode] = ModeMetrics(
                sample_count=len(responses),
                terminology_accuracy_pct=term_metric.percentage,
                changed_sentence_rate_pct=changed_rate,
                successful_correction_rate_pct=successful_correction_rate,
                successful_correction_improved_count=(
                    successful_correction_improved_count
                ),
                successful_correction_labeled_count=(
                    successful_correction_labeled_count
                ),
                mean_latency_ms=latency.mean_ms,
                median_latency_ms=latency.median_ms,
                p95_latency_ms=latency.p95_ms,
                mean_retry_count=fmean(retry_counts) if uses_agent else None,
                retry_distribution=retry_distribution if uses_agent else {},
                agent_judgment_labeled_count=agent_labeled_count,
                agent_judgment_accuracy_pct=agent_accuracy,
                agent_revision_recall_pct=agent_revision_recall,
                agent_unnecessary_revision_rate_pct=(
                    agent_unnecessary_revision_rate
                ),
                agent_confusion_counts=agent_confusion,
                error_counts=dict(sorted(error_counts.items())),
                manual_primary_error_counts=dict(
                    sorted(manual_primary_error_counts.items())
                ),
                manual_error_type_counts=dict(
                    sorted(manual_error_type_counts.items())
                ),
            )

        run_config = {
            "dataset_id": request.dataset_id,
            "dataset_sha256": dataset_sha256,
            "modes": [mode.value for mode in request.modes],
            "limit": request.limit,
            "warmup": request.warmup,
            "model_versions_by_mode": model_versions_by_mode,
        }
        config_sha256 = _json_sha256(run_config)

        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_directory / f"benchmark-{run_id}.json"
        artifact = {
            "run_id": str(run_id),
            "dataset_id": request.dataset_id,
            "dataset_sha256": dataset_sha256,
            "config_sha256": config_sha256,
            "run_config": run_config,
            "modes": [mode.value for mode in request.modes],
            "unavailable_metrics": unavailable_metrics,
            "metrics_by_mode": {
                mode.value: metrics.model_dump(mode="json")
                for mode, metrics in metrics_by_mode.items()
            },
            "results": result_records,
        }
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return BenchmarkResponse(
            run_id=run_id,
            dataset_id=request.dataset_id,
            metrics_by_mode=metrics_by_mode,
            artifact_path=str(artifact_path.resolve()),
            unavailable_metrics=unavailable_metrics,
            metadata={
                "sample_count": len(cases),
                "dataset_sha256": dataset_sha256,
                "config_sha256": config_sha256,
                "model_versions_by_mode": model_versions_by_mode,
                "warmup_runs_per_mode": 1 if request.warmup else 0,
                "reference_exposed_to_runtime": False,
                "manual_judgment_gold_exposed_to_runtime": False,
                "manual_outcome_gold_exposed_to_runtime": False,
            },
        )
