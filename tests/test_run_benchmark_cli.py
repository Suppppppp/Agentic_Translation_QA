from __future__ import annotations

import json

from scripts.run_benchmark import main
from translation_qa.schemas import BenchmarkRequest, ExecutionMode


class FakeResult:
    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps({"run_id": "fake"}, indent=indent)


class FakeRunner:
    def __init__(self) -> None:
        self.request: BenchmarkRequest | None = None

    def run(self, request: BenchmarkRequest) -> FakeResult:
        self.request = request
        return FakeResult()


def test_cli_defaults_to_reproducible_four_way_run(capsys: object) -> None:
    runner = FakeRunner()
    service = object()

    assert (
        main(
            [],
            service_factory=lambda: service,
            benchmark_factory=lambda received: runner,
        )
        == 0
    )

    assert runner.request is not None
    assert runner.request.dataset_id == "evaluation_v1"
    assert runner.request.modes == list(ExecutionMode)
    assert runner.request.warmup is True


def test_cli_accepts_selected_modes_limit_and_no_warmup() -> None:
    runner = FakeRunner()

    main(
        [
            "--dataset-id",
            "pilot_v1",
            "--mode",
            "baseline",
            "--mode",
            "agent_rag",
            "--limit",
            "7",
            "--no-warmup",
        ],
        service_factory=object,
        benchmark_factory=lambda service: runner,
    )

    assert runner.request is not None
    assert runner.request.dataset_id == "pilot_v1"
    assert runner.request.modes == [ExecutionMode.BASELINE, ExecutionMode.AGENT_RAG]
    assert runner.request.limit == 7
    assert runner.request.warmup is False
