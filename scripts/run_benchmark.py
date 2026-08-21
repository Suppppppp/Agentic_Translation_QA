"""Run the reproducible four-way benchmark without starting the HTTP server."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Protocol

from translation_qa.main import build_default_benchmark, build_default_service
from translation_qa.schemas import BenchmarkRequest, ExecutionMode


class JsonResult(Protocol):
    def model_dump_json(self, *, indent: int | None = None) -> str: ...


class BenchmarkRunnerLike(Protocol):
    def run(self, request: BenchmarkRequest) -> JsonResult: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Baseline, RAG-only, Agent-only, and Agent+RAG on one frozen "
            "JSONL dataset. Runtime components are selected with the same "
            "TRANSLATION_QA_* environment variables as the API."
        )
    )
    parser.add_argument("--dataset-id", default="evaluation_v1")
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in ExecutionMode],
        help="Repeat to select modes; omission runs all four ablations.",
    )
    parser.add_argument("--limit", type=int, choices=range(1, 51))
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable the default one-request warm-up for each mode.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[], object] = build_default_service,
    benchmark_factory: Callable[[object], BenchmarkRunnerLike] = build_default_benchmark,
) -> int:
    args = build_parser().parse_args(argv)
    modes = (
        [ExecutionMode(value) for value in args.mode]
        if args.mode
        else list(ExecutionMode)
    )
    service = service_factory()
    runner = benchmark_factory(service)
    response = runner.run(
        BenchmarkRequest(
            dataset_id=args.dataset_id,
            modes=modes,
            limit=args.limit,
            warmup=not args.no_warmup,
        )
    )
    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
