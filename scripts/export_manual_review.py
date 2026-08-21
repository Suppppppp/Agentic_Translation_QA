"""Export an offline review sheet by joining benchmark traces with references."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path


REVIEW_COLUMNS = [
    "case_id",
    "mode",
    "source_text",
    "reference_text",
    "initial_translation",
    "final_translation",
    "agent_initial_passed",
    "agent_initial_error_types",
    "retry_count",
    "stop_reason",
    "manual_initial_needs_revision",
    "manual_severity",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
]


def load_dataset(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"dataset line {line_number} has no case_id")
            if case_id in records:
                raise ValueError(f"duplicate dataset case_id: {case_id}")
            records[case_id] = record
    return records


def build_rows(
    dataset: dict[str, dict[str, object]],
    artifact: dict[str, object],
) -> list[dict[str, object]]:
    raw_results = artifact.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("benchmark artifact has no results array")

    rows: list[dict[str, object]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError("benchmark result must be an object")
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or case_id not in dataset:
            raise ValueError(f"artifact case is absent from dataset: {case_id}")
        response = result.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"artifact response is invalid for {case_id}")
        trace = response.get("trace")
        if not isinstance(trace, dict):
            raise ValueError(f"artifact trace is invalid for {case_id}")
        attempts = trace.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"artifact has no attempts for {case_id}")
        initial = attempts[0]
        if not isinstance(initial, dict):
            raise ValueError(f"initial attempt is invalid for {case_id}")
        candidate = initial.get("candidate")
        judgment = initial.get("judgment")
        if not isinstance(candidate, dict):
            raise ValueError(f"initial candidate is invalid for {case_id}")

        source_record = dataset[case_id]
        error_types = (
            judgment.get("error_types", []) if isinstance(judgment, dict) else []
        )
        rows.append(
            {
                "case_id": case_id,
                "mode": result.get("mode", ""),
                "source_text": source_record.get("source_text", ""),
                "reference_text": source_record.get("reference_text", ""),
                "initial_translation": candidate.get("text", ""),
                "final_translation": response.get("translation", ""),
                "agent_initial_passed": (
                    judgment.get("passed", "") if isinstance(judgment, dict) else ""
                ),
                "agent_initial_error_types": json.dumps(
                    error_types, ensure_ascii=False
                ),
                "retry_count": max(len(attempts) - 1, 0),
                "stop_reason": trace.get("stop_reason", ""),
                "manual_initial_needs_revision": "",
                "manual_severity": "",
                "manual_primary_error": "",
                "manual_error_types": "",
                "pairwise_outcome": "",
                "review_status": "",
                "reviewer": "",
                "note": "",
            }
        )
    return rows


def export_review(dataset_path: Path, artifact_path: Path, output_path: Path) -> int:
    dataset = load_dataset(dataset_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = build_rows(dataset, artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=REVIEW_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join a leak-free runtime artifact with offline references and create "
            "a blank human-review sheet."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = export_review(args.dataset, args.artifact, args.output)
    print(f"wrote {count} review rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
