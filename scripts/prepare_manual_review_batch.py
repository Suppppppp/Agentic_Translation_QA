"""Prepare a deterministic, blank human-review batch from a frozen benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


LEGACY_MANUAL_COLUMNS = [
    "manual_initial_needs_revision",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
]

MANUAL_COLUMNS = [
    "manual_initial_needs_revision",
    "manual_severity",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
]

LEGACY_REVIEW_COLUMNS = [
    "review_key",
    "case_id",
    "mode",
    *LEGACY_MANUAL_COLUMNS,
    "source_text",
    "reference_text",
    "initial_translation",
    "final_translation",
    "agent_initial_passed",
    "agent_initial_error_types",
    "agent_initial_summary",
    "retry_count",
    "stop_reason",
]

REVIEW_COLUMNS = [
    "review_key",
    "case_id",
    "mode",
    *MANUAL_COLUMNS,
    "source_text",
    "reference_text",
    "initial_translation",
    "final_translation",
    "agent_initial_passed",
    "agent_initial_error_types",
    "agent_initial_summary",
    "retry_count",
    "stop_reason",
]

ALLOWED_MODES = {"agent", "agent_rag"}
EXPECTED_BATCH1_PAIRS = [
    (case_id, mode)
    for case_id in (
        "evaluation-v1-004",
        "evaluation-v1-009",
        "evaluation-v1-019",
        "evaluation-v1-024",
        "evaluation-v1-027",
    )
    for mode in ("agent", "agent_rag")
]


def _manual_review_schema_version(selection: dict[str, Any]) -> int:
    value = selection.get("manual_review_schema_version", 1)
    if type(value) is not int or value not in {1, 2}:
        raise ValueError("manual_review_schema_version must be 1 or 2")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve_project_file(project_root: Path, value: object, field: str) -> Path:
    relative = Path(_require_string(value, field))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a project-relative path")
    return project_root / relative


def _load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"dataset line {line_number} must be an object")
            case_id = _require_string(record.get("case_id"), f"dataset line {line_number} case_id")
            if case_id in records:
                raise ValueError(f"duplicate dataset case_id: {case_id}")
            records[case_id] = record
    return records


def _validate_reference_provenance(
    dataset: dict[str, dict[str, Any]], selection: dict[str, Any]
) -> None:
    expected_review_hash = _require_string(
        selection.get("reference_review_sha256"), "reference_review_sha256"
    )
    expected_feedback_hash = _require_string(
        selection.get("source_feedback_sha256"), "source_feedback_sha256"
    )
    for case_id, record in dataset.items():
        provenance = record.get("reference_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"dataset reference_provenance is missing for {case_id}")
        if provenance.get("reference_review_sha256") != expected_review_hash:
            raise ValueError(
                f"dataset reference_review_sha256 does not match selection for {case_id}"
            )
        if provenance.get("source_feedback_sha256") != expected_feedback_hash:
            raise ValueError(
                f"dataset source_feedback_sha256 does not match selection for {case_id}"
            )


def _validate_selection(selection: dict[str, Any]) -> list[dict[str, str]]:
    if selection.get("schema_version") != 1:
        raise ValueError("selection schema_version must be 1")
    if selection.get("partial_representative_sample") is not True:
        raise ValueError("partial_representative_sample must be true")
    batch_id = _require_string(selection.get("batch_id"), "batch_id")

    raw_selected = selection.get("selected")
    if not isinstance(raw_selected, list) or not raw_selected:
        raise ValueError("selection must contain a non-empty selected array")

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_selected):
        if not isinstance(raw, dict):
            raise ValueError(f"selected[{index}] must be an object")
        case_id = _require_string(raw.get("case_id"), f"selected[{index}].case_id")
        mode = _require_string(raw.get("mode"), f"selected[{index}].mode")
        reason = _require_string(
            raw.get("selection_reason"), f"selected[{index}].selection_reason"
        )
        if mode not in ALLOWED_MODES:
            raise ValueError(f"selected[{index}].mode must be agent or agent_rag")
        key = (case_id, mode)
        if key in seen:
            raise ValueError(f"duplicate selected pair: {case_id}/{mode}")
        seen.add(key)
        selected.append(
            {"case_id": case_id, "mode": mode, "selection_reason": reason}
        )
    if batch_id == "evaluation_v1_batch1":
        actual_pairs = [(item["case_id"], item["mode"]) for item in selected]
        if actual_pairs != EXPECTED_BATCH1_PAIRS:
            raise ValueError(
                "evaluation_v1_batch1 selected pairs or order do not match the frozen batch"
            )
    return selected


def _validate_artifact(
    artifact: dict[str, Any], selection: dict[str, Any], dataset_hash: str
) -> None:
    expected_run_id = _require_string(selection.get("run_id"), "run_id")
    expected_dataset_id = _require_string(selection.get("dataset_id"), "dataset_id")
    expected_config_hash = _require_string(
        selection.get("artifact_config_sha256"), "artifact_config_sha256"
    )

    if artifact.get("run_id") != expected_run_id:
        raise ValueError("artifact run_id does not match selection")
    if artifact.get("dataset_id") != expected_dataset_id:
        raise ValueError("artifact dataset_id does not match selection")
    if artifact.get("dataset_sha256") != dataset_hash:
        raise ValueError("artifact dataset_sha256 does not match frozen dataset")
    if artifact.get("config_sha256") != expected_config_hash:
        raise ValueError("artifact config_sha256 does not match selection")

    run_config = artifact.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("artifact has no run_config object")
    if run_config.get("dataset_id") != expected_dataset_id:
        raise ValueError("artifact run_config dataset_id does not match selection")
    if run_config.get("dataset_sha256") != dataset_hash:
        raise ValueError("artifact run_config dataset_sha256 does not match dataset")


def _index_results(artifact: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    raw_results = artifact.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("artifact has no results array")

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            raise ValueError(f"artifact results[{index}] must be an object")
        case_id = raw.get("case_id")
        mode = raw.get("mode")
        if not isinstance(case_id, str) or not isinstance(mode, str):
            raise ValueError(f"artifact results[{index}] has invalid case_id or mode")
        key = (case_id, mode)
        if key in indexed:
            raise ValueError(f"duplicate artifact result: {case_id}/{mode}")
        indexed[key] = raw
    return indexed


def _build_row(
    case: dict[str, Any],
    result: dict[str, Any],
    case_id: str,
    mode: str,
    manual_columns: list[str],
) -> dict[str, object]:
    response = result.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"artifact response is invalid for {case_id}/{mode}")
    trace = response.get("trace")
    if not isinstance(trace, dict):
        raise ValueError(f"artifact trace is invalid for {case_id}/{mode}")
    attempts = trace.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"artifact has no attempts for {case_id}/{mode}")
    initial = attempts[0]
    if not isinstance(initial, dict):
        raise ValueError(f"initial attempt is invalid for {case_id}/{mode}")
    candidate = initial.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"initial candidate is invalid for {case_id}/{mode}")
    judgment = initial.get("judgment")
    if judgment is not None and not isinstance(judgment, dict):
        raise ValueError(f"initial judgment is invalid for {case_id}/{mode}")

    row: dict[str, object] = {
        "review_key": f"{case_id}::{mode}",
        "case_id": case_id,
        "mode": mode,
        **dict.fromkeys(manual_columns, ""),
        "source_text": case.get("source_text", ""),
        "reference_text": case.get("reference_text", ""),
        "initial_translation": candidate.get("text", ""),
        "final_translation": response.get("translation", ""),
        "agent_initial_passed": (
            judgment.get("passed", "") if isinstance(judgment, dict) else ""
        ),
        "agent_initial_error_types": json.dumps(
            judgment.get("error_types", []) if isinstance(judgment, dict) else [],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "agent_initial_summary": (
            judgment.get("summary", "") if isinstance(judgment, dict) else ""
        ),
        "retry_count": max(len(attempts) - 1, 0),
        "stop_reason": trace.get("stop_reason", ""),
    }
    return row


def prepare_batch(
    selection_path: Path, output_path: Path, project_root: Path
) -> int:
    selection_raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection_raw, dict):
        raise ValueError("selection manifest must be an object")
    manual_review_schema_version = _manual_review_schema_version(selection_raw)
    selected = _validate_selection(selection_raw)

    dataset_path = _resolve_project_file(
        project_root, selection_raw.get("dataset_file"), "dataset_file"
    )
    artifact_path = _resolve_project_file(
        project_root, selection_raw.get("artifact_file"), "artifact_file"
    )
    dataset_hash = _sha256(dataset_path)
    expected_dataset_hash = _require_string(
        selection_raw.get("dataset_sha256"), "dataset_sha256"
    )
    if dataset_hash != expected_dataset_hash:
        raise ValueError("dataset SHA-256 does not match selection")
    artifact_hash = _sha256(artifact_path)
    expected_artifact_hash = _require_string(
        selection_raw.get("artifact_sha256"), "artifact_sha256"
    )
    if artifact_hash != expected_artifact_hash:
        raise ValueError("artifact SHA-256 does not match selection")

    dataset = _load_dataset(dataset_path)
    _validate_reference_provenance(dataset, selection_raw)
    artifact_raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact_raw, dict):
        raise ValueError("benchmark artifact must be an object")
    _validate_artifact(artifact_raw, selection_raw, dataset_hash)
    indexed_results = _index_results(artifact_raw)

    manual_columns = (
        MANUAL_COLUMNS
        if manual_review_schema_version == 2
        else LEGACY_MANUAL_COLUMNS
    )
    rows: list[dict[str, object]] = []
    for item in selected:
        case_id, mode = item["case_id"], item["mode"]
        case = dataset.get(case_id)
        if case is None:
            raise ValueError(f"selected case is absent from dataset: {case_id}")
        result = indexed_results.get((case_id, mode))
        if result is None:
            raise ValueError(f"selected result is absent from artifact: {case_id}/{mode}")
        rows.append(_build_row(case, result, case_id, mode, manual_columns))

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite review labels: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_columns = (
        REVIEW_COLUMNS
        if manual_review_schema_version == 2
        else LEGACY_REVIEW_COLUMNS
    )
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=review_columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a hash-pinned, blank manual-review batch."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = prepare_batch(
        args.selection.resolve(), args.output.resolve(), args.project_root.resolve()
    )
    print(f"wrote {count} blank review rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
