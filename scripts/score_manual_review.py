"""Score a pinned, human-labeled representative Agent review batch offline.

This command never calls a model and never mutates the benchmark artifact or
frozen dataset.  It validates the review evidence against a hash-pinned
selection manifest before writing a separate JSON score file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from translation_qa.metrics import agent_judgment_metrics
from translation_qa.schemas import (
    ManualJudgmentLabel,
    ManualOutcomeLabel,
    ManualReviewStatus,
    TranslationErrorType,
)


EVIDENCE_COLUMNS = (
    "review_key",
    "case_id",
    "mode",
    "source_text",
    "reference_text",
    "initial_translation",
    "final_translation",
    "agent_initial_passed",
    "agent_initial_error_types",
    "agent_initial_summary",
    "retry_count",
    "stop_reason",
)
LEGACY_MANUAL_COLUMNS = (
    "manual_initial_needs_revision",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
)
MANUAL_COLUMNS = (
    "manual_initial_needs_revision",
    "manual_severity",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
)
LEGACY_REQUIRED_COLUMNS = EVIDENCE_COLUMNS + LEGACY_MANUAL_COLUMNS
REQUIRED_COLUMNS = EVIDENCE_COLUMNS + MANUAL_COLUMNS
AGENT_MODES = {"agent", "agent_rag"}
MANUAL_SEVERITIES = ("MAJOR", "MINOR")
HASH_FIELDS = (
    "artifact_sha256",
    "dataset_sha256",
    "artifact_config_sha256",
    "reference_review_sha256",
    "source_feedback_sha256",
)
CAVEAT = (
    "These metrics cover only a deliberately selected representative subset, "
    "not the full benchmark. They validate the manual-review and offline-scoring "
    "workflow and must not be presented as overall translation-quality claims."
)


def _manual_review_schema_version(manifest: dict[str, Any]) -> int:
    value = manifest.get("manual_review_schema_version", 1)
    if type(value) is not int or value not in {1, 2}:
        raise ValueError("manual_review_schema_version must be 1 or 2")
    return value


def _sha256(path: Path) -> str:
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


def _require_string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"selection manifest requires nonblank {field}")
    return value


def _require_hash(mapping: dict[str, Any], field: str) -> str:
    value = _require_string(mapping, field)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"selection manifest {field} must be lowercase SHA-256")
    return value


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid dataset JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"dataset line {line_number} must be an object")
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"dataset line {line_number} has no case_id")
            if case_id in records:
                raise ValueError(f"duplicate dataset case_id: {case_id}")
            records[case_id] = record
    if not records:
        raise ValueError("dataset is empty")
    return records


def _selection_keys(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection manifest requires a nonempty selected array")
    keys: list[tuple[str, str]] = []
    for index, item in enumerate(selected, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"selected item {index} must be an object")
        case_id = item.get("case_id")
        mode = item.get("mode")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"selected item {index} has no case_id")
        if mode not in AGENT_MODES:
            raise ValueError(
                f"selected item {index} mode must be agent or agent_rag"
            )
        keys.append((case_id, mode))
    if len(keys) != len(set(keys)):
        raise ValueError("selection manifest contains duplicate (case_id, mode) keys")
    return keys


def _artifact_index(artifact: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark artifact has no results array")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for position, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"artifact result {position} must be an object")
        case_id = result.get("case_id")
        mode = result.get("mode")
        if not isinstance(case_id, str) or not isinstance(mode, str):
            raise ValueError(f"artifact result {position} has an invalid key")
        key = (case_id, mode)
        if key in index:
            raise ValueError(f"benchmark artifact contains duplicate result key: {key}")
        index[key] = result
    return index


def _validate_manifest_and_inputs(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    artifact: dict[str, Any],
    artifact_path: Path,
    dataset_path: Path,
    project_root: Path,
    selection_keys: list[tuple[str, str]],
    dataset: dict[str, dict[str, Any]],
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("selection manifest schema_version must be 1")
    if manifest.get("partial_representative_sample") is not True:
        raise ValueError(
            "selection manifest must declare partial_representative_sample=true"
        )
    _manual_review_schema_version(manifest)
    _require_string(manifest, "batch_id")
    for field in HASH_FIELDS:
        _require_hash(manifest, field)

    declared_artifact = (
        project_root / _require_string(manifest, "artifact_file")
    ).resolve()
    declared_dataset = (
        project_root / _require_string(manifest, "dataset_file")
    ).resolve()
    if declared_artifact != artifact_path.resolve():
        raise ValueError("artifact path does not match selection manifest")
    if declared_dataset != dataset_path.resolve():
        raise ValueError("dataset path does not match selection manifest")
    if _sha256(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("artifact hash does not match selection manifest")
    if _sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("dataset hash does not match selection manifest")

    run_id = _require_string(manifest, "run_id")
    dataset_id = _require_string(manifest, "dataset_id")
    if artifact.get("run_id") != run_id:
        raise ValueError("artifact run_id does not match selection manifest")
    if artifact.get("dataset_id") != dataset_id:
        raise ValueError("artifact dataset_id does not match selection manifest")
    if artifact.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("artifact dataset hash does not match selection manifest")
    if artifact.get("config_sha256") != manifest["artifact_config_sha256"]:
        raise ValueError("artifact config hash does not match selection manifest")
    run_config = artifact.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("benchmark artifact has no run_config object")
    if _json_sha256(run_config) != artifact.get("config_sha256"):
        raise ValueError("artifact config_sha256 does not match run_config")
    if run_config.get("dataset_id") != dataset_id:
        raise ValueError("artifact run_config dataset_id is inconsistent")
    if run_config.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("artifact run_config dataset hash is inconsistent")

    for case_id, _ in selection_keys:
        record = dataset.get(case_id)
        if record is None:
            raise ValueError(f"selected case is absent from dataset: {case_id}")
        provenance = record.get("reference_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"selected case has no reference provenance: {case_id}")
        for field in ("reference_review_sha256", "source_feedback_sha256"):
            if provenance.get(field) != manifest[field]:
                raise ValueError(
                    f"selected case {field} does not match selection manifest: "
                    f"{case_id}"
                )

    # Include the manifest in the audit trail only after its own fields validate.
    if not manifest_path.is_file():
        raise ValueError("selection manifest does not exist")


def _expected_evidence(
    result: dict[str, Any],
    dataset_record: dict[str, Any],
) -> dict[str, str]:
    case_id = result.get("case_id")
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
        raise ValueError(f"artifact initial attempt is invalid for {case_id}")
    candidate = initial.get("candidate")
    judgment = initial.get("judgment")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
        raise ValueError(f"artifact initial candidate is invalid for {case_id}")
    if judgment is not None and not isinstance(judgment, dict):
        raise ValueError(f"artifact initial judgment is invalid for {case_id}")
    error_types = judgment.get("error_types", []) if judgment is not None else []
    if not isinstance(error_types, list):
        raise ValueError(f"artifact Agent error types are invalid for {case_id}")
    agent_passed = judgment.get("passed", "") if judgment is not None else ""
    if agent_passed != "" and type(agent_passed) is not bool:
        raise ValueError(f"artifact Agent passed label is invalid for {case_id}")
    agent_summary = judgment.get("summary", "") if judgment is not None else ""
    if not isinstance(agent_summary, str):
        raise ValueError(f"artifact Agent summary is invalid for {case_id}")
    if response.get("source_text") != dataset_record.get("source_text"):
        raise ValueError(f"artifact source text differs from dataset for {case_id}")
    mode = str(result.get("mode", ""))
    values = {
        "review_key": f"{case_id}::{mode}",
        "case_id": str(case_id),
        "mode": mode,
        "source_text": str(dataset_record.get("source_text", "")),
        "reference_text": str(dataset_record.get("reference_text", "")),
        "initial_translation": candidate["text"],
        "final_translation": str(response.get("translation", "")),
        "agent_initial_passed": str(agent_passed),
        "agent_initial_error_types": json.dumps(error_types, ensure_ascii=False),
        "agent_initial_summary": agent_summary,
        "retry_count": str(max(len(attempts) - 1, 0)),
        "stop_reason": str(trace.get("stop_reason", "")),
    }
    return values


def _load_review_rows(
    path: Path, required_columns: tuple[str, ...]
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("review CSV has no header")
        missing = [
            column for column in required_columns if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"review CSV is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _evidence_matches(column: str, actual: str, expected: str) -> bool:
    """Compare evidence exactly, except for immaterial JSON list whitespace."""

    if column != "agent_initial_error_types":
        return actual == expected
    try:
        actual_value = json.loads(actual)
        expected_value = json.loads(expected)
    except json.JSONDecodeError:
        return False
    return actual_value == expected_value


def _parse_bool(value: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(
        f"row {row_number} manual_initial_needs_revision must be true or false"
    )


def _parse_error_types(value: str, row_number: int) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"row {row_number} manual_error_types must be a JSON list"
        ) from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise ValueError(
            f"row {row_number} manual_error_types must be a JSON string list"
        )
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"row {row_number} manual_error_types contains duplicates")
    return parsed


def _parse_manual_labels(
    row: dict[str, str], row_number: int, manual_review_schema_version: int
) -> tuple[ManualJudgmentLabel, ManualOutcomeLabel, str, str | None]:
    manual_columns = (
        MANUAL_COLUMNS
        if manual_review_schema_version == 2
        else LEGACY_MANUAL_COLUMNS
    )
    manual = {column: row.get(column, "").strip() for column in manual_columns}
    if manual_review_schema_version == 2:
        # Severity is deliberately not stripped or case-normalized. It is a
        # human label and must arrive as one of the exact contracted values.
        manual["manual_severity"] = row.get("manual_severity", "")
    if not any(manual.values()):
        raise ValueError(
            f"row {row_number} is pending; complete all required manual labels"
        )

    for field in (
        "manual_initial_needs_revision",
        "pairwise_outcome",
        "review_status",
        "reviewer",
    ):
        if not manual[field]:
            raise ValueError(
                f"row {row_number} is partially filled; {field} is required"
            )

    needs_revision = _parse_bool(manual["manual_initial_needs_revision"], row_number)
    severity: str | None = None
    if manual_review_schema_version == 2:
        raw_severity = manual["manual_severity"]
        if needs_revision:
            if not raw_severity:
                raise ValueError(
                    f"row {row_number} manual_severity is required when "
                    "manual_initial_needs_revision is true"
                )
            if raw_severity not in MANUAL_SEVERITIES:
                raise ValueError(
                    f"row {row_number} manual_severity must be MAJOR or MINOR"
                )
            severity = raw_severity
        elif raw_severity:
            raise ValueError(
                f"row {row_number} manual_severity must be blank when "
                "manual_initial_needs_revision is false"
            )
    error_types = _parse_error_types(manual["manual_error_types"], row_number)
    try:
        judgment = ManualJudgmentLabel(
            needs_revision=needs_revision,
            review_status=manual["review_status"],
            primary_error=manual["manual_primary_error"] or None,
            error_types=error_types,
        )
        outcome = ManualOutcomeLabel(
            outcome=manual["pairwise_outcome"],
            review_status=manual["review_status"],
        )
    except ValidationError as exc:
        raise ValueError(
            f"row {row_number} contains invalid manual labels: {exc}"
        ) from exc

    has_other = (
        judgment.primary_error is TranslationErrorType.OTHER
        or TranslationErrorType.OTHER in judgment.error_types
    )
    if has_other and not manual["note"]:
        raise ValueError(f"row {row_number} requires note when other is used")
    if (
        judgment.review_status is ManualReviewStatus.AMBIGUOUS
        and not manual["note"]
    ):
        raise ValueError(
            f"row {row_number} requires note when review_status is ambiguous"
        )
    return judgment, outcome, manual["reviewer"], severity


def _blank_counts() -> dict[str, int]:
    return {value.value: 0 for value in TranslationErrorType}


def _score_mode(
    entries: list[dict[str, Any]], *, include_severity: bool = False
) -> dict[str, Any]:
    confirmed = [
        entry
        for entry in entries
        if entry["judgment"].review_status is ManualReviewStatus.CONFIRMED
    ]
    ambiguous = len(entries) - len(confirmed)
    scorable = [
        entry
        for entry in confirmed
        if not entry["component_failure"] and entry["agent_passed"] is not None
    ]
    manual_labels = [entry["judgment"].needs_revision for entry in scorable]
    agent_labels = [not entry["agent_passed"] for entry in scorable]
    metrics = agent_judgment_metrics(manual_labels, agent_labels)
    confusion = metrics.confusion

    reviewed_outcomes: Counter[str] = Counter(
        entry["outcome"].outcome.value for entry in entries
    )
    confirmed_outcomes: Counter[str] = Counter(
        entry["outcome"].outcome.value for entry in confirmed
    )
    primary_errors = _blank_counts()
    error_types = _blank_counts()
    for entry in confirmed:
        judgment = entry["judgment"]
        if judgment.primary_error is not None:
            primary_errors[judgment.primary_error.value] += 1
        for error in judgment.error_types:
            error_types[error.value] += 1

    eligible = [entry for entry in confirmed if entry["judgment"].needs_revision]
    improved = sum(entry["outcome"].outcome.value == "improved" for entry in eligible)
    outcome_names = ("improved", "same", "worse")
    result = {
        "selected": len(entries),
        "completed": len(entries),
        "confirmed": len(confirmed),
        "ambiguous": ambiguous,
        "unscorable": sum(
            entry["component_failure"] or entry["agent_passed"] is None
            for entry in entries
        ),
        "component_failure_count": sum(
            entry["component_failure"] for entry in entries
        ),
        "agent_judgment": {
            "denominator": confusion.total,
            "confusion": {
                "true_positive": confusion.true_positive,
                "true_negative": confusion.true_negative,
                "false_positive": confusion.false_positive,
                "false_negative": confusion.false_negative,
            },
            "accuracy_pct": metrics.accuracy.percentage,
            "revision_recall_pct": metrics.revision_recall.percentage,
            "unnecessary_revision_rate_pct": (
                metrics.unnecessary_revision_rate.percentage
            ),
        },
        "successful_correction": {
            "eligible": len(eligible),
            "improved": improved,
            "rate_pct": None if not eligible else improved / len(eligible) * 100.0,
        },
        "reviewed_outcome_counts": {
            name: reviewed_outcomes[name] for name in outcome_names
        },
        "confirmed_outcome_counts": {
            name: confirmed_outcomes[name] for name in outcome_names
        },
        "confirmed_primary_error_counts": primary_errors,
        "confirmed_error_type_counts": error_types,
    }
    if include_severity:
        result["confirmed_severity_counts"] = {
            severity: sum(
                entry["severity"] == severity for entry in confirmed
            )
            for severity in MANUAL_SEVERITIES
        }
        result["major_false_pass_count"] = sum(
            entry["judgment"].needs_revision
            and entry["severity"] == "MAJOR"
            and not entry["component_failure"]
            and entry["agent_passed"] is True
            for entry in confirmed
        )
    return result


def score_manual_review(
    *,
    artifact_path: Path,
    dataset_path: Path,
    selection_manifest_path: Path,
    review_csv_path: Path,
    output_path: Path,
    project_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and score a completed review batch, then write one JSON file."""

    input_paths = {
        artifact_path.resolve(),
        dataset_path.resolve(),
        selection_manifest_path.resolve(),
        review_csv_path.resolve(),
    }
    if output_path.resolve() in input_paths:
        raise ValueError("output must not overwrite an input file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    manifest = _load_json_object(selection_manifest_path, "selection manifest")
    manual_review_schema_version = _manual_review_schema_version(manifest)
    artifact = _load_json_object(artifact_path, "benchmark artifact")
    dataset = _load_dataset(dataset_path)
    keys = _selection_keys(manifest)
    _validate_manifest_and_inputs(
        manifest=manifest,
        manifest_path=selection_manifest_path,
        artifact=artifact,
        artifact_path=artifact_path,
        dataset_path=dataset_path,
        project_root=project_root.resolve(),
        selection_keys=keys,
        dataset=dataset,
    )
    artifact_by_key = _artifact_index(artifact)

    required_columns = (
        REQUIRED_COLUMNS
        if manual_review_schema_version == 2
        else LEGACY_REQUIRED_COLUMNS
    )
    rows = _load_review_rows(review_csv_path, required_columns)
    csv_keys = [(row["case_id"], row["mode"]) for row in rows]
    if len(csv_keys) != len(set(csv_keys)):
        raise ValueError("review CSV contains duplicate (case_id, mode) keys")
    if csv_keys != keys:
        raise ValueError(
            "review CSV keys/order do not exactly match selection manifest"
        )

    entries_by_mode: dict[str, list[dict[str, Any]]] = {
        "agent": [],
        "agent_rag": [],
    }
    all_entries: list[dict[str, Any]] = []
    for row_number, (row, key) in enumerate(zip(rows, keys, strict=True), start=2):
        result = artifact_by_key.get(key)
        if result is None:
            raise ValueError(f"selected key is absent from benchmark artifact: {key}")
        record = dataset[key[0]]
        expected = _expected_evidence(result, record)
        for column in EVIDENCE_COLUMNS:
            if not _evidence_matches(column, row[column], expected[column]):
                raise ValueError(
                    f"row {row_number} immutable evidence differs in {column}"
                )
        judgment, outcome, reviewer, severity = _parse_manual_labels(
            row, row_number, manual_review_schema_version
        )
        response = result["response"]
        first_attempt = response["trace"]["attempts"][0]
        first_judgment = first_attempt.get("judgment")
        entry = {
            "judgment": judgment,
            "outcome": outcome,
            "reviewer": reviewer,
            "severity": severity,
            "component_failure": expected["stop_reason"] == "component_failure",
            "agent_passed": (
                first_judgment.get("passed")
                if isinstance(first_judgment, dict)
                else None
            ),
        }
        entries_by_mode[key[1]].append(entry)
        all_entries.append(entry)

    metrics_by_mode = {
        mode: _score_mode(
            entries, include_severity=manual_review_schema_version == 2
        )
        for mode, entries in entries_by_mode.items()
        if entries
    }
    output = {
        "schema_version": manual_review_schema_version,
        "batch_id": manifest["batch_id"],
        "partial_representative_sample": True,
        "quality_claims_allowed": False,
        "caveat": CAVEAT,
        "provenance": {
            "artifact_file": manifest["artifact_file"],
            "artifact_sha256": manifest["artifact_sha256"],
            "run_id": manifest["run_id"],
            "dataset_id": manifest["dataset_id"],
            "dataset_file": manifest["dataset_file"],
            "dataset_sha256": manifest["dataset_sha256"],
            "artifact_config_sha256": manifest["artifact_config_sha256"],
            "reference_review_sha256": manifest["reference_review_sha256"],
            "source_feedback_sha256": manifest["source_feedback_sha256"],
            "selection_manifest_sha256": _sha256(selection_manifest_path),
            "review_csv_sha256": _sha256(review_csv_path),
        },
        "selected_count": len(keys),
        "completed_count": len(keys),
        "overall_metrics": _score_mode(
            all_entries, include_severity=manual_review_schema_version == 2
        ),
        "metrics_by_mode": metrics_by_mode,
    }
    if manual_review_schema_version == 2:
        output["manual_review_schema_version"] = 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and score a completed, hash-pinned manual review CSV."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = score_manual_review(
        artifact_path=args.artifact,
        dataset_path=args.dataset,
        selection_manifest_path=args.selection_manifest,
        review_csv_path=args.review_csv,
        output_path=args.output,
        project_root=args.project_root,
        overwrite=args.overwrite,
    )
    print(
        f"scored {result['completed_count']} representative review rows to "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
