"""Prepare a blank, separately tracked severity supplement for a reviewed batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


SUPPLEMENT_COLUMNS = (
    "review_key",
    "case_id",
    "mode",
    "manual_initial_needs_revision",
    "manual_severity",
    "severity_reviewer",
    "severity_note",
    "source_text",
    "initial_translation",
    "manual_primary_error",
    "manual_error_types",
)
REQUIRED_BASE_COLUMNS = (
    "review_key",
    "case_id",
    "mode",
    "manual_initial_needs_revision",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
    "source_text",
    "initial_translation",
)
PINNED_SELECTION_FIELDS = (
    "artifact_file",
    "artifact_sha256",
    "artifact_config_sha256",
    "run_id",
    "dataset_id",
    "dataset_file",
    "dataset_sha256",
    "reference_review_sha256",
    "source_feedback_sha256",
)
ALLOWED_ERROR_TYPES = {
    "term",
    "meaning",
    "omission_addition",
    "entity_value",
    "fluency_grammar",
    "other",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _relative_project_path(path: Path, project_root: Path, field: str) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field} must be inside the project root") from exc


def _selection_items(manifest: dict[str, Any]) -> list[dict[str, str]]:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("base selection requires a nonempty selected array")
    items: list[dict[str, str]] = []
    for index, raw in enumerate(selected, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"selected item {index} must be an object")
        case_id = raw.get("case_id")
        mode = raw.get("mode")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"selected item {index} has no case_id")
        if mode not in {"agent", "agent_rag"}:
            raise ValueError(
                f"selected item {index} mode must be agent or agent_rag"
            )
        reason = raw.get("selection_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"selected item {index} selection_reason is invalid")
        items.append(
            {
                "review_key": f"{case_id}::{mode}",
                "case_id": case_id,
                "mode": mode,
                "selection_reason": reason,
            }
        )
    keys = [item["review_key"] for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("base selection contains duplicate review keys")
    return items


def _load_base_review(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read base review CSV: {path}") from exc
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("base review CSV has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("base review CSV contains duplicate headers")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def _parse_error_types(value: str, row_number: int) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"base review row {row_number} manual_error_types is not JSON"
        ) from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise ValueError(
            f"base review row {row_number} manual_error_types must be a string list"
        )
    if len(parsed) != len(set(parsed)):
        raise ValueError(
            f"base review row {row_number} manual_error_types contains duplicates"
        )
    if any(item not in ALLOWED_ERROR_TYPES for item in parsed):
        raise ValueError(
            f"base review row {row_number} manual_error_types contains invalid values"
        )
    return parsed


def _validate_completed_base_row(row: dict[str, str], row_number: int) -> None:
    for field in (
        "manual_initial_needs_revision",
        "pairwise_outcome",
        "review_status",
        "reviewer",
    ):
        if not row[field] or not row[field].strip():
            raise ValueError(f"base review row {row_number} requires {field}")
    needs_revision = row["manual_initial_needs_revision"]
    if needs_revision not in {"true", "false"}:
        raise ValueError(
            f"base review row {row_number} needs_revision must be true or false"
        )
    primary = row["manual_primary_error"]
    errors = _parse_error_types(row["manual_error_types"], row_number)
    if needs_revision == "false" and (primary or errors):
        raise ValueError(f"base review row {row_number} PASS label has errors")
    if needs_revision == "true":
        if primary not in ALLOWED_ERROR_TYPES or not errors:
            raise ValueError(
                f"base review row {row_number} requires existing error labels"
            )
        if primary not in errors:
            raise ValueError(
                f"base review row {row_number} primary error is not in error_types"
            )
    if row["pairwise_outcome"] not in {"improved", "same", "worse"}:
        raise ValueError(f"base review row {row_number} outcome is invalid")
    if row["review_status"] not in {"confirmed", "ambiguous"}:
        raise ValueError(f"base review row {row_number} review_status is invalid")
    note_is_blank = not row["note"] or not row["note"].strip()
    if (primary == "other" or "other" in errors) and note_is_blank:
        raise ValueError(f"base review row {row_number} other requires note")
    if row["review_status"] == "ambiguous" and note_is_blank:
        raise ValueError(f"base review row {row_number} ambiguous requires note")


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=SUPPLEMENT_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def prepare_severity_supplement(
    *,
    base_selection_path: Path,
    base_review_csv_path: Path,
    output_selection_path: Path,
    output_csv_path: Path,
    project_root: Path,
    batch_id: str = "evaluation_v1_batch1_severity",
) -> tuple[dict[str, Any], int]:
    """Create a v2 manifest and a blank severity-only review template."""

    inputs = {base_selection_path.resolve(), base_review_csv_path.resolve()}
    outputs = {output_selection_path.resolve(), output_csv_path.resolve()}
    if len(outputs) != 2:
        raise ValueError("selection and CSV outputs must be different files")
    if inputs & outputs:
        raise ValueError("outputs must not overwrite base selection or base review CSV")
    for output in (output_selection_path, output_csv_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not batch_id or not batch_id.strip():
        raise ValueError("batch_id must be nonblank")

    project_root = project_root.resolve()
    base_selection_file = _relative_project_path(
        base_selection_path, project_root, "base_selection_file"
    )
    base_review_csv_file = _relative_project_path(
        base_review_csv_path, project_root, "base_review_csv_file"
    )
    review_template_file = _relative_project_path(
        output_csv_path, project_root, "review_template_file"
    )
    _relative_project_path(
        output_selection_path, project_root, "output_selection_file"
    )
    base_selection = _load_json_object(base_selection_path, "base selection")
    if base_selection.get("schema_version") != 1:
        raise ValueError("base selection schema_version must be 1")
    base_batch_id = base_selection.get("batch_id")
    if not isinstance(base_batch_id, str) or not base_batch_id:
        raise ValueError("base selection requires batch_id")
    if base_selection.get("partial_representative_sample") is not True:
        raise ValueError("base selection must be a partial representative sample")
    for field in PINNED_SELECTION_FIELDS:
        value = base_selection.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"base selection requires {field}")
    selected = _selection_items(base_selection)

    fieldnames, base_rows = _load_base_review(base_review_csv_path)
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"base review CSV is missing columns: {', '.join(missing)}")
    if "manual_severity" in fieldnames:
        raise ValueError("base review CSV must be the original pre-severity file")
    actual_keys = [
        (row["review_key"], row["case_id"], row["mode"]) for row in base_rows
    ]
    expected_keys = [
        (item["review_key"], item["case_id"], item["mode"])
        for item in selected
    ]
    if actual_keys != expected_keys:
        raise ValueError("base review CSV keys/order do not match base selection")

    supplement_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(base_rows, start=2):
        _validate_completed_base_row(row, row_number)
        supplement_rows.append(
            {
                "review_key": row["review_key"],
                "case_id": row["case_id"],
                "mode": row["mode"],
                "manual_initial_needs_revision": row[
                    "manual_initial_needs_revision"
                ],
                "manual_severity": "",
                "severity_reviewer": "",
                "severity_note": "",
                "source_text": row["source_text"],
                "initial_translation": row["initial_translation"],
                "manual_primary_error": row["manual_primary_error"],
                "manual_error_types": row["manual_error_types"],
            }
        )

    csv_payload = _csv_bytes(supplement_rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manual_review_schema_version": 2,
        "batch_id": batch_id,
        "review_kind": "severity_supplement",
        "supplements_batch_id": base_batch_id,
        "partial_representative_sample": True,
        "base_selection_file": base_selection_file,
        "base_selection_sha256": _sha256(base_selection_path),
        "base_review_csv_file": base_review_csv_file,
        "base_review_csv_sha256": _sha256(base_review_csv_path),
        "review_template_file": review_template_file,
        "review_template_sha256": hashlib.sha256(csv_payload).hexdigest(),
        **{field: base_selection[field] for field in PINNED_SELECTION_FIELDS},
        "severity_contract": {
            "allowed_values": ["MAJOR", "MINOR"],
            "required_when": "manual_initial_needs_revision=true",
            "must_be_blank_when": "manual_initial_needs_revision=false",
            "reviewer_required_when_severity_present": True,
            "automatic_labels_generated": False,
        },
        "selected": selected,
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    created_outputs: list[Path] = []
    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output_selection_path.parent.mkdir(parents=True, exist_ok=True)
        with output_csv_path.open("xb") as stream:
            created_outputs.append(output_csv_path)
            stream.write(csv_payload)
        with output_selection_path.open("xb") as stream:
            created_outputs.append(output_selection_path)
            stream.write(manifest_payload)
    except Exception:
        for output in reversed(created_outputs):
            output.unlink(missing_ok=True)
        raise
    return manifest, len(supplement_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a separately tracked, blank MAJOR/MINOR severity supplement "
            "from an already reviewed batch."
        )
    )
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--base-review-csv", type=Path, required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--batch-id", default="evaluation_v1_batch1_severity"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, count = prepare_severity_supplement(
        base_selection_path=args.base_selection,
        base_review_csv_path=args.base_review_csv,
        output_selection_path=args.output_selection,
        output_csv_path=args.output_csv,
        project_root=args.project_root,
        batch_id=args.batch_id,
    )
    print(f"wrote {count} blank severity rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
