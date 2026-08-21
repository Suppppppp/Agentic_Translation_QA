from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_severity_supplement import (
    SUPPLEMENT_COLUMNS,
    prepare_severity_supplement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    selection = tmp_path / "data" / "batch1" / "selection.json"
    review = tmp_path / "data" / "batch1" / "review_labels.csv"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "batch1",
                "partial_representative_sample": True,
                "artifact_file": "artifacts/frozen.json",
                "artifact_sha256": "a" * 64,
                "artifact_config_sha256": "b" * 64,
                "run_id": "run-1",
                "dataset_id": "evaluation",
                "dataset_file": "data/evaluation.jsonl",
                "dataset_sha256": "c" * 64,
                "reference_review_sha256": "d" * 64,
                "source_feedback_sha256": "e" * 64,
                "selected": [
                    {
                        "case_id": "case-1",
                        "mode": "agent",
                        "selection_reason": "retry contrast",
                    },
                    {
                        "case_id": "case-2",
                        "mode": "agent_rag",
                        "selection_reason": "pass contrast",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fieldnames = [
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
        "reference_text",
        "initial_translation",
        "final_translation",
        "agent_initial_passed",
        "agent_initial_error_types",
        "agent_initial_summary",
        "retry_count",
        "stop_reason",
    ]
    rows = [
        {
            "review_key": "case-1::agent",
            "case_id": "case-1",
            "mode": "agent",
            "manual_initial_needs_revision": "true",
            "manual_primary_error": "meaning",
            "manual_error_types": '["meaning","term"]',
            "pairwise_outcome": "improved",
            "review_status": "confirmed",
            "reviewer": "Sup",
            "note": "사람이 확정한 기존 라벨, 그대로 복사",
            "source_text": "원문, 1",
            "reference_text": "Reference 1",
            "initial_translation": "Initial, unchanged",
            "final_translation": "Final 1",
            "agent_initial_passed": "False",
            "agent_initial_error_types": '["meaning"]',
            "agent_initial_summary": "summary",
            "retry_count": "1",
            "stop_reason": "passed",
        },
        {
            "review_key": "case-2::agent_rag",
            "case_id": "case-2",
            "mode": "agent_rag",
            "manual_initial_needs_revision": "false",
            "manual_primary_error": "",
            "manual_error_types": "",
            "pairwise_outcome": "same",
            "review_status": "confirmed",
            "reviewer": "Sup",
            "note": "",
            "source_text": "원문 2",
            "reference_text": "Reference 2",
            "initial_translation": "Initial 2",
            "final_translation": "Initial 2",
            "agent_initial_passed": "True",
            "agent_initial_error_types": "[]",
            "agent_initial_summary": "summary 2",
            "retry_count": "0",
            "stop_reason": "passed",
        },
    ]
    with review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return selection, review


def test_prepare_supplement_copies_only_declared_evidence_and_blanks_new_labels(
    tmp_path: Path,
) -> None:
    selection, review = _fixture(tmp_path)
    output_dir = tmp_path / "data" / "severity"
    output_selection = output_dir / "selection.json"
    output_csv = output_dir / "severity_labels.csv"

    manifest, count = prepare_severity_supplement(
        base_selection_path=selection,
        base_review_csv_path=review,
        output_selection_path=output_selection,
        output_csv_path=output_csv,
        project_root=tmp_path,
        batch_id="batch1-severity",
    )

    assert count == 2
    assert manifest["schema_version"] == 1
    assert manifest["manual_review_schema_version"] == 2
    assert manifest["batch_id"] == "batch1-severity"
    assert manifest["review_kind"] == "severity_supplement"
    assert manifest["supplements_batch_id"] == "batch1"
    assert manifest["base_selection_sha256"] == _sha256(selection)
    assert manifest["base_review_csv_sha256"] == _sha256(review)
    assert manifest["review_template_sha256"] == _sha256(output_csv)
    assert manifest["severity_contract"] == {
        "allowed_values": ["MAJOR", "MINOR"],
        "required_when": "manual_initial_needs_revision=true",
        "must_be_blank_when": "manual_initial_needs_revision=false",
        "reviewer_required_when_severity_present": True,
        "automatic_labels_generated": False,
    }

    with output_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == list(SUPPLEMENT_COLUMNS)
    assert [row["review_key"] for row in rows] == [
        "case-1::agent",
        "case-2::agent_rag",
    ]
    assert all(row["manual_severity"] == "" for row in rows)
    assert all(row["severity_reviewer"] == "" for row in rows)
    assert all(row["severity_note"] == "" for row in rows)
    assert rows[0]["source_text"] == "원문, 1"
    assert rows[0]["initial_translation"] == "Initial, unchanged"
    assert rows[0]["manual_error_types"] == '["meaning","term"]'
    assert rows[1]["manual_initial_needs_revision"] == "false"


def test_prepare_supplement_refuses_overwrite_without_touching_existing_file(
    tmp_path: Path,
) -> None:
    selection, review = _fixture(tmp_path)
    output_selection = tmp_path / "severity" / "selection.json"
    output_csv = tmp_path / "severity" / "severity_labels.csv"
    output_csv.parent.mkdir()
    output_csv.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_severity_supplement(
            base_selection_path=selection,
            base_review_csv_path=review,
            output_selection_path=output_selection,
            output_csv_path=output_csv,
            project_root=tmp_path,
        )

    assert output_csv.read_text(encoding="utf-8") == "keep me"
    assert not output_selection.exists()


def test_prepare_supplement_rejects_reordered_or_incomplete_base_review(
    tmp_path: Path,
) -> None:
    selection, review = _fixture(tmp_path)
    with review.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows.reverse()
    with review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="keys/order"):
        prepare_severity_supplement(
            base_selection_path=selection,
            base_review_csv_path=review,
            output_selection_path=tmp_path / "out" / "selection.json",
            output_csv_path=tmp_path / "out" / "severity_labels.csv",
            project_root=tmp_path,
        )

    assert not (tmp_path / "out").exists()


def test_prepare_supplement_rejects_outside_project_output_without_outputs(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    selection, review = _fixture(project_root)
    output_selection = project_root / "severity" / "selection.json"
    output_csv = tmp_path / "outside" / "severity_labels.csv"

    with pytest.raises(ValueError, match="review_template_file must be inside"):
        prepare_severity_supplement(
            base_selection_path=selection,
            base_review_csv_path=review,
            output_selection_path=output_selection,
            output_csv_path=output_csv,
            project_root=project_root,
        )

    assert not output_selection.exists()
    assert not output_csv.exists()
    assert not output_selection.parent.exists()
    assert not output_csv.parent.exists()


def test_tracked_batch1_severity_template_is_blank_and_pinned_to_originals() -> None:
    supplement_dir = (
        PROJECT_ROOT
        / "data"
        / "manual_reviews"
        / "evaluation_v1_batch1_severity"
    )
    base_dir = (
        PROJECT_ROOT / "data" / "manual_reviews" / "evaluation_v1_batch1"
    )
    manifest_path = supplement_dir / "selection.json"
    template_path = supplement_dir / "severity_labels.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["manual_review_schema_version"] == 2
    assert manifest["review_kind"] == "severity_supplement"
    assert manifest["base_selection_sha256"] == _sha256(
        base_dir / "selection.json"
    )
    assert manifest["base_review_csv_sha256"] == _sha256(
        base_dir / "review_labels.csv"
    )
    assert manifest["review_template_sha256"] == _sha256(template_path)
    assert manifest["severity_contract"]["automatic_labels_generated"] is False

    with template_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    with (base_dir / "review_labels.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        base_rows = list(csv.DictReader(stream))
    assert reader.fieldnames == list(SUPPLEMENT_COLUMNS)
    assert len(rows) == 10
    assert all(row["manual_severity"] == "" for row in rows)
    assert all(row["severity_reviewer"] == "" for row in rows)
    assert all(row["severity_note"] == "" for row in rows)
    copied_columns = (
        "review_key",
        "case_id",
        "mode",
        "manual_initial_needs_revision",
        "source_text",
        "initial_translation",
        "manual_primary_error",
        "manual_error_types",
    )
    assert [
        tuple(row[column] for column in copied_columns) for row in rows
    ] == [
        tuple(row[column] for column in copied_columns) for row in base_rows
    ]
