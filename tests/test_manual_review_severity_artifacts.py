from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.ingest_manual_review_workbook import read_review_worksheet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "01a01f71-e0d2-7250-b573-d6bc5d0b1c59"
)


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def _read_csv(relative_path: str) -> tuple[list[str], list[dict[str, str]]]:
    with (PROJECT_ROOT / relative_path).open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def test_pre_severity_batch1_and_frozen_evidence_hashes_are_unchanged() -> None:
    protected_hashes = {
        "data/manual_reviews/evaluation_v1_batch1/review_labels.csv": (
            "3c2a8cb87bcbd972f66c40a48d1a455446f9b26f61186c0727e516761d0c62df"
        ),
        "data/manual_reviews/evaluation_v1_batch1/selection.json": (
            "301848b0097f7ed94ca6403ae3846831e7eb9e6b656f16cb1030f1db65cc0684"
        ),
        "data/manual_reviews/evaluation_v1_batch1/scores.json": (
            "b1f289738974d6281a30851f7749ec68da9b3fa471c96771395755647ad5dbb3"
        ),
        "data/manual_reviews/evaluation_v1_batch1/ingestion.json": (
            "986dc18ea41585d4767695a759e631d3dc3492e10784ee343e77c796ff87e827"
        ),
        "outputs/01a01c85-3942-7583-bfed-366c3c0b2b2f/"
        "evaluation_v1_manual_review_batch1.xlsx": (
            "09108937f4df0cbd4291f9f7f8e11a66f2035bc22763b7c523525968d0e87821"
        ),
        "outputs/01a01f71-e0d2-7250-b573-d6bc5d0b1c59/"
        "evaluation_v1_manual_review_batch1_sup_reviewed.xlsx": (
            "f61a06f1d7cefd0183b250040eb47b08a6c4f24d4e28d8445cdac8b2f3d438c5"
        ),
        "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json": (
            "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
        ),
        "data/evaluation_v1.jsonl": (
            "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
        ),
    }

    assert {
        path: _sha256(path) for path in protected_hashes
    } == protected_hashes


def test_batch2_v2_csv_and_workbook_contain_no_automatic_human_labels() -> None:
    manifest = json.loads(
        (
            PROJECT_ROOT
            / "data/manual_reviews/evaluation_v1_batch2/selection.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["manual_review_schema_version"] == 2
    assert manifest["review_template_file"] == (
        "data/manual_reviews/evaluation_v1_batch2/review_labels.csv"
    )
    assert manifest["review_template_sha256"] == _sha256(
        manifest["review_template_file"]
    )

    headers, rows = _read_csv(
        "data/manual_reviews/evaluation_v1_batch2/review_labels.csv"
    )
    assert headers[3:11] == [
        "manual_initial_needs_revision",
        "manual_severity",
        "manual_primary_error",
        "manual_error_types",
        "pairwise_outcome",
        "review_status",
        "reviewer",
        "note",
    ]
    assert len(rows) == 10
    assert all(not any(row[column] for column in headers[3:11]) for row in rows)

    workbook_headers, workbook_rows, _ = read_review_worksheet(
        OUTPUT_DIR / "evaluation_v1_manual_review_batch2.xlsx", "Review"
    )
    assert workbook_headers == [*headers, "row_status"]
    assert len(workbook_rows) == 10
    assert all(
        not any(row[column] for column in headers[3:11])
        for row in workbook_rows
    )
    assert {row["row_status"] for row in workbook_rows} == {"PENDING"}


def test_batch1_severity_supplement_is_blank_and_pins_legacy_review() -> None:
    manifest_path = (
        PROJECT_ROOT
        / "data/manual_reviews/evaluation_v1_batch1_severity/selection.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["manual_review_schema_version"] == 2
    assert manifest["review_kind"] == "severity_supplement"
    assert manifest["base_selection_sha256"] == _sha256(
        "data/manual_reviews/evaluation_v1_batch1/selection.json"
    )
    assert manifest["base_review_csv_sha256"] == _sha256(
        "data/manual_reviews/evaluation_v1_batch1/review_labels.csv"
    )
    assert manifest["severity_contract"]["automatic_labels_generated"] is False

    headers, rows = _read_csv(
        "data/manual_reviews/evaluation_v1_batch1_severity/severity_labels.csv"
    )
    assert headers == [
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
    ]
    assert len(rows) == 10
    assert all(row["manual_initial_needs_revision"] == "true" for row in rows)
    assert all(
        not row[column]
        for row in rows
        for column in ("manual_severity", "severity_reviewer", "severity_note")
    )

    workbook_headers, workbook_rows, _ = read_review_worksheet(
        OUTPUT_DIR / "evaluation_v1_manual_review_batch1_severity_supplement.xlsx",
        "Severity Review",
    )
    assert workbook_headers == [*headers, "row_status"]
    assert len(workbook_rows) == 10
    assert all(
        not row[column]
        for row in workbook_rows
        for column in ("manual_severity", "severity_reviewer", "severity_note")
    )
    assert {row["row_status"] for row in workbook_rows} == {"ADD SEVERITY"}

    # Completing the supplement creates separate outputs; the blank template
    # and its original workbook remain immutable inputs to that workflow.
