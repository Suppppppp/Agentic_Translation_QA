from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.ingest_manual_review_workbook import read_review_worksheet
from scripts.score_manual_review import score_manual_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = (
    PROJECT_ROOT / "data" / "manual_reviews" / "evaluation_v1_batch1_severity"
)
REVIEWED_WORKBOOK = (
    PROJECT_ROOT
    / "outputs"
    / "01a0205c-c18c-71b2-8bcd-dc64888d032b"
    / "evaluation_v1_manual_review_batch1_severity_supplement_sup_reviewed.xlsx"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completed_batch1_severity_reproduces_tracked_scores(
    tmp_path: Path,
) -> None:
    review_csv = BATCH_DIR / "review_labels_with_severity.csv"
    ingestion = json.loads((BATCH_DIR / "ingestion.json").read_text(encoding="utf-8"))
    tracked_scores = json.loads((BATCH_DIR / "scores.json").read_text(encoding="utf-8"))

    reproduced = score_manual_review(
        artifact_path=(
            PROJECT_ROOT
            / "artifacts"
            / "benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json"
        ),
        dataset_path=PROJECT_ROOT / "data" / "evaluation_v1.jsonl",
        selection_manifest_path=BATCH_DIR / "selection.json",
        review_csv_path=review_csv,
        output_path=tmp_path / "scores.json",
        project_root=PROJECT_ROOT,
    )

    assert reproduced == tracked_scores
    assert ingestion["ingest_kind"] == "severity-supplement"
    assert ingestion["manual_review_schema_version"] == 2
    assert ingestion["selected_count"] == 10
    assert ingestion["ready_count"] == 10
    assert ingestion["severity_required_count"] == 10
    assert ingestion["severity_counts"] == {"MAJOR": 10, "MINOR": 0}
    assert ingestion["automatic_severity_labels_generated"] is False
    assert ingestion["immutable_values_verified"] is True
    assert ingestion["base_review_csv_sha256"] == (
        "3c2a8cb87bcbd972f66c40a48d1a455446f9b26f61186c0727e516761d0c62df"
    )
    assert ingestion["base_selection_sha256"] == (
        "301848b0097f7ed94ca6403ae3846831e7eb9e6b656f16cb1030f1db65cc0684"
    )
    assert ingestion["source_workbook_sha256"] == (
        "2bed78cf44f4715281ec6ff91413b087ac217959de2fbd9dd7d13ccaa32b7cb2"
    )
    assert ingestion["output_csv_sha256"] == _sha256(review_csv)
    assert reproduced["provenance"]["review_csv_sha256"] == _sha256(review_csv)
    assert reproduced["provenance"]["selection_manifest_sha256"] == ingestion[
        "selection_manifest_sha256"
    ]
    assert reproduced["provenance"]["artifact_sha256"] == (
        "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
    )
    assert reproduced["provenance"]["dataset_sha256"] == (
        "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
    )

    workbook_headers, workbook_rows, source_range = read_review_worksheet(
        REVIEWED_WORKBOOK, "Severity Review"
    )
    assert source_range == "A1:L11"
    assert workbook_headers[-1] == "row_status"
    assert len(workbook_rows) == 10
    assert {row["row_status"] for row in workbook_rows} == {"READY"}
    assert {row["severity_reviewer"] for row in workbook_rows} == {"Sup"}
    assert {row["manual_severity"] for row in workbook_rows} == {"MAJOR"}

    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["reviewer"] for row in rows} == {"Sup"}
    assert {row["review_status"] for row in rows} == {"confirmed"}
    assert {row["manual_severity"] for row in rows} == {"MAJOR"}
    component_failure_rows = [
        row for row in rows if row["stop_reason"] == "component_failure"
    ]
    assert [row["review_key"] for row in component_failure_rows] == [
        "evaluation-v1-024::agent_rag"
    ]

    overall = reproduced["overall_metrics"]
    assert overall["confirmed"] == 10
    assert overall["confirmed_severity_counts"] == {"MAJOR": 10, "MINOR": 0}
    assert overall["component_failure_count"] == 1
    assert overall["unscorable"] == 1
    assert overall["agent_judgment"]["denominator"] == 9
    assert overall["agent_judgment"]["confusion"] == {
        "true_positive": 6,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 3,
    }
    assert overall["agent_judgment"]["accuracy_pct"] == pytest.approx(66.6666667)
    assert overall["major_false_pass_count"] == 3
    assert overall["successful_correction"] == {
        "eligible": 10,
        "improved": 5,
        "rate_pct": 50.0,
    }

    agent = reproduced["metrics_by_mode"]["agent"]
    agent_rag = reproduced["metrics_by_mode"]["agent_rag"]
    assert agent["confirmed_severity_counts"] == {"MAJOR": 5, "MINOR": 0}
    assert agent["major_false_pass_count"] == 1
    assert agent["component_failure_count"] == 0
    assert agent["agent_judgment"]["denominator"] == 5
    assert agent_rag["confirmed_severity_counts"] == {"MAJOR": 5, "MINOR": 0}
    assert agent_rag["major_false_pass_count"] == 2
    assert agent_rag["component_failure_count"] == 1
    assert agent_rag["agent_judgment"]["denominator"] == 4

    # The failed component remains in human severity/outcome review but is
    # excluded from the Agent judgment and MAJOR false-pass denominators.
    assert overall["confirmed"] == overall["agent_judgment"]["denominator"] + 1
    assert overall["confirmed_severity_counts"]["MAJOR"] == 10
    assert overall["major_false_pass_count"] == overall["agent_judgment"][
        "confusion"
    ]["false_negative"]
