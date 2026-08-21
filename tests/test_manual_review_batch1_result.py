from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_manual_review import score_manual_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_ROOT / "data" / "manual_reviews" / "evaluation_v1_batch1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmed_batch1_reproduces_tracked_offline_scores(tmp_path: Path) -> None:
    review_csv = BATCH_DIR / "review_labels.csv"
    tracked_scores = json.loads((BATCH_DIR / "scores.json").read_text(encoding="utf-8"))
    ingestion = json.loads((BATCH_DIR / "ingestion.json").read_text(encoding="utf-8"))

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
    assert ingestion["reviewer"] == "Sup"
    assert ingestion["ready_count"] == 10
    assert ingestion["confirmed_count"] == 10
    assert ingestion["immutable_evidence_verified"] is True
    assert ingestion["review_csv_sha256_after"] == _sha256(review_csv)
    assert reproduced["provenance"]["review_csv_sha256"] == ingestion[
        "review_csv_sha256_after"
    ]
    assert reproduced["provenance"]["selection_manifest_sha256"] == ingestion[
        "selection_manifest_sha256"
    ]
    assert reproduced["provenance"]["artifact_sha256"] == (
        "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
    )
    assert reproduced["provenance"]["dataset_sha256"] == (
        "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
    )
    assert reproduced["provenance"]["reference_review_sha256"] == (
        "9e5eefc70976104e4e77ee79a1401ea87a511d1f0d25b83ec9a2e6acbeace2ba"
    )
    assert reproduced["provenance"]["source_feedback_sha256"] == (
        "099b40811f3d3c9453c8c33e60f7d163d0ea84d2d529ff986ecdc65458d01e7b"
    )
    assert ingestion["source_workbook_sha256"] == (
        "f61a06f1d7cefd0183b250040eb47b08a6c4f24d4e28d8445cdac8b2f3d438c5"
    )

    overall = reproduced["overall_metrics"]
    assert overall["confirmed"] == 10
    assert overall["unscorable"] == 1
    assert overall["component_failure_count"] == 1
    assert overall["agent_judgment"]["denominator"] == 9
    assert overall["agent_judgment"]["confusion"] == {
        "true_positive": 6,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 3,
    }
    assert overall["agent_judgment"]["accuracy_pct"] == pytest.approx(66.6666667)
    assert overall["successful_correction"] == {
        "eligible": 10,
        "improved": 5,
        "rate_pct": 50.0,
    }
    assert overall["confirmed_outcome_counts"] == {
        "improved": 5,
        "same": 4,
        "worse": 1,
    }

    agent = reproduced["metrics_by_mode"]["agent"]
    agent_rag = reproduced["metrics_by_mode"]["agent_rag"]
    assert agent["agent_judgment"]["accuracy_pct"] == 80.0
    assert agent["successful_correction"]["rate_pct"] == 80.0
    assert agent["component_failure_count"] == 0
    assert agent_rag["agent_judgment"]["accuracy_pct"] == 50.0
    assert agent_rag["successful_correction"]["rate_pct"] == 20.0
    assert agent_rag["component_failure_count"] == 1

    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["reviewer"] for row in rows} == {"Sup"}
    assert {row["review_status"] for row in rows} == {"confirmed"}
    component_failure_rows = [
        row for row in rows if row["stop_reason"] == "component_failure"
    ]
    assert [row["review_key"] for row in component_failure_rows] == [
        "evaluation-v1-024::agent_rag"
    ]
