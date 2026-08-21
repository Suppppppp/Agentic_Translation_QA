from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.ingest_manual_review_workbook import read_review_worksheet
from scripts.score_manual_review import score_manual_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_ROOT / "data" / "manual_reviews" / "evaluation_v1_batch2"
REVIEWED_WORKBOOK = (
    PROJECT_ROOT
    / "outputs"
    / "01a02074-03fa-7530-a959-1f4de7b5c360"
    / "evaluation_v1_manual_review_batch2_sup_reviewed.xlsx"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_batch2_sources_and_frozen_evidence_are_unchanged() -> None:
    protected_hashes = {
        BATCH_DIR / "selection.json": (
            "9e96a2356ecc3554327e339f1c0d6127fa71d7d6a4ae8ee9ac029a5b2dadbfaa"
        ),
        BATCH_DIR / "review_labels.csv": (
            "b054fc87425e967c71d705e194833286e497c5e981a258ca8fe46f23670e30bb"
        ),
        PROJECT_ROOT
        / "outputs/01a01f71-e0d2-7250-b573-d6bc5d0b1c59"
        / "evaluation_v1_manual_review_batch2.xlsx": (
            "e54a6f94651d2a1af06bf5c31a9224e5ceea4ad911a9f5b54946dbad08d2a84e"
        ),
        PROJECT_ROOT
        / "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json": (
            "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
        ),
        PROJECT_ROOT / "data/evaluation_v1.jsonl": (
            "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
        ),
        PROJECT_ROOT / "data/reference_reviews/evaluation_v1.json": (
            "9e5eefc70976104e4e77ee79a1401ea87a511d1f0d25b83ec9a2e6acbeace2ba"
        ),
    }
    assert {path: _sha256(path) for path in protected_hashes} == protected_hashes


def test_batch2_reviewed_workbook_and_ingest_preserve_order_and_provenance() -> None:
    manifest = json.loads((BATCH_DIR / "selection.json").read_text(encoding="utf-8"))
    ingestion = json.loads((BATCH_DIR / "ingestion.json").read_text(encoding="utf-8"))
    reviewed_csv = BATCH_DIR / "review_labels_reviewed.csv"
    csv_rows = _csv_rows(reviewed_csv)
    workbook_headers, workbook_rows, source_range = read_review_worksheet(
        REVIEWED_WORKBOOK, "Review"
    )
    expected_keys = [
        (f"{item['case_id']}::{item['mode']}", item["case_id"], item["mode"])
        for item in manifest["selected"]
    ]

    assert _sha256(REVIEWED_WORKBOOK) == (
        "26fb34a945d88f31b0c06e66d0a1f0fbdb73a2600d6ba1076a7abbdf27084acb"
    )
    assert _sha256(reviewed_csv) == (
        "337e94f38fb54c365244d7c432f035b7444afa08b1b8745dd9e8fe3313b83c5d"
    )
    assert len(workbook_rows) == len(csv_rows) == len(expected_keys) == 10
    assert workbook_headers[-1] == "row_status"
    assert source_range == "A1:U11"
    assert [
        (row["review_key"], row["case_id"], row["mode"])
        for row in workbook_rows
    ] == expected_keys
    assert [
        (row["review_key"], row["case_id"], row["mode"]) for row in csv_rows
    ] == expected_keys
    assert {row["row_status"] for row in workbook_rows} == {"READY"}
    assert {row["reviewer"] for row in workbook_rows} == {"Sup"}
    assert {row["reviewer"] for row in csv_rows} == {"Sup"}
    assert {row["review_status"] for row in csv_rows} == {"confirmed"}

    needs_revision = [
        row for row in csv_rows if row["manual_initial_needs_revision"] == "true"
    ]
    accepted = [
        row for row in csv_rows if row["manual_initial_needs_revision"] == "false"
    ]
    assert len(needs_revision) == 4
    assert len(accepted) == 6
    assert {row["manual_severity"] for row in needs_revision} == {"MAJOR"}
    assert all(
        not row[column]
        for row in accepted
        for column in (
            "manual_severity",
            "manual_primary_error",
            "manual_error_types",
        )
    )
    assert {row["pairwise_outcome"] for row in csv_rows} == {"same"}

    assert ingestion["source_workbook_sha256"] == _sha256(REVIEWED_WORKBOOK)
    assert ingestion["selection_manifest_sha256"] == _sha256(
        BATCH_DIR / "selection.json"
    )
    assert ingestion["baseline_csv_sha256"] == _sha256(
        BATCH_DIR / "review_labels.csv"
    )
    assert ingestion["output_csv_sha256"] == _sha256(reviewed_csv)
    assert ingestion["immutable_values_verified"] is True
    assert ingestion["automatic_severity_labels_generated"] is False
    assert ingestion["ready_count"] == ingestion["selected_count"] == 10
    assert ingestion["severity_counts"] == {"MAJOR": 4, "MINOR": 0}


def test_batch2_offline_scores_are_reproducible_and_match_expected(
    tmp_path: Path,
) -> None:
    tracked_scores_path = BATCH_DIR / "scores.json"
    tracked_scores = json.loads(tracked_scores_path.read_text(encoding="utf-8"))
    reproduced = score_manual_review(
        artifact_path=(
            PROJECT_ROOT
            / "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json"
        ),
        dataset_path=PROJECT_ROOT / "data/evaluation_v1.jsonl",
        selection_manifest_path=BATCH_DIR / "selection.json",
        review_csv_path=BATCH_DIR / "review_labels_reviewed.csv",
        output_path=tmp_path / "scores.json",
        project_root=PROJECT_ROOT,
    )

    assert reproduced == tracked_scores
    assert _sha256(tracked_scores_path) == (
        "48f53f4c25bf353b10065f4a13ec9dc208ccc03307cbef261e4c737f27f7b0bc"
    )
    assert reproduced["partial_representative_sample"] is True
    assert reproduced["quality_claims_allowed"] is False
    assert reproduced["selected_count"] == reproduced["completed_count"] == 10
    assert reproduced["provenance"]["artifact_sha256"] == (
        "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
    )
    assert reproduced["provenance"]["dataset_sha256"] == (
        "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
    )
    assert reproduced["provenance"]["review_csv_sha256"] == _sha256(
        BATCH_DIR / "review_labels_reviewed.csv"
    )

    overall = reproduced["overall_metrics"]
    assert overall["confirmed"] == 10
    assert overall["unscorable"] == 0
    assert overall["component_failure_count"] == 0
    assert overall["agent_judgment"] == {
        "denominator": 10,
        "confusion": {
            "true_positive": 0,
            "true_negative": 6,
            "false_positive": 0,
            "false_negative": 4,
        },
        "accuracy_pct": 60.0,
        "revision_recall_pct": 0.0,
        "unnecessary_revision_rate_pct": 0.0,
    }
    assert overall["successful_correction"] == {
        "eligible": 4,
        "improved": 0,
        "rate_pct": 0.0,
    }
    assert overall["confirmed_severity_counts"] == {"MAJOR": 4, "MINOR": 0}
    assert overall["major_false_pass_count"] == 4
    assert overall["confirmed_outcome_counts"] == {
        "improved": 0,
        "same": 10,
        "worse": 0,
    }

    for mode in ("agent", "agent_rag"):
        metrics = reproduced["metrics_by_mode"][mode]
        assert metrics["selected"] == metrics["confirmed"] == 5
        assert metrics["component_failure_count"] == 0
        assert metrics["agent_judgment"]["denominator"] == 5
        assert metrics["agent_judgment"]["accuracy_pct"] == 60.0
        assert metrics["agent_judgment"]["confusion"] == {
            "true_positive": 0,
            "true_negative": 3,
            "false_positive": 0,
            "false_negative": 2,
        }
        assert metrics["confirmed_severity_counts"] == {"MAJOR": 2, "MINOR": 0}
        assert metrics["major_false_pass_count"] == 2
        assert metrics["successful_correction"] == {
            "eligible": 2,
            "improved": 0,
            "rate_pct": 0.0,
        }


def test_batch2_verification_record_matches_tracked_outputs_and_removed_copy() -> None:
    verification = json.loads(
        (BATCH_DIR / "verification.json").read_text(encoding="utf-8")
    )
    generated = verification["generated_outputs"]
    protected = verification["protected_inputs"]

    assert verification["workflow"] == {
        "model_rerun": False,
        "translation_rerun": False,
        "benchmark_rerun": False,
        "offline_scorer_only": True,
    }
    assert generated["review_csv_sha256"] == _sha256(
        BATCH_DIR / "review_labels_reviewed.csv"
    )
    assert generated["ingestion_sha256"] == _sha256(BATCH_DIR / "ingestion.json")
    assert generated["scores_sha256"] == _sha256(BATCH_DIR / "scores.json")
    assert protected["selection_manifest_sha256"] == _sha256(
        BATCH_DIR / "selection.json"
    )
    assert protected["blank_review_csv_sha256"] == _sha256(
        BATCH_DIR / "review_labels.csv"
    )
    assert protected["benchmark_artifact_sha256"] == _sha256(
        PROJECT_ROOT
        / "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json"
    )
    assert protected["dataset_sha256"] == _sha256(
        PROJECT_ROOT / "data/evaluation_v1.jsonl"
    )
    assert verification["actual_metrics"]["expected_preview_match"] is True
    assert verification["actual_metrics"]["component_failure_count"] == 0
    assert verification["tests"]["regression_test_sha256"] == _sha256(Path(__file__))

    removed = verification["removed_prior_copy"]
    assert removed["removed"] is True
    assert not (PROJECT_ROOT / removed["file"]).exists()
