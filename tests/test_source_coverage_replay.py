from __future__ import annotations

from pathlib import Path

from scripts.replay_source_coverage import replay


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_fn_and_tn_source_coverage_replay_without_models() -> None:
    result = replay(
        artifact_path=ROOT
        / "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json",
        review_batches=[
            (
                ROOT
                / "data/manual_reviews/evaluation_v1_batch1_severity/selection.json",
                ROOT
                / (
                    "data/manual_reviews/evaluation_v1_batch1_severity/"
                    "review_labels_with_severity.csv"
                ),
                ROOT
                / "data/manual_reviews/evaluation_v1_batch1_severity/ingestion.json",
            ),
            (
                ROOT / "data/manual_reviews/evaluation_v1_batch2/selection.json",
                ROOT
                / "data/manual_reviews/evaluation_v1_batch2/review_labels_reviewed.csv",
                ROOT / "data/manual_reviews/evaluation_v1_batch2/ingestion.json",
            ),
        ],
        project_root=ROOT,
    )

    assert result["provenance"]["artifact_sha256"] == (
        "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
    )
    assert result["provenance"]["dataset_sha256"] == (
        "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
    )
    assert result["provenance"]["artifact_config_sha256"] == (
        "2a2935ab0c366522aeae3cbbbb7c7e35c1fe96c654b3131bdcca164daaee16b1"
    )
    assert [batch["review_csv_sha256"] for batch in result["provenance"]["batches"]] == [
        "0bec42e8cf02d262687d2b5786a562fe809f9434228f437432cdaa5d4809ad92",
        "337e94f38fb54c365244d7c432f035b7444afa08b1b8745dd9e8fe3313b83c5d",
    ]
    assert result["source_selected_count"] == 20
    assert result["source_component_failure_count"] == 1
    assert result["replayed_count"] == 13
    assert result["distinct_source_case_count"] == 8
    assert result["baseline_false_negative_count"] == 7
    assert result["baseline_true_negative_count"] == 6
    assert result["coverage_confusion"] == {
        "true_positive": 6,
        "true_negative": 6,
        "false_positive": 0,
        "false_negative": 1,
        "denominator": 13,
        "accuracy_pct": 92.3076923076923,
        "revision_recall_pct": 85.71428571428571,
        "true_negative_rate_pct": 100.0,
    }
    assert result["recovery_action_counts"] == {
        "must_preserve": 3,
        "targeted_rag": 3,
    }

    missed = [
        row["review_key"]
        for row in result["rows"]
        if row["human_needs_revision"] and not row["coverage_requests_revision"]
    ]
    assert missed == ["evaluation-v1-009::agent_rag"]
    assert all(
        not row["coverage_requests_revision"]
        for row in result["rows"]
        if row["baseline_group"] == "true_negative"
    )
    assert result["quality_claims_allowed"] is False
