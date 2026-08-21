from __future__ import annotations

from pathlib import Path

import scripts.replay_alias_detector as replay_module
from scripts.replay_alias_detector import _source_metadata, replay


ROOT = Path(__file__).resolve().parents[1]


def _run_replay():
    return replay(
        artifact_path=ROOT
        / "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json",
        dataset_path=ROOT / "data/evaluation_v1.jsonl",
        glossary_paths=[
            ROOT / "data/glossary_evaluation_v2.csv",
            ROOT / "data/glossary_alias_v1.csv",
        ],
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


def test_source_projection_drops_reference_and_manual_gold() -> None:
    projected = _source_metadata(
        {
            "case_id": "case-1",
            "source_text": "로드 밸런싱은 중요하다.",
            "domain": "software",
            "reference_text": "SECRET_REFERENCE",
            "manual_judgments": {"agent": "SECRET_GOLD"},
        }
    )

    assert projected == {
        "case_id": "case-1",
        "source_text": "로드 밸런싱은 중요하다.",
        "domain": "software",
    }
    assert "SECRET_REFERENCE" not in repr(projected)
    assert "SECRET_GOLD" not in repr(projected)


def test_frozen_alias_detector_replay_has_expected_narrow_contribution() -> None:
    result = _run_replay()

    assert result["provenance"]["artifact_sha256"] == (
        "e0f858ca97ff8d0130e016660f037b421cce0b7c5fdb7721a2fae50c319590cf"
    )
    assert result["provenance"]["dataset_sha256"] == (
        "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
    )
    assert [item["sha256"] for item in result["provenance"]["glossaries"]] == [
        "476f0a3476e24c2b0bf66fae4e3e22dc97b4018dda91d68ef66d1c03d506a1d4",
        "0d47c385d7d8b85a2f117bd029b11c7e6f2888ad6942da4142f417145fa7b673",
    ]
    assert result["frozen_replay_count"] == 13
    assert result["frozen_human_revision_count"] == 7
    assert result["frozen_human_pass_count"] == 6
    assert [row["review_key"] for row in result["rows"]] == [
        "evaluation-v1-009::agent_rag",
        "evaluation-v1-019::agent_rag",
        "evaluation-v1-027::agent",
        "evaluation-v1-008::agent",
        "evaluation-v1-008::agent_rag",
        "evaluation-v1-012::agent",
        "evaluation-v1-012::agent_rag",
        "evaluation-v1-013::agent",
        "evaluation-v1-013::agent_rag",
        "evaluation-v1-015::agent",
        "evaluation-v1-015::agent_rag",
        "evaluation-v1-038::agent",
        "evaluation-v1-038::agent_rag",
    ]
    assert result["detector_component_failure_count"] == 0
    assert result["alias_only_confusion"] == {
        "true_positive": 1,
        "true_negative": 6,
        "false_positive": 0,
        "false_negative": 6,
        "denominator": 13,
        "accuracy_pct": 53.84615384615385,
        "revision_recall_pct": 14.285714285714285,
        "component_failure_count": 0,
    }
    assert result["combined_confusion"] == {
        "true_positive": 7,
        "true_negative": 6,
        "false_positive": 0,
        "false_negative": 0,
        "denominator": 13,
        "accuracy_pct": 100.0,
        "revision_recall_pct": 100.0,
        "component_failure_count": 0,
    }
    assert result["incremental_alias_recovery_keys"] == [
        "evaluation-v1-009::agent_rag"
    ]
    assert result["tn_alias_false_positive_keys"] == []
    assert result["acceptance"]["all_passed"] is True

    case_009 = next(
        row
        for row in result["rows"]
        if row["review_key"] == "evaluation-v1-009::agent_rag"
    )
    assert case_009["alias_requests_revision"] is True
    finding = next(
        item
        for item in case_009["alias_findings"]
        if item["alias_id"] == "sw-load-balancing-transliteration"
    )
    assert finding["status"] == "error"
    assert finding["evidence_kind"] == "confusable_variant"
    assert finding["matched_variants"] == ["road balance"]


def test_reserve_pair_checks_old_disallowed_and_accepted_aliases() -> None:
    reserve = _run_replay()["reserve"]

    assert reserve["reviewed_count"] == 2
    by_key = {row["review_key"]: row for row in reserve["rows"]}
    assert by_key["evaluation-v1-004::agent"]["alias_requests_revision"] is True
    assert (
        by_key["evaluation-v1-004::agent_rag"]["alias_requests_revision"]
        is False
    )
    rag_deployment = next(
        finding
        for finding in by_key["evaluation-v1-004::agent_rag"]["alias_findings"]
        if finding["alias_id"] == "sw-deployment"
    )
    assert rag_deployment["status"] == "verified"


def test_replay_guardrails_show_blind_detector_only_ordering() -> None:
    result = _run_replay()
    guardrails = result["guardrails"]

    assert guardrails["reference_exposed_to_detector"] is False
    assert guardrails["manual_labels_loaded_after_all_detector_decisions"] is True
    assert guardrails["judge_called"] is False
    assert guardrails["translator_called"] is False
    assert guardrails["retriever_called"] is False
    assert guardrails["reviser_called"] is False
    assert guardrails["retry_pipeline_called"] is False
    assert result["automatic_gold_labels_generated"] is False
    assert result["quality_claims_allowed"] is False


def test_review_csv_is_not_opened_until_all_selected_decisions_exist(
    monkeypatch,
) -> None:
    detection_count = 0
    original_detect = replay_module.AliasDetector.detect
    original_load_csv = replay_module._load_csv

    def counted_detect(self, **kwargs):
        nonlocal detection_count
        detection_count += 1
        return original_detect(self, **kwargs)

    def guarded_load_csv(path):
        assert detection_count == 20
        return original_load_csv(path)

    monkeypatch.setattr(replay_module.AliasDetector, "detect", counted_detect)
    monkeypatch.setattr(replay_module, "_load_csv", guarded_load_csv)

    result = _run_replay()

    assert detection_count == 20
    assert result["detector_component_failure_count"] == 0
