from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.replay_judgment_consistency import _is_component_failure, replay
from translation_qa.judgment import (
    canonicalize_llm_judgment,
    summary_explicitly_reports_error,
    summary_explicitly_reports_pass,
)
from translation_qa.schemas import (
    JudgmentConsistencyIssue,
    NextAction,
    QualityJudgment,
    TranslationErrorType,
)


ROOT = Path(__file__).resolve().parents[1]


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "passed": True,
        "quality_score": 0.8,
        "error_types": [],
        "summary": "The translation accurately conveys the source without errors.",
        "confidence": 0.9,
        "next_action": "accept",
        "suggested_query_terms": [],
    }
    payload.update(overrides)
    return payload


def test_structured_errors_override_reported_pass_and_accept() -> None:
    judgment = canonicalize_llm_judgment(
        _payload(
            error_types=["meaning"],
            summary="The candidate contains a meaning error.",
        )
    )

    assert not judgment.passed
    assert judgment.error_types == [TranslationErrorType.MEANING]
    assert judgment.next_action is NextAction.REVISE
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.reported_error_types == ["meaning"]
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH,
        JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH,
    ]


def test_empty_structured_errors_override_reported_failure_and_revision() -> None:
    judgment = canonicalize_llm_judgment(
        _payload(
            passed=False,
            next_action="revise",
            summary="The candidate is acceptable.",
        )
    )

    assert judgment.passed
    assert judgment.error_types == []
    assert judgment.next_action is NextAction.ACCEPT
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH,
        JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH,
    ]


def test_unknown_error_is_preserved_as_other_before_decision() -> None:
    judgment = canonicalize_llm_judgment(
        _payload(
            error_types=["pronoun"],
            summary="The pronoun choice is an error.",
        )
    )

    assert not judgment.passed
    assert judgment.error_types == [TranslationErrorType.OTHER]
    assert judgment.next_action is NextAction.REVISE
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.reported_error_types == ["pronoun"]
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH,
        JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH,
        JudgmentConsistencyIssue.UNSUPPORTED_ERROR_TYPE_NORMALIZED,
    ]


@pytest.mark.parametrize(
    "reported_action",
    [
        NextAction.REVISE,
        NextAction.RETRY_WITH_RAG,
        NextAction.RETRY_WITH_CONSTRAINTS,
        NextAction.STOP,
    ],
)
def test_non_accept_strategy_is_preserved_after_error_decision(
    reported_action: NextAction,
) -> None:
    judgment = canonicalize_llm_judgment(
        _payload(
            passed=False,
            error_types=["term", "term"],
            summary="A required term is missing.",
            next_action=reported_action.value,
        )
    )

    assert not judgment.passed
    assert judgment.error_types == [TranslationErrorType.TERM]
    assert judgment.next_action is reported_action
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.consistency_issues == []


def test_optional_query_terms_remain_backward_compatible() -> None:
    payload = _payload()
    del payload["suggested_query_terms"]

    judgment = canonicalize_llm_judgment(payload)

    assert judgment.passed
    assert judgment.suggested_query_terms == []


def test_code_added_error_does_not_hide_raw_llm_contradictions() -> None:
    judgment = canonicalize_llm_judgment(
        _payload(
            passed=False,
            error_types=[],
            summary="A required term is missing.",
            next_action="revise",
        ),
        additional_error_types=[TranslationErrorType.TERM],
        forced_error_action=NextAction.REVISE,
    )

    assert not judgment.passed
    assert judgment.error_types == [TranslationErrorType.TERM]
    assert judgment.decision_audit is not None
    assert judgment.decision_audit.reported_error_types == []
    assert judgment.decision_audit.code_added_error_types == [
        TranslationErrorType.TERM
    ]
    assert judgment.decision_audit.consistency_issues == [
        JudgmentConsistencyIssue.REPORTED_PASSED_MISMATCH,
        JudgmentConsistencyIssue.REPORTED_NEXT_ACTION_MISMATCH,
        JudgmentConsistencyIssue.SUMMARY_ERROR_WITHOUT_STRUCTURED_ERROR,
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"passed": "false"},
        {"error_types": "meaning"},
        {"error_types": [1]},
        {"next_action": "approve"},
    ],
)
def test_raw_redundant_claims_and_error_list_are_strict(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        canonicalize_llm_judgment(_payload(**overrides))


def test_structured_error_list_is_required() -> None:
    payload = _payload()
    del payload["error_types"]

    with pytest.raises(ValidationError):
        canonicalize_llm_judgment(payload)


def test_public_schema_rejects_failure_without_structured_error() -> None:
    with pytest.raises(ValidationError, match="passed must be derived"):
        QualityJudgment(
            passed=False,
            quality_score=0.5,
            error_types=[],
            summary="The model requested revision without an error.",
            confidence=0.5,
            next_action=NextAction.REVISE,
        )


def test_component_failure_stop_reason_excludes_cached_judgment() -> None:
    assert _is_component_failure(
        {
            "stop_reason": "component_failure",
            "attempts": [{"judgment": {"passed": True}}],
        }
    )


@pytest.mark.parametrize(
    "summary",
    [
        "The translation has a minor error instead of the required term.",
        "The translation adds text which isn't in the original source.",
        "The verb is ambiguous in this context.",
    ],
)
def test_summary_audit_detects_explicit_defect_claims(summary: str) -> None:
    assert summary_explicitly_reports_error(summary)


@pytest.mark.parametrize(
    "summary",
    [
        "The translation accurately conveys the original meaning without errors.",
        "The wording has minor redundancy and could be improved for clarity.",
        "The translation is mostly accurate with minor adjustments for precision.",
    ],
)
def test_summary_audit_does_not_treat_optional_polish_as_blocking_error(
    summary: str,
) -> None:
    assert not summary_explicitly_reports_error(summary)


@pytest.mark.parametrize(
    "summary",
    [
        "No terms are missing.",
        "There is no omission.",
        "No word is omitted.",
        "The wording is not ambiguous.",
        "The term is not incorrect, awkward, or wrong.",
    ],
)
def test_summary_error_audit_respects_negation(summary: str) -> None:
    assert not summary_explicitly_reports_error(summary)


@pytest.mark.parametrize(
    "summary",
    [
        "The candidate is not acceptable.",
        "The translation does not accurately convey the source.",
        "This is not high quality.",
    ],
)
def test_summary_pass_audit_respects_negation(summary: str) -> None:
    assert not summary_explicitly_reports_pass(summary)


def test_frozen_manual_batches_replay_without_models_or_decision_changes() -> None:
    result = replay(
        artifact_path=ROOT
        / "artifacts/benchmark-429d6c4a-a1b6-4514-bfd3-dab6966c4101.json",
        review_batches=[
            (
                ROOT
                / "data/manual_reviews/evaluation_v1_batch1_severity/selection.json",
                ROOT
                / "data/manual_reviews/evaluation_v1_batch1_severity/review_labels_with_severity.csv",
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
    assert result["provenance"]["run_id"] == (
        "429d6c4a-a1b6-4514-bfd3-dab6966c4101"
    )
    assert result["provenance"]["dataset_id"] == "evaluation_v1"
    assert result["provenance"]["dataset_sha256"] == (
        "cdef83d7a78de9071cb850890c1c408cbf0573e5501a21086026f232501e6650"
    )
    assert result["provenance"]["artifact_config_sha256"] == (
        "2a2935ab0c366522aeae3cbbbb7c7e35c1fe96c654b3131bdcca164daaee16b1"
    )
    assert [batch["selection_sha256"] for batch in result["provenance"]["batches"]] == [
        "b4ae590193c06dc0bf37c54889bde640314955f3bc0fc33fd05535019efca515",
        "9e96a2356ecc3554327e339f1c0d6127fa71d7d6a4ae8ee9ac029a5b2dadbfaa",
    ]
    assert [batch["review_csv_sha256"] for batch in result["provenance"]["batches"]] == [
        "0bec42e8cf02d262687d2b5786a562fe809f9434228f437432cdaa5d4809ad92",
        "337e94f38fb54c365244d7c432f035b7444afa08b1b8745dd9e8fe3313b83c5d",
    ]
    assert [batch["ingestion_sha256"] for batch in result["provenance"]["batches"]] == [
        "aafdae7c9496fb947f5a2286f5b9557fcd5a5faa82981160563228807e3c4543",
        "80b3b3b58d74bcf2c0507e71158e4cdbea19af2904f0657ae88df7a55d2b61a2",
    ]
    assert result["selected_count"] == 20
    assert result["scorable_judgment_count"] == 19
    assert result["component_failure_count"] == 1
    assert result["decision_change_count"] == 0
    assert result["next_action_change_count"] == 0
    assert result["before"] == result["after"]
    assert result["before"]["true_positive"] == 6
    assert result["before"]["true_negative"] == 6
    assert result["before"]["false_positive"] == 0
    assert result["before"]["false_negative"] == 7
    assert [item["review_key"] for item in result["summary_conflicts"]] == [
        "evaluation-v1-009::agent_rag",
        "evaluation-v1-019::agent_rag",
        "evaluation-v1-027::agent",
    ]
    assert result["consistency_issue_counts"] == {
        "reported_passed_mismatch": 0,
        "reported_next_action_mismatch": 0,
        "summary_error_without_structured_error": 3,
        "summary_pass_with_structured_error": 0,
        "unsupported_error_type_normalized": 0,
    }
