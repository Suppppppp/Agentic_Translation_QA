from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_manual_review import (
    LEGACY_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    score_manual_review,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "data" / "evaluation.jsonl"
    dataset.parent.mkdir(parents=True)
    reference_hash = "1" * 64
    feedback_hash = "2" * 64
    records = [
        {
            "case_id": "case-1",
            "source_text": "원문 1",
            "reference_text": "Reference 1",
            "reference_provenance": {
                "reference_review_sha256": reference_hash,
                "source_feedback_sha256": feedback_hash,
            },
        },
        {
            "case_id": "case-2",
            "source_text": "원문 2",
            "reference_text": "Reference 2",
            "reference_provenance": {
                "reference_review_sha256": reference_hash,
                "source_feedback_sha256": feedback_hash,
            },
        },
        {
            "case_id": "case-3",
            "source_text": "원문 3",
            "reference_text": "Reference 3",
            "reference_provenance": {
                "reference_review_sha256": reference_hash,
                "source_feedback_sha256": feedback_hash,
            },
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    run_config = {
        "dataset_id": "evaluation",
        "dataset_sha256": _sha(dataset),
        "modes": ["agent", "agent_rag"],
    }
    artifact_data = {
        "run_id": "run-1",
        "dataset_id": "evaluation",
        "dataset_sha256": _sha(dataset),
        "config_sha256": _json_sha(run_config),
        "run_config": run_config,
        "results": [
            {
                "case_id": "case-1",
                "mode": "agent",
                "response": {
                    "source_text": "원문 1",
                    "translation": "Final 1",
                    "trace": {
                        "stop_reason": "passed",
                        "attempts": [
                            {
                                "candidate": {"text": "Initial 1"},
                                "judgment": {"passed": True, "error_types": []},
                            }
                        ],
                    },
                },
            },
            {
                "case_id": "case-2",
                "mode": "agent",
                "response": {
                    "source_text": "원문 2",
                    "translation": "Final 2",
                    "trace": {
                        "stop_reason": "passed",
                        "attempts": [
                            {
                                "candidate": {"text": "Initial 2"},
                                "judgment": {
                                    "passed": False,
                                    "error_types": ["meaning"],
                                },
                            },
                            {"candidate": {"text": "Final 2"}},
                        ],
                    },
                },
            },
            {
                "case_id": "case-3",
                "mode": "agent_rag",
                "response": {
                    "source_text": "원문 3",
                    "translation": "Initial 3",
                    "trace": {
                        "stop_reason": "component_failure",
                        "attempts": [
                            {
                                "candidate": {"text": "Initial 3"},
                                "judgment": {"passed": True, "error_types": []},
                            }
                        ],
                    },
                },
            },
        ],
    }
    artifact = tmp_path / "artifacts" / "benchmark.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps(artifact_data), encoding="utf-8")
    manifest_data = {
        "schema_version": 1,
        "batch_id": "representative-v1",
        "partial_representative_sample": True,
        "artifact_file": "artifacts/benchmark.json",
        "artifact_sha256": _sha(artifact),
        "run_id": "run-1",
        "dataset_id": "evaluation",
        "dataset_file": "data/evaluation.jsonl",
        "dataset_sha256": _sha(dataset),
        "artifact_config_sha256": artifact_data["config_sha256"],
        "reference_review_sha256": reference_hash,
        "source_feedback_sha256": feedback_hash,
        "selected": [
            {"case_id": "case-1", "mode": "agent", "selection_reason": "pass"},
            {"case_id": "case-2", "mode": "agent", "selection_reason": "revise"},
            {
                "case_id": "case-3",
                "mode": "agent_rag",
                "selection_reason": "failure",
            },
        ],
    }
    manifest = tmp_path / "data" / "selection.json"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    rows = [
        {
            "review_key": "case-1::agent",
            "case_id": "case-1",
            "mode": "agent",
            "source_text": "원문 1",
            "reference_text": "Reference 1",
            "initial_translation": "Initial 1",
            "final_translation": "Final 1",
            "agent_initial_passed": "True",
            "agent_initial_error_types": "[]",
            "agent_initial_summary": "",
            "retry_count": "0",
            "stop_reason": "passed",
            "manual_initial_needs_revision": "false",
            "manual_primary_error": "",
            "manual_error_types": "",
            "pairwise_outcome": "same",
            "review_status": "confirmed",
            "reviewer": "Kim",
            "note": "",
        },
        {
            "review_key": "case-2::agent",
            "case_id": "case-2",
            "mode": "agent",
            "source_text": "원문 2",
            "reference_text": "Reference 2",
            "initial_translation": "Initial 2",
            "final_translation": "Final 2",
            "agent_initial_passed": "False",
            "agent_initial_error_types": '["meaning"]',
            "agent_initial_summary": "",
            "retry_count": "1",
            "stop_reason": "passed",
            "manual_initial_needs_revision": "true",
            "manual_primary_error": "meaning",
            "manual_error_types": '["meaning"]',
            "pairwise_outcome": "improved",
            "review_status": "confirmed",
            "reviewer": "Kim",
            "note": "",
        },
        {
            "review_key": "case-3::agent_rag",
            "case_id": "case-3",
            "mode": "agent_rag",
            "source_text": "원문 3",
            "reference_text": "Reference 3",
            "initial_translation": "Initial 3",
            "final_translation": "Initial 3",
            "agent_initial_passed": "True",
            "agent_initial_error_types": "[]",
            "agent_initial_summary": "",
            "retry_count": "0",
            "stop_reason": "component_failure",
            "manual_initial_needs_revision": "true",
            "manual_primary_error": "other",
            "manual_error_types": '["other"]',
            "pairwise_outcome": "same",
            "review_status": "confirmed",
            "reviewer": "Kim",
            "note": "Component failed before first judgment.",
        },
    ]
    review = tmp_path / "data" / "review.csv"
    with review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "artifact": artifact,
        "dataset": dataset,
        "manifest": manifest,
        "review": review,
        "output": tmp_path / "data" / "scores.json",
        "root": tmp_path,
    }


def _score(paths: dict[str, Path]) -> dict[str, object]:
    return score_manual_review(
        artifact_path=paths["artifact"],
        dataset_path=paths["dataset"],
        selection_manifest_path=paths["manifest"],
        review_csv_path=paths["review"],
        output_path=paths["output"],
        project_root=paths["root"],
    )


def _enable_v2(paths: dict[str, Path]) -> list[dict[str, str]]:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["manual_review_schema_version"] = 2
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with paths["review"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["manual_severity"] = ""
    rows[1]["manual_severity"] = "MAJOR"
    rows[2]["manual_severity"] = "MINOR"
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_score_manual_review_computes_metrics_and_keeps_unscorable_outcome(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    result = _score(paths)

    assert paths["output"].is_file()
    assert result["partial_representative_sample"] is True
    assert result["quality_claims_allowed"] is False
    assert result["schema_version"] == 1
    assert "manual_review_schema_version" not in result
    assert "confirmed_severity_counts" not in result["overall_metrics"]
    assert "major_false_pass_count" not in result["overall_metrics"]
    agent = result["metrics_by_mode"]["agent"]
    assert agent["agent_judgment"]["confusion"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert agent["agent_judgment"]["accuracy_pct"] == 100.0
    assert agent["successful_correction"] == {
        "eligible": 1,
        "improved": 1,
        "rate_pct": 100.0,
    }
    agent_rag = result["metrics_by_mode"]["agent_rag"]
    assert agent_rag["completed"] == 1
    assert agent_rag["unscorable"] == 1
    assert agent_rag["component_failure_count"] == 1
    assert agent_rag["agent_judgment"]["denominator"] == 0
    assert agent_rag["reviewed_outcome_counts"]["same"] == 1
    assert agent_rag["successful_correction"]["eligible"] == 1
    assert agent["component_failure_count"] == 0


def test_overall_metrics_are_recomputed_and_count_component_failures(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    result = _score(paths)

    overall = result["overall_metrics"]
    assert overall["selected"] == 3
    assert overall["component_failure_count"] == 1
    assert overall["agent_judgment"]["denominator"] == 2
    assert overall["agent_judgment"]["confusion"]["true_positive"] == 1
    assert overall["agent_judgment"]["confusion"]["false_negative"] == 0
    assert overall["agent_judgment"]["accuracy_pct"] == 100.0
    assert overall["successful_correction"] == {
        "eligible": 2,
        "improved": 1,
        "rate_pct": 50.0,
    }


def test_ambiguous_row_is_excluded_from_metrics(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = list(csv.DictReader(paths["review"].open(encoding="utf-8")))
    rows[0]["review_status"] = "ambiguous"
    rows[0]["note"] = "The source permits two reasonable readings."
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    result = _score(paths)

    agent = result["metrics_by_mode"]["agent"]
    assert agent["ambiguous"] == 1
    assert agent["confirmed"] == 1
    assert agent["agent_judgment"]["denominator"] == 1
    assert agent["reviewed_outcome_counts"] == {
        "improved": 1,
        "same": 1,
        "worse": 0,
    }
    assert agent["confirmed_outcome_counts"] == {
        "improved": 1,
        "same": 0,
        "worse": 0,
    }


def test_ambiguous_row_requires_note(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = list(csv.DictReader(paths["review"].open(encoding="utf-8")))
    rows[0]["review_status"] = "ambiguous"
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="review_status is ambiguous"):
        _score(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("column", "tampered"),
    [
        ("review_key", "case-1::agent_rag"),
        ("agent_initial_summary", "tampered Agent summary"),
    ],
)
def test_additional_immutable_evidence_tampering_creates_no_score_file(
    tmp_path: Path, column: str, tampered: str
) -> None:
    paths = _fixture(tmp_path)
    rows = list(csv.DictReader(paths["review"].open(encoding="utf-8")))
    rows[0][column] = tampered
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=f"immutable evidence differs in {column}"):
        _score(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "case_id": "case-1"}),
            "duplicate",
        ),
        (
            lambda rows: rows[0].__setitem__("initial_translation", "tampered"),
            "immutable evidence",
        ),
        (
            lambda rows: rows[0].__setitem__("manual_initial_needs_revision", ""),
            "partially filled",
        ),
        (
            lambda rows: rows[0].update(
                {
                    "manual_initial_needs_revision": "",
                    "manual_primary_error": "",
                    "manual_error_types": "",
                    "pairwise_outcome": "",
                    "review_status": "",
                    "reviewer": "",
                    "note": "",
                }
            ),
            "pending",
        ),
    ],
)
def test_invalid_or_pending_review_creates_no_score_file(
    tmp_path: Path, mutation: object, match: str
) -> None:
    paths = _fixture(tmp_path)
    rows = list(csv.DictReader(paths["review"].open(encoding="utf-8")))
    mutation(rows)  # type: ignore[operator]
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=match):
        _score(paths)

    assert not paths["output"].exists()


def test_other_requires_note_and_output_overwrite_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = list(csv.DictReader(paths["review"].open(encoding="utf-8")))
    rows[2]["note"] = ""
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEGACY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="requires note"):
        _score(paths)
    assert not paths["output"].exists()

    paths = _fixture(tmp_path / "valid")
    _score(paths)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _score(paths)


def test_manifest_hash_mismatch_creates_no_score_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["artifact_sha256"] = "0" * 64
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash"):
        _score(paths)

    assert not paths["output"].exists()


def test_v2_score_counts_severity_and_excludes_component_failure_false_pass(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    rows = _enable_v2(paths)
    rows[0].update(
        {
            "manual_initial_needs_revision": "true",
            "manual_severity": "MAJOR",
            "manual_primary_error": "meaning",
            "manual_error_types": '["meaning"]',
        }
    )
    # This component-failure row has a cached first judgment that passed. It is
    # still part of the human severity distribution, but not an Agent FN.
    rows[2]["manual_severity"] = "MAJOR"
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    result = _score(paths)

    assert result["schema_version"] == 2
    assert result["manual_review_schema_version"] == 2
    overall = result["overall_metrics"]
    assert overall["confirmed_severity_counts"] == {"MAJOR": 3, "MINOR": 0}
    assert overall["major_false_pass_count"] == 1
    assert overall["component_failure_count"] == 1
    assert result["metrics_by_mode"]["agent"]["confirmed_severity_counts"] == {
        "MAJOR": 2,
        "MINOR": 0,
    }
    assert result["metrics_by_mode"]["agent"]["major_false_pass_count"] == 1
    assert result["metrics_by_mode"]["agent_rag"][
        "confirmed_severity_counts"
    ] == {"MAJOR": 1, "MINOR": 0}
    assert (
        result["metrics_by_mode"]["agent_rag"]["major_false_pass_count"] == 0
    )
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result


def test_v2_score_accepts_blank_pass_severity_and_both_revision_values(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _enable_v2(paths)

    result = _score(paths)

    assert result["overall_metrics"]["confirmed_severity_counts"] == {
        "MAJOR": 1,
        "MINOR": 1,
    }
    assert result["overall_metrics"]["major_false_pass_count"] == 0


@pytest.mark.parametrize(
    ("row_index", "severity", "match"),
    [
        (1, "", "manual_severity is required"),
        (0, "MAJOR", "manual_severity must be blank"),
        (1, "major", "manual_severity must be MAJOR or MINOR"),
        (1, " MAJOR", "manual_severity must be MAJOR or MINOR"),
        (1, "CRITICAL", "manual_severity must be MAJOR or MINOR"),
    ],
)
def test_v2_rejects_missing_forbidden_or_nonexact_severity_without_output(
    tmp_path: Path, row_index: int, severity: str, match: str
) -> None:
    paths = _fixture(tmp_path)
    rows = _enable_v2(paths)
    rows[row_index]["manual_severity"] = severity
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=match):
        _score(paths)

    assert not paths["output"].exists()


def test_v2_true_ambiguous_row_still_requires_human_severity(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    rows = _enable_v2(paths)
    rows[1]["manual_severity"] = ""
    rows[1]["review_status"] = "ambiguous"
    rows[1]["note"] = "Two defensible readings."
    with paths["review"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="manual_severity is required"):
        _score(paths)

    assert not paths["output"].exists()


def test_v2_requires_severity_column(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["manual_review_schema_version"] = 2
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns: manual_severity"):
        _score(paths)

    assert not paths["output"].exists()


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_score_rejects_invalid_manual_review_schema_version(
    tmp_path: Path, version: object
) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["manual_review_schema_version"] = version
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manual_review_schema_version"):
        _score(paths)

    assert not paths["output"].exists()
