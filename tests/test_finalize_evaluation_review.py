from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_evaluation_review import finalize_review, main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, case_count: int = 40) -> dict[str, Path]:
    root = tmp_path
    candidates = root / "artifacts" / "eda" / "candidates.jsonl"
    candidates.parent.mkdir(parents=True)
    records = [
        {
            "source_record_id": f"row-{index:03d}",
            "korean": f"source {index}",
            "english_reference": f"Reference {index}.",
            "hit_terms": ["term"],
        }
        for index in range(case_count)
    ]
    candidates.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = root / "data" / "selection_draft.json"
    overlay = root / "data" / "reference_reviews" / "review_draft.json"
    selected = [
        {"source_record_id": record["source_record_id"], "selection_note": "keep"}
        for record in records
    ]
    selected_ids = [record["source_record_id"] for record in records]
    corrections = [
        {
            "case_order": 2,
            "source_record_id": records[1]["source_record_id"],
            "original_reference_sha256": _text_sha256(
                records[1]["english_reference"]
            ),
            "corrected_reference_text": "Corrected reference 1.",
            "rationale": "Fix the draft reference.",
        },
        {
            "case_order": 7,
            "source_record_id": records[6]["source_record_id"],
            "original_reference_sha256": _text_sha256(
                records[6]["english_reference"]
            ),
            "corrected_reference_text": "Corrected reference 6.",
            "rationale": "Improve source alignment.",
        },
    ]
    _write_json(
        overlay,
        {
            "schema_version": 1,
            "review_status": "ai_assisted_draft",
            "human_confirmed": False,
            "candidate_file": "artifacts/eda/candidates.jsonl",
            "candidate_sha256": _sha256(candidates),
            "selection_manifest_file": "data/selection_draft.json",
            "ordered_source_id_sha256": _text_sha256("\n".join(selected_ids)),
            "feedback_file_label": "feedback.md",
            "feedback_sha256": "f" * 64,
            "reference_policy": {
                "public_candidate_reference_preserved": True,
                "corrections_are_overlay_only": True,
            },
            "corrections": corrections,
        },
    )
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "status": "AI_ASSISTED_DRAFT_AWAITING_HUMAN_CONFIRMATION",
            "human_confirmed": False,
            "candidate_file": "artifacts/eda/candidates.jsonl",
            "candidate_sha256": _sha256(candidates),
            "glossary_file": "data/glossary.csv",
            "glossary_sha256": "a" * 64,
            "reference_review_file": "reference_reviews/review_draft.json",
            "reference_review_sha256": _sha256(overlay),
            "selection_metadata": {
                "reviewer_type": "ai_assisted_draft",
                "human_review_required": "Review all cases.",
            },
            "selected": selected,
        },
    )
    workbook = root / "artifacts" / "reviews" / "review.xlsx"
    workbook.parent.mkdir(parents=True)
    workbook.write_bytes(b"confirmed workbook fixture")
    return {
        "root": root,
        "candidates": candidates,
        "manifest": manifest,
        "overlay": overlay,
        "workbook": workbook,
        "output_manifest": root / "data" / "selection_v1.json",
        "output_overlay": root / "data" / "reference_reviews" / "review_v1.json",
    }


def _finalize(paths: dict[str, Path]) -> dict[str, object]:
    return finalize_review(
        paths["candidates"],
        paths["manifest"],
        paths["overlay"],
        paths["workbook"],
        paths["output_manifest"],
        paths["output_overlay"],
        reviewer="project-owner",
        reviewed_at_utc="2026-08-20T12:34:56Z",
        confirmation_basis="User explicitly said the review was complete.",
        project_root=paths["root"],
    )


def _assert_no_absolute_strings(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_strings(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_strings(child)
    elif isinstance(value, str):
        assert not Path(value).is_absolute()


def test_finalize_preserves_corrections_and_fills_all_other_decisions(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    summary = _finalize(paths)

    overlay = json.loads(paths["output_overlay"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["output_manifest"].read_text(encoding="utf-8"))
    assert summary["case_count"] == 40
    assert summary["decision_counts"] == {"corrected": 2, "keep_original": 38}
    assert overlay["human_confirmed"] is True
    assert overlay["review_status"] == "human_confirmed"
    assert overlay["reviewer"] == "project-owner"
    assert overlay["reviewed_at_utc"] == "2026-08-20T12:34:56Z"
    assert overlay["review_workbook_file"] == "artifacts/reviews/review.xlsx"
    assert overlay["review_workbook_sha256"] == _sha256(paths["workbook"])
    assert overlay["confirmation"] == {
        "kind": "explicit_user_confirmation_in_conversation",
        "basis": "User explicitly said the review was complete.",
    }
    assert len(overlay["decisions"]) == 40
    assert [decision["case_order"] for decision in overlay["decisions"]] == list(
        range(1, 41)
    )
    assert overlay["decisions"][0] == {
        "case_order": 1,
        "source_record_id": "row-000",
        "decision": "keep_original",
        "original_reference_sha256": _text_sha256("Reference 0."),
    }
    assert overlay["decisions"][1] == {
        "case_order": 2,
        "source_record_id": "row-001",
        "decision": "corrected",
        "original_reference_sha256": _text_sha256("Reference 1."),
        "corrected_reference_text": "Corrected reference 1.",
        "rationale": "Fix the draft reference.",
    }
    assert manifest["human_confirmed"] is True
    assert manifest["status"] == "HUMAN_CONFIRMED_FINAL"
    assert manifest["reference_review_file"] == "reference_reviews/review_v1.json"
    assert manifest["reference_review_sha256"] == _sha256(paths["output_overlay"])
    assert manifest["review_confirmation"]["kind"] == (
        "explicit_user_confirmation_in_conversation"
    )
    assert manifest["selection_metadata"]["reviewer_type"] == "human_user"
    assert manifest["selection_metadata"]["human_review_completed"] is True
    assert "human_review_required" not in manifest["selection_metadata"]
    _assert_no_absolute_strings(overlay)
    _assert_no_absolute_strings(manifest)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("manifest_candidate_hash", "candidate hash does not match draft manifest"),
        ("overlay_candidate_hash", "candidate hash does not match draft overlay"),
        ("ordered_hash", "ordered source ID hash"),
        ("original_hash", "original reference hash"),
        ("already_final_manifest", "draft manifest is already final"),
        ("already_final_overlay", "draft overlay is already final"),
    ],
)
def test_finalize_rejects_invalid_or_already_final_inputs(
    tmp_path: Path, problem: str, message: str
) -> None:
    paths = _fixture(tmp_path)
    target = paths["manifest"] if problem in {
        "manifest_candidate_hash",
        "already_final_manifest",
    } else paths["overlay"]
    payload = json.loads(target.read_text(encoding="utf-8"))
    if problem in {"manifest_candidate_hash", "overlay_candidate_hash"}:
        payload["candidate_sha256"] = "0" * 64
    elif problem == "ordered_hash":
        payload["ordered_source_id_sha256"] = "0" * 64
    elif problem == "original_hash":
        payload["corrections"][0]["original_reference_sha256"] = "0" * 64
    elif problem in {"already_final_manifest", "already_final_overlay"}:
        payload["human_confirmed"] = True
    _write_json(target, payload)
    if target == paths["overlay"]:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["reference_review_sha256"] = _sha256(paths["overlay"])
        _write_json(paths["manifest"], manifest)

    with pytest.raises(ValueError, match=message):
        _finalize(paths)


@pytest.mark.parametrize("case_count", [39, 41])
def test_finalize_requires_exactly_40_selected_cases(
    tmp_path: Path, case_count: int
) -> None:
    paths = _fixture(tmp_path, case_count=case_count)

    with pytest.raises(ValueError, match="exactly 40 selected cases"):
        _finalize(paths)


def test_cli_writes_final_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _fixture(tmp_path)

    exit_code = main(
        [
            "--candidates",
            str(paths["candidates"]),
            "--draft-manifest",
            str(paths["manifest"]),
            "--draft-reference-review",
            str(paths["overlay"]),
            "--review-workbook",
            str(paths["workbook"]),
            "--output-manifest",
            str(paths["output_manifest"]),
            "--output-reference-review",
            str(paths["output_overlay"]),
            "--reviewer",
            "project-owner",
            "--reviewed-at-utc",
            "2026-08-20T12:34:56Z",
            "--confirmation-basis",
            "User explicitly said the review was complete.",
            "--project-root",
            str(paths["root"]),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["case_count"] == 40
    assert paths["output_manifest"].is_file()
    assert paths["output_overlay"].is_file()
