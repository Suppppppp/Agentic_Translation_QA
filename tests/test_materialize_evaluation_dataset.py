from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_evaluation_dataset import materialize
from translation_qa.benchmark import JsonlDatasetRepository


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path, *, human_confirmed: bool) -> tuple[Path, Path, Path]:
    terms = [(f"용어-{index}", f"term-{index}") for index in range(5)]
    glossary = tmp_path / "glossary.csv"
    glossary.write_text(
        "glossary_version,term_id,domain,source_term_ko,preferred_target_en,"
        "accepted_variants_json,disallowed_variants_json,replacement_rules_json,"
        "definition,source,notes\n"
        + "".join(
            f'v1,t{index},software,{source},{target},"[]","[]","{{}}",d,s,n\n'
            for index, (source, target) in enumerate(terms)
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.jsonl"
    candidate_records = []
    selected = []
    for index in range(30):
        source_term, _ = terms[index % len(terms)]
        source_id = f"row-{index:03d}"
        candidate_records.append(
            {
                "source_record_id": source_id,
                "korean": f"{source_term} 소스 문장 {index}",
                "english_reference": f"Reference sentence {index}",
                "hit_terms": [source_term],
            }
        )
        selected.append(
            {"source_record_id": source_id, "selection_note": "Reviewed fixture."}
        )
    candidates.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_records),
        encoding="utf-8",
    )
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "human_confirmed": human_confirmed,
                "candidate_sha256": _sha256(candidates),
                "glossary_sha256": _sha256(glossary),
                "selected": selected,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return candidates, manifest, glossary


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_reference_review(
    candidates: Path,
    manifest: Path,
    *,
    human_confirmed: bool,
    decisions: list[dict[str, object]],
) -> Path:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    selected_ids = [item["source_record_id"] for item in manifest_payload["selected"]]
    overlay = manifest.parent / "reference-review.json"
    overlay.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "human_confirmed": human_confirmed,
                "candidate_sha256": _sha256(candidates),
                "ordered_source_id_sha256": hashlib.sha256(
                    "\n".join(selected_ids).encode("utf-8")
                ).hexdigest(),
                "reviewer_type": (
                    "bilingual_human" if human_confirmed else "ai_assisted_draft"
                ),
                "reviewer": "reviewer-1" if human_confirmed else None,
                "reviewed_at_utc": (
                    "2026-08-20T00:00:00Z" if human_confirmed else None
                ),
                "source_feedback_sha256": "f" * 64,
                "decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_payload["reference_review_file"] = overlay.name
    manifest_payload["reference_review_sha256"] = _sha256(overlay)
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return overlay


def _decision(
    candidate: dict[str, object],
    decision: str,
    *,
    corrected_reference_text: str | None = None,
    rationale: str | None = None,
) -> dict[str, object]:
    reference = candidate["english_reference"]
    assert isinstance(reference, str)
    value: dict[str, object] = {
        "source_record_id": candidate["source_record_id"],
        "decision": decision,
        "original_reference_sha256": hashlib.sha256(
            reference.encode("utf-8")
        ).hexdigest(),
    }
    if corrected_reference_text is not None:
        value["corrected_reference_text"] = corrected_reference_text
    if rationale is not None:
        value["rationale"] = rationale
    return value


def test_materialize_builds_valid_frozen_dataset_with_five_terms(tmp_path: Path) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=True)
    output = tmp_path / "evaluation_v1.jsonl"
    summary_path = tmp_path / "evaluation_v1.summary.json"

    summary = materialize(candidates, manifest, glossary, output, summary_path)

    assert summary["status"] == "HUMAN_CONFIRMED_FROZEN"
    assert summary["case_count"] == 30
    assert summary["distinct_term_count"] == 5
    assert summary["benchmark_allowed"] is True
    cases = JsonlDatasetRepository(tmp_path).load("evaluation_v1")
    assert len(cases) == 30
    assert cases[0].to_translation_request().model_dump() == {
        "text": "용어-0 소스 문장 0"
    }
    assert cases[0].reference_provenance is not None
    assert cases[0].reference_provenance.review_status.value == "unreviewed"
    assert cases[0].reference_provenance.original_reference_text == (
        "Reference sentence 0"
    )


def test_materialize_requires_explicit_override_for_unconfirmed_draft(
    tmp_path: Path,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=False)

    with pytest.raises(ValueError, match="not human-confirmed"):
        materialize(
            candidates,
            manifest,
            glossary,
            tmp_path / "draft.jsonl",
            tmp_path / "draft.summary.json",
        )

    summary = materialize(
        candidates,
        manifest,
        glossary,
        tmp_path / "draft.jsonl",
        tmp_path / "draft.summary.json",
        allow_unconfirmed_draft=True,
    )
    assert summary["benchmark_allowed"] is False
    assert summary["human_confirmed"] is False


def test_materialize_applies_pinned_draft_overlay_without_mutating_public_reference(
    tmp_path: Path,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=False)
    records = _jsonl_records(candidates)
    original_candidate_hash = _sha256(candidates)
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=False,
        decisions=[
            _decision(
                records[0],
                "corrected",
                corrected_reference_text="Human-readable corrected reference 0.",
                rationale="The public reference omitted source information.",
            ),
            _decision(records[1], "keep_original"),
        ],
    )
    output = tmp_path / "draft.jsonl"

    with pytest.raises(ValueError, match="not human-confirmed"):
        materialize(
            candidates,
            manifest,
            glossary,
            output,
            tmp_path / "draft.summary.json",
            reference_review_path=overlay,
        )

    summary = materialize(
        candidates,
        manifest,
        glossary,
        output,
        tmp_path / "draft.summary.json",
        allow_unconfirmed_draft=True,
        reference_review_path=overlay,
    )

    cases = JsonlDatasetRepository(tmp_path).load("draft")
    corrected = cases[0]
    provenance = corrected.reference_provenance
    assert provenance is not None
    assert corrected.reference_text == "Human-readable corrected reference 0."
    assert provenance.original_reference_text == "Reference sentence 0"
    assert provenance.effective_origin.value == "reviewer_correction"
    assert provenance.review_status.value == "ai_assisted_draft"
    assert provenance.source_feedback_sha256 == "f" * 64
    assert corrected.to_translation_request().model_dump() == {
        "text": "용어-0 소스 문장 0"
    }
    assert cases[1].reference_text == "Reference sentence 1"
    assert cases[1].reference_provenance is not None
    assert cases[1].reference_provenance.decision.value == "keep_original"
    assert cases[2].reference_provenance is not None
    assert cases[2].reference_provenance.review_status.value == "unreviewed"
    assert _sha256(candidates) == original_candidate_hash
    assert summary["reference_review_sha256"] == _sha256(overlay)
    assert summary["reference_review_human_confirmed"] is False
    assert summary["reference_decision_counts"] == {
        "corrected": 1,
        "keep_original": 1,
        "unreviewed": 28,
    }
    assert (
        summary["original_reference_set_sha256"]
        != summary["effective_reference_set_sha256"]
    )


def test_materialize_accepts_canonical_corrections_only_overlay(tmp_path: Path) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=False)
    record = _jsonl_records(candidates)[0]
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=False,
        decisions=[
            _decision(
                record,
                "corrected",
                corrected_reference_text="Canonical corrected reference.",
                rationale="Correct the public reference before evaluation.",
            )
        ],
    )
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    payload["review_status"] = payload.pop("reviewer_type")
    payload["feedback_sha256"] = payload.pop("source_feedback_sha256")
    payload["corrections"] = payload.pop("decisions")
    payload["corrections"][0].pop("decision")
    overlay.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["reference_review_sha256"] = _sha256(overlay)
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = materialize(
        candidates,
        manifest,
        glossary,
        tmp_path / "draft.jsonl",
        tmp_path / "draft.summary.json",
        allow_unconfirmed_draft=True,
        reference_review_path=overlay,
    )

    case = JsonlDatasetRepository(tmp_path).load("draft")[0]
    assert case.reference_text == "Canonical corrected reference."
    assert case.reference_provenance is not None
    assert case.reference_provenance.reviewer_type == "ai_assisted_draft"
    assert case.reference_provenance.source_feedback_sha256 == "f" * 64
    assert summary["reference_decision_counts"] == {
        "corrected": 1,
        "unreviewed": 29,
    }


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("candidate_hash", "candidate_sha256"),
        ("ordered_hash", "ordered_source_id_sha256"),
        ("original_hash", "original reference hash"),
        ("duplicate", "duplicate reference review decision"),
        ("missing_text", "requires corrected_reference_text"),
        ("missing_rationale", "requires a rationale"),
    ],
)
def test_materialize_rejects_invalid_reference_overlay_content(
    tmp_path: Path,
    problem: str,
    message: str,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=False)
    record = _jsonl_records(candidates)[0]
    valid_decision = _decision(
        record,
        "corrected",
        corrected_reference_text="Corrected reference.",
        rationale="Fix an omission.",
    )
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=False,
        decisions=[valid_decision],
    )
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    if problem == "candidate_hash":
        payload["candidate_sha256"] = "0" * 64
    elif problem == "ordered_hash":
        payload["ordered_source_id_sha256"] = "0" * 64
    elif problem == "original_hash":
        payload["decisions"][0]["original_reference_sha256"] = "0" * 64
    elif problem == "duplicate":
        payload["decisions"].append(dict(payload["decisions"][0]))
    elif problem == "missing_text":
        payload["decisions"][0].pop("corrected_reference_text")
    elif problem == "missing_rationale":
        payload["decisions"][0].pop("rationale")
    overlay.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["reference_review_sha256"] = _sha256(overlay)
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        materialize(
            candidates,
            manifest,
            glossary,
            tmp_path / "draft.jsonl",
            tmp_path / "draft.summary.json",
            allow_unconfirmed_draft=True,
            reference_review_path=overlay,
        )


def test_materialize_rejects_unconfirmed_overlay_for_confirmed_manifest(
    tmp_path: Path,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=True)
    record = _jsonl_records(candidates)[0]
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=False,
        decisions=[_decision(record, "keep_original")],
    )

    with pytest.raises(ValueError, match="cannot use an unconfirmed"):
        materialize(
            candidates,
            manifest,
            glossary,
            tmp_path / "evaluation_v1.jsonl",
            tmp_path / "evaluation_v1.summary.json",
            allow_unconfirmed_draft=True,
            reference_review_path=overlay,
        )


def test_manifest_hash_pin_detects_reference_overlay_tampering(tmp_path: Path) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=False)
    record = _jsonl_records(candidates)[0]
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=False,
        decisions=[_decision(record, "keep_original")],
    )
    overlay.write_text(
        overlay.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash does not match"):
        materialize(
            candidates,
            manifest,
            glossary,
            tmp_path / "draft.jsonl",
            tmp_path / "draft.summary.json",
            allow_unconfirmed_draft=True,
            reference_review_path=overlay,
        )


def test_confirmed_overlay_requires_one_decision_per_selected_case(
    tmp_path: Path,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=True)
    record = _jsonl_records(candidates)[0]
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=True,
        decisions=[_decision(record, "keep_original")],
    )

    with pytest.raises(ValueError, match="one decision for every"):
        materialize(
            candidates,
            manifest,
            glossary,
            tmp_path / "evaluation_v1.jsonl",
            tmp_path / "evaluation_v1.summary.json",
            reference_review_path=overlay,
        )


@pytest.mark.parametrize("missing_field", ["reviewer", "reviewed_at_utc"])
def test_confirmed_overlay_requires_reviewer_identity_and_review_time(
    tmp_path: Path,
    missing_field: str,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=True)
    records = _jsonl_records(candidates)
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=True,
        decisions=[_decision(record, "keep_original") for record in records],
    )
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    payload[missing_field] = None
    overlay.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["reference_review_sha256"] = _sha256(overlay)
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires reviewer and reviewed_at_utc"):
        materialize(
            candidates,
            manifest,
            glossary,
            tmp_path / "evaluation_v1.jsonl",
            tmp_path / "evaluation_v1.summary.json",
            reference_review_path=overlay,
        )


def test_confirmed_overlay_materializes_when_every_case_has_one_decision(
    tmp_path: Path,
) -> None:
    candidates, manifest, glossary = _fixtures(tmp_path, human_confirmed=True)
    records = _jsonl_records(candidates)
    overlay = _write_reference_review(
        candidates,
        manifest,
        human_confirmed=True,
        decisions=[_decision(record, "keep_original") for record in records],
    )

    summary = materialize(
        candidates,
        manifest,
        glossary,
        tmp_path / "evaluation_v1.jsonl",
        tmp_path / "evaluation_v1.summary.json",
        reference_review_path=overlay,
    )

    cases = JsonlDatasetRepository(tmp_path).load("evaluation_v1")
    assert summary["benchmark_allowed"] is True
    assert summary["reference_review_human_confirmed"] is True
    assert summary["reference_decision_counts"] == {"keep_original": 30}
    assert (
        summary["original_reference_set_sha256"]
        == summary["effective_reference_set_sha256"]
    )
    assert cases[0].reference_provenance is not None
    assert cases[0].reference_provenance.review_status.value == "human_confirmed"
