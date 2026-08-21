"""Materialize a reviewed 30--50 case dataset from an explicit selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from translation_qa.retrieval import GlossaryEntry, load_glossary_csv
from translation_qa.schemas import (
    EvaluationCase,
    ExpectedTerm,
    ReferenceOrigin,
    ReferenceProvenance,
    ReferenceReviewDecision,
    ReferenceReviewStatus,
)


_SHA256_LENGTH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _checked_manifest(
    path: Path,
    *,
    candidate_path: Path,
    glossary_path: Path,
    reference_review_path: Path | None,
    allow_unconfirmed_draft: bool,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("selection manifest must be a JSON object")
    human_confirmed = manifest.get("human_confirmed")
    if not isinstance(human_confirmed, bool):
        raise ValueError("manifest human_confirmed must be a boolean")
    if not human_confirmed and not allow_unconfirmed_draft:
        raise ValueError(
            "selection is not human-confirmed; use --allow-unconfirmed-draft only "
            "for a clearly named provisional output"
        )
    expected_candidate_hash = manifest.get("candidate_sha256")
    if expected_candidate_hash and expected_candidate_hash != _sha256(candidate_path):
        raise ValueError("candidate file hash does not match the selection manifest")
    expected_glossary_hash = manifest.get("glossary_sha256")
    if expected_glossary_hash and expected_glossary_hash != _sha256(glossary_path):
        raise ValueError("glossary hash does not match the selection manifest")

    pinned_review_file = manifest.get("reference_review_file")
    pinned_review_hash = manifest.get("reference_review_sha256")
    review_is_declared = pinned_review_file is not None or pinned_review_hash is not None
    if reference_review_path is None:
        if review_is_declared:
            raise ValueError(
                "manifest pins a reference review, but no reference review path was "
                "provided"
            )
    else:
        if not isinstance(pinned_review_file, str) or not pinned_review_file.strip():
            raise ValueError("manifest must pin reference_review_file")
        if not _is_sha256(pinned_review_hash):
            raise ValueError("manifest must pin a valid reference_review_sha256")
        declared_path = Path(pinned_review_file)
        expected_path = (
            declared_path.resolve()
            if declared_path.is_absolute()
            else (path.parent / declared_path).resolve()
        )
        if expected_path != reference_review_path.resolve():
            raise ValueError(
                "reference review path does not match reference_review_file in the "
                "selection manifest"
            )
        if not reference_review_path.is_file():
            raise ValueError("reference review file does not exist")
        if pinned_review_hash != _sha256(reference_review_path):
            raise ValueError(
                "reference review file hash does not match the selection manifest"
            )
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not 30 <= len(selected) <= 50:
        raise ValueError("manifest must select between 30 and 50 cases")
    return manifest


def _glossary_by_source(entries: Sequence[GlossaryEntry]) -> dict[str, GlossaryEntry]:
    mapping: dict[str, GlossaryEntry] = {}
    for entry in entries:
        if entry.source_term in mapping:
            raise ValueError(f"duplicate glossary source term: {entry.source_term}")
        mapping[entry.source_term] = entry
    return mapping


def _selected_source_ids(
    selected_items: Sequence[Any],
    candidate_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    selected_ids: list[str] = []
    for item in selected_items:
        if not isinstance(item, dict):
            raise ValueError("each selected manifest item must be an object")
        source_id = item.get("source_record_id")
        if not isinstance(source_id, str) or source_id not in candidate_by_id:
            raise ValueError(f"selected source_record_id is not in candidates: {source_id}")
        if source_id in selected_ids:
            raise ValueError(f"duplicate selected source_record_id: {source_id}")
        selected_ids.append(source_id)
    return selected_ids


def _optional_nonempty_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"reference review {field_name} must be a non-empty string")
    return value.strip()


def _load_reference_review(
    path: Path,
    *,
    candidate_path: Path,
    candidate_by_id: dict[str, dict[str, Any]],
    selected_ids: Sequence[str],
    manifest_human_confirmed: bool,
    allow_unconfirmed_draft: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(overlay, dict):
        raise ValueError("reference review overlay must be a JSON object")

    human_confirmed = overlay.get("human_confirmed")
    if not isinstance(human_confirmed, bool):
        raise ValueError("reference review human_confirmed must be a boolean")
    if not human_confirmed and not allow_unconfirmed_draft:
        raise ValueError(
            "reference review is not human-confirmed; use "
            "--allow-unconfirmed-draft only for provisional output"
        )
    if manifest_human_confirmed and not human_confirmed:
        raise ValueError(
            "a human-confirmed selection manifest cannot use an unconfirmed "
            "reference review"
        )

    overlay_candidate_hash = overlay.get("candidate_sha256")
    if not _is_sha256(overlay_candidate_hash):
        raise ValueError("reference review must contain a valid candidate_sha256")
    if overlay_candidate_hash != _sha256(candidate_path):
        raise ValueError("reference review candidate_sha256 does not match candidates")

    ordered_id_hash = _text_sha256("\n".join(selected_ids))
    overlay_ordered_hash = overlay.get("ordered_source_id_sha256")
    if not _is_sha256(overlay_ordered_hash):
        raise ValueError(
            "reference review must contain a valid ordered_source_id_sha256"
        )
    if overlay_ordered_hash != ordered_id_hash:
        raise ValueError(
            "reference review ordered_source_id_sha256 does not match selection"
        )

    reviewer_type = _optional_nonempty_text(
        overlay.get("reviewer_type", overlay.get("review_status")),
        "reviewer_type/review_status",
    )
    reviewer = _optional_nonempty_text(overlay.get("reviewer"), "reviewer")
    reviewed_at_utc = _optional_nonempty_text(
        overlay.get("reviewed_at_utc"), "reviewed_at_utc"
    )
    if human_confirmed and (reviewer is None or reviewed_at_utc is None):
        raise ValueError(
            "a human-confirmed reference review requires reviewer and "
            "reviewed_at_utc"
        )
    feedback_hash = overlay.get(
        "source_feedback_sha256", overlay.get("feedback_sha256")
    )
    if feedback_hash is not None and not _is_sha256(feedback_hash):
        raise ValueError(
            "reference review source_feedback_sha256 must be a lowercase SHA-256"
        )

    raw_decisions = overlay.get("decisions")
    raw_corrections = overlay.get("corrections")
    if raw_decisions is not None and raw_corrections is not None:
        raise ValueError(
            "reference review must use either decisions or corrections, not both"
        )
    corrections_only = raw_decisions is None and raw_corrections is not None
    if corrections_only:
        raw_decisions = raw_corrections
    if not isinstance(raw_decisions, list):
        raise ValueError("reference review decisions/corrections must be an array")
    selected_id_set = set(selected_ids)
    decisions: dict[str, dict[str, Any]] = {}
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("each reference review decision must be an object")
        source_id = raw.get("source_record_id")
        if not isinstance(source_id, str) or source_id not in selected_id_set:
            raise ValueError(
                f"reference review decision targets an unselected case: {source_id}"
            )
        if source_id in decisions:
            raise ValueError(f"duplicate reference review decision: {source_id}")

        raw_decision = (
            ReferenceReviewDecision.CORRECTED.value
            if corrections_only
            else raw.get("decision")
        )
        try:
            decision = ReferenceReviewDecision(raw_decision)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid reference review decision for {source_id}"
            ) from exc
        original_hash = raw.get("original_reference_sha256")
        if not _is_sha256(original_hash):
            raise ValueError(
                f"reference review decision has an invalid original reference hash: "
                f"{source_id}"
            )
        original_reference = candidate_by_id[source_id].get("english_reference")
        if not isinstance(original_reference, str) or not original_reference.strip():
            raise ValueError(f"selected candidate has no English reference: {source_id}")
        if original_hash != _text_sha256(original_reference):
            raise ValueError(
                f"original reference hash does not match candidate: {source_id}"
            )

        corrected_text = raw.get("corrected_reference_text")
        rationale = _optional_nonempty_text(raw.get("rationale"), "rationale")
        if decision is ReferenceReviewDecision.CORRECTED:
            if not isinstance(corrected_text, str) or not corrected_text.strip():
                raise ValueError(
                    f"corrected decision requires corrected_reference_text: {source_id}"
                )
            corrected_text = corrected_text.strip()
            if corrected_text == original_reference.strip():
                raise ValueError(
                    f"corrected reference must differ from the original: {source_id}"
                )
            if rationale is None:
                raise ValueError(
                    f"corrected decision requires a rationale: {source_id}"
                )
        elif corrected_text not in (None, ""):
            raise ValueError(
                f"keep_original decision cannot contain corrected text: {source_id}"
            )

        decisions[source_id] = {
            "decision": decision,
            "original_reference_sha256": original_hash,
            "corrected_reference_text": corrected_text,
            "rationale": rationale,
        }

    if human_confirmed and set(decisions) != selected_id_set:
        raise ValueError(
            "a human-confirmed reference review must contain one decision for every "
            "selected case"
        )

    return decisions, {
        "human_confirmed": human_confirmed,
        "reviewer_type": reviewer_type,
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at_utc,
        "source_feedback_sha256": feedback_hash,
    }


def materialize(
    candidate_path: Path,
    manifest_path: Path,
    glossary_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    allow_unconfirmed_draft: bool = False,
    reference_review_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _checked_manifest(
        manifest_path,
        candidate_path=candidate_path,
        glossary_path=glossary_path,
        reference_review_path=reference_review_path,
        allow_unconfirmed_draft=allow_unconfirmed_draft,
    )
    candidates = _load_jsonl(candidate_path)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for record in candidates:
        source_id = record.get("source_record_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("candidate has no source_record_id")
        if source_id in candidate_by_id:
            raise ValueError(f"duplicate candidate source_record_id: {source_id}")
        candidate_by_id[source_id] = record

    glossary = _glossary_by_source(load_glossary_csv(glossary_path))
    selected_items = manifest["selected"]
    selected_ids = _selected_source_ids(selected_items, candidate_by_id)
    ordered_id_hash = _text_sha256("\n".join(selected_ids))
    review_sha256 = (
        _sha256(reference_review_path) if reference_review_path is not None else None
    )
    review_decisions: dict[str, dict[str, Any]] = {}
    review_metadata: dict[str, Any] = {
        "human_confirmed": None,
        "reviewer_type": None,
        "reviewer": None,
        "reviewed_at_utc": None,
        "source_feedback_sha256": None,
    }
    if reference_review_path is not None:
        review_decisions, review_metadata = _load_reference_review(
            reference_review_path,
            candidate_path=candidate_path,
            candidate_by_id=candidate_by_id,
            selected_ids=selected_ids,
            manifest_human_confirmed=manifest["human_confirmed"],
            allow_unconfirmed_draft=allow_unconfirmed_draft,
        )

    cases: list[EvaluationCase] = []
    term_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    original_reference_records: list[dict[str, str]] = []
    effective_reference_records: list[dict[str, str]] = []
    for index, (item, source_id) in enumerate(
        zip(selected_items, selected_ids, strict=True), start=1
    ):
        assert isinstance(item, dict)
        candidate = candidate_by_id[source_id]
        hit_terms = candidate.get("hit_terms")
        if not isinstance(hit_terms, list) or not hit_terms:
            raise ValueError(f"selected candidate has no hit_terms: {source_id}")

        expected_terms: list[ExpectedTerm] = []
        for source_term in hit_terms:
            if not isinstance(source_term, str) or source_term not in glossary:
                raise ValueError(
                    f"candidate term is absent from frozen glossary: {source_term}"
                )
            entry = glossary[source_term]
            targets = list(dict.fromkeys([entry.target_term, *entry.accepted_variants]))
            expected_terms.append(
                ExpectedTerm(source_term=source_term, accepted_targets=targets)
            )
            term_counts[source_term] += 1

        source_text = candidate.get("korean")
        original_reference = candidate.get("english_reference")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"selected candidate has no Korean source: {source_id}")
        if not isinstance(original_reference, str) or not original_reference.strip():
            raise ValueError(f"selected candidate has no English reference: {source_id}")
        original_reference_hash = _text_sha256(original_reference)
        decision_record = review_decisions.get(source_id)
        if decision_record is None:
            effective_reference = original_reference
            effective_origin = ReferenceOrigin.PUBLIC_DATASET
            review_status = ReferenceReviewStatus.UNREVIEWED
            decision = None
            rationale = None
            decision_counts[ReferenceReviewStatus.UNREVIEWED.value] += 1
        else:
            decision = decision_record["decision"]
            rationale = decision_record["rationale"]
            if decision is ReferenceReviewDecision.CORRECTED:
                effective_reference = decision_record["corrected_reference_text"]
                effective_origin = ReferenceOrigin.REVIEWER_CORRECTION
            else:
                effective_reference = original_reference
                effective_origin = ReferenceOrigin.PUBLIC_DATASET
            review_status = (
                ReferenceReviewStatus.HUMAN_CONFIRMED
                if review_metadata["human_confirmed"]
                else ReferenceReviewStatus.AI_ASSISTED_DRAFT
            )
            decision_counts[decision.value] += 1

        original_reference_records.append(
            {"source_record_id": source_id, "reference_text": original_reference}
        )
        effective_reference_records.append(
            {"source_record_id": source_id, "reference_text": effective_reference}
        )
        note = item.get("selection_note")
        if not isinstance(note, str) or not note.strip():
            note = "Selected from the source-only pool after bilingual draft review."

        tags = ["public_dataset", "glossary_term"]
        if len(hit_terms) > 1:
            tags.append("multiple_terms")
        cases.append(
            EvaluationCase(
                case_id=f"evaluation-v1-{index:03d}",
                source_record_id=source_id,
                source_text=source_text,
                reference_text=effective_reference,
                reference_provenance=ReferenceProvenance(
                    original_reference_text=original_reference,
                    original_reference_sha256=original_reference_hash,
                    effective_origin=effective_origin,
                    review_status=review_status,
                    decision=decision,
                    reviewer_type=review_metadata["reviewer_type"],
                    reviewer=review_metadata["reviewer"],
                    reviewed_at_utc=review_metadata["reviewed_at_utc"],
                    rationale=rationale,
                    reference_review_sha256=review_sha256,
                    source_feedback_sha256=review_metadata[
                        "source_feedback_sha256"
                    ],
                ),
                domain="software",
                scenario_tags=tags,
                selection_note=note,
                expected_terms=expected_terms,
            )
        )

    if len(term_counts) < 5:
        raise ValueError("selected dataset must cover at least five distinct terms")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )
    summary = {
        "status": (
            "HUMAN_CONFIRMED_FROZEN"
            if manifest["human_confirmed"]
            else "AI_ASSISTED_DRAFT_NOT_HUMAN_CONFIRMED"
        ),
        "human_confirmed": manifest["human_confirmed"],
        "benchmark_allowed": manifest["human_confirmed"],
        "case_count": len(cases),
        "distinct_term_count": len(term_counts),
        "term_case_counts": dict(sorted(term_counts.items())),
        "ordered_source_id_sha256": ordered_id_hash,
        "candidate_sha256": _sha256(candidate_path),
        "manifest_sha256": _sha256(manifest_path),
        "glossary_sha256": _sha256(glossary_path),
        "reference_review_sha256": review_sha256,
        "reference_review_human_confirmed": review_metadata["human_confirmed"],
        "reference_decision_counts": dict(sorted(decision_counts.items())),
        "source_feedback_sha256": review_metadata["source_feedback_sha256"],
        "original_reference_set_sha256": _json_sha256(
            original_reference_records
        ),
        "effective_reference_set_sha256": _json_sha256(
            effective_reference_records
        ),
        "dataset_sha256": _sha256(output_path),
        "reference_exposed_to_runtime": False,
        "manual_agent_labels_included": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a reviewed, fixed-size evaluation JSONL dataset."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--reference-review",
        type=Path,
        help=(
            "Explicit reference-review overlay pinned by file and SHA-256 in the "
            "selection manifest."
        ),
    )
    parser.add_argument("--allow-unconfirmed-draft", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = materialize(
        args.candidates,
        args.manifest,
        args.glossary,
        args.output,
        args.summary_output,
        allow_unconfirmed_draft=args.allow_unconfirmed_draft,
        reference_review_path=args.reference_review,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
