"""Promote a reviewed evaluation draft to human-confirmed metadata.

This command does not infer review decisions from the workbook.  It records an
explicit conversation confirmation, preserves every draft correction, and
adds ``keep_original`` for every other selected case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CASE_COUNT = 40
SHA256_LENGTH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"candidate line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"candidate line {line_number} is not an object")
            source_id = record.get("source_record_id")
            reference = record.get("english_reference")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(
                    f"candidate line {line_number} has no source_record_id"
                )
            if source_id in candidates:
                raise ValueError(f"duplicate candidate source_record_id: {source_id}")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"candidate has no English reference: {source_id}"
                )
            candidates[source_id] = record
    return candidates


def _relative_posix(path: Path, *, base: Path, label: str) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside project_root") from exc
    if relative == Path("."):
        raise ValueError(f"{label} must name a file")
    return relative.as_posix()


def _manifest_relative_path(path: Path, *, manifest_path: Path) -> str:
    value = os.path.relpath(path.resolve(), start=manifest_path.parent.resolve())
    if Path(value).is_absolute():  # pragma: no cover - defensive on unusual platforms
        raise ValueError("reference review path must be relative to the manifest")
    return Path(value).as_posix()


def _validate_reviewed_at_utc(value: str) -> str:
    if not value.endswith("Z"):
        raise ValueError("reviewed_at_utc must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("reviewed_at_utc must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("reviewed_at_utc must be in UTC")
    return value


def _required_nonempty(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must be a non-empty string")
    return stripped


def _selected_ids(
    manifest: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> list[str]:
    raw_selected = manifest.get("selected")
    if not isinstance(raw_selected, list) or len(raw_selected) != EXPECTED_CASE_COUNT:
        actual = len(raw_selected) if isinstance(raw_selected, list) else "invalid"
        raise ValueError(
            f"draft manifest must contain exactly {EXPECTED_CASE_COUNT} selected "
            f"cases; found {actual}"
        )
    selected_ids: list[str] = []
    for item in raw_selected:
        if not isinstance(item, dict):
            raise ValueError("each selected manifest item must be an object")
        source_id = item.get("source_record_id")
        if not isinstance(source_id, str) or source_id not in candidates:
            raise ValueError(f"selected source_record_id is not in candidates: {source_id}")
        if source_id in selected_ids:
            raise ValueError(f"duplicate selected source_record_id: {source_id}")
        selected_ids.append(source_id)
    return selected_ids


def _draft_decisions(
    overlay: dict[str, Any],
    *,
    selected_ids: Sequence[str],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    raw_decisions = overlay.get("decisions")
    raw_corrections = overlay.get("corrections")
    if raw_decisions is not None and raw_corrections is not None:
        raise ValueError("draft overlay must use either decisions or corrections")
    corrections_only = raw_decisions is None and raw_corrections is not None
    if corrections_only:
        raw_decisions = raw_corrections
    if not isinstance(raw_decisions, list):
        raise ValueError("draft overlay decisions/corrections must be an array")

    selected_set = set(selected_ids)
    decisions: dict[str, dict[str, Any]] = {}
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("each draft overlay decision must be an object")
        source_id = raw.get("source_record_id")
        if not isinstance(source_id, str) or source_id not in selected_set:
            raise ValueError(f"draft overlay targets an unselected case: {source_id}")
        if source_id in decisions:
            raise ValueError(f"duplicate draft overlay decision: {source_id}")
        decision = "corrected" if corrections_only else raw.get("decision")
        if decision not in {"corrected", "keep_original"}:
            raise ValueError(f"invalid draft overlay decision for {source_id}")

        original_reference = candidates[source_id]["english_reference"]
        expected_original_hash = _text_sha256(original_reference)
        if raw.get("original_reference_sha256") != expected_original_hash:
            raise ValueError(f"original reference hash does not match candidate: {source_id}")

        corrected_text = raw.get("corrected_reference_text")
        rationale = raw.get("rationale")
        if decision == "corrected":
            if not isinstance(corrected_text, str) or not corrected_text.strip():
                raise ValueError(
                    f"corrected decision requires corrected_reference_text: {source_id}"
                )
            if corrected_text.strip() == original_reference.strip():
                raise ValueError(
                    f"corrected reference must differ from the original: {source_id}"
                )
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"corrected decision requires a rationale: {source_id}")
        elif corrected_text not in (None, ""):
            raise ValueError(
                f"keep_original decision cannot contain corrected text: {source_id}"
            )

        canonical: dict[str, Any] = {
            "source_record_id": source_id,
            "decision": decision,
            "original_reference_sha256": expected_original_hash,
        }
        if decision == "corrected":
            canonical["corrected_reference_text"] = corrected_text.strip()
            canonical["rationale"] = rationale.strip()
        elif isinstance(rationale, str) and rationale.strip():
            canonical["rationale"] = rationale.strip()
        decisions[source_id] = canonical
    return decisions


def _reject_absolute_strings(value: object, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_strings(child, f"{location}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"output metadata contains an absolute path at {location}")


def finalize_review(
    candidate_path: Path,
    draft_manifest_path: Path,
    draft_reference_review_path: Path,
    review_workbook_path: Path,
    output_manifest_path: Path,
    output_reference_review_path: Path,
    *,
    reviewer: str,
    reviewed_at_utc: str,
    confirmation_basis: str,
    project_root: Path,
) -> dict[str, Any]:
    """Write final human-confirmed manifest and reference-review overlay."""

    reviewer = _required_nonempty(reviewer, "reviewer")
    reviewed_at_utc = _validate_reviewed_at_utc(reviewed_at_utc)
    confirmation_basis = _required_nonempty(
        confirmation_basis, "confirmation_basis"
    )
    project_root = project_root.resolve()

    paths = {
        "candidate_path": candidate_path,
        "draft_manifest_path": draft_manifest_path,
        "draft_reference_review_path": draft_reference_review_path,
        "review_workbook_path": review_workbook_path,
        "output_manifest_path": output_manifest_path,
        "output_reference_review_path": output_reference_review_path,
    }
    relative_paths = {
        label: _relative_posix(path, base=project_root, label=label)
        for label, path in paths.items()
    }
    if output_manifest_path.resolve() in {
        draft_manifest_path.resolve(),
        draft_reference_review_path.resolve(),
    } or output_reference_review_path.resolve() in {
        draft_manifest_path.resolve(),
        draft_reference_review_path.resolve(),
    }:
        raise ValueError("final outputs must not overwrite draft inputs")
    if output_manifest_path.exists() or output_reference_review_path.exists():
        raise ValueError("final output already exists; refusing to overwrite")
    if not review_workbook_path.is_file() or review_workbook_path.stat().st_size == 0:
        raise ValueError("confirmed review workbook must be a non-empty file")

    manifest = _read_json_object(draft_manifest_path, "draft manifest")
    overlay = _read_json_object(draft_reference_review_path, "draft overlay")
    if manifest.get("human_confirmed") is not False:
        raise ValueError("draft manifest is already final or lacks human_confirmed=false")
    if overlay.get("human_confirmed") is not False:
        raise ValueError("draft overlay is already final or lacks human_confirmed=false")

    actual_candidate_hash = _sha256(candidate_path)
    if manifest.get("candidate_sha256") != actual_candidate_hash:
        raise ValueError("candidate hash does not match draft manifest")
    if overlay.get("candidate_sha256") != actual_candidate_hash:
        raise ValueError("candidate hash does not match draft overlay")

    declared_review_file = manifest.get("reference_review_file")
    declared_review_hash = manifest.get("reference_review_sha256")
    if not isinstance(declared_review_file, str) or Path(declared_review_file).is_absolute():
        raise ValueError("draft manifest must pin a relative reference_review_file")
    expected_draft_overlay = (
        draft_manifest_path.parent / declared_review_file
    ).resolve()
    if expected_draft_overlay != draft_reference_review_path.resolve():
        raise ValueError("draft reference review path does not match manifest pin")
    if not _is_sha256(declared_review_hash) or declared_review_hash != _sha256(
        draft_reference_review_path
    ):
        raise ValueError("draft reference review hash does not match manifest pin")

    candidates = _load_candidates(candidate_path)
    selected_ids = _selected_ids(manifest, candidates)
    ordered_id_hash = _text_sha256("\n".join(selected_ids))
    if overlay.get("ordered_source_id_sha256") != ordered_id_hash:
        raise ValueError("ordered source ID hash does not match draft selection")

    draft_decisions = _draft_decisions(
        overlay, selected_ids=selected_ids, candidates=candidates
    )
    final_decisions: list[dict[str, Any]] = []
    for case_order, source_id in enumerate(selected_ids, start=1):
        decision = draft_decisions.get(source_id)
        if decision is None:
            reference = candidates[source_id]["english_reference"]
            decision = {
                "source_record_id": source_id,
                "decision": "keep_original",
                "original_reference_sha256": _text_sha256(reference),
            }
        final_decisions.append({"case_order": case_order, **decision})
    if len(final_decisions) != EXPECTED_CASE_COUNT:
        raise ValueError(f"final overlay must contain {EXPECTED_CASE_COUNT} decisions")

    confirmation = {
        "kind": "explicit_user_confirmation_in_conversation",
        "basis": confirmation_basis,
    }
    workbook_relative = relative_paths["review_workbook_path"]
    workbook_hash = _sha256(review_workbook_path)

    final_overlay: dict[str, Any] = {
        "schema_version": 1,
        "review_status": "human_confirmed",
        "human_confirmed": True,
        "candidate_file": relative_paths["candidate_path"],
        "candidate_sha256": actual_candidate_hash,
        "selection_manifest_file": relative_paths["output_manifest_path"],
        "ordered_source_id_sha256": ordered_id_hash,
        "reviewer_type": "human_user",
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at_utc,
        "review_workbook_file": workbook_relative,
        "review_workbook_sha256": workbook_hash,
        "confirmation": confirmation,
        "original_reference_sha256_contract": overlay.get(
            "original_reference_sha256_contract",
            "SHA-256 of the exact UTF-8 english_reference string from the anchored "
            "candidate file, with no trailing newline.",
        ),
        "reference_policy": overlay.get(
            "reference_policy",
            {
                "public_candidate_reference_preserved": True,
                "corrections_are_overlay_only": True,
                "application_boundary": (
                    "explicit_offline_materialization_only_never_runtime"
                ),
            },
        ),
        "decisions": final_decisions,
    }
    for source_key in (
        "feedback_file_label",
        "feedback_sha256",
        "source_feedback_sha256",
    ):
        if source_key in overlay:
            final_overlay[source_key] = overlay[source_key]

    final_manifest = dict(manifest)
    final_manifest.update(
        {
            "status": "HUMAN_CONFIRMED_FINAL",
            "human_confirmed": True,
            "candidate_file": relative_paths["candidate_path"],
            "reference_review_file": _manifest_relative_path(
                output_reference_review_path, manifest_path=output_manifest_path
            ),
            "review_workbook_file": workbook_relative,
            "review_workbook_sha256": workbook_hash,
            "review_confirmation": {
                **confirmation,
                "reviewer": reviewer,
                "reviewed_at_utc": reviewed_at_utc,
            },
        }
    )
    selection_metadata = final_manifest.get("selection_metadata")
    if not isinstance(selection_metadata, dict):
        selection_metadata = {}
    else:
        selection_metadata = dict(selection_metadata)
    selection_metadata["reviewer_type"] = "human_user"
    selection_metadata["human_review_completed"] = True
    selection_metadata.pop("human_review_required", None)
    final_manifest["selection_metadata"] = selection_metadata

    _reject_absolute_strings(final_overlay)
    overlay_bytes = (
        json.dumps(final_overlay, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    overlay_hash = hashlib.sha256(overlay_bytes).hexdigest()
    final_manifest["reference_review_sha256"] = overlay_hash
    _reject_absolute_strings(final_manifest)
    manifest_bytes = (
        json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    output_reference_review_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_reference_review_path.write_bytes(overlay_bytes)
    output_manifest_path.write_bytes(manifest_bytes)

    decision_counts = {
        decision: sum(item["decision"] == decision for item in final_decisions)
        for decision in ("corrected", "keep_original")
    }
    return {
        "status": "HUMAN_CONFIRMED_FINAL",
        "case_count": len(final_decisions),
        "decision_counts": decision_counts,
        "manifest_file": relative_paths["output_manifest_path"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reference_review_file": relative_paths["output_reference_review_path"],
        "reference_review_sha256": overlay_hash,
        "review_workbook_file": workbook_relative,
        "review_workbook_sha256": workbook_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a 40-case draft after explicit human review confirmation."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--draft-reference-review", type=Path, required=True)
    parser.add_argument("--review-workbook", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-reference-review", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--confirmation-basis", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize_review(
        args.candidates,
        args.draft_manifest,
        args.draft_reference_review,
        args.review_workbook,
        args.output_manifest,
        args.output_reference_review,
        reviewer=args.reviewer,
        reviewed_at_utc=args.reviewed_at_utc,
        confirmation_basis=args.confirmation_basis,
        project_root=args.project_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
