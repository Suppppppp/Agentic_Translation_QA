"""Ingest a completed manual-review XLSX without changing source artifacts.

The importer intentionally uses only the Python standard library.  It reads the
``Review`` worksheet from the XLSX container, validates its order and immutable
cells against a pinned CSV template, validates the human labels, and writes a
new CSV plus a provenance record.  It never evaluates formulas, calls a model,
or mutates any input file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import re
import zipfile
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


MANUAL_COLUMNS = (
    "manual_initial_needs_revision",
    "manual_severity",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
)
KEY_COLUMNS = ("review_key", "case_id", "mode")
SUPPLEMENT_COLUMNS = (
    "review_key",
    "case_id",
    "mode",
    "manual_initial_needs_revision",
    "manual_severity",
    "severity_reviewer",
    "severity_note",
    "source_text",
    "initial_translation",
    "manual_primary_error",
    "manual_error_types",
)
SUPPLEMENT_MANUAL_COLUMNS = (
    "manual_severity",
    "severity_reviewer",
    "severity_note",
)
ALLOWED_SEVERITIES = ("MAJOR", "MINOR")
ALLOWED_ERROR_TYPES = {
    "term",
    "meaning",
    "omission_addition",
    "entity_value",
    "fluency_grammar",
    "other",
}
ALLOWED_OUTCOMES = {"improved", "same", "worse"}
ALLOWED_REVIEW_STATUSES = {"confirmed", "ambiguous"}
INGEST_KINDS = {"full", "severity-supplement"}

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read baseline CSV: {path}") from exc
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("baseline CSV has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("baseline CSV contains duplicate headers")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def _project_file(project_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"selection manifest requires nonblank {field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"selection manifest {field} must be project-relative")
    return (project_root / relative).resolve()


def _manifest_keys(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection manifest requires a nonempty selected array")
    keys: list[tuple[str, str, str]] = []
    for index, item in enumerate(selected, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"selected item {index} must be an object")
        case_id = item.get("case_id")
        mode = item.get("mode")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"selected item {index} has no case_id")
        if mode not in {"agent", "agent_rag"}:
            raise ValueError(
                f"selected item {index} mode must be agent or agent_rag"
            )
        expected_review_key = f"{case_id}::{mode}"
        review_key = item.get("review_key", expected_review_key)
        if review_key != expected_review_key:
            raise ValueError(f"selected item {index} has an invalid review_key")
        keys.append((expected_review_key, case_id, mode))
    if len(keys) != len(set(keys)):
        raise ValueError("selection manifest contains duplicate review keys")
    return keys


def _column_index(reference: str) -> int:
    match = _CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(raw)
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{_MAIN_NS}}}t"))
        for item in root.findall(f"{{{_MAIN_NS}}}si")
    ]


def _review_sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
    except (KeyError, ElementTree.ParseError) as exc:
        raise ValueError("XLSX workbook metadata is invalid") from exc

    relationship_id: str | None = None
    for sheet in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
        if sheet.get("name") == sheet_name:
            relationship_id = sheet.get(f"{{{_OFFICE_REL_NS}}}id")
            break
    if not relationship_id:
        raise ValueError(f'XLSX has no worksheet named "{sheet_name}"')

    target: str | None = None
    for relation in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship"):
        if relation.get("Id") == relationship_id:
            if relation.get("TargetMode") == "External":
                raise ValueError("Review worksheet relationship cannot be external")
            target = relation.get("Target")
            break
    if not target:
        raise ValueError("XLSX Review worksheet relationship is missing")

    if target.startswith("/"):
        resolved = target.lstrip("/")
    else:
        resolved = posixpath.normpath(posixpath.join("xl", target))
    if resolved.startswith("../") or resolved not in archive.namelist():
        raise ValueError("XLSX Review worksheet target is invalid")
    return resolved


def _cell_text(cell: ElementTree.Element, shared: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(
            node.text or "" for node in inline.findall(f".//{{{_MAIN_NS}}}t")
        )

    value_node = cell.find(f"{{{_MAIN_NS}}}v")
    value = "" if value_node is None or value_node.text is None else value_node.text
    if cell_type == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError) as exc:
            raise ValueError("XLSX contains an invalid shared-string index") from exc
    if cell_type == "b":
        if value == "1":
            return "True"
        if value == "0":
            return "False"
        raise ValueError("XLSX contains an invalid boolean cell")
    return value


def read_review_worksheet(
    workbook_path: Path, sheet_name: str = "Review"
) -> tuple[list[str], list[dict[str, str]], str]:
    """Return exact displayed cell payloads from the named worksheet.

    Strings are not trimmed, case-folded, or rewritten.  XLSX boolean evidence
    is represented as ``True``/``False`` to match Python CSV serialization.
    """

    try:
        archive = zipfile.ZipFile(workbook_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot read XLSX workbook: {workbook_path}") from exc
    with archive:
        shared = _shared_strings(archive)
        sheet_path = _review_sheet_path(archive, sheet_name)
        try:
            root = ElementTree.fromstring(archive.read(sheet_path))
        except (KeyError, ElementTree.ParseError) as exc:
            raise ValueError("XLSX Review worksheet XML is invalid") from exc

    dimension = root.find(f"{{{_MAIN_NS}}}dimension")
    source_range = dimension.get("ref", "") if dimension is not None else ""
    indexed_rows: dict[int, dict[int, str]] = {}
    formula_cells: set[tuple[int, int]] = set()
    for row in root.findall(f".//{{{_MAIN_NS}}}sheetData/{{{_MAIN_NS}}}row"):
        raw_row_number = row.get("r")
        if raw_row_number is None:
            raise ValueError("XLSX Review row has no row number")
        try:
            row_number = int(raw_row_number)
        except ValueError as exc:
            raise ValueError("XLSX Review row number is invalid") from exc
        cells: dict[int, str] = {}
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            reference = cell.get("r")
            if not reference:
                raise ValueError("XLSX Review cell has no reference")
            column = _column_index(reference)
            if column in cells:
                raise ValueError(f"XLSX Review row {row_number} has duplicate cells")
            if cell.find(f"{{{_MAIN_NS}}}f") is not None:
                formula_cells.add((row_number, column))
            cells[column] = _cell_text(cell, shared)
        indexed_rows[row_number] = cells

    header_cells = indexed_rows.get(1)
    if not header_cells:
        raise ValueError("XLSX Review worksheet has no header row")
    last_header_column = max(header_cells)
    headers = [header_cells.get(index, "") for index in range(last_header_column + 1)]
    if any(not header for header in headers):
        raise ValueError("XLSX Review header contains a blank cell")
    if len(headers) != len(set(headers)):
        raise ValueError("XLSX Review header contains duplicates")
    for row_number, column in formula_cells:
        header = headers[column] if column < len(headers) else ""
        if header != "row_status":
            location = header or "an extra column"
            raise ValueError(
                f"XLSX Review row {row_number} contains a formula in {location}"
            )

    rows: list[dict[str, str]] = []
    for row_number in sorted(number for number in indexed_rows if number > 1):
        cells = indexed_rows[row_number]
        values = [cells.get(index, "") for index in range(len(headers))]
        if not any(values):
            continue
        rows.append(dict(zip(headers, values, strict=True)))
    if not rows:
        raise ValueError("XLSX Review worksheet has no review rows")
    if not source_range:
        source_range = f"A1:{_column_name(len(headers) - 1)}{len(rows) + 1}"
    return headers, rows, source_range


def _column_name(index: int) -> str:
    result = ""
    number = index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _validate_manual_labels(row: dict[str, str], row_number: int) -> str | None:
    manual = {column: row[column] for column in MANUAL_COLUMNS}
    if not any(manual.values()):
        raise ValueError(f"row {row_number} is pending; complete the manual labels")
    for field in (
        "manual_initial_needs_revision",
        "pairwise_outcome",
        "review_status",
        "reviewer",
    ):
        if not manual[field] or not manual[field].strip():
            raise ValueError(f"row {row_number} requires {field}")

    needs_revision = manual["manual_initial_needs_revision"]
    severity = manual["manual_severity"]
    if needs_revision not in {"true", "false"}:
        raise ValueError(
            f"row {row_number} manual_initial_needs_revision must be true or false"
        )
    if needs_revision == "true":
        if severity not in ALLOWED_SEVERITIES:
            raise ValueError(
                f"row {row_number} manual_severity must be MAJOR or MINOR "
                "when manual_initial_needs_revision is true"
            )
    elif severity:
        raise ValueError(
            f"row {row_number} manual_severity must be blank when "
            "manual_initial_needs_revision is false"
        )

    primary_error = manual["manual_primary_error"]
    error_types_text = manual["manual_error_types"]
    if needs_revision == "false":
        if primary_error or error_types_text:
            raise ValueError(
                f"row {row_number} PASS label cannot contain manual error types"
            )
        parsed_errors: list[str] = []
    else:
        if primary_error not in ALLOWED_ERROR_TYPES:
            raise ValueError(f"row {row_number} requires a valid manual_primary_error")
        if not error_types_text:
            raise ValueError(f"row {row_number} requires manual_error_types")
        try:
            parsed = json.loads(error_types_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"row {row_number} manual_error_types must be a JSON list"
            ) from exc
        if not isinstance(parsed, list) or any(
            not isinstance(item, str) for item in parsed
        ):
            raise ValueError(
                f"row {row_number} manual_error_types must be a JSON string list"
            )
        parsed_errors = parsed
        if not parsed_errors:
            raise ValueError(f"row {row_number} requires manual_error_types")
        if len(parsed_errors) != len(set(parsed_errors)):
            raise ValueError(
                f"row {row_number} manual_error_types contains duplicates"
            )
        if any(error not in ALLOWED_ERROR_TYPES for error in parsed_errors):
            raise ValueError(
                f"row {row_number} manual_error_types contains an invalid value"
            )
        if primary_error not in parsed_errors:
            raise ValueError(
                f"row {row_number} manual_primary_error must appear in "
                "manual_error_types"
            )

    if manual["pairwise_outcome"] not in ALLOWED_OUTCOMES:
        raise ValueError(f"row {row_number} pairwise_outcome is invalid")
    if manual["review_status"] not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"row {row_number} review_status is invalid")
    note_is_blank = not manual["note"] or not manual["note"].strip()
    if (primary_error == "other" or "other" in parsed_errors) and note_is_blank:
        raise ValueError(f"row {row_number} requires note when other is used")
    if manual["review_status"] == "ambiguous" and note_is_blank:
        raise ValueError(
            f"row {row_number} requires note when review_status is ambiguous"
        )
    return severity or None


def _validate_severity_supplement(
    row: dict[str, str], row_number: int
) -> str | None:
    needs_revision = row["manual_initial_needs_revision"]
    severity = row["manual_severity"]
    reviewer = row["severity_reviewer"]
    note = row["severity_note"]
    if needs_revision not in {"true", "false"}:
        raise ValueError(
            f"row {row_number} manual_initial_needs_revision must be true or false"
        )
    if needs_revision == "true":
        if severity not in ALLOWED_SEVERITIES:
            raise ValueError(
                f"row {row_number} manual_severity must be MAJOR or MINOR "
                "when manual_initial_needs_revision is true"
            )
        if not reviewer or not reviewer.strip():
            raise ValueError(f"row {row_number} requires severity_reviewer")
        return severity
    if severity:
        raise ValueError(
            f"row {row_number} manual_severity must be blank when "
            "manual_initial_needs_revision is false"
        )
    if reviewer or note:
        raise ValueError(
            f"row {row_number} severity supplement fields must be blank when "
            "manual_initial_needs_revision is false"
        )
    return None


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def ingest_manual_review_workbook(
    *,
    workbook_path: Path,
    selection_manifest_path: Path,
    baseline_csv_path: Path,
    output_csv_path: Path,
    provenance_output_path: Path,
    project_root: Path,
    ingest_kind: str,
    sheet_name: str = "Review",
) -> dict[str, Any]:
    """Validate one completed workbook and write new, auditable outputs."""

    if ingest_kind not in INGEST_KINDS:
        raise ValueError("ingest_kind must be full or severity-supplement")
    inputs = {
        workbook_path.resolve(),
        selection_manifest_path.resolve(),
        baseline_csv_path.resolve(),
    }
    outputs = {output_csv_path.resolve(), provenance_output_path.resolve()}
    if len(outputs) != 2:
        raise ValueError("CSV output and provenance output must be different files")
    if inputs & outputs:
        raise ValueError("outputs must not overwrite workbook, manifest, or baseline CSV")
    for output in (output_csv_path, provenance_output_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    manifest = _load_json_object(selection_manifest_path, "selection manifest")
    schema_version = manifest.get("schema_version")
    if schema_version != 1:
        raise ValueError("selection manifest schema_version must be 1")
    if manifest.get("manual_review_schema_version") != 2:
        raise ValueError("selection manifest manual_review_schema_version must be 2")
    if ingest_kind == "severity-supplement":
        if manifest.get("review_kind") != "severity_supplement":
            raise ValueError(
                "severity supplement manifest must declare review_kind=severity_supplement"
            )
    keys = _manifest_keys(manifest)

    project_root = project_root.resolve()
    declared_base_selection: Path | None = None
    if "base_selection_file" in manifest or "base_selection_sha256" in manifest:
        declared_base_selection = _project_file(
            project_root,
            manifest.get("base_selection_file"),
            "base_selection_file",
        )
        expected_base_selection_hash = manifest.get("base_selection_sha256")
        if (
            not isinstance(expected_base_selection_hash, str)
            or not declared_base_selection.is_file()
            or _sha256(declared_base_selection) != expected_base_selection_hash
        ):
            raise ValueError("base selection hash does not match selection manifest")

    if (
        "review_template_file" not in manifest
        or "review_template_sha256" not in manifest
    ):
        raise ValueError(
            "selection manifest requires review_template_file and "
            "review_template_sha256"
        )
    declared_template = _project_file(
        project_root, manifest.get("review_template_file"), "review_template_file"
    )
    if declared_template != baseline_csv_path.resolve():
        raise ValueError("baseline CSV path does not match selection manifest")
    expected_template_hash = manifest.get("review_template_sha256")
    if (
        not isinstance(expected_template_hash, str)
        or _sha256(baseline_csv_path) != expected_template_hash
    ):
        raise ValueError("baseline CSV hash does not match selection manifest")

    declared_base: Path | None = None
    if "base_review_csv_file" in manifest or "base_review_csv_sha256" in manifest:
        declared_base = _project_file(
            project_root, manifest.get("base_review_csv_file"), "base_review_csv_file"
        )
        expected_base_hash = manifest.get("base_review_csv_sha256")
        if (
            not isinstance(expected_base_hash, str)
            or not declared_base.is_file()
            or _sha256(declared_base) != expected_base_hash
        ):
            raise ValueError("base review CSV hash does not match selection manifest")
    if ingest_kind == "severity-supplement" and declared_base is None:
        raise ValueError("severity supplement manifest requires base_review_csv_file")
    if ingest_kind == "severity-supplement" and declared_base_selection is None:
        raise ValueError("severity supplement manifest requires base_selection_file")

    fieldnames, baseline_rows = _load_csv(baseline_csv_path)
    expected_columns = (
        list(SUPPLEMENT_COLUMNS)
        if ingest_kind == "severity-supplement"
        else None
    )
    if expected_columns is not None and fieldnames != expected_columns:
        raise ValueError(
            "severity supplement baseline CSV columns/order do not match the v2 contract"
        )
    required_columns = (
        SUPPLEMENT_COLUMNS
        if ingest_kind == "severity-supplement"
        else (*KEY_COLUMNS, *MANUAL_COLUMNS)
    )
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(f"baseline CSV is missing columns: {', '.join(missing)}")
    baseline_keys = [
        tuple(row[column] for column in KEY_COLUMNS) for row in baseline_rows
    ]
    if baseline_keys != keys:
        raise ValueError("baseline CSV keys/order do not match selection manifest")

    base_full_fieldnames: list[str] | None = None
    base_full_rows: list[dict[str, str]] | None = None
    output_fieldnames = fieldnames
    if ingest_kind == "severity-supplement":
        assert declared_base is not None
        base_full_fieldnames, base_full_rows = _load_csv(declared_base)
        if "manual_severity" in base_full_fieldnames:
            raise ValueError("base review CSV must be the original pre-severity file")
        required_base_columns = {
            *KEY_COLUMNS,
            "manual_initial_needs_revision",
            "manual_primary_error",
            "manual_error_types",
            "pairwise_outcome",
            "review_status",
            "reviewer",
            "note",
            "source_text",
            "initial_translation",
        }
        missing_base = sorted(required_base_columns - set(base_full_fieldnames))
        if missing_base:
            raise ValueError(
                "base review CSV is missing columns: " + ", ".join(missing_base)
            )
        base_full_keys = [
            tuple(row[column] for column in KEY_COLUMNS) for row in base_full_rows
        ]
        if base_full_keys != keys:
            raise ValueError(
                "base review CSV keys/order do not match selection manifest"
            )
        copied_columns = (
            "review_key",
            "case_id",
            "mode",
            "manual_initial_needs_revision",
            "source_text",
            "initial_translation",
            "manual_primary_error",
            "manual_error_types",
        )
        for row_number, (template_row, base_row) in enumerate(
            zip(baseline_rows, base_full_rows, strict=True), start=2
        ):
            for column in copied_columns:
                if template_row[column] != base_row[column]:
                    raise ValueError(
                        f"severity template row {row_number} differs from base "
                        f"review CSV in {column}"
                    )
        severity_position = base_full_fieldnames.index(
            "manual_initial_needs_revision"
        ) + 1
        output_fieldnames = [*base_full_fieldnames]
        output_fieldnames.insert(severity_position, "manual_severity")

    workbook_headers, workbook_rows, source_range = read_review_worksheet(
        workbook_path, sheet_name
    )
    allowed_headers = [fieldnames, [*fieldnames, "row_status"]]
    if workbook_headers not in allowed_headers:
        raise ValueError(
            "XLSX Review headers/order must exactly match the baseline CSV, "
            "with only an optional trailing row_status"
        )
    if len(workbook_rows) != len(baseline_rows):
        raise ValueError("XLSX Review row count does not match baseline CSV")
    workbook_keys = [
        tuple(row[column] for column in KEY_COLUMNS) for row in workbook_rows
    ]
    if workbook_keys != keys:
        raise ValueError("XLSX Review keys/order do not match selection manifest")

    mutable_columns = (
        MANUAL_COLUMNS if ingest_kind == "full" else SUPPLEMENT_MANUAL_COLUMNS
    )
    mutable = set(mutable_columns)
    immutable = [column for column in fieldnames if column not in mutable]
    output_rows: list[dict[str, str]] = []
    severity_counts: Counter[str] = Counter()
    required_severity_count = 0
    severity_annotations: list[dict[str, str]] = []
    for row_number, (workbook_row, baseline_row) in enumerate(
        zip(workbook_rows, baseline_rows, strict=True), start=2
    ):
        for column in immutable:
            if workbook_row[column] != baseline_row[column]:
                raise ValueError(
                    f"row {row_number} immutable value differs in {column}"
                )
        ingested_template_row = {
            column: (
                workbook_row[column] if column in mutable else baseline_row[column]
            )
            for column in fieldnames
        }
        if ingest_kind == "full":
            output_row = ingested_template_row
            severity = _validate_manual_labels(output_row, row_number)
        else:
            severity = _validate_severity_supplement(
                ingested_template_row, row_number
            )
            assert base_full_rows is not None
            base_full_row = base_full_rows[row_number - 2]
            output_row = {
                column: (
                    ingested_template_row["manual_severity"]
                    if column == "manual_severity"
                    else base_full_row[column]
                )
                for column in output_fieldnames
            }
            _validate_manual_labels(output_row, row_number)
            severity_annotations.append(
                {
                    "review_key": ingested_template_row["review_key"],
                    "manual_severity": ingested_template_row["manual_severity"],
                    "severity_reviewer": ingested_template_row[
                        "severity_reviewer"
                    ],
                    "severity_note": ingested_template_row["severity_note"],
                }
            )
        if output_row["manual_initial_needs_revision"] == "true":
            required_severity_count += 1
        if severity is not None:
            severity_counts[severity] += 1
        output_rows.append(output_row)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=output_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    provenance = {
        "schema_version": 1,
        "manual_review_schema_version": 2,
        "batch_id": manifest.get("batch_id"),
        "ingest_kind": ingest_kind,
        "ingested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "selected_count": len(keys),
        "ready_count": len(output_rows),
        "severity_required_count": required_severity_count,
        "severity_counts": {
            severity: severity_counts[severity] for severity in ALLOWED_SEVERITIES
        },
        "automatic_severity_labels_generated": False,
        "immutable_values_verified": True,
        "manual_fields_copied": list(mutable_columns),
        "source_sheet": sheet_name,
        "source_range": source_range,
        "source_workbook_file": _display_path(workbook_path, project_root),
        "source_workbook_sha256": _sha256(workbook_path),
        "selection_manifest_file": _display_path(
            selection_manifest_path, project_root
        ),
        "selection_manifest_sha256": _sha256(selection_manifest_path),
        "baseline_csv_file": _display_path(baseline_csv_path, project_root),
        "baseline_csv_sha256": _sha256(baseline_csv_path),
        "output_csv_file": _display_path(output_csv_path, project_root),
        "output_csv_sha256": _sha256(output_csv_path),
    }
    if ingest_kind == "severity-supplement":
        assert declared_base is not None
        assert declared_base_selection is not None
        provenance["base_selection_file"] = _display_path(
            declared_base_selection, project_root
        )
        provenance["base_selection_sha256"] = _sha256(declared_base_selection)
        provenance["base_review_csv_file"] = _display_path(
            declared_base, project_root
        )
        provenance["base_review_csv_sha256"] = _sha256(declared_base)
        provenance["severity_annotations"] = severity_annotations
    provenance_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with provenance_output_path.open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        output_csv_path.unlink(missing_ok=True)
        raise
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed Review worksheet and ingest its human labels "
            "without modifying any input."
        )
    )
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--kind", choices=sorted(INGEST_KINDS), default="full")
    parser.add_argument("--sheet-name", default="Review")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = ingest_manual_review_workbook(
        workbook_path=args.workbook,
        selection_manifest_path=args.selection_manifest,
        baseline_csv_path=args.baseline_csv,
        output_csv_path=args.output_csv,
        provenance_output_path=args.provenance_output,
        project_root=args.project_root,
        ingest_kind=args.kind,
        sheet_name=args.sheet_name,
    )
    print(
        f"ingested {result['ready_count']} {args.kind} review rows to "
        f"{args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
