from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import pytest

from scripts.ingest_manual_review_workbook import ingest_manual_review_workbook
from scripts.prepare_severity_supplement import prepare_severity_supplement


FULL_COLUMNS = [
    "review_key",
    "case_id",
    "mode",
    "manual_initial_needs_revision",
    "manual_severity",
    "manual_primary_error",
    "manual_error_types",
    "pairwise_outcome",
    "review_status",
    "reviewer",
    "note",
    "source_text",
    "reference_text",
    "initial_translation",
    "final_translation",
    "agent_initial_passed",
    "agent_initial_error_types",
    "agent_initial_summary",
    "retry_count",
    "stop_reason",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _column_name(index: int) -> str:
    result = ""
    number = index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _write_xlsx(
    path: Path,
    *,
    sheet_name: str,
    headers: list[str],
    rows: list[dict[str, str]],
    use_shared_strings: bool = False,
    typed_evidence: bool = False,
    formula_cells: set[tuple[int, str]] | None = None,
) -> None:
    all_rows = [dict(zip(headers, headers, strict=True)), *rows]
    shared_values: list[str] = []
    shared_index: dict[str, int] = {}

    formula_cells = formula_cells or set()

    def cell_xml(
        reference: str, row_number: int, header: str, value: str
    ) -> str:
        if (row_number, header) in formula_cells:
            return (
                f'<c r="{reference}" t="str"><f>"{escape(value)}"</f>'
                f"<v>{escape(value)}</v></c>"
            )
        if typed_evidence and header == "agent_initial_passed":
            boolean = {"True": "1", "False": "0"}.get(value)
            if boolean is not None:
                return f'<c r="{reference}" t="b"><v>{boolean}</v></c>'
        if typed_evidence and header == "retry_count" and value.isdigit():
            return f'<c r="{reference}"><v>{value}</v></c>'
        if use_shared_strings:
            if value not in shared_index:
                shared_index[value] = len(shared_values)
                shared_values.append(value)
            return f'<c r="{reference}" t="s"><v>{shared_index[value]}</v></c>'
        return (
            f'<c r="{reference}" t="inlineStr"><is>'
            f'<t xml:space="preserve">{escape(value)}</t></is></c>'
        )

    row_xml: list[str] = []
    for row_number, row in enumerate(all_rows, start=1):
        cells = []
        for column, header in enumerate(headers):
            value = row.get(header, "")
            if value == "":
                continue
            cells.append(
                cell_xml(
                    f"{_column_name(column)}{row_number}",
                    row_number,
                    header,
                    value,
                )
            )
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    last_cell = f"{_column_name(len(headers) - 1)}{len(all_rows)}"
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_cell}"/><sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name={quoteattr(sheet_name)} sheetId="1" r:id="rId1"/>'
        "</sheets></workbook>"
    )
    workbook_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    overrides = (
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        if use_shared_strings
        else ""
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        f"{overrides}</Types>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        if use_shared_strings:
            items = "".join(
                f'<si><t xml:space="preserve">{escape(value)}</t></si>'
                for value in shared_values
            )
            archive.writestr(
                "xl/sharedStrings.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                f'count="{len(shared_values)}" uniqueCount="{len(shared_values)}">'
                f"{items}</sst>",
            )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _full_fixture(tmp_path: Path) -> dict[str, object]:
    baseline = tmp_path / "data" / "batch2" / "review_labels.csv"
    baseline_rows = [
        {
            "review_key": "case-1::agent",
            "case_id": "case-1",
            "mode": "agent",
            "manual_initial_needs_revision": "",
            "manual_severity": "",
            "manual_primary_error": "",
            "manual_error_types": "",
            "pairwise_outcome": "",
            "review_status": "",
            "reviewer": "",
            "note": "",
            "source_text": "원문 1",
            "reference_text": "Reference 1",
            "initial_translation": "Initial 1",
            "final_translation": "Initial 1",
            "agent_initial_passed": "True",
            "agent_initial_error_types": "[]",
            "agent_initial_summary": "summary 1",
            "retry_count": "0",
            "stop_reason": "passed",
        },
        {
            "review_key": "case-2::agent_rag",
            "case_id": "case-2",
            "mode": "agent_rag",
            "manual_initial_needs_revision": "",
            "manual_severity": "",
            "manual_primary_error": "",
            "manual_error_types": "",
            "pairwise_outcome": "",
            "review_status": "",
            "reviewer": "",
            "note": "",
            "source_text": "원문 2",
            "reference_text": "Reference 2",
            "initial_translation": "Initial 2",
            "final_translation": "Final 2",
            "agent_initial_passed": "False",
            "agent_initial_error_types": '["meaning"]',
            "agent_initial_summary": "summary 2",
            "retry_count": "1",
            "stop_reason": "passed",
        },
    ]
    _write_csv(baseline, FULL_COLUMNS, baseline_rows)
    manifest = tmp_path / "data" / "batch2" / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manual_review_schema_version": 2,
                "batch_id": "batch2",
                "review_template_file": "data/batch2/review_labels.csv",
                "review_template_sha256": _sha256(baseline),
                "selected": [
                    {
                        "case_id": "case-1",
                        "mode": "agent",
                        "selection_reason": "pass",
                    },
                    {
                        "case_id": "case-2",
                        "mode": "agent_rag",
                        "selection_reason": "retry",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    completed_rows = [dict(row) for row in baseline_rows]
    completed_rows[0].update(
        {
            "manual_initial_needs_revision": "false",
            "manual_severity": "",
            "pairwise_outcome": "same",
            "review_status": "confirmed",
            "reviewer": "  Sup Exact  ",
        }
    )
    completed_rows[1].update(
        {
            "manual_initial_needs_revision": "true",
            "manual_severity": "MINOR",
            "manual_primary_error": "meaning",
            "manual_error_types": '["meaning"]',
            "pairwise_outcome": "improved",
            "review_status": "confirmed",
            "reviewer": "Sup",
            "note": "localized fix",
        }
    )
    workbook_rows = [
        {**row, "row_status": "BROKEN FORMULA CACHE"} for row in completed_rows
    ]
    workbook = tmp_path / "outputs" / "batch2_reviewed.xlsx"
    _write_xlsx(
        workbook,
        sheet_name="Review",
        headers=[*FULL_COLUMNS, "row_status"],
        rows=workbook_rows,
        use_shared_strings=True,
        typed_evidence=True,
    )
    return {
        "baseline": baseline,
        "manifest": manifest,
        "workbook": workbook,
        "completed_rows": completed_rows,
    }


def _base_selection_and_review(tmp_path: Path) -> tuple[Path, Path]:
    selection = tmp_path / "data" / "batch1" / "selection.json"
    review = tmp_path / "data" / "batch1" / "review_labels.csv"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "batch1",
                "partial_representative_sample": True,
                "artifact_file": "artifacts/frozen.json",
                "artifact_sha256": "a" * 64,
                "artifact_config_sha256": "b" * 64,
                "run_id": "run-1",
                "dataset_id": "evaluation",
                "dataset_file": "data/evaluation.jsonl",
                "dataset_sha256": "c" * 64,
                "reference_review_sha256": "d" * 64,
                "source_feedback_sha256": "e" * 64,
                "selected": [
                    {
                        "case_id": "case-1",
                        "mode": "agent",
                        "selection_reason": "reviewed revision",
                    },
                    {
                        "case_id": "case-2",
                        "mode": "agent_rag",
                        "selection_reason": "reviewed pass",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    legacy_columns = [column for column in FULL_COLUMNS if column != "manual_severity"]
    rows = [
        {
            "review_key": "case-1::agent",
            "case_id": "case-1",
            "mode": "agent",
            "manual_initial_needs_revision": "true",
            "manual_primary_error": "meaning",
            "manual_error_types": '["meaning"]',
            "pairwise_outcome": "improved",
            "review_status": "confirmed",
            "reviewer": "Sup",
            "note": "existing human note",
            "source_text": "원문 1",
            "reference_text": "Reference 1",
            "initial_translation": "Initial 1",
            "final_translation": "Final 1",
            "agent_initial_passed": "False",
            "agent_initial_error_types": '["meaning"]',
            "agent_initial_summary": "summary 1",
            "retry_count": "1",
            "stop_reason": "passed",
        },
        {
            "review_key": "case-2::agent_rag",
            "case_id": "case-2",
            "mode": "agent_rag",
            "manual_initial_needs_revision": "false",
            "manual_primary_error": "",
            "manual_error_types": "",
            "pairwise_outcome": "same",
            "review_status": "confirmed",
            "reviewer": "Sup",
            "note": "",
            "source_text": "원문 2",
            "reference_text": "Reference 2",
            "initial_translation": "Initial 2",
            "final_translation": "Initial 2",
            "agent_initial_passed": "True",
            "agent_initial_error_types": "[]",
            "agent_initial_summary": "summary 2",
            "retry_count": "0",
            "stop_reason": "passed",
        },
    ]
    _write_csv(review, legacy_columns, rows)
    return selection, review


def _supplement_fixture(tmp_path: Path) -> dict[str, Path]:
    base_selection, base_review = _base_selection_and_review(tmp_path)
    supplement_dir = tmp_path / "data" / "batch1_severity"
    manifest_path = supplement_dir / "selection.json"
    template_path = supplement_dir / "severity_labels.csv"
    prepare_severity_supplement(
        base_selection_path=base_selection,
        base_review_csv_path=base_review,
        output_selection_path=manifest_path,
        output_csv_path=template_path,
        project_root=tmp_path,
        batch_id="batch1-severity",
    )
    with template_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0].update(
        {
            "manual_severity": "MAJOR",
            "severity_reviewer": "Sup",
            "severity_note": "core meaning impact",
        }
    )
    rows[1]["row_status"] = "READY SHOULD BE IGNORED"
    workbook = tmp_path / "outputs" / "batch1_severity_reviewed.xlsx"
    _write_xlsx(
        workbook,
        sheet_name="Severity Review",
        headers=[*headers, "row_status"],
        rows=rows,
    )
    return {
        "base_review": base_review,
        "manifest": manifest_path,
        "template": template_path,
        "workbook": workbook,
    }


def test_full_v2_ingest_preserves_exact_labels_and_ignores_row_status(
    tmp_path: Path,
) -> None:
    paths = _full_fixture(tmp_path)
    output = tmp_path / "data" / "ingested" / "review_labels.csv"
    provenance_path = tmp_path / "data" / "ingested" / "ingestion.json"

    provenance = ingest_manual_review_workbook(
        workbook_path=paths["workbook"],  # type: ignore[arg-type]
        selection_manifest_path=paths["manifest"],  # type: ignore[arg-type]
        baseline_csv_path=paths["baseline"],  # type: ignore[arg-type]
        output_csv_path=output,
        provenance_output_path=provenance_path,
        project_root=tmp_path,
        ingest_kind="full",
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == paths["completed_rows"]
    assert rows[0]["reviewer"] == "  Sup Exact  "
    assert provenance["ready_count"] == 2
    assert provenance["severity_required_count"] == 1
    assert provenance["severity_counts"] == {"MAJOR": 0, "MINOR": 1}
    assert provenance["automatic_severity_labels_generated"] is False
    assert provenance["output_csv_sha256"] == _sha256(output)
    assert json.loads(provenance_path.read_text()) == provenance


def test_full_v2_ingest_requires_pinned_template_without_creating_outputs(
    tmp_path: Path,
) -> None:
    paths = _full_fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest.pop("review_template_file")
    manifest.pop("review_template_sha256")
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "out" / "review.csv"
    provenance = tmp_path / "out" / "ingestion.json"

    with pytest.raises(ValueError, match="requires review_template_file"):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["baseline"],
            output_csv_path=output,
            provenance_output_path=provenance,
            project_root=tmp_path,
            ingest_kind="full",
        )

    assert not output.exists()
    assert not provenance.exists()


def test_full_v2_ingest_rejects_tampered_pinned_template_without_outputs(
    tmp_path: Path,
) -> None:
    paths = _full_fixture(tmp_path)
    with paths["baseline"].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = tmp_path / "out" / "review.csv"
    provenance = tmp_path / "out" / "ingestion.json"

    with pytest.raises(ValueError, match="baseline CSV hash"):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["baseline"],
            output_csv_path=output,
            provenance_output_path=provenance,
            project_root=tmp_path,
            ingest_kind="full",
        )

    assert not output.exists()
    assert not provenance.exists()


def test_severity_supplement_ingest_outputs_full_v2_csv_and_keeps_annotations(
    tmp_path: Path,
) -> None:
    paths = _supplement_fixture(tmp_path)
    output = tmp_path / "data" / "ingested" / "review_labels.csv"
    provenance_path = tmp_path / "data" / "ingested" / "ingestion.json"

    provenance = ingest_manual_review_workbook(
        workbook_path=paths["workbook"],
        selection_manifest_path=paths["manifest"],
        baseline_csv_path=paths["template"],
        output_csv_path=output,
        provenance_output_path=provenance_path,
        project_root=tmp_path,
        ingest_kind="severity-supplement",
        sheet_name="Severity Review",
    )

    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == FULL_COLUMNS
    assert rows[0]["manual_severity"] == "MAJOR"
    assert rows[0]["reviewer"] == "Sup"
    assert rows[0]["note"] == "existing human note"
    assert rows[1]["manual_severity"] == ""
    assert provenance["severity_counts"] == {"MAJOR": 1, "MINOR": 0}
    assert provenance["base_review_csv_sha256"] == _sha256(paths["base_review"])
    assert provenance["severity_annotations"] == [
        {
            "review_key": "case-1::agent",
            "manual_severity": "MAJOR",
            "severity_reviewer": "Sup",
            "severity_note": "core meaning impact",
        },
        {
            "review_key": "case-2::agent_rag",
            "manual_severity": "",
            "severity_reviewer": "",
            "severity_note": "",
        },
    ]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("manual_severity", "", "manual_severity must be MAJOR or MINOR"),
        ("manual_severity", "major", "manual_severity must be MAJOR or MINOR"),
        ("severity_reviewer", "", "requires severity_reviewer"),
    ],
)
def test_supplement_rejects_missing_or_noncanonical_human_input(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    paths = _supplement_fixture(tmp_path)
    with paths["template"].open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0].update(
        {
            "manual_severity": "MAJOR",
            "severity_reviewer": "Sup",
            field: value,
        }
    )
    _write_xlsx(
        paths["workbook"],
        sheet_name="Severity Review",
        headers=[*headers, "row_status"],
        rows=rows,
    )
    output = tmp_path / "out" / "review.csv"
    provenance = tmp_path / "out" / "ingestion.json"

    with pytest.raises(ValueError, match=match):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["template"],
            output_csv_path=output,
            provenance_output_path=provenance,
            project_root=tmp_path,
            ingest_kind="severity-supplement",
            sheet_name="Severity Review",
        )

    assert not output.exists()
    assert not provenance.exists()


def test_supplement_rejects_immutable_change_and_refuses_output_overwrite(
    tmp_path: Path,
) -> None:
    paths = _supplement_fixture(tmp_path)
    with paths["template"].open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0].update(
        {
            "manual_severity": "MAJOR",
            "severity_reviewer": "Sup",
            "initial_translation": "tampered",
        }
    )
    _write_xlsx(
        paths["workbook"],
        sheet_name="Severity Review",
        headers=[*headers, "row_status"],
        rows=rows,
    )
    output = tmp_path / "out" / "review.csv"
    provenance = tmp_path / "out" / "ingestion.json"

    with pytest.raises(ValueError, match="immutable value differs"):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["template"],
            output_csv_path=output,
            provenance_output_path=provenance,
            project_root=tmp_path,
            ingest_kind="severity-supplement",
            sheet_name="Severity Review",
        )
    assert not output.exists()
    assert not provenance.exists()

    output.parent.mkdir(parents=True)
    output.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["template"],
            output_csv_path=output,
            provenance_output_path=provenance,
            project_root=tmp_path,
            ingest_kind="severity-supplement",
            sheet_name="Severity Review",
        )
    assert output.read_text(encoding="utf-8") == "do not replace"


def test_supplement_rejects_severity_on_no_revision_row(tmp_path: Path) -> None:
    paths = _supplement_fixture(tmp_path)
    with paths["template"].open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0].update(
        {"manual_severity": "MAJOR", "severity_reviewer": "Sup"}
    )
    rows[1].update(
        {"manual_severity": "MINOR", "severity_reviewer": "Sup"}
    )
    _write_xlsx(
        paths["workbook"],
        sheet_name="Severity Review",
        headers=[*headers, "row_status"],
        rows=rows,
    )

    with pytest.raises(ValueError, match="must be blank"):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["template"],
            output_csv_path=tmp_path / "out" / "review.csv",
            provenance_output_path=tmp_path / "out" / "ingestion.json",
            project_root=tmp_path,
            ingest_kind="severity-supplement",
            sheet_name="Severity Review",
        )


def test_supplement_rejects_formula_generated_severity(tmp_path: Path) -> None:
    paths = _supplement_fixture(tmp_path)
    with paths["template"].open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0].update(
        {"manual_severity": "MAJOR", "severity_reviewer": "Sup"}
    )
    _write_xlsx(
        paths["workbook"],
        sheet_name="Severity Review",
        headers=[*headers, "row_status"],
        rows=rows,
        formula_cells={(2, "manual_severity")},
    )

    with pytest.raises(ValueError, match="contains a formula in manual_severity"):
        ingest_manual_review_workbook(
            workbook_path=paths["workbook"],
            selection_manifest_path=paths["manifest"],
            baseline_csv_path=paths["template"],
            output_csv_path=tmp_path / "out" / "review.csv",
            provenance_output_path=tmp_path / "out" / "ingestion.json",
            project_root=tmp_path,
            ingest_kind="severity-supplement",
            sheet_name="Severity Review",
        )
