from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from translation_qa.retrieval import GlossaryEntry
from translation_qa.schemas import RetrievalMatchType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_retrieval.py"
SPEC = importlib.util.spec_from_file_location("compare_retrieval_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_retrieval_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare_retrieval_script
SPEC.loader.exec_module(compare_retrieval_script)

QueryCase = compare_retrieval_script.QueryCase
compare_cases = compare_retrieval_script.compare_cases
main = compare_retrieval_script.main


class FakeEmbedder:
    """Network-free injected vectors for ranking assertions."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [
                float("배포" in text),
                float("캐시" in text),
                float("회의" in text),
            ]
            for text in texts
        ]


def test_compare_cases_uses_injected_embedder_and_deduplicates_hybrid() -> None:
    entries = [
        GlossaryEntry("deploy", "배포", "deployment", "software"),
        GlossaryEntry("cache", "캐시", "cache", "software"),
    ]
    cases = [
        QueryCase(
            case_id="case-1",
            source_text="새 버전을 배포한다.",
            domain="software",
        )
    ]

    [result] = compare_cases(entries, cases, FakeEmbedder(), top_k=2, min_score=0.5)

    assert [hit.term_id for hit in result["exact"]] == ["deploy"]
    assert [hit.term_id for hit in result["vector"]] == ["deploy"]
    assert [hit.term_id for hit in result["hybrid"]] == ["deploy"]
    assert [hit.term_id for hit in result["safe_hybrid"]] == ["deploy"]
    assert result["hybrid"][0].match_type is RetrievalMatchType.HYBRID


def test_cli_default_embedder_runs_pilot_without_model_download(capsys) -> None:
    exit_code = main(
        [
            "--glossary",
            str(PROJECT_ROOT / "data" / "glossary_pilot.csv"),
            "--pilot",
            str(PROJECT_ROOT / "data" / "pilot_v1.jsonl"),
            "--case-id",
            "pilot-001",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["config"]["embedder"] == "deterministic"
    assert payload["config"]["model_id"] is None
    assert payload["cases"][0]["case_id"] == "pilot-001"
    assert payload["cases"][0]["exact"][0]["term_id"] == "sw-deployment"
    hybrid_ids = [hit["term_id"] for hit in payload["cases"][0]["hybrid"]]
    assert len(hybrid_ids) == len(set(hybrid_ids))


def test_cli_no_match_can_be_reproduced_with_strict_threshold(capsys) -> None:
    exit_code = main(
        [
            "--case-id",
            "pilot-012",
            "--min-score",
            "0.99",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["cases"][0]["exact"] == []
    assert payload["cases"][0]["vector"] == []
    assert payload["cases"][0]["hybrid"] == []


def test_cli_sentence_transformer_path_is_optional_and_injected(monkeypatch, capsys) -> None:
    created: list[str] = []

    class FakeSentenceTransformerEmbedder(FakeEmbedder):
        def __init__(self, model_id: str) -> None:
            created.append(model_id)

    monkeypatch.setattr(
        compare_retrieval_script,
        "SentenceTransformerEmbedder",
        FakeSentenceTransformerEmbedder,
    )

    exit_code = main(
        [
            "--text",
            "새 버전을 배포한다.",
            "--domain",
            "software",
            "--embedder",
            "sentence-transformer",
            "--model-id",
            "fake/model",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created == ["fake/model"]
    assert payload["config"]["embedder"] == "sentence-transformer"
    assert payload["config"]["model_id"] == "fake/model"
