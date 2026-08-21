from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from translation_qa.schemas import TermConstraint, TranslationCandidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_injection.py"
SPEC = importlib.util.spec_from_file_location("compare_injection_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
spike = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spike
SPEC.loader.exec_module(spike)


class RecordingTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[TermConstraint]]] = []

    @property
    def model_id(self) -> str:
        return "recording-marian"

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        frozen_constraints = list(constraints or ())
        self.calls.append((source_text, frozen_constraints))
        if frozen_constraints:
            output = "deployment completed"
        else:
            output = f"output {source_text}"
        return TranslationCandidate(text=output, model_id=self.model_id)


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    glossary = tmp_path / "glossary.csv"
    glossary.write_text(
        "term_id,domain,source_term,target_term,accepted_variants_json\n"
        'deploy,software,배포,deployment,"[""deploy""]"\n',
        encoding="utf-8",
    )
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "source_text": "새 버전을 배포한다.",
                "reference_text": "SECRET REFERENCE MUST NEVER REACH TRANSLATOR",
                "domain": "software",
                "expected_terms": [
                    {"source_term": "배포", "accepted_targets": ["deployment"]}
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return pilot, glossary


def test_all_five_arms_share_backend_and_never_receive_reference(tmp_path: Path) -> None:
    pilot_path, glossary_path = _write_fixture_files(tmp_path)
    translator = RecordingTranslator()
    cases = spike.load_pilot_cases(pilot_path)

    [result] = spike.compare_injections(cases, glossary_path, translator)

    assert [arm["method"] for arm in result["arms"]] == list(spike.METHODS)
    assert len(translator.calls) == 5
    assert all(
        "SECRET REFERENCE" not in source_text
        for source_text, _constraints in translator.calls
    )
    assert translator.calls[0] == ("새 버전을 배포한다.", [])
    assert translator.calls[1][0] == "새 버전을 배포한다."
    assert translator.calls[1][1][0].target_term == "deployment"
    assert translator.calls[2][0] == "새 버전을 deployment한다."
    assert translator.calls[3][0] == "새 버전을 배포 (deployment)한다."
    assert translator.calls[4][0].startswith('[TERM "배포" = "deployment"]\n')


def test_output_includes_candidate_term_hits_degeneracy_and_failure(tmp_path: Path) -> None:
    pilot_path, glossary_path = _write_fixture_files(tmp_path)

    class MixedOutcomeTranslator(RecordingTranslator):
        def translate(
            self,
            source_text: str,
            constraints: Sequence[TermConstraint] | None = None,
        ) -> TranslationCandidate:
            if constraints:
                raise RuntimeError("forced decoding failed")
            if "(deployment)" in source_text:
                repeated = " ".join(["alpha beta gamma"] * 8)
                return TranslationCandidate(text=repeated, model_id=self.model_id)
            return TranslationCandidate(text=f"result deployment {source_text}", model_id=self.model_id)

    [result] = spike.compare_injections(
        spike.load_pilot_cases(pilot_path),
        glossary_path,
        MixedOutcomeTranslator(),
    )
    arms = {arm["method"]: arm for arm in result["arms"]}

    assert arms["baseline"]["candidate"] is not None
    assert arms["baseline"]["term_hits"][0]["hit"]
    assert arms["lexical_constraints"]["status"] == "failure"
    assert arms["lexical_constraints"]["failure"]["type"] == "RuntimeError"
    assert arms["parenthetical"]["status"] == "degenerate"
    assert arms["parenthetical"]["degenerate"]


def test_default_cli_is_offline_and_emits_json(capsys) -> None:
    exit_code = spike.main(
        [
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
    assert payload["config"]["backend"] == "fake"
    assert payload["warning"] is not None
    assert [arm["method"] for arm in payload["cases"][0]["arms"]] == list(spike.METHODS)
    for arm in payload["cases"][0]["arms"]:
        assert set(("candidate", "term_hits", "degenerate", "failure")) <= arm.keys()


def test_real_model_flag_constructs_one_injected_backend(monkeypatch, capsys) -> None:
    translator = RecordingTranslator()
    calls: list[object] = []

    def fake_real_translator(args: object) -> RecordingTranslator:
        calls.append(args)
        return translator

    monkeypatch.setattr(spike, "_real_translator", fake_real_translator)
    exit_code = spike.main(
        [
            "--case-id",
            "pilot-001",
            "--real-model",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(calls) == 1
    assert len(translator.calls) == 5
    assert payload["config"]["backend"] == "marian"
    assert payload["warning"] is None
