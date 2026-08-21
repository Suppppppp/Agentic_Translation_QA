"""Translation backends.

The Marian backend deliberately loads optional ML dependencies and model weights
only on the first translation request. Importing the API never downloads a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from threading import RLock
from time import perf_counter
from typing import Any

from translation_qa.errors import (
    ComponentExecutionError,
    ComponentUnavailableError,
    ConstraintApplicationError,
)
from translation_qa.schemas import TermConstraint, TranslationCandidate


class MarianTranslator:
    """Lazy Hugging Face MarianMT adapter with optional lexical constraints."""

    def __init__(
        self,
        model_id: str = "Helsinki-NLP/opus-mt-ko-en",
        *,
        device: str = "auto",
        num_beams: int = 6,
        max_new_tokens: int = 256,
    ) -> None:
        if num_beams < 1:
            raise ValueError("num_beams must be at least one")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one")
        self._model_id = model_id
        self._requested_device = device
        self._num_beams = num_beams
        self._max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._lock = RLock()
        self.startup_ms: float | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    def _resolve_device(self, torch: Any) -> str:
        requested = self._requested_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ComponentUnavailableError("CUDA was requested but is unavailable")
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ComponentUnavailableError("MPS was requested but is unavailable")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ComponentUnavailableError(f"unsupported device: {requested}")
        return requested

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            started = perf_counter()
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ComponentUnavailableError(
                    "MarianMT dependencies are missing; install the project with the ml extra"
                ) from exc

            try:
                tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(self._model_id)
                device = self._resolve_device(torch)
                model.to(device)
                model.eval()
            except ComponentUnavailableError:
                raise
            except Exception as exc:  # model hub and backend errors vary by version
                raise ComponentUnavailableError(
                    f"could not load translation model {self._model_id}: {exc}"
                ) from exc

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self.startup_ms = (perf_counter() - started) * 1000.0

    def _forced_word_ids(
        self, constraints: Sequence[TermConstraint] | None
    ) -> list[list[int] | list[list[int]]]:
        if not constraints:
            return []
        assert self._tokenizer is not None
        seen_groups: set[tuple[str, ...]] = set()
        forced: list[list[int] | list[list[int]]] = []
        for constraint in constraints:
            targets = list(
                dict.fromkeys(
                    target.strip()
                    for target in [
                        constraint.target_term,
                        *constraint.target_variants,
                    ]
                    if target.strip()
                )
            )
            normalized_group = tuple(target.casefold() for target in targets)
            if not targets or normalized_group in seen_groups:
                continue
            candidate_ids = [
                self._tokenizer(target, add_special_tokens=False).input_ids
                for target in targets
            ]
            candidate_ids = [token_ids for token_ids in candidate_ids if token_ids]
            # Transformers' disjunctive trie rejects a phrase when it is a
            # complete prefix of another phrase. Prefer the shorter accepted
            # form (for example ``deploy`` over ``deployed``) and keep distinct
            # non-prefix alternatives such as ``deployment``.
            pruned_ids: list[list[int]] = []
            for token_ids in sorted(candidate_ids, key=len):
                if token_ids in pruned_ids:
                    continue
                if any(
                    token_ids[: len(existing)] == existing
                    for existing in pruned_ids
                ):
                    continue
                pruned_ids.append(token_ids)
            candidate_ids = pruned_ids
            if not candidate_ids:
                continue
            forced.append(candidate_ids[0] if len(candidate_ids) == 1 else candidate_ids)
            seen_groups.add(normalized_group)
        return forced

    def translate(
        self,
        source_text: str,
        constraints: Sequence[TermConstraint] | None = None,
    ) -> TranslationCandidate:
        source_text = source_text.strip()
        if not source_text:
            raise ValueError("source_text must not be empty")
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        with self._lock:
            try:
                encoded = self._tokenizer(
                    source_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
                generation_args: dict[str, Any] = {
                    "max_new_tokens": min(
                        self._max_new_tokens,
                        max(32, len(source_text) * 2),
                    ),
                    "num_beams": self._num_beams,
                    "do_sample": False,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 3,
                    "remove_invalid_values": True,
                    "renormalize_logits": True,
                }
                forced_word_ids = self._forced_word_ids(constraints)
                if forced_word_ids:
                    generation_args["force_words_ids"] = forced_word_ids
                    generation_args["num_beams"] = max(self._num_beams, 4)

                with self._torch.inference_mode():
                    output_ids = self._model.generate(**encoded, **generation_args)
                translated = self._tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                ).strip()
            except Exception as exc:
                if constraints:
                    raise ConstraintApplicationError(
                        f"lexical constraint generation failed: {exc}"
                    ) from exc
                raise ComponentExecutionError(f"translation failed: {exc}") from exc

        if not translated:
            raise ComponentExecutionError("translation model returned an empty result")
        if constraints and self._looks_degenerate(source_text, translated):
            raise ConstraintApplicationError(
                "lexical constraints produced a degenerate candidate; using an "
                "unconstrained fallback is required"
            )
        return TranslationCandidate(text=translated, model_id=self._model_id)

    @staticmethod
    def _looks_degenerate(source_text: str, translated: str) -> bool:
        """Catch obvious constraint-induced runaway output without judging style."""

        if len(translated) > max(120, len(source_text) * 4):
            return True
        if len(re.findall(r"\([^)]{0,40}\)", translated)) >= 3:
            return True
        tokens = translated.casefold().split()
        if len(tokens) < 12:
            return False
        trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
        if not trigrams:
            return False
        most_common = max(trigrams.count(ngram) for ngram in set(trigrams))
        return most_common >= 4 and most_common / len(trigrams) >= 0.2
