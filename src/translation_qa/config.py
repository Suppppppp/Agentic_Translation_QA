"""Validated runtime configuration for the translation QA application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Application settings with conservative local-model defaults.

    ``max_retries`` counts *additional* attempts after the initial translation.
    The assignment limit of two retries therefore permits at most three
    translation candidates for one request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_name: str = "Agentic Translation QA"
    source_language: Literal["ko"] = "ko"
    target_language: Literal["en"] = "en"

    translator_model_id: str = "Helsinki-NLP/opus-mt-ko-en"
    agent_model_id: str = "qwen3:1.7b"
    embedding_model_id: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    vector_backend: Literal["memory", "faiss", "chroma"] = "memory"

    retrieval_top_k: int = Field(default=3, ge=1, le=20)
    max_retries: int = Field(default=2, ge=0, le=2)
    max_input_characters: int = Field(default=5_000, ge=1)

    @property
    def max_attempts(self) -> int:
        """Maximum number of candidates, including the initial translation."""

        return 1 + self.max_retries
