"""FastAPI application and dependency wiring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from translation_qa.agent import OllamaTranslationAgent, RuleBasedTermAgent
from translation_qa.benchmark import BenchmarkRunner, JsonlDatasetRepository
from translation_qa.config import Settings
from translation_qa.contracts import BenchmarkService, TranslationService
from translation_qa.errors import (
    ComponentUnavailableError,
    DatasetNotFoundError,
    TranslationQAError,
)
from translation_qa.pipeline import TranslationPipeline
from translation_qa.postedit import GlossaryPostEditingTranslator
from translation_qa.retrieval import (
    ExactGlossaryRetriever,
    ExactFirstHybridGlossaryRetriever,
    SentenceTransformerEmbedder,
    VectorGlossaryRetriever,
    load_glossary_csv,
)
from translation_qa.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ExecutionMode,
    TranslationRequest,
    TranslationResponse,
)
from translation_qa.translator import MarianTranslator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_default_service() -> TranslationPipeline:
    """Build lazy local components without downloading model weights."""

    settings = Settings(
        translator_model_id=os.getenv(
            "TRANSLATION_QA_TRANSLATOR_MODEL",
            Settings().translator_model_id,
        ),
        agent_model_id=os.getenv(
            "TRANSLATION_QA_AGENT_MODEL",
            Settings().agent_model_id,
        ),
        device=os.getenv("TRANSLATION_QA_DEVICE", "auto"),
    )
    base_translator = MarianTranslator(
        settings.translator_model_id,
        device=settings.device,
    )
    translator = GlossaryPostEditingTranslator(base_translator)

    glossary_path = Path(
        os.getenv(
            "TRANSLATION_QA_GLOSSARY_PATH",
            str(PROJECT_ROOT / "data" / "glossary_pilot.csv"),
        )
    )
    entries = load_glossary_csv(glossary_path) if glossary_path.is_file() else []
    exact_retriever = ExactGlossaryRetriever(entries)
    retriever_kind = os.getenv("TRANSLATION_QA_RETRIEVER", "exact").casefold()
    if retriever_kind == "exact":
        retriever = exact_retriever
    elif retriever_kind in {"vector", "hybrid"}:
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model_id,
            device=None if settings.device == "auto" else settings.device,
        )
        vector_retriever = VectorGlossaryRetriever(entries, embedder)
        retriever = (
            vector_retriever
            if retriever_kind == "vector"
            else ExactFirstHybridGlossaryRetriever(exact_retriever, vector_retriever)
        )
    else:
        raise ValueError(
            "TRANSLATION_QA_RETRIEVER must be exact, vector, or hybrid"
        )

    agent_kind = os.getenv("TRANSLATION_QA_AGENT_BACKEND", "ollama").casefold()
    if agent_kind == "ollama":
        agent = OllamaTranslationAgent(
            settings.agent_model_id,
            base_url=os.getenv(
                "TRANSLATION_QA_OLLAMA_URL",
                "http://127.0.0.1:11434",
            ),
        )
        reviser = agent
    elif agent_kind == "rule":
        agent = RuleBasedTermAgent()
        reviser = None
    else:
        raise ValueError("TRANSLATION_QA_AGENT_BACKEND must be ollama or rule")

    return TranslationPipeline(
        settings,
        translator,
        retriever=retriever,
        agent=agent,
        reviser=reviser,
    )


def build_default_benchmark(service: TranslationService) -> BenchmarkRunner:
    return BenchmarkRunner(
        service,
        JsonlDatasetRepository(PROJECT_ROOT / "data"),
        artifact_directory=PROJECT_ROOT / "artifacts",
    )


def get_translation_service(request: Request) -> TranslationService:
    return request.app.state.translation_service


def get_benchmark_service(request: Request) -> BenchmarkService:
    return request.app.state.benchmark_service


TranslationServiceDependency = Annotated[
    TranslationService,
    Depends(get_translation_service),
]
BenchmarkServiceDependency = Annotated[
    BenchmarkService,
    Depends(get_benchmark_service),
]


def _translate(
    service: TranslationService,
    payload: TranslationRequest,
    mode: ExecutionMode,
) -> TranslationResponse:
    try:
        return service.translate(payload, mode)
    except ComponentUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except TranslationQAError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def create_app(
    *,
    translation_service: TranslationService | None = None,
    benchmark_service: BenchmarkService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Agentic Translation QA",
        version="0.1.0",
        description="Local Korean-to-English translation with RAG and Agent QA.",
    )
    service = translation_service or build_default_service()
    app.state.translation_service = service
    app.state.benchmark_service = benchmark_service or build_default_benchmark(service)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "models": "lazy"}

    @app.post("/translate/baseline", response_model=TranslationResponse)
    def translate_baseline(
        payload: TranslationRequest,
        service: TranslationServiceDependency,
    ) -> TranslationResponse:
        return _translate(service, payload, ExecutionMode.BASELINE)

    @app.post("/translate/agent-rag", response_model=TranslationResponse)
    def translate_agent_rag(
        payload: TranslationRequest,
        service: TranslationServiceDependency,
    ) -> TranslationResponse:
        return _translate(service, payload, ExecutionMode.AGENT_RAG)

    @app.post("/benchmark", response_model=BenchmarkResponse)
    def benchmark(
        payload: BenchmarkRequest,
        service: BenchmarkServiceDependency,
    ) -> BenchmarkResponse:
        try:
            return service.run(payload)
        except DatasetNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ComponentUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    return app


app = create_app()
