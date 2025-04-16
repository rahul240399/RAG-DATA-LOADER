"""End-to-end indexing: ingest a PDF, embed its chunks, and store the vectors."""

import time
from pathlib import Path

from langchain_core.runnables import Runnable, RunnableLambda

from rag_loader.factories import build_embeddings
from rag_loader.ingest.pipeline import build_ingestion_runnable
from rag_loader.models.config import PipelineConfig
from rag_loader.models.results import DocumentProcessingResult
from rag_loader.models.text_chunk import TextChunk
from rag_loader.settings import Settings
from rag_loader.store import VectorBackend, build_vector_store


def build_index_runnable(
    config: PipelineConfig, store: VectorBackend
) -> Runnable[str | Path, list[str]]:
    """LCEL chain mapping a PDF path to the ids of its stored chunks."""
    ingest = build_ingestion_runnable(config)
    add: RunnableLambda[list[TextChunk], list[str]] = RunnableLambda(
        lambda chunks: store.add_chunks(chunks)
    )
    return ingest | add


class Indexer:
    """Indexes PDFs into a vector store, returning per-document results.

    The store and config are injected so the indexer can be unit-tested with an
    in-memory backend; ``from_settings`` wires the real, configured stack.
    """

    def __init__(self, store: VectorBackend, config: PipelineConfig) -> None:
        self._store = store
        self._config = config
        self._runnable = build_index_runnable(config, store)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Indexer":
        store = build_vector_store(settings, build_embeddings(settings))
        return cls(store, settings.pipeline_config())

    @property
    def store(self) -> VectorBackend:
        return self._store

    def index_pdf(self, path: str | Path) -> DocumentProcessingResult:
        """Ingest, embed, and store one PDF, reporting success or failure.

        Failures (missing file, parse error, store outage) are captured in the
        result instead of raising, so a batch can continue past a bad document.
        """
        start = time.perf_counter()
        source = str(path)
        try:
            ids = self._runnable.invoke(path)
        except Exception as exc:
            return DocumentProcessingResult(
                success=False,
                chunks_processed=0,
                embeddings_generated=0,
                storage_success=False,
                processing_time=time.perf_counter() - start,
                errors=[f"{type(exc).__name__}: {exc}"],
                source_document=source,
            )
        return DocumentProcessingResult(
            success=True,
            chunks_processed=len(ids),
            embeddings_generated=len(ids),
            storage_success=True,
            processing_time=time.perf_counter() - start,
            errors=[],
            source_document=source,
        )
