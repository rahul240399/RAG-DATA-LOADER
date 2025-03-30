"""Ingestion pipeline composing PDF loading and splitting as an LCEL runnable."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from rag_loader.ingest.loader import load_pdf
from rag_loader.ingest.splitter import split_documents
from rag_loader.models.config import PipelineConfig
from rag_loader.models.text_chunk import TextChunk


def build_ingestion_runnable(config: PipelineConfig) -> Runnable[str | Path, list[TextChunk]]:
    """Build an LCEL runnable: ``PDF path -> pages -> TextChunks``.

    Expressing ingestion as a runnable lets it compose with the rest of the
    pipeline (batching, async, and tracing come for free from LCEL).
    """
    load = RunnableLambda(load_pdf)
    split: RunnableLambda[list[Document], list[TextChunk]] = RunnableLambda(
        lambda docs: split_documents(docs, config)
    )
    return load | split


def ingest_pdf(path: str | Path, config: PipelineConfig) -> list[TextChunk]:
    """Run the ingestion runnable for a single PDF and return its chunks."""
    return build_ingestion_runnable(config).invoke(path)
