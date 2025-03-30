"""Tests for the composed ingestion runnable."""

from pathlib import Path

from rag_loader.ingest.pipeline import build_ingestion_runnable, ingest_pdf
from rag_loader.models.config import PipelineConfig
from rag_loader.models.text_chunk import TextChunk


def test_ingest_pdf_returns_chunks_for_every_page(sample_pdf: tuple[Path, list[str]]):
    path, pages_text = sample_pdf
    chunks = ingest_pdf(path, PipelineConfig(chunk_size=120, chunk_overlap=20))
    assert chunks
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.metadata["source"] == path.name for c in chunks)
    assert {c.metadata["page"] for c in chunks} == set(range(1, len(pages_text) + 1))


def test_runnable_is_invocable(sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    runnable = build_ingestion_runnable(PipelineConfig())
    chunks = runnable.invoke(path)
    assert chunks
    assert all(isinstance(c, TextChunk) for c in chunks)
