"""End-to-end tests for the Indexer using an in-memory Chroma store."""

import uuid
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.indexing import Indexer
from rag_loader.models.config import PipelineConfig
from rag_loader.store.chroma import ChromaStore


def _indexer() -> Indexer:
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    return Indexer(store, PipelineConfig(chunk_size=200, chunk_overlap=20))


def test_index_pdf_succeeds(sample_pdf: tuple[Path, list[str]]):
    path, pages = sample_pdf
    result = _indexer().index_pdf(path)
    assert result.success
    assert result.chunks_processed >= len(pages)
    assert result.embeddings_generated == result.chunks_processed
    assert result.errors == []


def test_index_missing_pdf_reports_failure():
    result = _indexer().index_pdf("/no/such/file.pdf")
    assert not result.success
    assert result.errors
    assert result.chunks_processed == 0


def test_reindexing_same_pdf_is_idempotent(sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    indexer = _indexer()
    first = indexer.index_pdf(path)
    indexer.index_pdf(path)  # second pass must not duplicate
    hits = indexer.store.similarity_search("page", k=100)
    assert len(hits) == first.chunks_processed
