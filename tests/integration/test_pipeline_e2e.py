"""End-to-end test of the production assembly (Settings -> Indexer -> retrieval).

A deterministic fake embedding is injected so the full wiring runs without a
provider download, while still exercising from_settings, the vector store
factory, ingestion, indexing, and similarity search together.
"""

import uuid
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader import indexing
from rag_loader.indexing import Indexer
from rag_loader.settings import Settings


def test_from_settings_indexes_and_retrieves(
    monkeypatch: pytest.MonkeyPatch, sample_pdf: tuple[Path, list[str]]
):
    monkeypatch.setattr(
        indexing, "build_embeddings", lambda settings: DeterministicFakeEmbedding(size=16)
    )
    monkeypatch.setenv("RAG_VECTOR_STORE", "chroma")
    monkeypatch.setenv("RAG_CHROMADB_COLLECTION", f"rag_{uuid.uuid4().hex}")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "200")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "20")

    indexer = Indexer.from_settings(Settings())
    path, pages = sample_pdf

    result = indexer.index_pdf(path)
    assert result.success
    assert result.chunks_processed >= len(pages)

    hits = indexer.store.similarity_search("page", k=result.chunks_processed)
    assert len(hits) == result.chunks_processed
    assert all("page" in hit.metadata and "source" in hit.metadata for hit in hits)
    assert all(hit.page_content for hit in hits)
