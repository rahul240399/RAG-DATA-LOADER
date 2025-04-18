"""Batch indexing tests (sequential and bounded-async)."""

import asyncio
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


def test_index_paths_aggregates_success_and_failure(sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    batch = _indexer().index_paths([path, path, "/missing.pdf"])
    assert batch.total_documents == 3
    assert batch.successful_documents == 2
    assert batch.failed_documents == 1


def test_aindex_paths_matches_sequential(sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    batch = asyncio.run(_indexer().aindex_paths([path, path], concurrency=2))
    assert batch.total_documents == 2
    assert batch.successful_documents == 2
