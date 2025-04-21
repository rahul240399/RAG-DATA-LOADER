"""Batch result timestamps and summary statistics."""

import uuid
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.indexing import Indexer
from rag_loader.models.config import PipelineConfig
from rag_loader.store.chroma import ChromaStore


def test_batch_has_timestamps_and_summary(sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    batch = Indexer(store, PipelineConfig()).index_paths([path])

    assert batch.started_at is not None
    assert batch.completed_at is not None

    stats = batch.get_summary_statistics()
    assert stats["total_documents"] == 1
    assert stats["successful_documents"] == 1
    assert stats["success_rate"] == 1.0
