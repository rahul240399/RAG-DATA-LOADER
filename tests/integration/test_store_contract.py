"""Contract tests run against every vector store backend.

Both backends run fully in-memory (Chroma embedded, Qdrant ``:memory:``) with
deterministic fake embeddings, so the suite needs no servers or model downloads.
"""

import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.base import VectorBackend
from rag_loader.store.chroma import ChromaStore
from rag_loader.store.qdrant import QdrantStore


def _chunk(content: str, start: int) -> TextChunk:
    return TextChunk(
        content=content,
        metadata={"source": "d.pdf", "page": 1},
        start_index=start,
        end_index=start + len(content),
        source_document="d.pdf",
    )


@pytest.fixture(params=["chroma", "qdrant"])
def backend(request: pytest.FixtureRequest) -> VectorBackend:
    embeddings = DeterministicFakeEmbedding(size=16)
    collection = f"rag_{uuid.uuid4().hex}"
    if request.param == "chroma":
        return ChromaStore(embeddings, collection_name=collection)
    return QdrantStore(embeddings, collection_name=collection)


class TestVectorBackendContract:
    def test_satisfies_protocol(self, backend: VectorBackend):
        assert isinstance(backend, VectorBackend)

    def test_add_returns_chunk_ids(self, backend: VectorBackend):
        chunks = [_chunk("alpha", 0), _chunk("beta", 10)]
        assert backend.add_chunks(chunks) == [c.chunk_id for c in chunks]

    def test_empty_add_returns_empty(self, backend: VectorBackend):
        assert backend.add_chunks([]) == []

    def test_search_returns_documents_with_metadata(self, backend: VectorBackend):
        backend.add_chunks([_chunk("alpha apple", 0), _chunk("beta banana", 20)])
        results = backend.similarity_search("alpha", k=2)
        assert results
        assert all(r.metadata["source"] == "d.pdf" for r in results)

    def test_upsert_is_idempotent(self, backend: VectorBackend):
        chunks = [_chunk("a", 0), _chunk("b", 10), _chunk("c", 20)]
        backend.add_chunks(chunks)
        backend.add_chunks(chunks)  # re-adding identical chunks must not duplicate
        results = backend.similarity_search("a", k=10)
        assert len(results) == 3
