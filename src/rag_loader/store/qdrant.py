"""Qdrant vector store backend."""

import uuid
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from rag_loader.embeddings import embedding_dimension
from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.base import chunk_to_document


def _point_id(chunk_id: str) -> str:
    """Map a chunk id to a deterministic UUID (Qdrant requires UUID/int ids)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """VectorBackend backed by Qdrant (in-memory, local, or remote server)."""

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "rag_documents",
        location: str = ":memory:",
        url: str | None = None,
    ) -> None:
        client = QdrantClient(url=url) if url else QdrantClient(location=location)
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimension(embeddings),
                    distance=Distance.COSINE,
                ),
            )
        self._store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    def add_chunks(self, chunks: Sequence[TextChunk]) -> list[str]:
        """Upsert chunks; deterministic point ids make re-indexing idempotent."""
        if not chunks:
            return []
        documents = [chunk_to_document(chunk) for chunk in chunks]
        self._store.add_documents(documents, ids=[_point_id(c.chunk_id) for c in chunks])
        return [chunk.chunk_id for chunk in chunks]

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self._store.similarity_search(query, k=k)
