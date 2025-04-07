"""Chroma vector store backend."""

from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.base import chunk_to_document


class ChromaStore:
    """VectorBackend backed by Chroma (embedded by default, or persistent on disk)."""

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "rag_documents",
        persist_directory: str | Path | None = None,
    ) -> None:
        self._store = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_directory) if persist_directory else None,
        )

    def add_chunks(self, chunks: Sequence[TextChunk]) -> list[str]:
        """Upsert chunks; re-adding the same ids overwrites rather than duplicates."""
        if not chunks:
            return []
        documents = [chunk_to_document(chunk) for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        self._store.add_documents(documents, ids=ids)
        return ids

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self._store.similarity_search(query, k=k)
