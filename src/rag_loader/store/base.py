"""Vector store abstraction and TextChunk-to-Document mapping."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_loader.models.text_chunk import TextChunk


def chunk_to_document(chunk: TextChunk) -> Document:
    """Convert a TextChunk into a LangChain Document, preserving id and metadata.

    Using the chunk's deterministic id as the Document id is what makes storage
    idempotent: re-indexing the same content upserts in place instead of
    duplicating.
    """
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.content,
        metadata=dict(chunk.metadata),
    )


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal interface every vector store backend implements."""

    def add_chunks(self, chunks: Sequence[TextChunk]) -> list[str]:
        """Upsert chunks into the store and return their ids."""
        ...

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the ``k`` documents most similar to ``query``."""
        ...

    def as_retriever(self, k: int = 4, score_threshold: float | None = None) -> BaseRetriever:
        """Return a LangChain retriever over this store for use in chains."""
        ...
