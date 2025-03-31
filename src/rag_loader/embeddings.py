"""Embedding helpers bridging TextChunks to a LangChain Embeddings provider."""

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from rag_loader.models.text_chunk import TextChunk


def embed_chunks(chunks: Sequence[TextChunk], embeddings: Embeddings) -> list[list[float]]:
    """Embed the text content of chunks, returning one vector per chunk."""
    return embeddings.embed_documents([chunk.content for chunk in chunks])


def embedding_dimension(embeddings: Embeddings) -> int:
    """Probe the vector dimension of an embeddings provider.

    Needed by stores such as Qdrant that require the vector size up front.
    """
    return len(embeddings.embed_query("dimension probe"))
