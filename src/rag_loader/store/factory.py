"""Build a vector store backend from settings."""

from langchain_core.embeddings import Embeddings

from rag_loader.settings import Settings
from rag_loader.store.base import VectorBackend
from rag_loader.store.chroma import ChromaStore
from rag_loader.store.qdrant import QdrantStore


def build_vector_store(settings: Settings, embeddings: Embeddings) -> VectorBackend:
    """Construct the vector store backend selected by ``settings``."""
    if settings.vector_store == "qdrant":
        return QdrantStore(
            embeddings,
            collection_name=settings.chromadb_collection,
            url=settings.qdrant_url,
        )
    return ChromaStore(
        embeddings,
        collection_name=settings.chromadb_collection,
        persist_directory=settings.chroma_persist_directory,
    )
