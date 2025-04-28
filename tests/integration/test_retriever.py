"""Tests for store-backed retrievers across both backends."""

import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever

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
    name = f"rag_{uuid.uuid4().hex}"
    store: VectorBackend = (
        ChromaStore(embeddings, collection_name=name)
        if request.param == "chroma"
        else QdrantStore(embeddings, collection_name=name)
    )
    store.add_chunks(
        [_chunk("alpha apple", 0), _chunk("beta banana", 20), _chunk("gamma grape", 40)]
    )
    return store


def test_as_retriever_returns_documents(backend: VectorBackend):
    retriever = backend.as_retriever(k=2)
    assert isinstance(retriever, BaseRetriever)
    docs = retriever.invoke("alpha")
    assert docs
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) <= 2
