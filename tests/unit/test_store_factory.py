"""Tests for settings-driven vector store selection."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.settings import Settings
from rag_loader.store.chroma import ChromaStore
from rag_loader.store.factory import build_vector_store
from rag_loader.store.qdrant import QdrantStore


def test_default_builds_chroma():
    store = build_vector_store(Settings(), DeterministicFakeEmbedding(size=8))
    assert isinstance(store, ChromaStore)


def test_qdrant_selected_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_VECTOR_STORE", "qdrant")
    store = build_vector_store(Settings(), DeterministicFakeEmbedding(size=8))
    assert isinstance(store, QdrantStore)
