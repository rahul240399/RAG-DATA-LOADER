"""Tests for the on-disk embedding cache."""

from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from rag_loader.embeddings import CachingEmbeddings


class _CountingEmbeddings(Embeddings):
    """Counts how many documents the underlying provider actually embeds."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = DeterministicFakeEmbedding(size=8)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def test_cache_returns_same_vectors_as_underlying(tmp_path: Path):
    base = DeterministicFakeEmbedding(size=8)
    cached = CachingEmbeddings(base, tmp_path, namespace="fake")
    expected = base.embed_documents(["hello", "world"])
    assert cached.embed_documents(["hello", "world"]) == expected
    # A second call is served entirely from the cache.
    assert cached.embed_documents(["hello", "world"]) == expected


def test_cache_writes_one_file_per_unique_text(tmp_path: Path):
    cached = CachingEmbeddings(DeterministicFakeEmbedding(size=8), tmp_path, namespace="fake")
    cached.embed_documents(["alpha", "beta"])
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_cache_avoids_recomputation(tmp_path: Path):
    base = _CountingEmbeddings()
    cached = CachingEmbeddings(base, tmp_path, namespace="count")
    cached.embed_documents(["x", "y"])
    cached.embed_documents(["x", "y", "z"])  # x, y are cached; only z is new
    assert base.calls == 3
