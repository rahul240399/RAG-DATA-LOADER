"""Embedding helpers bridging TextChunks to a LangChain Embeddings provider."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

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


class CachingEmbeddings(Embeddings):
    """Wrap an embeddings provider with a persistent, content-addressed cache.

    Document embeddings are keyed by a hash of ``(namespace, text)`` and stored on
    disk, so repeated or overlapping content (re-indexing, resumed runs) is
    embedded once and reused, cutting both latency and provider cost. Queries are
    passed straight through.
    """

    def __init__(self, embeddings: Embeddings, cache_dir: str | Path, namespace: str = "") -> None:
        self._embeddings = embeddings
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._namespace = namespace

    def _key(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self._namespace}\x00{text}".encode()).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cached: dict[int, list[float]] = {}
        pending: dict[int, str] = {}
        for index, text in enumerate(texts):
            key = self._key(text)
            if key.exists():
                cached[index] = json.loads(key.read_text())
            else:
                pending[index] = text

        computed: dict[int, list[float]] = {}
        if pending:
            indices = list(pending)
            vectors = self._embeddings.embed_documents([pending[i] for i in indices])
            for i, vector in zip(indices, vectors, strict=True):
                self._key(pending[i]).write_text(json.dumps(vector))
                computed[i] = vector

        merged = {**cached, **computed}
        return [merged[i] for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
