"""Tests for embedding helpers using deterministic fake embeddings."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.embeddings import embed_chunks, embedding_dimension
from rag_loader.models.text_chunk import TextChunk


def _chunk(content: str) -> TextChunk:
    return TextChunk(
        content=content,
        metadata={"source": "doc.pdf", "page": 1},
        start_index=0,
        end_index=len(content),
        source_document="doc.pdf",
    )


def test_embed_chunks_returns_one_vector_per_chunk():
    emb = DeterministicFakeEmbedding(size=24)
    chunks = [_chunk("alpha"), _chunk("beta"), _chunk("gamma")]
    vectors = embed_chunks(chunks, emb)
    assert len(vectors) == len(chunks)
    assert all(len(v) == 24 for v in vectors)


def test_embedding_dimension_probes_size():
    emb = DeterministicFakeEmbedding(size=24)
    assert embedding_dimension(emb) == 24


def test_embed_chunks_is_deterministic():
    emb = DeterministicFakeEmbedding(size=8)
    chunks = [_chunk("repeatable")]
    assert embed_chunks(chunks, emb) == embed_chunks(chunks, emb)
