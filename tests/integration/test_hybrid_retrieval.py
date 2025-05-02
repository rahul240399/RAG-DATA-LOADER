"""Tests for hybrid dense + BM25 retrieval."""

import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.models.text_chunk import TextChunk
from rag_loader.retrieval import build_bm25_retriever, build_hybrid_retriever
from rag_loader.store.chroma import ChromaStore


def _chunk(content: str, start: int) -> TextChunk:
    return TextChunk(
        content=content,
        metadata={"source": "d.pdf", "page": 1},
        start_index=start,
        end_index=start + len(content),
        source_document="d.pdf",
    )


def test_bm25_ranks_keyword_match_first():
    # A realistic corpus size so BM25's IDF can discriminate the rare term.
    chunks = [
        _chunk("apple pie recipe instructions", 0),
        _chunk("dog elephant frog zoo", 20),
        _chunk("vector search retrieval system", 40),
        _chunk("quantum physics lecture notes", 60),
        _chunk("financial market quarterly report", 80),
    ]
    docs = build_bm25_retriever(chunks, k=1).invoke("apple")
    assert docs[0].page_content.startswith("apple")


def test_hybrid_merges_dense_and_bm25():
    chunks = [
        _chunk("apple banana cherry", 0),
        _chunk("dog elephant frog", 20),
        _chunk("vector search retrieval system", 40),
    ]
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    store.add_chunks(chunks)
    hybrid = build_hybrid_retriever(
        store.as_retriever(k=3), build_bm25_retriever(chunks, k=3), weights=(0.5, 0.5), k=2
    )
    docs = hybrid.invoke("apple")
    assert docs
    assert len(docs) <= 2
    assert all(isinstance(doc, Document) for doc in docs)
