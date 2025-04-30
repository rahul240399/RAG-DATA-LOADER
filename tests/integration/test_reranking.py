"""Tests for the reranking retriever using a deterministic fake compressor."""

import uuid
from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.models.text_chunk import TextChunk
from rag_loader.retrieval import build_reranking_retriever
from rag_loader.store.chroma import ChromaStore


class _LongestFirstCompressor(BaseDocumentCompressor):
    """Deterministic stand-in for a cross-encoder: keep the longest documents."""

    top_n: int = 1

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        return sorted(documents, key=lambda doc: len(doc.page_content), reverse=True)[: self.top_n]


def _chunk(content: str, start: int) -> TextChunk:
    return TextChunk(
        content=content,
        metadata={"source": "d.pdf", "page": 1},
        start_index=start,
        end_index=start + len(content),
        source_document="d.pdf",
    )


def test_reranking_retriever_applies_compressor():
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    store.add_chunks(
        [
            _chunk("short", 0),
            _chunk("a considerably longer document chunk with more content", 10),
            _chunk("medium length text", 80),
        ]
    )
    reranked = build_reranking_retriever(store.as_retriever(k=3), _LongestFirstCompressor(top_n=1))
    docs = reranked.invoke("anything")
    assert len(docs) == 1
    assert "longer" in docs[0].page_content
