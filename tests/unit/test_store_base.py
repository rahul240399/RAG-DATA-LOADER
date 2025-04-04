"""Tests for the vector store mapping helpers."""

from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.base import chunk_to_document


def test_chunk_to_document_preserves_id_content_and_metadata():
    chunk = TextChunk(
        content="retrieval augmented generation",
        metadata={"source": "doc.pdf", "page": 2},
        start_index=0,
        end_index=30,
        source_document="doc.pdf",
    )
    doc = chunk_to_document(chunk)
    assert doc.id == chunk.chunk_id
    assert doc.page_content == chunk.content
    assert doc.metadata == {"source": "doc.pdf", "page": 2}
