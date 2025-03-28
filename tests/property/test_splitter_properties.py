"""Property-based tests for the text splitter."""

import string

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document

from rag_loader.ingest.splitter import split_documents
from rag_loader.models.config import PipelineConfig

words = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12)
document_text = st.lists(words, min_size=20, max_size=400).map(" ".join)


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "doc.pdf", "page": 1})


class TestSplitterProperties:
    """Boundary, location, and overlap invariants across many inputs."""

    @given(
        text=document_text,
        chunk_size=st.integers(min_value=40, max_value=400),
        overlap_frac=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=120, deadline=None)
    def test_chunks_are_bounded_and_locate_back_into_source(
        self, text: str, chunk_size: int, overlap_frac: float
    ):
        overlap = int(chunk_size * overlap_frac)
        config = PipelineConfig(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = split_documents([_doc(text)], config)

        assert chunks, "non-empty input should yield at least one chunk"
        for chunk in chunks:
            assert chunk.content
            assert len(chunk.content) <= chunk_size
            # Recorded span must index back to the exact chunk text.
            assert text[chunk.start_index : chunk.end_index] == chunk.content

    @given(text=document_text)
    @settings(max_examples=50, deadline=None)
    def test_chunk_ids_are_unique_and_deterministic(self, text: str):
        config = PipelineConfig(chunk_size=80, chunk_overlap=20)
        first = split_documents([_doc(text)], config)
        second = split_documents([_doc(text)], config)
        ids = [c.chunk_id for c in first]
        assert len(set(ids)) == len(ids)
        assert ids == [c.chunk_id for c in second]
        assert [c.content for c in first] == [c.content for c in second]

    @given(text=document_text)
    @settings(max_examples=50, deadline=None)
    def test_chunk_indices_are_sequential(self, text: str):
        config = PipelineConfig(chunk_size=80, chunk_overlap=20)
        chunks = split_documents([_doc(text)], config)
        total = len(chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(total))
        assert all(c.metadata["total_chunks"] == total for c in chunks)

    def test_overlap_duplicates_content_on_continuous_text(self):
        text = "x" * 500  # no separators, so the splitter falls back to fixed windows
        config = PipelineConfig(chunk_size=100, chunk_overlap=20)
        chunks = split_documents([_doc(text)], config)
        assert len(chunks) > 1
        # Overlapping windows duplicate characters, so total content exceeds source.
        assert sum(len(c.content) for c in chunks) > len(text)

    def test_no_overlap_does_not_duplicate_continuous_text(self):
        text = "y" * 500
        config = PipelineConfig(chunk_size=100, chunk_overlap=0)
        chunks = split_documents([_doc(text)], config)
        assert sum(len(c.content) for c in chunks) == len(text)
