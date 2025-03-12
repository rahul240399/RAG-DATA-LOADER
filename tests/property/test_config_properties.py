"""Property-based tests for PipelineConfig validation."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from rag_loader.models.config import PipelineConfig

# Printable, non-whitespace text so model/collection/host names round-trip exactly.
clean_text = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
)


@st.composite
def valid_size_and_overlap(draw: st.DrawFn) -> tuple[int, int]:
    """Draw a (chunk_size, chunk_overlap) pair where overlap < size."""
    size = draw(st.integers(min_value=1, max_value=100_000))
    overlap = draw(st.integers(min_value=0, max_value=size - 1))
    return size, overlap


class TestPipelineConfigProperties:
    """Validation invariants exercised across a wide range of inputs."""

    @given(
        size_overlap=valid_size_and_overlap(),
        embedding_model=clean_text,
        chromadb_collection=clean_text,
        chromadb_host=clean_text,
        chromadb_port=st.integers(min_value=1, max_value=65_535),
    )
    @settings(max_examples=150)
    def test_valid_parameters_construct_and_round_trip(
        self,
        size_overlap: tuple[int, int],
        embedding_model: str,
        chromadb_collection: str,
        chromadb_host: str,
        chromadb_port: int,
    ):
        chunk_size, chunk_overlap = size_overlap
        config = PipelineConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            chromadb_collection=chromadb_collection,
            chromadb_host=chromadb_host,
            chromadb_port=chromadb_port,
        )
        assert config.chunk_size == chunk_size
        assert config.chunk_overlap == chunk_overlap
        assert config.embedding_model == embedding_model
        assert config.chromadb_port == chromadb_port

    @given(chunk_size=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_non_positive_chunk_size_always_rejected(self, chunk_size: int):
        with pytest.raises(ValidationError):
            PipelineConfig(chunk_size=chunk_size)

    @given(
        chunk_size=st.integers(min_value=1, max_value=10_000),
        chunk_overlap=st.integers(min_value=0, max_value=20_000),
    )
    @settings(max_examples=150)
    def test_overlap_must_be_smaller_than_size(self, chunk_size: int, chunk_overlap: int):
        if chunk_overlap < chunk_size:
            config = PipelineConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            assert config.chunk_overlap < config.chunk_size
        else:
            with pytest.raises(ValidationError):
                PipelineConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @given(port=st.one_of(st.integers(max_value=0), st.integers(min_value=65_536)))
    @settings(max_examples=100)
    def test_ports_outside_range_rejected(self, port: int):
        with pytest.raises(ValidationError):
            PipelineConfig(chromadb_port=port)

    def test_default_configuration_is_valid(self):
        config = PipelineConfig.get_defaults()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
