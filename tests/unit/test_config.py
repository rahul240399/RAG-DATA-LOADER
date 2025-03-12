"""Unit tests for the PipelineConfig model."""

import pytest
from pydantic import ValidationError

from rag_loader.models.config import PipelineConfig


class TestDefaults:
    """Default values and the get_defaults helper."""

    def test_default_values(self):
        config = PipelineConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.chromadb_collection == "rag_documents"
        assert config.chromadb_host == "localhost"
        assert config.chromadb_port == 8000

    def test_get_defaults_returns_independent_equal_instances(self):
        a = PipelineConfig.get_defaults()
        b = PipelineConfig.get_defaults()
        assert isinstance(a, PipelineConfig)
        assert a == b
        assert a is not b

    def test_config_is_frozen(self):
        config = PipelineConfig()
        with pytest.raises(ValidationError):
            config.chunk_size = 2000


class TestValidConfigurations:
    """Configurations that should construct successfully."""

    def test_custom_values(self):
        config = PipelineConfig(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="custom-model",
            chromadb_collection="custom_collection",
            chromadb_host="192.168.1.100",
            chromadb_port=9000,
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.chromadb_port == 9000

    def test_boundary_values(self):
        assert PipelineConfig(chunk_size=1, chunk_overlap=0).chunk_size == 1
        assert PipelineConfig(chromadb_port=1).chromadb_port == 1
        assert PipelineConfig(chromadb_port=65535).chromadb_port == 65535

    def test_string_fields_are_stripped(self):
        config = PipelineConfig(embedding_model="  model-name  ")
        assert config.embedding_model == "model-name"


class TestInvalidConfigurations:
    """Configurations that must raise at construction time."""

    @pytest.mark.parametrize("chunk_size", [0, -1, -100])
    def test_non_positive_chunk_size(self, chunk_size):
        with pytest.raises(ValidationError):
            PipelineConfig(chunk_size=chunk_size)

    def test_negative_overlap(self):
        with pytest.raises(ValidationError):
            PipelineConfig(chunk_overlap=-1)

    @pytest.mark.parametrize(("size", "overlap"), [(1000, 1000), (500, 600), (10, 10)])
    def test_overlap_not_smaller_than_size(self, size, overlap):
        with pytest.raises(ValidationError):
            PipelineConfig(chunk_size=size, chunk_overlap=overlap)

    @pytest.mark.parametrize("value", ["", "   ", "  \t\n  "])
    def test_blank_string_fields(self, value):
        with pytest.raises(ValidationError):
            PipelineConfig(embedding_model=value)
        with pytest.raises(ValidationError):
            PipelineConfig(chromadb_collection=value)
        with pytest.raises(ValidationError):
            PipelineConfig(chromadb_host=value)

    @pytest.mark.parametrize("value", [None, 123])
    def test_non_string_fields(self, value):
        with pytest.raises(ValidationError):
            PipelineConfig(embedding_model=value)

    @pytest.mark.parametrize("port", [0, -1, 65536, 70000])
    def test_port_out_of_range(self, port):
        with pytest.raises(ValidationError):
            PipelineConfig(chromadb_port=port)

    def test_fractional_port_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig(chromadb_port=8000.5)

    def test_multiple_failures_report_together(self):
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(chunk_size=-1, embedding_model="", chromadb_port=0)
        # Pydantic aggregates every violation into a single error.
        assert len(exc_info.value.errors()) >= 3
