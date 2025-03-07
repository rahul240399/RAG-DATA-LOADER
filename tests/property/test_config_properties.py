"""
Property-based tests for PipelineConfig model.

This module contains property-based tests that verify the correctness
of PipelineConfig validation behavior across a wide range of inputs using Hypothesis.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rag_loader.models.config import PipelineConfig


class TestPipelineConfigProperties:
    """Property-based tests for PipelineConfig model."""

    @given(
        chunk_size=st.integers(),
        chunk_overlap=st.integers(),
        embedding_model=st.text(),
        chromadb_collection=st.text(),
        chromadb_host=st.text(),
        chromadb_port=st.integers(),
    )
    @settings(max_examples=100)
    def test_configuration_parameter_validation(
        self,
        chunk_size,
        chunk_overlap,
        embedding_model,
        chromadb_collection,
        chromadb_host,
        chromadb_port,
    ):
        """
        **Feature: rag-indexing-pipeline, Property 5: Configuration parameter validation**

        Property: For any configuration parameters provided to the pipeline, the system
        should validate parameter ranges and apply changes to subsequent operations.

        This test validates that:
        1. Valid configurations return True from validate()
        2. Invalid configurations return False from validate()
        3. Validation rules are consistently applied across all parameters
        4. Parameter ranges are properly enforced

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        config = PipelineConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            chromadb_collection=chromadb_collection,
            chromadb_host=chromadb_host,
            chromadb_port=chromadb_port,
        )

        validation_result = config.validate()

        # Determine if the configuration should be valid based on validation rules
        should_be_valid = (
            chunk_size > 0  # chunk_size must be positive
            and chunk_overlap >= 0  # chunk_overlap must be non-negative
            and chunk_overlap < chunk_size  # chunk_overlap must be less than chunk_size
            and isinstance(embedding_model, str)
            and len(embedding_model.strip()) > 0  # embedding_model must be non-empty string
            and isinstance(chromadb_collection, str)
            and len(chromadb_collection.strip()) > 0  # chromadb_collection must be non-empty string
            and isinstance(chromadb_host, str)
            and len(chromadb_host.strip()) > 0  # chromadb_host must be non-empty string
            and isinstance(chromadb_port, int)
            and 1 <= chromadb_port <= 65535  # chromadb_port must be in valid port range
        )

        # Property: Validation result should match expected validity
        assert validation_result == should_be_valid, (
            f"Validation result {validation_result} does not match "
            f"expected validity {should_be_valid} "
            f"for config: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"embedding_model='{embedding_model}', chromadb_collection='{chromadb_collection}', "
            f"chromadb_host='{chromadb_host}', chromadb_port={chromadb_port}"
        )

    @given(
        chunk_size=st.integers(min_value=1, max_value=10000),
        chunk_overlap=st.integers(min_value=0, max_value=9999),
        embedding_model=st.text(min_size=1).filter(lambda x: len(x.strip()) > 0),
        chromadb_collection=st.text(min_size=1).filter(lambda x: len(x.strip()) > 0),
        chromadb_host=st.text(min_size=1).filter(lambda x: len(x.strip()) > 0),
        chromadb_port=st.integers(min_value=1, max_value=65535),
    )
    @settings(max_examples=100)
    def test_valid_configurations_pass_validation(
        self,
        chunk_size,
        chunk_overlap,
        embedding_model,
        chromadb_collection,
        chromadb_host,
        chromadb_port,
    ):
        """
        Property: All configurations with valid parameters should pass validation.

        This test ensures that when all parameters are within valid ranges,
        the validate() method returns True.
        """
        # Ensure chunk_overlap is less than chunk_size
        assume(chunk_overlap < chunk_size)

        config = PipelineConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            chromadb_collection=chromadb_collection,
            chromadb_host=chromadb_host,
            chromadb_port=chromadb_port,
        )

        # Property: Valid configurations should always pass validation
        assert config.validate(), (
            f"Valid configuration should pass validation: "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"embedding_model='{embedding_model}', chromadb_collection='{chromadb_collection}', "
            f"chromadb_host='{chromadb_host}', chromadb_port={chromadb_port}"
        )

    def test_default_configuration_is_valid(self):
        """
        Property: The default configuration should always be valid.

        This test ensures that PipelineConfig.get_defaults() returns a valid configuration.
        """
        default_config = PipelineConfig.get_defaults()

        # Property: Default configuration should always be valid
        assert default_config.validate(), f"Default configuration should be valid: {default_config}"

        # Verify default values match expected defaults
        assert default_config.chunk_size == 1000
        assert default_config.chunk_overlap == 200
        assert default_config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert default_config.chromadb_collection == "rag_documents"
        assert default_config.chromadb_host == "localhost"
        assert default_config.chromadb_port == 8000

    @given(
        chunk_size=st.one_of(
            st.integers(max_value=0),  # Invalid: non-positive chunk_size
            st.integers(min_value=1, max_value=10000),  # Valid chunk_size
        ),
        chunk_overlap=st.integers(min_value=-1000, max_value=10000),
    )
    @settings(max_examples=100)
    def test_chunk_size_and_overlap_validation(self, chunk_size, chunk_overlap):
        """
        Property: Chunk size and overlap validation rules should be consistently enforced.

        This test specifically validates the relationship between chunk_size and chunk_overlap.
        """
        config = PipelineConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        validation_result = config.validate()

        # Determine expected validity based on chunk size and overlap rules
        expected_valid = (
            chunk_size > 0  # chunk_size must be positive
            and chunk_overlap >= 0  # chunk_overlap must be non-negative
            and chunk_overlap < chunk_size  # chunk_overlap must be less than chunk_size
        )

        # Property: Validation should match expected result for chunk parameters
        assert validation_result == expected_valid, (
            f"Chunk validation failed: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"expected_valid={expected_valid}, actual_result={validation_result}"
        )

    @given(
        port=st.one_of(
            st.integers(max_value=0),  # Invalid: non-positive port
            st.integers(min_value=65536),  # Invalid: port too high
            st.integers(min_value=1, max_value=65535),  # Valid port range
        )
    )
    @settings(max_examples=100)
    def test_port_validation(self, port):
        """
        Property: Port validation should enforce valid port range (1-65535).

        This test validates that chromadb_port is properly validated.
        """
        config = PipelineConfig(chromadb_port=port)

        validation_result = config.validate()
        expected_valid = 1 <= port <= 65535

        # Property: Port validation should match expected result
        assert validation_result == expected_valid, (
            f"Port validation failed: port={port}, "
            f"expected_valid={expected_valid}, actual_result={validation_result}"
        )

    @given(
        string_param=st.one_of(
            st.text(max_size=0),  # Empty string
            st.text().filter(lambda x: len(x.strip()) == 0),  # Whitespace-only string
            st.text(min_size=1).filter(lambda x: len(x.strip()) > 0),  # Valid non-empty string
        ),
        param_name=st.sampled_from(["embedding_model", "chromadb_collection", "chromadb_host"]),
    )
    @settings(max_examples=100)
    def test_string_parameter_validation(self, string_param, param_name):
        """
        Property: String parameters should be validated as non-empty after stripping whitespace.

        This test validates that embedding_model, chromadb_collection, and chromadb_host
        are properly validated as non-empty strings.
        """
        kwargs = {param_name: string_param}
        config = PipelineConfig(**kwargs)

        validation_result = config.validate()
        expected_valid = isinstance(string_param, str) and len(string_param.strip()) > 0

        # Property: String parameter validation should match expected result
        assert validation_result == expected_valid, (
            f"String parameter validation failed: {param_name}='{string_param}', "
            f"expected_valid={expected_valid}, actual_result={validation_result}"
        )
