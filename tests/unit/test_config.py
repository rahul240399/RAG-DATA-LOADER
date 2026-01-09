"""
Unit tests for PipelineConfig data model.

Tests PipelineConfig validation edge cases and default value handling.
Requirements: 5.1
"""

import pytest
from src.models.config import PipelineConfig


class TestPipelineConfigValidation:
    """Test PipelineConfig validation method edge cases."""
    
    def test_valid_default_configuration(self):
        """Test that default configuration is valid."""
        config = PipelineConfig()
        assert config.validate() is True
    
    def test_valid_custom_configuration(self):
        """Test valid custom configuration."""
        config = PipelineConfig(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="custom-model",
            chromadb_collection="custom_collection",
            chromadb_host="192.168.1.100",
            chromadb_port=9000
        )
        assert config.validate() is True
    
    # Chunk size validation edge cases
    def test_chunk_size_zero_invalid(self):
        """Test that chunk_size of 0 is invalid."""
        config = PipelineConfig(chunk_size=0)
        assert config.validate() is False
    
    def test_chunk_size_negative_invalid(self):
        """Test that negative chunk_size is invalid."""
        config = PipelineConfig(chunk_size=-100)
        assert config.validate() is False
    
    def test_chunk_size_one_valid(self):
        """Test that chunk_size of 1 is valid (minimum positive)."""
        config = PipelineConfig(chunk_size=1, chunk_overlap=0)
        assert config.validate() is True
    
    # Chunk overlap validation edge cases
    def test_chunk_overlap_negative_invalid(self):
        """Test that negative chunk_overlap is invalid."""
        config = PipelineConfig(chunk_overlap=-1)
        assert config.validate() is False
    
    def test_chunk_overlap_zero_valid(self):
        """Test that chunk_overlap of 0 is valid."""
        config = PipelineConfig(chunk_overlap=0)
        assert config.validate() is True
    
    def test_chunk_overlap_equal_to_chunk_size_invalid(self):
        """Test that chunk_overlap equal to chunk_size is invalid."""
        config = PipelineConfig(chunk_size=1000, chunk_overlap=1000)
        assert config.validate() is False
    
    def test_chunk_overlap_greater_than_chunk_size_invalid(self):
        """Test that chunk_overlap greater than chunk_size is invalid."""
        config = PipelineConfig(chunk_size=500, chunk_overlap=600)
        assert config.validate() is False
    
    def test_chunk_overlap_one_less_than_chunk_size_valid(self):
        """Test that chunk_overlap one less than chunk_size is valid."""
        config = PipelineConfig(chunk_size=1000, chunk_overlap=999)
        assert config.validate() is True
    
    # Embedding model validation edge cases
    def test_embedding_model_empty_string_invalid(self):
        """Test that empty embedding_model string is invalid."""
        config = PipelineConfig(embedding_model="")
        assert config.validate() is False
    
    def test_embedding_model_whitespace_only_invalid(self):
        """Test that whitespace-only embedding_model is invalid."""
        config = PipelineConfig(embedding_model="   ")
        assert config.validate() is False
    
    def test_embedding_model_none_invalid(self):
        """Test that None embedding_model is invalid."""
        config = PipelineConfig(embedding_model=None)
        assert config.validate() is False
    
    def test_embedding_model_non_string_invalid(self):
        """Test that non-string embedding_model is invalid."""
        config = PipelineConfig(embedding_model=123)
        assert config.validate() is False
    
    def test_embedding_model_single_character_valid(self):
        """Test that single character embedding_model is valid."""
        config = PipelineConfig(embedding_model="a")
        assert config.validate() is True
    
    # ChromaDB collection validation edge cases
    def test_chromadb_collection_empty_string_invalid(self):
        """Test that empty chromadb_collection string is invalid."""
        config = PipelineConfig(chromadb_collection="")
        assert config.validate() is False
    
    def test_chromadb_collection_whitespace_only_invalid(self):
        """Test that whitespace-only chromadb_collection is invalid."""
        config = PipelineConfig(chromadb_collection="  \t\n  ")
        assert config.validate() is False
    
    def test_chromadb_collection_none_invalid(self):
        """Test that None chromadb_collection is invalid."""
        config = PipelineConfig(chromadb_collection=None)
        assert config.validate() is False
    
    def test_chromadb_collection_non_string_invalid(self):
        """Test that non-string chromadb_collection is invalid."""
        config = PipelineConfig(chromadb_collection=456)
        assert config.validate() is False
    
    # ChromaDB host validation edge cases
    def test_chromadb_host_empty_string_invalid(self):
        """Test that empty chromadb_host string is invalid."""
        config = PipelineConfig(chromadb_host="")
        assert config.validate() is False
    
    def test_chromadb_host_whitespace_only_invalid(self):
        """Test that whitespace-only chromadb_host is invalid."""
        config = PipelineConfig(chromadb_host="   ")
        assert config.validate() is False
    
    def test_chromadb_host_none_invalid(self):
        """Test that None chromadb_host is invalid."""
        config = PipelineConfig(chromadb_host=None)
        assert config.validate() is False
    
    def test_chromadb_host_non_string_invalid(self):
        """Test that non-string chromadb_host is invalid."""
        config = PipelineConfig(chromadb_host=789)
        assert config.validate() is False
    
    # ChromaDB port validation edge cases
    def test_chromadb_port_zero_invalid(self):
        """Test that chromadb_port of 0 is invalid."""
        config = PipelineConfig(chromadb_port=0)
        assert config.validate() is False
    
    def test_chromadb_port_negative_invalid(self):
        """Test that negative chromadb_port is invalid."""
        config = PipelineConfig(chromadb_port=-1)
        assert config.validate() is False
    
    def test_chromadb_port_one_valid(self):
        """Test that chromadb_port of 1 is valid (minimum)."""
        config = PipelineConfig(chromadb_port=1)
        assert config.validate() is True
    
    def test_chromadb_port_65535_valid(self):
        """Test that chromadb_port of 65535 is valid (maximum)."""
        config = PipelineConfig(chromadb_port=65535)
        assert config.validate() is True
    
    def test_chromadb_port_65536_invalid(self):
        """Test that chromadb_port of 65536 is invalid (above maximum)."""
        config = PipelineConfig(chromadb_port=65536)
        assert config.validate() is False
    
    def test_chromadb_port_non_integer_invalid(self):
        """Test that non-integer chromadb_port is invalid."""
        config = PipelineConfig(chromadb_port="8000")
        assert config.validate() is False
    
    def test_chromadb_port_float_invalid(self):
        """Test that float chromadb_port is invalid."""
        config = PipelineConfig(chromadb_port=8000.5)
        assert config.validate() is False


class TestPipelineConfigDefaults:
    """Test PipelineConfig default values and get_defaults method."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PipelineConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.chromadb_collection == "rag_documents"
        assert config.chromadb_host == "localhost"
        assert config.chromadb_port == 8000
    
    def test_get_defaults_method(self):
        """Test get_defaults class method returns correct defaults."""
        config = PipelineConfig.get_defaults()
        
        assert isinstance(config, PipelineConfig)
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.chromadb_collection == "rag_documents"
        assert config.chromadb_host == "localhost"
        assert config.chromadb_port == 8000
    
    def test_get_defaults_returns_valid_config(self):
        """Test that get_defaults returns a valid configuration."""
        config = PipelineConfig.get_defaults()
        assert config.validate() is True
    
    def test_get_defaults_creates_new_instance(self):
        """Test that get_defaults creates a new instance each time."""
        config1 = PipelineConfig.get_defaults()
        config2 = PipelineConfig.get_defaults()
        
        assert config1 is not config2  # Different instances
        assert config1.chunk_size == config2.chunk_size  # Same values


class TestPipelineConfigEdgeCases:
    """Test PipelineConfig edge cases and boundary conditions."""
    
    def test_multiple_validation_failures(self):
        """Test configuration with multiple validation failures."""
        config = PipelineConfig(
            chunk_size=-1,  # Invalid
            chunk_overlap=-1,  # Invalid
            embedding_model="",  # Invalid
            chromadb_collection="",  # Invalid
            chromadb_host="",  # Invalid
            chromadb_port=0  # Invalid
        )
        assert config.validate() is False
    
    def test_boundary_values_valid(self):
        """Test configuration with boundary values that should be valid."""
        config = PipelineConfig(
            chunk_size=1,
            chunk_overlap=0,
            embedding_model="a",
            chromadb_collection="c",
            chromadb_host="h",
            chromadb_port=1
        )
        assert config.validate() is True
    
    def test_large_valid_values(self):
        """Test configuration with large but valid values."""
        config = PipelineConfig(
            chunk_size=100000,
            chunk_overlap=50000,
            embedding_model="very-long-model-name-with-many-characters",
            chromadb_collection="very_long_collection_name_with_underscores",
            chromadb_host="very.long.hostname.with.many.dots.example.com",
            chromadb_port=65535
        )
        assert config.validate() is True
    
    def test_special_characters_in_strings(self):
        """Test configuration with special characters in string fields."""
        config = PipelineConfig(
            embedding_model="model/with-special_chars.v2",
            chromadb_collection="collection-with_special.chars",
            chromadb_host="host-with-dashes.example.com"
        )
        assert config.validate() is True