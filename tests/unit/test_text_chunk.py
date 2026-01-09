"""
Unit tests for TextChunk data model.

Tests TextChunk instantiation, chunk ID generation, and method behavior.
Requirements: 2.1
"""

import pytest
from src.models.text_chunk import TextChunk


class TestTextChunkInstantiation:
    """Test TextChunk instantiation and basic functionality."""
    
    def test_text_chunk_creation_with_all_fields(self):
        """Test creating TextChunk with all fields provided."""
        chunk = TextChunk(
            content="This is test content",
            metadata={"page": 1, "section": "intro"},
            start_index=0,
            end_index=20,
            source_document="test.pdf",
            chunk_id="custom_id"
        )
        
        assert chunk.content == "This is test content"
        assert chunk.metadata == {"page": 1, "section": "intro"}
        assert chunk.start_index == 0
        assert chunk.end_index == 20
        assert chunk.source_document == "test.pdf"
        assert chunk.chunk_id == "custom_id"
    
    def test_text_chunk_creation_without_chunk_id(self):
        """Test creating TextChunk without chunk_id triggers auto-generation."""
        chunk = TextChunk(
            content="Test content",
            metadata={},
            start_index=10,
            end_index=22,
            source_document="doc.pdf"
        )
        
        # chunk_id should be auto-generated
        assert chunk.chunk_id != ""
        assert isinstance(chunk.chunk_id, str)
        assert len(chunk.chunk_id) == 32  # MD5 hash length
    
    def test_text_chunk_creation_with_empty_chunk_id(self):
        """Test creating TextChunk with empty chunk_id triggers auto-generation."""
        chunk = TextChunk(
            content="Test content",
            metadata={},
            start_index=10,
            end_index=22,
            source_document="doc.pdf",
            chunk_id=""
        )
        
        # chunk_id should be auto-generated when empty
        assert chunk.chunk_id != ""
        assert isinstance(chunk.chunk_id, str)
        assert len(chunk.chunk_id) == 32  # MD5 hash length


class TestTextChunkIdGeneration:
    """Test TextChunk ID generation behavior."""
    
    def test_chunk_id_generation_consistency(self):
        """Test that identical chunks generate identical IDs."""
        chunk1 = TextChunk(
            content="Same content",
            metadata={"key": "value"},
            start_index=0,
            end_index=12,
            source_document="same.pdf"
        )
        
        chunk2 = TextChunk(
            content="Different content",  # Content doesn't affect ID
            metadata={"different": "metadata"},  # Metadata doesn't affect ID
            start_index=0,
            end_index=12,
            source_document="same.pdf"
        )
        
        # IDs should be identical because position and source are the same
        assert chunk1.chunk_id == chunk2.chunk_id
    
    def test_chunk_id_generation_uniqueness(self):
        """Test that different positions generate different IDs."""
        chunk1 = TextChunk(
            content="Content",
            metadata={},
            start_index=0,
            end_index=7,
            source_document="doc.pdf"
        )
        
        chunk2 = TextChunk(
            content="Content",
            metadata={},
            start_index=8,  # Different start position
            end_index=15,
            source_document="doc.pdf"
        )
        
        # IDs should be different because positions are different
        assert chunk1.chunk_id != chunk2.chunk_id
    
    def test_chunk_id_generation_different_sources(self):
        """Test that different source documents generate different IDs."""
        chunk1 = TextChunk(
            content="Content",
            metadata={},
            start_index=0,
            end_index=7,
            source_document="doc1.pdf"
        )
        
        chunk2 = TextChunk(
            content="Content",
            metadata={},
            start_index=0,
            end_index=7,
            source_document="doc2.pdf"  # Different source
        )
        
        # IDs should be different because source documents are different
        assert chunk1.chunk_id != chunk2.chunk_id
    
    def test_private_generate_chunk_id_method(self):
        """Test the private _generate_chunk_id method directly."""
        chunk = TextChunk(
            content="Test",
            metadata={},
            start_index=5,
            end_index=9,
            source_document="test.pdf",
            chunk_id="initial_id"  # Provide initial ID
        )
        
        # Call private method directly
        generated_id = chunk._generate_chunk_id()
        
        # Should generate a valid MD5 hash
        assert isinstance(generated_id, str)
        assert len(generated_id) == 32
        assert all(c in '0123456789abcdef' for c in generated_id)


class TestTextChunkEdgeCases:
    """Test TextChunk edge cases and boundary conditions."""
    
    def test_empty_content(self):
        """Test TextChunk with empty content."""
        chunk = TextChunk(
            content="",
            metadata={},
            start_index=0,
            end_index=0,
            source_document="empty.pdf"
        )
        
        assert chunk.content == ""
        assert chunk.chunk_id != ""  # Should still generate ID
    
    def test_empty_metadata(self):
        """Test TextChunk with empty metadata."""
        chunk = TextChunk(
            content="Content",
            metadata={},
            start_index=0,
            end_index=7,
            source_document="doc.pdf"
        )
        
        assert chunk.metadata == {}
        assert chunk.chunk_id != ""
    
    def test_zero_indices(self):
        """Test TextChunk with zero start and end indices."""
        chunk = TextChunk(
            content="",
            metadata={},
            start_index=0,
            end_index=0,
            source_document="doc.pdf"
        )
        
        assert chunk.start_index == 0
        assert chunk.end_index == 0
        assert chunk.chunk_id != ""
    
    def test_large_indices(self):
        """Test TextChunk with large index values."""
        chunk = TextChunk(
            content="Content at end of large document",
            metadata={},
            start_index=1000000,
            end_index=1000032,
            source_document="large.pdf"
        )
        
        assert chunk.start_index == 1000000
        assert chunk.end_index == 1000032
        assert chunk.chunk_id != ""
    
    def test_special_characters_in_source_document(self):
        """Test TextChunk with special characters in source document name."""
        chunk = TextChunk(
            content="Content",
            metadata={},
            start_index=0,
            end_index=7,
            source_document="doc with spaces & symbols!.pdf"
        )
        
        assert chunk.source_document == "doc with spaces & symbols!.pdf"
        assert chunk.chunk_id != ""  # Should handle special characters
    
    def test_unicode_content(self):
        """Test TextChunk with Unicode content."""
        chunk = TextChunk(
            content="Content with émojis 🚀 and ñoñó",
            metadata={"language": "mixed"},
            start_index=0,
            end_index=32,
            source_document="unicode.pdf"
        )
        
        assert "émojis" in chunk.content
        assert "🚀" in chunk.content
        assert "ñoñó" in chunk.content
        assert chunk.chunk_id != ""