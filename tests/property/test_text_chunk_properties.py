"""
Property-based tests for TextChunk model.

This module contains property-based tests that verify the correctness
of TextChunk behavior across a wide range of inputs using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, settings
from src.models.text_chunk import TextChunk


class TestTextChunkProperties:
    """Property-based tests for TextChunk model."""
    
    @given(
        content=st.text(min_size=1),
        metadata=st.dictionaries(st.text(min_size=1), st.text()),
        start_index=st.integers(min_value=0, max_value=10000),
        end_index=st.integers(min_value=0, max_value=10000),
        source_document=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_chunk_id_generation_consistency(self, content, metadata, start_index, end_index, source_document):
        """
        **Feature: rag-indexing-pipeline, Property 1: Chunk ID generation consistency**
        
        Property: For any TextChunk with the same source_document, start_index, and end_index,
        the chunk_id should be generated consistently and deterministically.
        
        This test validates that:
        1. Chunk IDs are generated consistently for the same inputs (idempotency)
        2. Chunk IDs are deterministic MD5 hashes
        3. Multiple TextChunk instances with the same positioning data have identical chunk_ids
        4. The pipeline supports idempotent operations for duplicate processing
        """
        # Ensure end_index >= start_index for valid chunks
        if end_index < start_index:
            start_index, end_index = end_index, start_index
        
        # Create first TextChunk instance
        chunk1 = TextChunk(
            content=content,
            metadata=metadata,
            start_index=start_index,
            end_index=end_index,
            source_document=source_document
        )
        
        # Create second TextChunk instance with same positioning data but different content/metadata
        chunk2 = TextChunk(
            content=content + "_different",  # Different content
            metadata={**metadata, "extra": "data"},  # Different metadata
            start_index=start_index,
            end_index=end_index,
            source_document=source_document
        )
        
        # Property 1: Chunk IDs should be identical for same positioning data (idempotency)
        assert chunk1.chunk_id == chunk2.chunk_id, (
            f"Chunk IDs should be identical for same positioning data. "
            f"Got {chunk1.chunk_id} and {chunk2.chunk_id}"
        )
        
        # Property 2: Chunk ID should be a valid MD5 hash (32 hex characters)
        import re
        md5_pattern = re.compile(r'^[a-f0-9]{32}$')
        assert md5_pattern.match(chunk1.chunk_id), (
            f"Chunk ID should be a valid MD5 hash (32 hex characters). "
            f"Got {chunk1.chunk_id}"
        )
        
        # Property 3: Chunk ID should be deterministic - same inputs always produce same hash
        import hashlib
        expected_id_string = f"{source_document}_{start_index}_{end_index}"
        expected_hash = hashlib.md5(expected_id_string.encode('utf-8')).hexdigest()
        assert chunk1.chunk_id == expected_hash, (
            f"Chunk ID should be deterministic MD5 hash. "
            f"Expected {expected_hash}, got {chunk1.chunk_id}"
        )
        
        # Property 4: Chunk ID should be automatically generated when not provided
        assert chunk1.chunk_id != "", "Chunk ID should not be empty"
        assert chunk2.chunk_id != "", "Chunk ID should not be empty"
        
        # Property 5: Manually set chunk_id should be preserved
        manual_chunk_id = "manual_id_123"
        chunk3 = TextChunk(
            content=content,
            metadata=metadata,
            start_index=start_index,
            end_index=end_index,
            source_document=source_document,
            chunk_id=manual_chunk_id
        )
        assert chunk3.chunk_id == manual_chunk_id, (
            f"Manually set chunk_id should be preserved. "
            f"Expected {manual_chunk_id}, got {chunk3.chunk_id}"
        )
    
    @given(
        content=st.text(min_size=1),
        metadata=st.dictionaries(st.text(min_size=1), st.text()),
        start_index1=st.integers(min_value=0, max_value=5000),
        end_index1=st.integers(min_value=0, max_value=5000),
        start_index2=st.integers(min_value=0, max_value=5000),
        end_index2=st.integers(min_value=0, max_value=5000),
        source_document=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_chunk_id_uniqueness_for_different_positions(self, content, metadata, 
                                                       start_index1, end_index1,
                                                       start_index2, end_index2,
                                                       source_document):
        """
        Property: TextChunks with different positioning data should have different chunk_ids.
        
        This ensures that chunk_id serves as a unique identifier within a document.
        """
        # Ensure valid index ranges
        if end_index1 < start_index1:
            start_index1, end_index1 = end_index1, start_index1
        if end_index2 < start_index2:
            start_index2, end_index2 = end_index2, start_index2
        
        # Skip if the positioning data is identical
        if start_index1 == start_index2 and end_index1 == end_index2:
            return
        
        chunk1 = TextChunk(
            content=content,
            metadata=metadata,
            start_index=start_index1,
            end_index=end_index1,
            source_document=source_document
        )
        
        chunk2 = TextChunk(
            content=content,
            metadata=metadata,
            start_index=start_index2,
            end_index=end_index2,
            source_document=source_document
        )
        
        # Property: Different positioning should result in different chunk_ids
        assert chunk1.chunk_id != chunk2.chunk_id, (
            f"TextChunks with different positioning should have different chunk_ids. "
            f"Both have chunk_id: {chunk1.chunk_id}"
        )