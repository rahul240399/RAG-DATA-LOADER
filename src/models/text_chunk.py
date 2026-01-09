"""
TextChunk Data Model

This module defines the TextChunk class, which represents a segment of text
extracted from a document with associated metadata and positioning information.
"""

from dataclasses import dataclass
from typing import Dict, Any
import hashlib


@dataclass
class TextChunk:
    """
    Represents a chunk of text extracted from a document.
    
    A TextChunk contains the actual text content along with metadata about
    its position within the source document and other relevant information
    for indexing and retrieval.
    
    Attributes:
        content (str): The actual text content of the chunk
        metadata (Dict[str, Any]): Additional metadata associated with the chunk
        start_index (int): Starting character position in the source document
        end_index (int): Ending character position in the source document
        source_document (str): Identifier or path of the source document
        chunk_id (str): Unique identifier for this chunk
    """
    
    content: str
    metadata: Dict[str, Any]
    start_index: int
    end_index: int
    source_document: str
    chunk_id: str = ""
    
    def __post_init__(self) -> None:
        """
        Post-initialization method that automatically generates a chunk_id
        if one was not provided during instantiation.
        """
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
    
    def _generate_chunk_id(self) -> str:
        """
        Generate a deterministic unique identifier for this chunk.
        
        The chunk ID is generated using an MD5 hash of the source document,
        start index, and end index. This ensures that:
        1. Identical chunks from the same location always get the same ID
        2. The pipeline is idempotent - reprocessing the same file produces the same IDs
        3. ChromaDB can handle duplicates via upsert operations
        
        Returns:
            str: A deterministic unique identifier based on MD5 hash
        """
        # Create a deterministic string from positioning data
        id_string = f"{self.source_document}_{self.start_index}_{self.end_index}"
        
        # Generate MD5 hash for consistent, collision-resistant ID
        hash_object = hashlib.md5(id_string.encode('utf-8'))
        return hash_object.hexdigest()