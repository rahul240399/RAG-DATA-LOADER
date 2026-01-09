"""
PipelineConfig Data Model

This module defines the PipelineConfig class, which manages configuration
parameters for the RAG indexing pipeline including chunking, embedding,
and storage settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """
    Configuration parameters for the RAG indexing pipeline.
    
    This class encapsulates all configurable parameters for the pipeline
    including text chunking settings, embedding model configuration,
    and ChromaDB connection details.
    
    Attributes:
        chunk_size (int): Maximum size of text chunks in characters (default: 1000)
        chunk_overlap (int): Number of overlapping characters between chunks (default: 200)
        embedding_model (str): Name/path of the embedding model to use
        chromadb_collection (str): Name of the ChromaDB collection (default: "rag_documents")
        chromadb_host (str): ChromaDB server host address (default: "localhost")
        chromadb_port (int): ChromaDB server port number (default: 8000)
    """
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chromadb_collection: str = "rag_documents"
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    
    def validate(self) -> bool:
        """
        Validate configuration parameters for correctness and consistency.
        
        Performs comprehensive validation of all configuration parameters
        to ensure they are within acceptable ranges and logically consistent.
        
        Returns:
            bool: True if all parameters are valid, False otherwise
            
        Validation Rules:
            - chunk_size must be positive (> 0)
            - chunk_overlap must be non-negative (>= 0)
            - chunk_overlap must be less than chunk_size
            - embedding_model must be a non-empty string
            - chromadb_collection must be a non-empty string
            - chromadb_host must be a non-empty string
            - chromadb_port must be in valid port range (1-65535)
        """
        # Validate chunk_size is positive
        if self.chunk_size <= 0:
            return False
        
        # Validate chunk_overlap is non-negative
        if self.chunk_overlap < 0:
            return False
        
        # Validate chunk_overlap is less than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            return False
        
        # Validate embedding_model is non-empty string
        if not isinstance(self.embedding_model, str) or len(self.embedding_model.strip()) == 0:
            return False
        
        # Validate chromadb_collection is non-empty string
        if not isinstance(self.chromadb_collection, str) or len(self.chromadb_collection.strip()) == 0:
            return False
        
        # Validate chromadb_host is non-empty string
        if not isinstance(self.chromadb_host, str) or len(self.chromadb_host.strip()) == 0:
            return False
        
        # Validate chromadb_port is in valid range
        if not isinstance(self.chromadb_port, int) or self.chromadb_port < 1 or self.chromadb_port > 65535:
            return False
        
        return True
    
    @classmethod
    def get_defaults(cls) -> 'PipelineConfig':
        """
        Create a PipelineConfig instance with default values.
        
        This class method provides a convenient way to obtain a configuration
        object with all default values, useful for initialization and testing.
        
        Returns:
            PipelineConfig: A new instance with default configuration values
        """
        return cls()