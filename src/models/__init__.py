"""
RAG Indexing Pipeline - Models Package

This package contains data models for the RAG indexing pipeline including:
- TextChunk: Represents a chunk of text with metadata
- PipelineConfig: Configuration parameters for the pipeline
- DocumentProcessingResult: Results from document processing operations
- BatchProcessingResult: Results from batch processing operations
"""

__version__ = "0.1.0"
__author__ = "RAG Pipeline Team"
__description__ = "Data models for RAG indexing pipeline operations"
__license__ = "MIT"

# Import models
from .text_chunk import TextChunk
from .config import PipelineConfig
from .results import DocumentProcessingResult, BatchProcessingResult

# Package exports
__all__ = [
    "TextChunk",
    "PipelineConfig", 
    "DocumentProcessingResult",
    "BatchProcessingResult",
]