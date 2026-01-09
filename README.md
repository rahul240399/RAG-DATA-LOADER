# RAG Indexing Pipeline

A robust Python library for processing PDF documents into searchable vector embeddings for Retrieval-Augmented Generation (RAG) applications.

## Features

- **PDF Text Extraction**: Extract text content from PDF documents
- **Intelligent Text Chunking**: Recursive character splitting with configurable overlap
- **Vector Embeddings**: Generate embeddings using sentence transformers
- **ChromaDB Integration**: Store and retrieve embeddings efficiently
- **Comprehensive Testing**: 95 tests including property-based testing with Hypothesis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models import PipelineConfig, TextChunk

# Configure the pipeline
config = PipelineConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Create text chunks
chunk = TextChunk(
    content="Your document text here",
    metadata={"source": "document.pdf"},
    start_index=0,
    end_index=100,
    source_document="document.pdf"
)
```

## Project Structure

```
src/
├── models/           # Data models
│   ├── config.py     # Pipeline configuration
│   ├── results.py    # Processing results
│   └── text_chunk.py # Text chunk model
tests/
├── unit/            # Unit tests
└── property/        # Property-based tests
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run property-based tests only:
```bash
pytest tests/property/ -v
```

## Models

### TextChunk
Represents a segment of text with metadata and automatic ID generation.

### PipelineConfig  
Configuration for the RAG pipeline with validation.

### DocumentProcessingResult
Results and statistics for processing individual documents.

### BatchProcessingResult
Aggregated results for batch processing operations.

## Development Status

✅ **Phase 1 Complete**: Core data models with comprehensive testing
- All 95 tests passing
- Property-based testing with Hypothesis
- Full validation and error handling

🚧 **Next Phase**: PDF loading, text splitting, embedding generation, and ChromaDB storage

## License

MIT License