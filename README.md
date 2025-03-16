# RAG Data Loader

A LangChain-based Retrieval-Augmented Generation (RAG) pipeline: ingest PDFs, chunk
and embed them, store the vectors, then retrieve and generate grounded answers.
Embedding, vector-store, and LLM providers are pluggable and selected entirely
through configuration.

> **Status:** foundations in place (domain models, settings, provider factories).
> Ingestion → retrieval → generation are being built out iteratively.

## Architecture

```
PDF → Load → Split → Embed → Vector Store      (indexing)
                                  |
Question → Retrieve → Rerank → LLM → Answer    (query)
```

Each stage is a LangChain component chosen from `Settings`, so the same code runs
locally (HuggingFace + Chroma) or in production (OpenAI/Voyage + Qdrant) unchanged.

## Stack

- **Orchestration:** LangChain / LCEL
- **Config:** Pydantic v2 + pydantic-settings
- **Embeddings:** HuggingFace · OpenAI · Voyage (via `init_embeddings`)
- **LLM:** Anthropic · OpenAI · Ollama (via `init_chat_model`)
- **Vector store:** Chroma (dev) · Qdrant (prod)
- **Tooling:** uv · ruff · mypy (strict) · pytest + Hypothesis

## Quickstart

```bash
uv sync           # install dependencies
uv run pytest     # run the test suite
```

## Configuration

Settings are read from `RAG_`-prefixed environment variables or a local `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | `1000` | Max characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_EMBEDDING_PROVIDER` | `huggingface` | Embedding backend |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_LLM_PROVIDER` | `anthropic` | Chat model backend |
| `RAG_LLM_MODEL` | `claude-3-5-sonnet-latest` | Chat model |

## Project layout

```
src/rag_loader/
├── models/       # domain models (chunks, config, results)
├── settings.py   # environment-driven configuration
└── factories.py  # provider selection (chat model / embeddings)
tests/            # unit + property-based tests
```

## License

MIT
