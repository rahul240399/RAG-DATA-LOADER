# Deployment

## Docker Compose (API + Qdrant + Redis)

```bash
docker compose up --build
```

This builds the multi-stage image and starts the API on port 8000 with a Qdrant
vector store and Redis. The API container runs as a non-root user with a `/health`
healthcheck.

## Configuration

All settings are read from `RAG_`-prefixed environment variables or a `.env` file.

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `200` | Chunking parameters |
| `RAG_CHUNK_STRATEGY` | `recursive` | `recursive` or `token` |
| `RAG_EMBEDDING_PROVIDER` / `RAG_EMBEDDING_MODEL` | `huggingface` / MiniLM | Embedding backend |
| `RAG_LLM_PROVIDER` / `RAG_LLM_MODEL` | `anthropic` / Claude | Chat backend |
| `RAG_VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| `RAG_QDRANT_URL` | `null` | Qdrant URL when using the Qdrant backend |
| `RAG_LOG_JSON` | `false` | Emit JSON logs in production |
| `RAG_LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |

## Provider extras

Install only the providers you need:

```bash
uv sync --extra huggingface   # local sentence-transformers embeddings
uv sync --extra openai        # OpenAI embeddings + chat
uv sync --extra anthropic     # Anthropic chat (Claude)
uv sync --extra eval          # Ragas evaluation
```

## Evaluation

```bash
uv run pytest tests/integration/test_evaluation.py
```

The offline keyword-recall gate runs in CI; full Ragas metrics
(faithfulness, answer relevancy, context precision/recall) run when an evaluator
LLM is configured.
