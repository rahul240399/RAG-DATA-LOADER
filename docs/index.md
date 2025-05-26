# RAG Data Loader

A LangChain-based Retrieval-Augmented Generation pipeline: ingest PDFs, chunk and
embed them, store the vectors, then retrieve and generate grounded, cited answers.
Embedding, vector-store, and LLM providers are pluggable and selected entirely
through configuration.

## Highlights

- **Ingestion** — PyMuPDF loading and recursive/token-aware chunking as LCEL runnables
- **Pluggable providers** — HuggingFace / OpenAI / Voyage embeddings; Anthropic / OpenAI / Ollama LLMs
- **Vector stores** — Chroma for development, Qdrant for production, behind one protocol
- **Retrieval** — dense, cross-encoder reranking, and hybrid (dense + BM25) via reciprocal rank fusion
- **Serving** — FastAPI with ingest/query/chat endpoints and SSE token streaming
- **Operability** — structured logging, LangSmith tracing, Prometheus metrics, Ragas evaluation

## Quickstart

```bash
uv sync           # install dependencies
uv run pytest     # run the test suite
uv run rag-loader index document.pdf
uv run uvicorn rag_loader.api.app:app --reload
```
