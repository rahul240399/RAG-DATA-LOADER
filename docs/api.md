# API Reference

Run the service:

```bash
uv run uvicorn rag_loader.api.app:app --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks the vector store) |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/ingest` | Upload and index one or more PDFs |
| `POST` | `/query` | Similarity search, returns chunks |
| `POST` | `/chat` | RAG answer with citations |
| `POST` | `/chat/stream` | RAG answer streamed token-by-token over SSE |

## Examples

```bash
# Ingest
curl -F "files=@document.pdf" http://localhost:8000/ingest

# Query (retrieval only)
curl -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "what is RAG?", "k": 4}'

# Chat (retrieval + generation)
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "what is RAG?"}'

# Streaming chat (server-sent events)
curl -N -X POST http://localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"question": "what is RAG?"}'
```

A `/chat` response includes the answer and deduplicated `[source:page]` citations:

```json
{ "answer": "RAG ... [doc.pdf:1].", "sources": [{ "source": "doc.pdf", "page": 1 }] }
```
