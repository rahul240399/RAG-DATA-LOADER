# Architecture

```
PDF → Load → Split → Embed → Vector Store      (indexing)
                                  |
Question → Retrieve → Rerank → LLM → Answer    (query)
```

Every stage is a LangChain component chosen from `Settings`, so the same code runs
locally (HuggingFace + Chroma) or in production (OpenAI/Voyage + Qdrant) unchanged.

## Components

| Layer | Module | Notes |
| --- | --- | --- |
| Configuration | `settings.py` | `pydantic-settings`, `RAG_`-prefixed env / `.env` |
| Provider factories | `factories.py` | `init_chat_model` / `init_embeddings` |
| Ingestion | `ingest/` | PyMuPDF loader, recursive/token splitter, LCEL runnable |
| Embeddings | `embeddings.py` | batch embed + content-addressed disk cache |
| Vector stores | `store/` | `VectorBackend` protocol, Chroma + Qdrant |
| Indexing | `indexing.py` | idempotent upserts, bounded-async batches |
| Retrieval | `retrieval.py` | reranking + hybrid (RRF) retrievers |
| Generation | `generation.py` | LCEL RAG chain with inline citations |
| API | `api/` | FastAPI endpoints + SSE streaming |
| Observability | `observability/` | structlog, LangSmith tracing |

## Key design decisions

- **Deterministic chunk IDs** make re-indexing idempotent: the same content upserts
  in place instead of duplicating.
- **Protocol-based stores** keep the backend swappable; both satisfy a shared
  contract test that runs fully in-memory.
- **Custom RRF hybrid and reranking retrievers** are version-robust and avoid
  coupling to fast-moving framework internals.
- **Dependency injection** in the API and indexer lets the whole stack be tested
  with fakes — no model downloads or servers required.
