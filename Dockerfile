# syntax=docker/dockerfile:1

# --- Builder: resolve and install dependencies with uv ---
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev --no-editable

# --- Runtime: copy the prebuilt virtualenv into a slim image ---
FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY eval ./eval
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    RAG_LOG_JSON=true
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "rag_loader.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
