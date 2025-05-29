# Contributing

Thanks for your interest in improving RAG Data Loader.

## Setup

```bash
uv sync --dev
uv run pre-commit install
cp .env.example .env   # optional: customize configuration
```

## Workflow

- `make check` runs lint, type-check, and tests — keep it green before pushing.
- Code is formatted and linted with **ruff**; types are checked with **mypy --strict** (on `src`).
- Add tests for new behavior. Integration tests run fully in-memory (Chroma embedded,
  Qdrant `:memory:`) with deterministic fake embeddings — no API keys or servers needed.
- Commits follow **Conventional Commits**: `feat:`, `fix:`, `refactor:`, `test:`,
  `docs:`, `chore:`, `ci:`, `perf:`.

## Project layout

See [docs/architecture.md](docs/architecture.md) for the module map and design decisions.
