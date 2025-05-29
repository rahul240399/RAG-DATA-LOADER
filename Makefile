.PHONY: install lint format typecheck test check run docker docs

install:
	uv sync --dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck test

run:
	uv run uvicorn rag_loader.api.app:app --reload

docker:
	docker compose up --build

docs:
	uv run mkdocs serve
