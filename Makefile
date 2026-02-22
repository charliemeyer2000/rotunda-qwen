.PHONY: lint format typecheck test test-all

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/unit/

test-all:
	uv run pytest
