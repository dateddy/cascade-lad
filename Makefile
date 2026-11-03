.PHONY: data test lint format

data:
	@echo "Stage 1: scripts/download_data.py not written yet"

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .