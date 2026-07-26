.PHONY: help install lint type-check format test test-unit test-integration test-quality test-performance test-all test-cov run serve clean

help:
	@echo "ast-grep-mcp — Development commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies (uv sync)"
	@echo "  make clean          Remove __pycache__ and .pytest_cache"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run unit tests (default target)"
	@echo "  make test-unit      Run unit tests only (fast)"
	@echo "  make test-integration Run integration tests (slower)"
	@echo "  make test-quality   Run quality regression tests"
	@echo "  make test-performance Run performance benchmarks (slow)"
	@echo "  make test-all       Run all tests with verbose output"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run ruff linter"
	@echo "  make type-check     Run mypy type checker"
	@echo "  make format         Format code with ruff"
	@echo ""
	@echo "Running:"
	@echo "  make run            Run MCP server locally"
	@echo "  make serve          Run MCP server with Doppler secrets"

install:
	uv sync

lint:
	uv run ruff check .

type-check:
	uv run mypy src/

format:
	uv run ruff format .

test:
	uv run pytest tests/unit/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-quality:
	uv run pytest tests/quality/ -v

test-performance:
	uv run pytest tests/performance/ -v

test-all:
	uv run pytest -v

test-cov:
	uv run pytest --cov=src/ast_grep_mcp --cov-report=term-missing --cov-report=html

run:
	uv run main.py

serve:
	doppler run -- uv run main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ 2>/dev/null || true
