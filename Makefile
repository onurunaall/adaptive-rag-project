.PHONY: help install install-dev install-mcp test clean run format lint check-env setup

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "InsightEngine - Makefile Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with core dependencies
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"

install-mcp: ## Install with MCP support (Python 3.10+ required)
	pip install -e ".[mcp]"

install-all: ## Install everything (dev + mcp)
	pip install -e ".[dev,mcp]"

test: ## Run tests
	pytest -v

test-coverage: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term

run: check-env ## Run the Streamlit application
	streamlit run src/main_app.py

format: ## Format code with black
	black src/ tests/

lint: ## Run linting checks
	flake8 src/ tests/ --max-line-length=127 --exclude=__pycache__,*.pyc

type-check: ## Run type checking with mypy
	mypy src/ --ignore-missing-imports

check-env: ## Check if .env file exists
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found!"; \
		echo "📝 Creating from template..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env and add your API keys!"; \
		exit 1; \
	fi
	@echo "✅ .env file found"

setup: ## Initial setup for the project
	@echo "🚀 Setting up InsightEngine..."
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"
	mkdir -p chroma_storage mcp_data/memory_storage logs
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "✨ Setup complete!"

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete
	rm -rf build/ dist/
	@echo "✨ Cleanup complete!"

clean-db: ## Clean up test databases
	rm -rf test_chroma_db agent_test_chroma_db_e2e
	rm -rf chroma_storage/test_*
	@echo "🗑️  Test databases cleaned"

reset: clean clean-db ## Full reset (clean + clean-db)
	@echo "🔄 Full reset complete!"

dev: install-dev check-env ## Setup development environment and check configuration
	@echo "✅ Development environment ready!"

quick-test: ## Run quick smoke tests
	pytest tests/test_config.py -v

docker-build: ## Build Docker image
	docker build -t insight-engine:latest .

docker-run: ## Run Docker container
	docker run -p 8501:8501 --env-file .env insight-engine:latest

info: ## Show project information
	@echo "Project: InsightEngine"
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Streamlit: $$(streamlit --version 2>/dev/null || echo 'Not installed')"
	@echo "Pytest: $$(pytest --version 2>/dev/null || echo 'Not installed')"