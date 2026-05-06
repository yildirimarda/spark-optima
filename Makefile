# Spark Optima - Makefile for Common Development Tasks
# Usage: make <target>

.PHONY: help install install-dev test test-unit test-integration lint format type-check security docs clean build docker-build docker-run

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := .venv/bin/python
POETRY := poetry  # Installed via Homebrew, available in PATH
VENV_BIN := .venv/bin
PACKAGE_NAME := spark_optima
SRC_DIR := src/$(PACKAGE_NAME)
TEST_DIR := tests

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "Spark Optima - Available Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================
install: ## Install production dependencies
	$(POETRY) install --no-dev

install-dev: ## Install development dependencies
	$(POETRY) install --all-extras
	$(VENV_BIN)/pre-commit install

install-docs: ## Install documentation dependencies
	$(POETRY) install --only dev

# =============================================================================
# Testing
# =============================================================================
test: ## Run all tests
	$(PYTHON) -m pytest

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest -m unit -v

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest -m integration -v

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-coverage-xml: ## Run tests with XML coverage report
	$(PYTHON) -m pytest --cov=$(SRC_DIR) --cov-report=xml

test-fast: ## Run tests in parallel (faster)
	$(PYTHON) -m pytest -x -n auto

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run Ruff linter
	$(PYTHON) -m ruff check .

lint-fix: ## Run Ruff linter with auto-fix
	$(PYTHON) -m ruff check . --fix

format: ## Run Ruff formatter
	$(PYTHON) -m ruff format .

format-check: ## Check formatting without modifying files
	$(PYTHON) -m ruff format . --check

type-check: ## Run MyPy type checker
	$(PYTHON) -m mypy $(SRC_DIR)

security: ## Run security scans (Bandit, Safety)
	$(PYTHON) -m bandit -r $(SRC_DIR)
	$(PYTHON) -m safety check

check-all: lint format-check type-check security ## Run all checks

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit: ## Run pre-commit hooks on all files
	$(VENV_BIN)/pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	$(VENV_BIN)/pre-commit install

# =============================================================================
# Documentation
# =============================================================================
docs: ## Build documentation
	$(PYTHON) -m mkdocs build

docs-serve: ## Serve documentation locally
	$(PYTHON) -m mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	$(PYTHON) -m mkdocs gh-deploy

# =============================================================================
# Building & Packaging
# =============================================================================
build: ## Build package
	$(POETRY) build

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# =============================================================================
# Docker
# =============================================================================
docker-build: ## Build Docker image (production)
	docker build -f docker/Dockerfile --target production -t spark-optima:latest .

docker-build-dev: ## Build Docker image (development)
	docker build -f docker/Dockerfile --target development -t spark-optima:dev .

docker-run: ## Run Docker container
	docker run --rm -it spark-optima:latest

docker-compose-up: ## Start services with Docker Compose
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-down: ## Stop services with Docker Compose
	docker-compose -f docker/docker-compose.yml down

docker-compose-logs: ## View Docker Compose logs
	docker-compose -f docker/docker-compose.yml logs -f

# =============================================================================
# Development Server
# =============================================================================
run-api: ## Run FastAPI development server
	$(PYTHON) -m uvicorn spark_optima.api.main:app --reload --host 0.0.0.0 --port 8000

run-cli: ## Run CLI in interactive mode
	$(PYTHON) -m spark_optima.cli.main

# =============================================================================
# Release
# =============================================================================
version-patch: ## Bump patch version
	$(POETRY) version patch

version-minor: ## Bump minor version
	$(POETRY) version minor

version-major: ## Bump major version
	$(POETRY) version major

publish-test: ## Publish to TestPyPI
	$(POETRY) config repositories.testpypi https://test.pypi.org/legacy/
	$(POETRY) publish -r testpypi --build

publish: ## Publish to PyPI
	$(POETRY) publish --build

# =============================================================================
# Utilities
# =============================================================================
shell: ## Open Poetry shell
	$(POETRY) shell

update: ## Update dependencies
	$(POETRY) update

lock: ## Update poetry.lock
	$(POETRY) lock

requirements: ## Export requirements.txt
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes

requirements-dev: ## Export requirements-dev.txt
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes
