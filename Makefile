# Spark Optima - Makefile for Common Development Tasks
# Usage: make <target>

.PHONY: help install install-dev test test-unit test-integration lint format type-check security docs clean build docker-build docker-run

# Default target
.DEFAULT_GOAL := help

# Variables
UV := uv
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
	uv sync --no-dev

install-dev: ## Install development dependencies
	uv sync
	uv run pre-commit install

install-docs: ## Install documentation dependencies
	uv sync --only-group dev

# =============================================================================
# Testing
# =============================================================================
test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest -m unit -v

test-integration: ## Run integration tests only
	uv run pytest -m integration -v

test-coverage: ## Run tests with coverage report
	uv run pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-coverage-xml: ## Run tests with XML coverage report
	uv run pytest --cov=$(SRC_DIR) --cov-report=xml

test-fast: ## Run tests in parallel (faster)
	uv run pytest -x -n auto

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run Ruff linter
	uv run ruff check .

lint-fix: ## Run Ruff linter with auto-fix
	uv run ruff check . --fix

format: ## Run Ruff formatter
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format . --check

type-check: ## Run MyPy type checker
	uv run mypy $(SRC_DIR)

security: ## Run security scans (Bandit, Safety)
	uv run bandit -r $(SRC_DIR)
	uv run safety check

check-all: lint format-check type-check security ## Run all checks

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

# =============================================================================
# Documentation
# =============================================================================
docs: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	uv run mkdocs gh-deploy

# =============================================================================
# Building & Packaging
# =============================================================================
build: ## Build package
	uv build

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
	uv run uvicorn spark_optima.api.main:app --reload --host 0.0.0.0 --port 8000

run-cli: ## Run CLI in interactive mode
	uv run python -m spark_optima.cli.main

# =============================================================================
# Release
# =============================================================================
version-patch: ## Bump patch version (edit version in pyproject.toml manually)
	# UV does not have a built-in version bump command; edit [project] version in pyproject.toml manually.

version-minor: ## Bump minor version (edit version in pyproject.toml manually)
	# UV does not have a built-in version bump command; edit [project] version in pyproject.toml manually.

version-major: ## Bump major version (edit version in pyproject.toml manually)
	# UV does not have a built-in version bump command; edit [project] version in pyproject.toml manually.

publish-test: ## Publish to TestPyPI
	uv publish --index testpypi

publish: ## Publish to PyPI
	uv publish

# =============================================================================
# Utilities
# =============================================================================
shell: ## Open an interactive shell in the project environment
	uv run bash

update: ## Update dependencies
	uv lock --upgrade

lock: ## Update uv.lock
	uv lock

requirements: ## Export requirements.txt (no editable project entry)
	uv export --no-hashes --no-dev --no-emit-project -o requirements.txt

requirements-dev: ## Export requirements-dev.txt (no editable project entry)
	uv export --no-hashes --no-emit-project -o requirements-dev.txt
