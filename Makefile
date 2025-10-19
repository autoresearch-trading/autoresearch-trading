.PHONY: help install test lint clean docker-up docker-down

COMPOSE_FILE := deploy/docker-compose.yml

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	cd signal-engine && pip install -r requirements.txt

test: ## Run all tests
	python -m pytest tests/ -v
	cd signal-engine && python -m pytest tests/ -v

test-unit: ## Run unit tests only
	cd signal-engine && python -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	cd signal-engine && python -m pytest tests/integration/ -v

test-no-questdb: ## Run tests without QuestDB
	cd signal-engine && SKIP_QUESTDB_TESTS=true python -m pytest tests/unit/ -v

test-ci: ## Run tests suitable for CI
	python -m pytest tests/ -v
	cd signal-engine && python -m pytest tests/unit/ -v

lint: ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check .
	isort --check-only .
	mypy . --ignore-missing-imports

lint-fix: ## Fix linting issues
	black .
	isort .

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ coverage.xml

docker-up: ## Start services with Docker Compose
	docker compose -f $(COMPOSE_FILE) up -d questdb
	@echo "Waiting for QuestDB to be ready..."
	@bash -c 'for i in {1..30}; do if curl -fsS http://localhost:9000/ >/dev/null; then exit 0; fi; sleep 2; done; exit 1'
	@echo "QuestDB is ready!"

docker-down: ## Stop Docker services
	docker compose -f $(COMPOSE_FILE) down

docker-test: ## Run tests in Docker
	docker compose -f $(COMPOSE_FILE) up --build --abort-on-container-exit

signal-pipeline: ## Run signal pipeline with sample data
	cd signal-engine && python scripts/run_signal_pipeline.py --symbols BTC --date 2025-10-08 --dry-run --skip-regime

collect-data: ## Start data collection
	python scripts/collect_data.py live --symbols BTC,ETH --max-rps 2

dev-setup: install questdb-local ## Complete development setup
	@echo "Development environment ready!"
	@echo "QuestDB available at: http://localhost:9000"
	@echo "Run 'make collect-data' to start data collection"
	@echo "Run 'make signal-pipeline' to test signal processing"

questdb-local: ## Setup QuestDB locally (no Docker required)
	cd signal-engine && python scripts/setup_questdb_local.py

questdb-docker: ## Setup QuestDB with Docker
	docker compose -f $(COMPOSE_FILE) up -d questdb
	@echo "Waiting for QuestDB to be ready..."
	@bash -c 'for i in {1..30}; do if curl -fsS http://localhost:9000/ >/dev/null; then exit 0; fi; sleep 2; done; exit 1'
	@echo "QuestDB is ready!"
