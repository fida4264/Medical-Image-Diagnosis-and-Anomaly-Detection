# ==============================================================================
# Medical Image Diagnosis & Anomaly Detection Platform
# Master Developer Makefile
# Module 22 — Docker, Deployment & Production Packaging
# ==============================================================================

.PHONY: help install run api ui docker-build docker-run docker-stop docker-dev benchmark test lint clean

PYTHON ?= python
DOCKER_COMPOSE ?= docker compose

help:
	@echo "======================================================================="
	@echo " Medical AI Platform — Developer Commands"
	@echo "======================================================================="
	@echo " make install      - Install all Python dependencies"
	@echo " make run          - Launch both FastAPI API and Gradio UI locally"
	@echo " make api          - Launch FastAPI REST API server only"
	@echo " make ui           - Launch Gradio Web UI dashboard only"
	@echo " make docker-build - Build production Docker image"
	@echo " make docker-run   - Run production stack using docker-compose"
	@echo " make docker-stop  - Stop running docker-compose container stack"
	@echo " make docker-dev   - Run development stack with live volume reload"
	@echo " make benchmark    - Run platform-wide AI model benchmarking"
	@echo " make test         - Execute unit and security test suites"
	@echo " make clean        - Remove temporary files, caches, and build artifacts"
	@echo "======================================================================="

install:
	@echo "[Install] Upgrading pip and installing requirements..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	@echo "[Run] Launching local platform services..."
	@bash deployment/run.sh || deployment\run.bat

api:
	@echo "[API] Launching FastAPI server on http://127.0.0.1:8000..."
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

ui:
	@echo "[UI] Launching Gradio Web UI on http://127.0.0.1:7860..."
	$(PYTHON) app.py

docker-build:
	@echo "[Docker] Building production container image..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml build

docker-run:
	@echo "[Docker] Starting production stack via docker-compose..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d

docker-stop:
	@echo "[Docker] Stopping docker-compose stack..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down

docker-dev:
	@echo "[Docker] Starting development stack with live reloading..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.dev.yml up --build

benchmark:
	@echo "[Benchmark] Executing benchmark suite..."
	$(PYTHON) -c "from medical_ai_engine.benchmark import BenchmarkEngine; print(BenchmarkEngine().run_full_benchmark())"

test:
	@echo "[Test] Executing complete platform test suite..."
	$(PYTHON) -m unittest discover -s tests

clean:
	@echo "[Clean] Cleaning cache, build artifacts, and logs..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info 2>/dev/null || true
	@echo "[Clean] Done."
