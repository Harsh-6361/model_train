.PHONY: install install-dev install-yolo lint format test train train-yolo serve docker-build docker-run clean help

help:
	@echo "Available commands:"
	@echo "  make install         - Install core dependencies"
	@echo "  make install-dev     - Install core + dev dependencies"
	@echo "  make install-yolo    - Install YOLO dependencies"
	@echo "  make lint            - Run code linters (black, isort, flake8)"
	@echo "  make format          - Format code with black and isort"
	@echo "  make test            - Run tests with coverage"
	@echo "  make train           - Train tabular model"
	@echo "  make train-yolo      - Train YOLO model"
	@echo "  make serve           - Start API server"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make clean           - Clean artifacts and cache"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

install-yolo:
	pip install -r requirements-yolo.txt

lint:
	black --check src/ tests/ scripts/ || true
	isort --check-only src/ tests/ scripts/ || true
	flake8 src/ tests/ scripts/ --max-line-length=100 --ignore=E203,W503 || true

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

test:
	pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing

train:
	python scripts/train.py --model tabular --config configs/training_config.yaml

train-yolo:
	python scripts/train.py --model yolo --config configs/yolo_config.yaml

serve:
	python scripts/serve.py --port 8000

docker-build:
	docker build -t ml-pipeline:latest .

docker-run:
	docker run -p 8000:8000 ml-pipeline:latest

clean:
	rm -rf artifacts/models/*.pt artifacts/models/*.pth artifacts/models/*.onnx
	rm -rf artifacts/logs/*.log
	rm -rf artifacts/metrics/*.json
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
