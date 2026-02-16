.PHONY: install lint test train serve clean docker-build help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

lint:  ## Run linters
	black --check src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/
	mypy src/

format:  ## Format code
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

train:  ## Run training with default config
	python scripts/train.py --config configs/training_config.yaml

train-tabular:  ## Train tabular model
	python scripts/train.py --config configs/training_config.yaml --model-type tabular

train-vision:  ## Train vision model
	python scripts/train.py --config configs/training_config.yaml --model-type vision

evaluate:  ## Evaluate trained model
	python scripts/evaluate.py --model-path artifacts/models/best_model.pth

predict:  ## Run prediction
	python scripts/predict.py --model-path artifacts/models/best_model.pth --input data/sample/

serve:  ## Start API server
	python scripts/serve.py --port 8000

docker-build:  ## Build Docker image
	docker build -t model-train:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 model-train:latest

clean:  ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/

setup:  ## Set up project structure and sample data
	python -c "from scripts.setup_data import setup_sample_data; setup_sample_data()"
