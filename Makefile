.PHONY: help install install-yolo install-gpu clean lint test train-yolo train-distributed validate-data export-yolo

help:
	@echo "Available targets:"
	@echo "  install              - Install basic dependencies"
	@echo "  install-yolo         - Install YOLO dependencies"
	@echo "  install-gpu          - Install GPU-enabled dependencies"
	@echo "  clean                - Clean artifacts and cache"
	@echo "  lint                 - Run code linting"
	@echo "  test                 - Run tests"
	@echo "  prepare-data         - Prepare dataset for training"
	@echo "  validate-data        - Validate dataset"
	@echo "  train-yolo           - Train YOLO model"
	@echo "  train-distributed    - Train with distributed setup"
	@echo "  evaluate             - Evaluate trained model"
	@echo "  export-yolo          - Export YOLO model to ONNX"

install:
	pip install -r requirements.txt

install-yolo:
	pip install -r requirements-yolo.txt
	pip install ultralytics

install-gpu:
	pip install -r requirements.txt
	pip install -r requirements-yolo.txt
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

clean:
	rm -rf artifacts/models/*
	rm -rf artifacts/checkpoints/*
	rm -rf artifacts/predictions/*
	rm -rf artifacts/evaluation/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@echo "Running code linting..."
	@which pylint > /dev/null && pylint src/ || echo "pylint not installed"
	@which flake8 > /dev/null && flake8 src/ || echo "flake8 not installed"

test:
	@echo "Running tests..."
	@which pytest > /dev/null && pytest tests/ || echo "pytest not installed"

prepare-data:
	python scripts/prepare_data.py \
		--input data/raw/ \
		--output data/processed/ \
		--config configs/data_config.yaml \
		--split

validate-data:
	python scripts/validate_data.py --config configs/data_config.yaml

train-yolo:
	python scripts/train.py \
		--model yolo \
		--config configs/yolo_config.yaml \
		--data data/processed/data.yaml

train-distributed:
	torchrun --nproc_per_node=auto scripts/auto_train.py \
		--model yolo \
		--config configs/large_scale_training.yaml \
		--data-config configs/data_config.yaml \
		--experiment-name distributed-training

evaluate:
	python scripts/evaluate.py \
		--model artifacts/models/yolo/weights/best.pt \
		--test-data data/processed/data.yaml

export-yolo:
	python scripts/export.py \
		--model artifacts/models/yolo/weights/best.pt \
		--format onnx

predict:
	python scripts/predict.py \
		--model artifacts/models/yolo/weights/best.pt \
		--source data/test/images/

dvc-init:
	dvc init
	dvc remote add -d storage artifacts/dvc-storage

dvc-push:
	dvc push

dvc-pull:
	dvc pull
