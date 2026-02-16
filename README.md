# Multi-Modal ML Pipeline

A comprehensive, production-ready machine learning pipeline that handles multi-modal data (CSV, images, and YOLO object detection) with full automation, observability, and deployment capabilities.

## Features

- **Multi-Modal Data Support**: CSV tabular data, images, and YOLO object detection
- **Flexible Model Architecture**: MLP for tabular data, CNN/ResNet for images, YOLOv5/v8 for object detection
- **Production-Ready**: FastAPI serving, Prometheus metrics, MLflow tracking
- **Scalable Training**: Distributed training support, mixed precision, gradient accumulation
- **Complete Pipeline**: Data ingestion → Validation → Preprocessing → Training → Evaluation → Deployment
- **CI/CD Integration**: Automated testing, training workflows, model versioning

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Harsh-6361/model_train.git
cd model_train

# Install dependencies
make install

# For development
make install-dev

# For YOLO support
make install-yolo
```

### Train a Model

```bash
# Prepare sample data
python scripts/prepare_data.py --type tabular

# Train tabular model
python scripts/train.py --model tabular

# Train vision model (requires image data)
python scripts/train.py --model vision

# Train YOLO model (requires YOLO dataset)
python scripts/train.py --model yolo
```

### Run Inference

```bash
# Start API server
python scripts/serve.py --model artifacts/models/best.pt --model-type tabular --port 8000

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

## Project Structure

See `docs/ARCHITECTURE.md` for detailed architecture documentation.

## Usage Examples

### Tabular Data Training

```python
from src.data.loaders.unified_loader import UnifiedLoader
from src.models.registry import ModelRegistry
from src.training.trainer import Trainer
from src.utils.config_loader import load_config

# Load configurations
data_config = load_config('configs/data_config.yaml')
model_config = load_config('configs/model_config.yaml')
training_config = load_config('configs/training_config.yaml')

# Get data loaders
loader = UnifiedLoader(data_config)
train_loader, val_loader, test_loader = loader.get_loaders('tabular')

# Create model
model = ModelRegistry.create('tabular', model_config, input_size=10, output_size=2)

# Train
trainer = Trainer(model, training_config)
trainer.fit(train_loader, val_loader)

# Evaluate
metrics = trainer.evaluate(test_loader)
print(metrics)
```

## API Documentation

Once the server is running, visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Tabular prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

# Image prediction
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@image.jpg"

# Model info
curl http://localhost:8000/model/info
```

## Development

```bash
# Install dev dependencies
make install-dev

# Format code
make format

# Run linters
make lint

# Run tests
make test

# Clean artifacts
make clean
```

## Docker Deployment

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker-compose
docker-compose up -d
```

## Configuration

All configuration is done via YAML files in `configs/`:

- **data_config.yaml**: Data paths, splits, preprocessing
- **model_config.yaml**: Model architectures and hyperparameters
- **training_config.yaml**: Training settings, optimizer, scheduler
- **deployment_config.yaml**: API server settings
- **yolo_config.yaml**: YOLO-specific settings

Environment variables can override config values:
```bash
export ML_TRAINING_EPOCHS=200
export ML_TRAINING_BATCH_SIZE=64
python scripts/train.py --model tabular
```

## CI/CD

The project includes GitHub Actions workflows:

- **ci.yaml**: Linting and testing on every push
- **train.yaml**: Manual training workflow
- **auto_train.yaml**: Scheduled automated training

## Documentation

- **SPEC.md**: Complete specification and requirements
- **ARCHITECTURE.md**: System architecture and design
- **API.md**: API reference documentation

## Monitoring

- **Prometheus**: Metrics collection at `/metrics`
- **MLflow**: Experiment tracking (if configured)
- **Structured Logging**: JSON logs in `artifacts/logs/`

## License

MIT License

## Support

For issues and questions:
- GitHub Issues: https://github.com/Harsh-6361/model_train/issues