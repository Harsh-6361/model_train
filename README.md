# Multi-Modal ML Training System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An industrial-ready, data-driven ML product that handles multi-modal data (images and CSV) with full pipeline automation, observability, and local deployment.

## Features

- 🔄 **Multi-Modal Support**: Train models on both tabular (CSV) and image data
- 🎯 **Unified Interface**: Single training pipeline for all model types
- ⚙️ **Configuration-Driven**: Fully configurable via YAML files
- 📊 **Built-in Observability**: Structured logging, metrics tracking, health checks
- 🚀 **FastAPI Serving**: Production-ready REST API for inference
- 🧪 **Comprehensive Testing**: Unit and integration tests with pytest
- 🐳 **Docker Support**: Containerized deployment
- 📝 **Full Documentation**: Detailed specs, architecture, and API docs

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Harsh-6361/model_train.git
cd model_train

# Install dependencies
make install

# Setup sample data
python scripts/setup_data.py
```

### Training a Model

#### Tabular Model (Iris Dataset)

```bash
# Train tabular model
python scripts/train.py --model-type tabular

# Or using make
make train-tabular
```

#### Vision Model (Synthetic Images)

```bash
# Train vision model
python scripts/train.py --model-type vision

# Or using make
make train-vision
```

### Serving the Model

```bash
# Start API server
python scripts/serve.py --model-type tabular

# Or using make
make serve
```

The API will be available at `http://localhost:8000`

### Making Predictions

#### Via API

```bash
# Tabular prediction
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'

# Image prediction
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@image.jpg"
```

#### Via Script

```bash
# Predict on CSV data
python scripts/predict.py \
  --model-path artifacts/models/best_model.pth \
  --input data/sample/iris.csv \
  --model-type tabular

# Predict on image
python scripts/predict.py \
  --model-path artifacts/models/best_model.pth \
  --input image.jpg \
  --model-type vision
```

## Project Structure

```
model_train/
├── configs/                    # Configuration files
│   ├── data_config.yaml       # Data paths and preprocessing
│   ├── model_config.yaml      # Model architecture
│   ├── training_config.yaml   # Training parameters
│   └── deployment_config.yaml # Serving configuration
├── src/
│   ├── data/                  # Data handling
│   │   ├── adapters/          # CSV and image adapters
│   │   ├── validators/        # Schema validation
│   │   ├── preprocessors/     # Data preprocessing
│   │   └── loaders/           # PyTorch data loaders
│   ├── models/                # Model implementations
│   │   ├── base_model.py      # Abstract base class
│   │   ├── tabular_model.py   # MLP classifier
│   │   ├── vision_model.py    # ResNet/EfficientNet
│   │   └── registry.py        # Model versioning
│   ├── training/              # Training components
│   │   ├── trainer.py         # Unified trainer
│   │   ├── callbacks.py       # Training callbacks
│   │   └── metrics.py         # Evaluation metrics
│   ├── inference/             # Inference components
│   │   ├── predictor.py       # Prediction engine
│   │   └── serving.py         # FastAPI server
│   ├── observability/         # Monitoring
│   │   ├── logger.py          # Structured logging
│   │   ├── metrics_collector.py
│   │   └── health_check.py
│   └── utils/                 # Utilities
├── scripts/                   # Entry point scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── predict.py            # Prediction script
│   ├── serve.py              # API serving script
│   └── setup_data.py         # Setup sample data
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── data/                      # Data directory
│   ├── sample/               # Sample datasets
│   └── raw/                  # Raw data
├── artifacts/                 # Training artifacts
│   ├── models/               # Saved models
│   ├── metrics/              # Training metrics
│   └── logs/                 # Log files
├── docs/                      # Documentation
│   ├── SPEC.md               # Product specification
│   ├── ARCHITECTURE.md       # Architecture docs
│   └── API.md                # API reference
├── .github/workflows/         # CI/CD workflows
├── Dockerfile                 # Docker configuration
├── Makefile                   # Common commands
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Configuration

### Data Configuration (`configs/data_config.yaml`)

```yaml
data:
  tabular:
    path: "data/sample/iris.csv"
    target_column: "species"
    train_split: 0.7
    validation_split: 0.15
    test_split: 0.15
    preprocessing:
      normalize: true
      handle_missing: "mean"
      categorical_encoding: "onehot"
  
  image:
    path: "data/sample/images/"
    image_size: [224, 224]
    augmentation:
      horizontal_flip: true
      rotation_range: 15
      normalize: true
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  type: "tabular"  # or "vision"
  name: "mlp_classifier"
  architecture:
    hidden_layers: [128, 64, 32]
    dropout: 0.3
    activation: "relu"
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  loss: "cross_entropy"
  early_stopping:
    enabled: true
    patience: 10
  checkpoint:
    enabled: true
    save_best: true
```

## Usage Examples

### Custom Training

```python
from src.data.adapters.csv_adapter import CSVAdapter
from src.models.tabular_model import TabularModel
from src.training.trainer import Trainer

# Load data
adapter = CSVAdapter("data.csv", target_column="label")
adapter.load()
splits = adapter.split_data()

# Preprocess
from src.data.preprocessors.tabular_preprocessor import TabularPreprocessor
preprocessor = TabularPreprocessor()
X_train = preprocessor.fit_transform(splits['train'][0])

# Create model
model = TabularModel.create(
    architecture='mlp',
    input_dim=X_train.shape[1],
    output_dim=3
)

# Train
trainer = Trainer(model, train_loader, val_loader)
trainer.train(epochs=50)
```

### Custom Prediction

```python
from src.inference.predictor import Predictor
from src.models.tabular_model import MLPClassifier

# Load model
model = MLPClassifier(input_dim=4, output_dim=3)
predictor = Predictor.from_checkpoint("model.pth", model)

# Predict
predictions = predictor.predict_tabular([[5.1, 3.5, 1.4, 0.2]])
```

## Available Commands

```bash
make install          # Install dependencies
make lint             # Run linters
make format           # Format code
make test             # Run tests
make train            # Train with default config
make train-tabular    # Train tabular model
make train-vision     # Train vision model
make evaluate         # Evaluate model
make serve            # Start API server
make docker-build     # Build Docker image
make docker-run       # Run Docker container
make clean            # Clean generated files
```

## Docker Deployment

```bash
# Build image
docker build -t model-train:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  model-train:latest
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_tabular_model.py

# Run integration tests only
pytest tests/integration/
```

## API Documentation

Interactive API documentation is available when the server is running:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

See [docs/API.md](docs/API.md) for detailed API documentation.

## Architecture

The system follows a layered architecture:

1. **Data Layer**: Adapters, validators, preprocessors, loaders
2. **Model Layer**: Base models, implementations, registry
3. **Training Layer**: Trainer, callbacks, metrics
4. **Inference Layer**: Predictor, API serving
5. **Observability**: Logging, metrics, health checks

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Style

We use:
- **black** for code formatting
- **flake8** for linting
- **isort** for import sorting
- **mypy** for type checking

```bash
# Format code
make format

# Check style
make lint
```

### Adding New Models

1. Create new model class in `src/models/`
2. Inherit from `BaseModel`
3. Implement `forward()` and `get_config()`
4. Add factory method
5. Update configurations

### Adding New Data Sources

1. Create new adapter in `src/data/adapters/`
2. Implement `load()` and `split_data()`
3. Update data configuration
4. Add validation schema

## Troubleshooting

### Common Issues

**Issue**: Import errors
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size in training config
# training_config.yaml
training:
  batch_size: 16  # Reduce from 32
```

**Issue**: Model not found
```bash
# Solution: Train a model first
python scripts/train.py --model-type tabular
```

## Performance

### Benchmarks (CPU)

| Task | Dataset | Time | Throughput |
|------|---------|------|------------|
| Training (Tabular) | Iris (150) | ~30s | 250 samples/s |
| Training (Vision) | Synthetic (30) | ~2m | 10 samples/s |
| Inference (Tabular) | Single | ~20ms | 50 req/s |
| Inference (Vision) | Single | ~100ms | 10 req/s |

*Note: GPU training is 5-10x faster*

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{model_train,
  title = {Multi-Modal ML Training System},
  author = {ML Team},
  year = {2024},
  url = {https://github.com/Harsh-6361/model_train}
}
```

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/Harsh-6361/model_train/issues)
- Discussions: [GitHub Discussions](https://github.com/Harsh-6361/model_train/discussions)

## Roadmap

- [x] MVP with tabular and vision models
- [x] FastAPI serving
- [x] CI/CD workflows
- [ ] Experiment tracking (MLflow)
- [ ] Distributed training support
- [ ] Model explainability (SHAP)
- [ ] Hyperparameter optimization
- [ ] Model compression
- [ ] Streaming inference
- [ ] Multi-modal fusion models

## Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI team for the web framework
- scikit-learn team for preprocessing and metrics
- Open source community

---

**Built with ❤️ by the ML Team**