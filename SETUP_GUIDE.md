# Complete Setup Guide for Model Training System

This guide provides comprehensive instructions for setting up, running, and using the multi-modal ML training system locally.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.8 minimum)
- **RAM**: 4GB minimum (8GB+ recommended for vision models)
- **Disk Space**: 2GB minimum
- **OS**: Linux, macOS, or Windows

### Optional (for GPU acceleration)

- NVIDIA GPU with CUDA 11.0+
- cuDNN library

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Harsh-6361/model_train.git
cd model_train
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Dependencies installed:**
- PyTorch 2.0+
- torchvision
- FastAPI (for serving)
- pandas, numpy, scikit-learn
- Pillow (image processing)
- PyYAML (configuration)
- pytest (testing)
- structlog (logging)

---

## Data Setup

### Option 1: Use Sample Data (Quick Start)

The repository includes a script to generate sample datasets:

```bash
python scripts/setup_data.py
```

This creates:
- **Iris dataset** (150 samples, 3 classes) at `data/sample/iris.csv`
- **Synthetic images** (30 images, 3 classes) at `data/sample/images/`

### Option 2: Use Your Own Data

#### For Tabular Data (CSV)

1. Place your CSV file in `data/raw/` or `data/sample/`
2. Ensure your CSV has:
   - Feature columns (numeric or categorical)
   - A target column with class labels
3. Update `configs/data_config.yaml`:

```yaml
data:
  tabular:
    path: "data/raw/your_data.csv"
    target_column: "your_target_column"
    train_split: 0.7
    validation_split: 0.15
    test_split: 0.15
```

#### For Image Data

1. Organize images in class folders:
   ```
   data/sample/images/
   ├── class_0/
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── class_1/
   │   └── ...
   └── class_2/
       └── ...
   ```

2. Update `configs/data_config.yaml`:

```yaml
data:
  image:
    path: "data/raw/images/"
    image_size: [224, 224]
    augmentation:
      horizontal_flip: true
      rotation_range: 15
```

---

## Configuration

The system is fully configuration-driven. Key config files:

### 1. Data Configuration (`configs/data_config.yaml`)

Controls data loading and preprocessing:

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
      handle_missing: "mean"  # or "median", "drop"
      categorical_encoding: "onehot"  # or "label"
```

### 2. Model Configuration (`configs/model_config.yaml`)

Defines model architecture:

```yaml
model:
  type: "tabular"  # or "vision"
  name: "mlp_classifier"
  architecture:
    hidden_layers: [128, 64, 32]
    dropout: 0.3
    activation: "relu"
```

### 3. Training Configuration (`configs/training_config.yaml`)

Training hyperparameters:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"  # or "sgd", "adamw"
  loss: "cross_entropy"
  early_stopping:
    enabled: true
    patience: 10
  checkpoint:
    enabled: true
    save_best: true
```

---

## Running the System

### Training Models

#### Quick Start (Using Makefile)

```bash
# Train tabular model
make train-tabular

# Train vision model
make train-vision
```

#### Using Python Scripts

**Tabular Model:**

```bash
python scripts/train.py \
  --model-type tabular \
  --config configs/training_config.yaml
```

**Vision Model:**

```bash
python scripts/train.py \
  --model-type vision \
  --config configs/training_config.yaml
```

**Output:**
- Trained models saved to `artifacts/models/`
- Training metrics in `artifacts/metrics/`
- Logs in `artifacts/logs/`

### Making Predictions

#### Predict on CSV Data

```bash
python scripts/predict.py \
  --model-path artifacts/models/best_model.pth \
  --input data/sample/iris.csv \
  --model-type tabular
```

#### Predict on Images

```bash
python scripts/predict.py \
  --model-path artifacts/models/best_model.pth \
  --input path/to/image.jpg \
  --model-type vision
```

### Serving via API

Start the FastAPI server:

```bash
# Using Python
python scripts/serve.py --port 8000

# Or using Makefile
make serve
```

**API Endpoints:**

Access interactive docs at `http://localhost:8000/docs`

**Tabular Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/tabular" \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

**Image Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@image.jpg"
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

---

## Testing

### Run All Tests

```bash
# Using pytest
pytest tests/ -v

# Using make
make test
```

### Run Specific Test Types

```bash
# Unit tests only
pytest tests/unit/ -v
make test-unit

# Integration tests
pytest tests/integration/ -v
make test-integration

# With coverage report
pytest --cov=src --cov-report=html
```

### Manual Verification

```bash
# 1. Setup data
python scripts/setup_data.py

# 2. Train a model
python scripts/train.py --model-type tabular

# 3. Verify model saved
ls -la artifacts/models/

# 4. Test prediction
python scripts/predict.py \
  --model-path artifacts/models/best_model.pth \
  --input data/sample/iris.csv \
  --model-type tabular
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
pip install -e .
```

#### 2. CUDA Out of Memory

**Problem:** GPU memory errors during training

**Solution:** Reduce batch size in `configs/training_config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

#### 3. Data Loading Errors

**Problem:** CSV file not found or columns missing

**Solution:**
- Verify file path in config
- Check CSV has correct columns
- Run data setup: `python scripts/setup_data.py`

#### 4. Model Not Found During Prediction

**Problem:** `FileNotFoundError` when loading model

**Solution:** Train a model first:
```bash
python scripts/train.py --model-type tabular
```

#### 5. Port Already in Use (API Serving)

**Problem:** `Address already in use` error

**Solution:** Use a different port:
```bash
python scripts/serve.py --port 8001
```

### Performance Issues

If training is slow:

1. **Use GPU if available:**
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`
   - Install GPU version of PyTorch

2. **Reduce model complexity:**
   - Decrease hidden layer sizes
   - Reduce number of epochs

3. **Optimize data loading:**
   - Increase `num_workers` in data loaders
   - Use smaller batch sizes

### Getting Help

1. Check the [README.md](README.md) for quick reference
2. Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
3. See [docs/API.md](docs/API.md) for API details
4. Check test files in `tests/` for usage examples
5. Open an issue on GitHub

---

## Quick Command Reference

```bash
# Installation
pip install -r requirements.txt
pip install -e .

# Setup
python scripts/setup_data.py

# Training
make train-tabular        # Train tabular model
make train-vision         # Train vision model

# Evaluation
python scripts/evaluate.py --model-path artifacts/models/best_model.pth

# Serving
make serve                # Start API server

# Testing
make test                 # Run all tests
make lint                 # Check code style
make format               # Format code

# Cleanup
make clean                # Remove generated files
```

---

## Next Steps

After setup:

1. ✅ Train your first model: `make train-tabular`
2. ✅ Test predictions: Try the predict script
3. ✅ Start the API: `make serve` and visit `http://localhost:8000/docs`
4. ✅ Explore configurations: Customize `configs/` files
5. ✅ Review optimizations: See [PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md)

---

**Need help?** Open an issue or check the documentation in `docs/`
