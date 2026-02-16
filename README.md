# ML Training Pipeline with YOLO Object Detection

A comprehensive machine learning training pipeline with YOLO (You Only Look Once) object detection capabilities and large-scale automated training infrastructure.

## Features

### YOLO Object Detection
- ✅ Support for YOLOv5 and YOLOv8
- ✅ Configurable model sizes (nano, small, medium, large, xlarge)
- ✅ Transfer learning with pretrained weights
- ✅ Multi-scale detection with NMS tuning
- ✅ Export to ONNX, TensorRT, CoreML for deployment

### Large-Scale Training
- ✅ Multi-GPU distributed training (DataParallel, DistributedDataParallel)
- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation for large batch sizes
- ✅ Memory-efficient streaming data loading
- ✅ Automatic checkpoint management and resume

### Data Handling
- ✅ Support for multiple annotation formats (YOLO, COCO, Pascal VOC)
- ✅ Automatic format conversion
- ✅ Dataset validation and splitting
- ✅ Data versioning with DVC

### Experiment Tracking
- ✅ Multiple backends (W&B, MLflow, TensorBoard, Custom)
- ✅ Automatic metric logging
- ✅ Model versioning and registry

### Automation
- ✅ GitHub Actions workflow for automated training
- ✅ Configurable training pipelines
- ✅ Automated evaluation and reporting

## Project Structure

```
model_train/
├── configs/                      # Configuration files
│   ├── yolo_config.yaml         # YOLO model configuration
│   ├── large_scale_training.yaml # Large-scale training settings
│   └── data_config.yaml         # Data configuration
├── src/                         # Source code
│   ├── data/
│   │   ├── adapters/            # Data format adapters
│   │   └── loaders/             # Data loaders
│   ├── models/
│   │   └── yolo_model.py        # YOLO model implementation
│   ├── training/
│   │   └── distributed_trainer.py # Distributed training
│   └── observability/
│       └── experiment_tracker.py  # Experiment tracking
├── scripts/                     # Utility scripts
│   ├── auto_train.py           # Automated training orchestrator
│   ├── prepare_data.py         # Data preparation
│   ├── train.py                # Training script
│   ├── predict.py              # Prediction script
│   ├── evaluate.py             # Evaluation script
│   ├── export.py               # Model export script
│   └── deploy.py               # Deployment script
├── data/                        # Data storage
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── test/                   # Test data
├── artifacts/                   # Training artifacts
│   ├── models/                 # Trained models
│   ├── checkpoints/            # Training checkpoints
│   ├── evaluation/             # Evaluation results
│   └── logs/                   # Training logs
├── .github/workflows/          # GitHub Actions
│   └── auto_train.yaml        # Automated training workflow
├── dvc.yaml                    # DVC pipeline
├── Makefile                    # Common commands
└── README.md                   # This file
```

## Installation

### Basic Installation

```bash
# Install basic dependencies
make install

# Install YOLO dependencies
make install-yolo
```

### GPU Installation

```bash
# Install with GPU support
make install-gpu
```

### Manual Installation

```bash
# Basic dependencies
pip install -r requirements.txt

# YOLO dependencies
pip install -r requirements-yolo.txt
pip install ultralytics
```

## Quick Start

### 1. Prepare Your Dataset

```bash
# Prepare dataset (converts to YOLO format if needed)
python scripts/prepare_data.py \
  --input data/raw/images/ \
  --annotations data/raw/annotations/ \
  --format coco \
  --output data/processed/ \
  --split

# Or use Make
make prepare-data
```

### 2. Validate Dataset

```bash
# Validate the prepared dataset
python scripts/validate_data.py --config configs/data_config.yaml

# Or use Make
make validate-data
```

### 3. Train YOLO Model

```bash
# Train YOLO model
python scripts/train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data data/processed/data.yaml

# Or use Make
make train-yolo
```

### 4. Run Predictions

```bash
# Run detection on images
python scripts/predict.py \
  --model artifacts/models/yolo/weights/best.pt \
  --source data/test/images/ \
  --conf 0.25

# Or use Make
make predict
```

### 5. Export for Deployment

```bash
# Export to ONNX format
python scripts/export.py \
  --model artifacts/models/yolo/weights/best.pt \
  --format onnx

# Or use Make
make export-yolo
```

## Advanced Usage

### Distributed Training

For multi-GPU training:

```bash
# Using torchrun
torchrun --nproc_per_node=auto scripts/auto_train.py \
  --model yolo \
  --config configs/large_scale_training.yaml \
  --data-config configs/data_config.yaml \
  --experiment-name multi-gpu-training

# Or use Make
make train-distributed
```

### Automated Training with Experiment Tracking

```bash
# With Weights & Biases
python scripts/auto_train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data-config configs/data_config.yaml \
  --experiment-name my-experiment \
  --backend wandb

# With MLflow
python scripts/auto_train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data-config configs/data_config.yaml \
  --experiment-name my-experiment \
  --backend mlflow
```

### Resume Training

```bash
# Resume from last checkpoint
python scripts/auto_train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data-config configs/data_config.yaml \
  --experiment-name my-experiment \
  --resume
```

## Configuration

### YOLO Configuration (`configs/yolo_config.yaml`)

```yaml
yolo:
  version: "v8"  # v5, v8
  model_size: "medium"  # nano, small, medium, large, xlarge
  pretrained: true
  
  training:
    img_size: 640
    batch_size: 16
    epochs: 300
    lr0: 0.01
    
  detection:
    conf_threshold: 0.25
    iou_threshold: 0.45
```

### Data Configuration (`configs/data_config.yaml`)

```yaml
data:
  root_dir: "data/"
  format: "yolo"
  
  preprocessing:
    resize: [640, 640]
    normalize: true
  
  split:
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
```

### Large-Scale Training (`configs/large_scale_training.yaml`)

```yaml
large_scale_training:
  hardware:
    gpus: "auto"
    precision: "mixed"
    
  memory:
    gradient_accumulation_steps: 4
    gradient_checkpointing: true
    
  data_pipeline:
    num_workers: 8
    streaming: true
```

## Data Format Support

### YOLO Format
```
images/
  image1.jpg
  image2.jpg
labels/
  image1.txt  # class x_center y_center width height (normalized)
  image2.txt
```

### COCO Format
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

### Pascal VOC Format
```
JPEGImages/
  image1.jpg
Annotations/
  image1.xml
```

## Model Export

Export trained models to various formats:

```bash
# Export to ONNX
python scripts/export.py --model best.pt --format onnx

# Export to multiple formats
python scripts/export.py --model best.pt --format onnx tensorrt coreml

# With optimization
python scripts/export.py --model best.pt --format onnx --optimize --half
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
  --model artifacts/models/yolo/weights/best.pt \
  --test-data data/processed/data.yaml

# Generate report
python scripts/generate_report.py
```

## DVC Pipeline

```bash
# Initialize DVC
make dvc-init

# Run pipeline
dvc repro

# Push data and models
make dvc-push
```

## GitHub Actions Workflow

The automated training workflow can be triggered:

1. **Scheduled**: Weekly on Sundays
2. **Manual**: Via workflow_dispatch

Configure secrets:
- `WANDB_API_KEY`: For Weights & Biases tracking (optional)

## Development

### Run Tests

```bash
make test
```

### Code Linting

```bash
make lint
```

### Clean Artifacts

```bash
make clean
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Enable `gradient_accumulation_steps`
- Use `gradient_checkpointing`

### Data Loading Slow
- Increase `num_workers`
- Enable `persistent_workers`
- Use `streaming` mode for large datasets

### Model Not Converging
- Check learning rate (`lr0`)
- Verify data augmentation settings
- Ensure proper data normalization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review configuration examples

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [DVC](https://dvc.org/)