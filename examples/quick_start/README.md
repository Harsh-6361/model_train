# Quick Start Example

This example demonstrates the basic workflow for training a YOLO model.

## Prerequisites

```bash
# Install dependencies
make install-yolo
```

## Step-by-Step Guide

### 1. Prepare Sample Data

First, create a sample dataset structure:

```bash
# Create sample data structure
mkdir -p data/raw/images data/raw/labels

# Add your images to data/raw/images/
# Add corresponding label files to data/raw/labels/
```

For YOLO format, each label file should have the format:
```
class_id x_center y_center width height
```

All values should be normalized (0-1).

### 2. Prepare Data for Training

```bash
# Convert and split dataset
python scripts/prepare_data.py \
  --input data/raw/ \
  --output data/processed/ \
  --format yolo \
  --split \
  --config configs/data_config.yaml
```

This will:
- Validate the dataset
- Split it into train/val/test sets
- Create a `data.yaml` configuration file

### 3. Validate Dataset

```bash
# Verify the prepared dataset
python scripts/validate_data.py --config configs/data_config.yaml
```

### 4. Train Model

```bash
# Train YOLO model
python scripts/train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data data/processed/data.yaml \
  --epochs 10 \
  --batch-size 8
```

For a quick test, use fewer epochs and smaller batch size.

### 5. Run Predictions

```bash
# Run detection on test images
python scripts/predict.py \
  --model artifacts/models/exp/weights/best.pt \
  --source data/test/images/ \
  --conf 0.25
```

### 6. Evaluate Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
  --model artifacts/models/exp/weights/best.pt \
  --test-data data/processed/data.yaml
```

### 7. Export Model

```bash
# Export to ONNX for deployment
python scripts/export.py \
  --model artifacts/models/exp/weights/best.pt \
  --format onnx
```

## Using Make Commands

All steps can also be executed using the Makefile:

```bash
# Prepare data
make prepare-data

# Validate data
make validate-data

# Train model
make train-yolo

# Run predictions
make predict

# Evaluate model
make evaluate

# Export model
make export-yolo
```

## Configuration Customization

### Adjust Model Size

Edit `configs/yolo_config.yaml`:

```yaml
yolo:
  model_size: "small"  # nano, small, medium, large, xlarge
```

### Adjust Training Parameters

```yaml
yolo:
  training:
    epochs: 100        # Number of training epochs
    batch_size: 16     # Batch size per GPU
    lr0: 0.01          # Initial learning rate
    img_size: 640      # Input image size
```

### Adjust Data Augmentation

```yaml
yolo:
  training:
    augmentation:
      fliplr: 0.5      # Horizontal flip probability
      mosaic: 1.0      # Mosaic augmentation
      scale: 0.5       # Scale augmentation
```

## Expected Results

After training for 100 epochs on a typical object detection dataset:

- Training time: 1-4 hours (depending on GPU)
- mAP@0.5: 0.40-0.60 (varies by dataset)
- Model size: 25-90 MB (depending on model size)

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or image size

```bash
python scripts/train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data data/processed/data.yaml \
  --batch-size 4 \
  --img-size 416
```

### Issue: Training Too Slow

**Solution**: Reduce number of workers or enable mixed precision

Edit `configs/large_scale_training.yaml`:

```yaml
large_scale_training:
  hardware:
    precision: "mixed"  # Enable mixed precision
  data_pipeline:
    num_workers: 4      # Reduce workers
```

### Issue: Model Not Converging

**Solution**: Adjust learning rate

```bash
python scripts/train.py \
  --model yolo \
  --config configs/yolo_config.yaml \
  --data data/processed/data.yaml
  # Edit config to set lr0: 0.001 for smaller learning rate
```

## Next Steps

1. **Experiment Tracking**: Enable W&B or MLflow for tracking experiments
2. **Distributed Training**: Train on multiple GPUs
3. **Hyperparameter Tuning**: Use the auto-tune feature
4. **Production Deployment**: Export and deploy your model

See the main [README.md](../../README.md) for more advanced usage.
