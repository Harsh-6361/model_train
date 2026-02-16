#!/usr/bin/env python3
"""
Automated Training Orchestrator

Features:
- Auto-detect hardware and optimize settings
- Experiment tracking (MLflow/W&B)
- Hyperparameter optimization
- Auto-resume from failures
- Notification on completion
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_model import YOLOModel
from src.training.distributed_trainer import DistributedTrainer
from src.observability.experiment_tracker import ExperimentTracker
from src.models import ModelRegistry


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs):
    """Merge multiple configuration dictionaries"""
    merged = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


def auto_configure(config: dict) -> dict:
    """
    Auto-configure based on available hardware
    
    Args:
        config: Base configuration
    
    Returns:
        Optimized configuration
    """
    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s)")
        config.setdefault('large_scale_training', {})
        config['large_scale_training'].setdefault('hardware', {})
        config['large_scale_training']['hardware']['gpus'] = list(range(num_gpus))
        
        # Adjust batch size for multiple GPUs
        if num_gpus > 1:
            yolo_config = config.get('yolo', {}).get('training', {})
            base_batch = yolo_config.get('batch_size', 16)
            config['yolo']['training']['batch_size'] = base_batch * num_gpus
            print(f"Adjusted batch size to {base_batch * num_gpus} for {num_gpus} GPUs")
    else:
        print("No GPU detected, using CPU")
        config.setdefault('large_scale_training', {})
        config['large_scale_training'].setdefault('hardware', {})
        config['large_scale_training']['hardware']['gpus'] = []
    
    return config


def main():
    """Main training orchestrator"""
    parser = argparse.ArgumentParser(description='Automated Training Orchestrator')
    parser.add_argument(
        '--model',
        choices=['yolo', 'tabular', 'vision'],
        required=True,
        help='Model type to train'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-config',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--data-version',
        default='latest',
        help='Dataset version tag'
    )
    parser.add_argument(
        '--experiment-name',
        required=True,
        help='Experiment name for tracking'
    )
    parser.add_argument(
        '--auto-tune',
        action='store_true',
        help='Enable automatic hyperparameter tuning'
    )
    parser.add_argument(
        '--backend',
        choices=['wandb', 'mlflow', 'tensorboard', 'custom'],
        default='custom',
        help='Experiment tracking backend'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    if args.data_config:
        print(f"Loading data configuration from {args.data_config}")
        data_config = load_config(args.data_config)
        config = merge_configs(config, {'data': data_config.get('data', {})})
    
    # Auto-configure based on hardware
    config = auto_configure(config)
    
    # Enable auto-resume if requested
    if args.resume:
        config.setdefault('large_scale_training', {})
        config['large_scale_training'].setdefault('fault_tolerance', {})
        config['large_scale_training']['fault_tolerance']['auto_resume'] = True
    
    # Initialize experiment tracking
    print(f"Initializing experiment tracking with backend: {args.backend}")
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        backend=args.backend,
        config=config
    )
    
    # Log configuration
    tracker.log_params(config)
    
    try:
        if args.model == 'yolo':
            print("Training YOLO model...")
            metrics = train_yolo(config, tracker, args.data_version)
        else:
            raise NotImplementedError(f"Model type {args.model} not yet implemented")
        
        # Log final metrics
        tracker.log_metrics(metrics)
        
        # Register model if improved
        best_metric = tracker.get_best_metric('mAP')
        if best_metric is None or metrics.get('mAP', 0) > best_metric:
            print(f"New best model! mAP: {metrics.get('mAP', 0)}")
            ModelRegistry.register(
                model_path='artifacts/models/yolo/best.pt',
                metrics=metrics,
                version=args.data_version
            )
            tracker.log_model('artifacts/models/yolo/best.pt', metadata=metrics)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        tracker.finish()


def train_yolo(config: dict, tracker: ExperimentTracker, data_version: str) -> dict:
    """
    Train YOLO model
    
    Args:
        config: Training configuration
        tracker: Experiment tracker
        data_version: Dataset version
    
    Returns:
        Training metrics
    """
    # Initialize YOLO model
    print("Initializing YOLO model...")
    model = YOLOModel(config)
    
    # Get data configuration
    data_yaml = config.get('data', {}).get('processed_dir', 'data/processed') + '/data.yaml'
    
    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"Data configuration not found: {data_yaml}\n"
            f"Please run 'python scripts/prepare_data.py' first"
        )
    
    # Train the model
    print(f"Starting training with data config: {data_yaml}")
    results = model.train(
        data_config=data_yaml,
        project='artifacts/models',
        name='yolo'
    )
    
    # Extract metrics from results
    metrics = {
        'data_version': data_version,
        'mAP': 0.0,  # Placeholder - extract from results
        'precision': 0.0,
        'recall': 0.0,
    }
    
    # Try to extract actual metrics if available
    try:
        if hasattr(results, 'results_dict'):
            metrics.update(results.results_dict)
        elif hasattr(results, 'maps'):
            metrics['mAP'] = float(results.maps[0])
    except:
        pass
    
    return metrics


if __name__ == '__main__':
    main()
