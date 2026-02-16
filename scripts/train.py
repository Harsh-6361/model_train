#!/usr/bin/env python3
"""
Training Script

Main script for training models
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_model import YOLOModel


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--model',
        choices=['yolo'],
        default='yolo',
        help='Model type to train'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to data.yaml configuration'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        help='Image size (overrides config)'
    )
    parser.add_argument(
        '--device',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--project',
        default='artifacts/models',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        default='exp',
        help='Experiment name'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Verify data file exists
    if not Path(args.data).exists():
        print(f"Error: Data configuration not found: {args.data}")
        print("Please run 'python scripts/prepare_data.py' first")
        sys.exit(1)
    
    if args.model == 'yolo':
        print("Initializing YOLO model...")
        model = YOLOModel(config)
        
        # Prepare training arguments
        train_kwargs = {
            'data_config': args.data,
            'project': args.project,
            'name': args.name,
        }
        
        if args.epochs:
            train_kwargs['epochs'] = args.epochs
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.img_size:
            train_kwargs['img_size'] = args.img_size
        if args.device:
            train_kwargs['device'] = args.device
        if args.resume:
            train_kwargs['resume'] = True
        
        # Train the model
        print(f"Starting training...")
        print(f"  Data: {args.data}")
        print(f"  Project: {args.project}")
        print(f"  Name: {args.name}")
        
        results = model.train(**train_kwargs)
        
        print("\nTraining completed!")
        print(f"Results saved to: {args.project}/{args.name}")
        
        # Print summary
        try:
            if hasattr(results, 'results_dict'):
                print("\nTraining Summary:")
                for key, value in results.results_dict.items():
                    print(f"  {key}: {value}")
        except:
            pass
    
    else:
        print(f"Error: Model type '{args.model}' not implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()
