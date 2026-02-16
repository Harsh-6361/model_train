#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates trained models on test data
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_model import YOLOModel


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--test-data',
        required=True,
        help='Path to test data.yaml or test directory'
    )
    parser.add_argument(
        '--config',
        help='Path to model configuration (optional)'
    )
    parser.add_argument(
        '--save-dir',
        default='artifacts/evaluation',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--device',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create minimal config if not provided
    if not config:
        config = {
            'yolo': {
                'version': 'v8',
                'model_size': 'medium'
            }
        }
    
    # Initialize model
    print(f"Loading model from {args.model}")
    model = YOLOModel(config)
    model.load(args.model)
    
    # Run validation
    print(f"Evaluating on {args.test_data}")
    
    val_kwargs = {'data': args.test_data}
    if args.device:
        val_kwargs['device'] = args.device
    
    results = model.validate(**val_kwargs)
    
    # Extract metrics
    metrics = {}
    try:
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        elif hasattr(results, 'maps'):
            metrics['mAP50'] = float(results.maps[0])
            metrics['mAP50-95'] = float(results.maps[1]) if len(results.maps) > 1 else 0
    except:
        print("Warning: Could not extract detailed metrics")
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = save_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"{'='*60}")
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
    else:
        print("No metrics available")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
