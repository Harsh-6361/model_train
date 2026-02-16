#!/usr/bin/env python3
"""
Prediction Script

Run predictions on images using trained models
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_model import YOLOModel


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Run predictions')
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Path to image(s) or video'
    )
    parser.add_argument(
        '--config',
        help='Path to model configuration (optional)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--save-dir',
        default='artifacts/predictions',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--device',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display predictions'
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Initialize model
    print(f"Loading model from {args.model}")
    
    # Create minimal config if not provided
    if not config:
        config = {
            'yolo': {
                'version': 'v8',
                'model_size': 'medium',
                'detection': {
                    'conf_threshold': args.conf,
                    'iou_threshold': args.iou
                }
            }
        }
    
    model = YOLOModel(config)
    model.load(args.model)
    
    # Run predictions
    print(f"Running predictions on {args.source}")
    results = model.detect(
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save=True,
        project=args.save_dir,
        show=args.show
    )
    
    print(f"\nPredictions complete!")
    print(f"Results saved to: {args.save_dir}")
    
    # Print detection summary
    try:
        for r in results:
            if hasattr(r, 'boxes'):
                print(f"\nDetected {len(r.boxes)} objects in {r.path}")
    except:
        pass


if __name__ == '__main__':
    main()
