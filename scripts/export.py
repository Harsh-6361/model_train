#!/usr/bin/env python3
"""
Model Export Script

Export trained models to various formats for deployment
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_model import YOLOModel


def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export model to deployment format')
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--format',
        nargs='+',
        default=['onnx'],
        help='Export format(s): onnx, tensorrt, coreml, etc.'
    )
    parser.add_argument(
        '--config',
        help='Path to model configuration (optional)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        default=True,
        help='Apply optimization'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 half precision'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        default=True,
        help='Enable dynamic batch size'
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
                'export': {
                    'optimize': args.optimize,
                    'half_precision': args.half,
                    'dynamic_batch': args.dynamic
                }
            }
        }
    
    # Initialize model
    print(f"Loading model from {args.model}")
    model = YOLOModel(config)
    model.load(args.model)
    
    # Export to each format
    exported_paths = []
    for fmt in args.format:
        print(f"\nExporting to {fmt.upper()} format...")
        try:
            export_path = model.export(
                format=fmt,
                optimize=args.optimize,
                half_precision=args.half,
                dynamic_batch=args.dynamic
            )
            exported_paths.append(export_path)
            print(f"✓ Export successful: {export_path}")
        except Exception as e:
            print(f"✗ Export failed for {fmt}: {e}")
    
    print(f"\n{'='*60}")
    print("Export Summary:")
    print(f"{'='*60}")
    print(f"Source model: {args.model}")
    print(f"Exported formats: {', '.join(args.format)}")
    print(f"Exported files:")
    for path in exported_paths:
        print(f"  - {path}")


if __name__ == '__main__':
    main()
