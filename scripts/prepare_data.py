#!/usr/bin/env python3
"""
Data Preparation Script

Prepares and converts datasets for YOLO training
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.adapters.yolo_adapter import YOLODataAdapter


def main():
    """Main data preparation function"""
    parser = argparse.ArgumentParser(description='Prepare data for YOLO training')
    parser.add_argument(
        '--input',
        required=True,
        help='Input data directory'
    )
    parser.add_argument(
        '--annotations',
        help='Annotations directory (if separate from input)'
    )
    parser.add_argument(
        '--format',
        choices=['yolo', 'coco', 'voc'],
        default='yolo',
        help='Input data format'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--config',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split dataset into train/val/test'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get class names
    class_names = config.get('data', {}).get('classes', ['class_0', 'class_1', 'class_2'])
    
    print(f"Loading {args.format} format data from {args.input}")
    
    # Initialize adapter
    adapter = YOLODataAdapter(root_dir=args.output)
    
    # Load data
    input_path = Path(args.input)
    adapter.load(input_path, format=args.format)
    
    print(f"Loaded {len(adapter.images)} images with {len(adapter.annotations)} annotations")
    
    # Validate annotations
    print("Validating annotations...")
    validation_report = adapter.validate_annotations()
    print(f"Validation report: {validation_report}")
    
    if validation_report.get('issues'):
        print("Warning: Issues found in annotations:")
        for issue in validation_report['issues']:
            print(f"  - {issue['type']}: {issue['count']}")
    
    # Convert to YOLO format if needed
    if args.format != 'yolo':
        print(f"Converting from {args.format} to YOLO format...")
        adapter.convert(args.format, 'yolo', args.output)
        # Reload in YOLO format
        adapter.load(Path(args.output), format='yolo')
    
    # Split dataset if requested
    if args.split:
        print(f"Splitting dataset (train: {args.train_ratio}, val: {args.val_ratio}, test: {args.test_ratio})...")
        adapter.split_dataset(
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    # Create data.yaml
    data_yaml_path = Path(args.output) / 'data.yaml'
    print(f"Creating data.yaml at {data_yaml_path}")
    adapter.create_data_yaml(
        output_path=str(data_yaml_path),
        class_names=class_names,
        train_path='train/images' if args.split else 'images',
        val_path='val/images' if args.split else 'images',
        test_path='test/images' if args.split else None
    )
    
    print(f"Data preparation complete! Output saved to {args.output}")
    print(f"Use data.yaml at: {data_yaml_path}")


if __name__ == '__main__':
    main()
