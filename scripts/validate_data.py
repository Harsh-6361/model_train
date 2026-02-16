#!/usr/bin/env python3
"""
Data Validation Script

Validates dataset for training
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.adapters.yolo_adapter import YOLODataAdapter


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate dataset')
    parser.add_argument(
        '--data',
        help='Path to data directory or data.yaml'
    )
    parser.add_argument(
        '--config',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--version',
        help='Dataset version to validate'
    )
    parser.add_argument(
        '--format',
        choices=['yolo', 'coco', 'voc'],
        default='yolo',
        help='Data format'
    )
    
    args = parser.parse_args()
    
    # Determine data path
    data_path = args.data
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if not data_path:
                data_path = config.get('data', {}).get('processed_dir', 'data/processed')
    
    if not data_path:
        data_path = 'data/processed'
    
    data_path = Path(data_path)
    
    print(f"Validating dataset at: {data_path}")
    print(f"Format: {args.format}")
    if args.version:
        print(f"Version: {args.version}")
    
    # Initialize adapter
    adapter = YOLODataAdapter(root_dir=str(data_path))
    
    # Check if data.yaml exists
    data_yaml = data_path / 'data.yaml'
    if data_yaml.exists():
        print(f"\n✓ Found data.yaml: {data_yaml}")
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            print(f"  Number of classes: {data_config.get('nc', 'unknown')}")
            print(f"  Classes: {data_config.get('names', 'unknown')}")
    else:
        print(f"\n✗ data.yaml not found at: {data_yaml}")
        return 1
    
    # Load and validate data
    try:
        adapter.load(data_path, format=args.format)
        print(f"\n✓ Successfully loaded dataset")
        print(f"  Images: {len(adapter.images)}")
        print(f"  Annotations: {len(adapter.annotations)}")
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        return 1
    
    # Validate annotations
    print("\nValidating annotations...")
    validation_report = adapter.validate_annotations()
    
    print(f"\nValidation Report:")
    print(f"  Total images: {validation_report['total_images']}")
    print(f"  Total annotations: {validation_report['total_annotations']}")
    
    if validation_report.get('issues'):
        print(f"\n⚠ Issues found:")
        for issue in validation_report['issues']:
            print(f"  - {issue['type']}: {issue['count']}")
            if 'samples' in issue:
                print(f"    Samples: {', '.join(issue['samples'][:3])}")
        return 1
    else:
        print(f"\n✓ No issues found! Dataset is valid.")
    
    # Check for split directories
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if split_dir.exists():
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                num_images = len(list(images_dir.glob('*')))
                num_labels = len(list(labels_dir.glob('*.txt')))
                print(f"\n✓ {split.capitalize()} split:")
                print(f"  Images: {num_images}")
                print(f"  Labels: {num_labels}")
            else:
                print(f"\n⚠ {split.capitalize()} split incomplete")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
