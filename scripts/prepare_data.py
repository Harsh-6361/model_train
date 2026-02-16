#!/usr/bin/env python3
"""
Data preparation script.
Usage: python scripts/prepare_data.py --type tabular
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.observability.logger import get_logger


def generate_tabular_data(output_path: str, n_samples: int = 1000, n_features: int = 10):
    """Generate synthetic tabular data."""
    logger = get_logger()
    logger.info(f"Generating {n_samples} samples with {n_features} features")
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary classification target
    # Make it somewhat correlated with features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved tabular data to {output_path}")
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")


def create_image_structure(output_dir: str):
    """Create image directory structure."""
    logger = get_logger()
    logger.info(f"Creating image directory structure at {output_dir}")
    
    output_dir = Path(output_dir)
    
    # Create class directories
    for class_name in ['class_0', 'class_1', 'class_2']:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Image directory structure created")
    logger.info("Please add images to the class subdirectories")


def create_yolo_structure(output_dir: str):
    """Create YOLO directory structure."""
    logger = get_logger()
    logger.info(f"Creating YOLO directory structure at {output_dir}")
    
    output_dir = Path(output_dir)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create sample data.yaml
    data_yaml = output_dir / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"""# YOLO dataset configuration
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: class_0
  1: class_1
  2: class_2

nc: 3  # number of classes
""")
    
    logger.info("YOLO directory structure created")
    logger.info(f"data.yaml created at {data_yaml}")
    logger.info("Please add images and labels to the respective directories")


def main():
    parser = argparse.ArgumentParser(description='Prepare sample data')
    parser.add_argument('--type', choices=['tabular', 'image', 'yolo'], required=True,
                       help='Type of data to prepare')
    parser.add_argument('--output', default=None,
                       help='Output path (default: data/sample/{type}/)')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples for tabular data')
    parser.add_argument('--n-features', type=int, default=10,
                       help='Number of features for tabular data')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        if args.type == 'tabular':
            args.output = 'data/sample/tabular/sample_data.csv'
        elif args.type == 'image':
            args.output = 'data/sample/images'
        elif args.type == 'yolo':
            args.output = 'data/sample/yolo'
    
    # Generate data
    if args.type == 'tabular':
        generate_tabular_data(args.output, args.n_samples, args.n_features)
    elif args.type == 'image':
        create_image_structure(args.output)
    elif args.type == 'yolo':
        create_yolo_structure(args.output)


if __name__ == '__main__':
    main()
