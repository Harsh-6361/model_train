#!/usr/bin/env python3
"""
Compute Data Hash Script

Computes a hash of the dataset for version tracking
"""

import hashlib
from pathlib import Path
import sys


def compute_directory_hash(directory: Path, extensions=None) -> str:
    """
    Compute hash of all files in directory
    
    Args:
        directory: Directory to hash
        extensions: List of file extensions to include (e.g., ['.jpg', '.txt'])
    
    Returns:
        SHA256 hash of directory contents
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.txt', '.json', '.xml']
    
    hasher = hashlib.sha256()
    
    # Get all files recursively
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}'))
    
    # Sort for consistent ordering
    files = sorted(files)
    
    for file_path in files:
        try:
            # Hash file path (relative)
            rel_path = file_path.relative_to(directory)
            hasher.update(str(rel_path).encode())
            
            # Hash file contents
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        except Exception as e:
            print(f"Warning: Could not hash {file_path}: {e}", file=sys.stderr)
    
    return hasher.hexdigest()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute dataset hash')
    parser.add_argument(
        '--data-dir',
        default='data/processed',
        help='Data directory to hash'
    )
    parser.add_argument(
        '--output',
        help='Output file for hash (default: stdout)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Computing hash for: {data_dir}", file=sys.stderr)
    hash_value = compute_directory_hash(data_dir)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(hash_value)
        print(f"Hash saved to: {args.output}", file=sys.stderr)
    else:
        # Output only hash to stdout (for use in scripts)
        print(hash_value)


if __name__ == '__main__':
    main()
