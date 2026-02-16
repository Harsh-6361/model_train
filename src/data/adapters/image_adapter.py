"""Image data adapter."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from ..validators.schema_validator import SchemaValidator


class ImageAdapter:
    """Adapter for loading and validating image data."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        schema_path: Optional[Union[str, Path]] = None,
        image_extensions: Optional[List[str]] = None
    ):
        """Initialize image adapter.
        
        Args:
            data_dir: Directory containing images (class-based subdirectories)
            schema_path: Optional path to schema file for validation
            image_extensions: List of valid image extensions
        """
        self.data_dir = Path(data_dir)
        self.schema_path = schema_path
        self.validator = SchemaValidator(schema_path) if schema_path else None
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        
        self.image_paths: List[Path] = []
        self.labels: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        
    def load(self) -> Tuple[List[Path], List[int]]:
        """Load image paths and labels from directory structure.
        
        Expected structure:
            data_dir/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    img3.jpg
                    img4.jpg
        
        Returns:
            Tuple of (image_paths, labels)
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_dir}")
        
        # Create class mappings
        class_names = sorted([d.name for d in class_dirs])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in this class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")
        
        return self.image_paths, self.labels
    
    def validate(self, sample_size: Optional[int] = 10) -> Dict[str, Any]:
        """Validate a sample of images against schema.
        
        Args:
            sample_size: Number of images to validate (None = all)
            
        Returns:
            Validation result dictionary
        """
        if not self.image_paths:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Sample images to validate
        if sample_size is None or sample_size >= len(self.image_paths):
            sample_paths = self.image_paths
        else:
            indices = np.random.choice(len(self.image_paths), sample_size, replace=False)
            sample_paths = [self.image_paths[i] for i in indices]
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_count': len(sample_paths),
            'total_count': len(self.image_paths)
        }
        
        if self.validator is None:
            results['warnings'].append("No schema provided, skipping validation")
            return results
        
        # Validate each sampled image - optimized with early stopping
        errors_found = []
        warnings_found = []
        
        for img_path in sample_paths:
            validation = self.validator.validate_image(img_path)
            
            if not validation['valid']:
                results['valid'] = False
                errors_found.extend([
                    f"{img_path.name}: {err}" for err in validation['errors']
                ])
                # Early stop if we have too many errors (sample is representative)
                if len(errors_found) > 5:
                    errors_found.append("... (additional errors truncated)")
                    break
            
            warnings_found.extend([
                f"{img_path.name}: {warn}" for warn in validation['warnings']
            ])
        
        results['errors'] = errors_found
        results['warnings'] = warnings_found[:10]  # Limit warnings to keep output manageable
        
        return results
    
    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, Tuple[List[Path], List[int]]]:
        """Split data into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if not self.image_paths:
            raise ValueError("Data not loaded. Call load() first.")
        
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        # First split: train and temp (val + test)
        temp_ratio = val_ratio + test_ratio
        paths_train, paths_temp, labels_train, labels_temp = train_test_split(
            self.image_paths,
            self.labels,
            test_size=temp_ratio,
            random_state=random_state,
            stratify=self.labels
        )
        
        # Second split: val and test
        if test_ratio > 0:
            val_ratio_adjusted = val_ratio / temp_ratio
            paths_val, paths_test, labels_val, labels_test = train_test_split(
                paths_temp,
                labels_temp,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_state,
                stratify=labels_temp
            )
        else:
            paths_val, labels_val = paths_temp, labels_temp
            paths_test, labels_test = [], []
        
        return {
            'train': (paths_train, labels_train),
            'val': (paths_val, labels_val),
            'test': (paths_test, labels_test)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        if not self.image_paths:
            return {'loaded': False}
        
        # Calculate class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = {
            self.idx_to_class[idx]: int(count) 
            for idx, count in zip(unique, counts)
        }
        
        return {
            'loaded': True,
            'total_images': len(self.image_paths),
            'num_classes': len(self.class_to_idx),
            'classes': list(self.class_to_idx.keys()),
            'class_to_idx': self.class_to_idx,
            'class_distribution': class_distribution
        }
