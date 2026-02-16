"""Image data adapter for computer vision tasks."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .base_adapter import BaseDataAdapter


class ImageDataset(Dataset):
    """PyTorch Dataset for images."""
    
    def __init__(self, image_paths: List[str], labels: Optional[List[int]] = None, 
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        """Get item by index."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image, -1


class ImageAdapter(BaseDataAdapter):
    """Adapter for loading and processing image data."""
    
    def load(self, path: str, **kwargs) -> Tuple[List[str], List[int]]:
        """
        Load image paths from directory structure.
        
        Expected structure:
            path/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    img3.jpg
        
        Args:
            path: Path to image directory
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (image_paths, labels)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image directory not found: {path}")
        
        image_paths = []
        labels = []
        class_names = sorted([d.name for d in path.iterdir() if d.is_dir()])
        
        if not class_names:
            # No subdirectories, load all images with label -1
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend([str(p) for p in path.glob(ext)])
            labels = [-1] * len(image_paths)
            return image_paths, labels
        
        # Load from class subdirectories
        for label, class_name in enumerate(class_names):
            class_dir = path / class_name
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(label)
        
        return image_paths, labels
    
    def validate(self, data: Tuple[List[str], List[int]]) -> bool:
        """
        Validate image data.
        
        Args:
            data: Tuple of (image_paths, labels)
            
        Returns:
            True if valid
        """
        image_paths, labels = data
        
        if len(image_paths) == 0:
            raise ValueError("No images found")
        
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
        
        # Validate that files exist
        for img_path in image_paths[:min(10, len(image_paths))]:  # Check first 10
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
        
        return True
    
    def preprocess(self, data: Tuple[List[str], List[int]], 
                   config: Optional[Dict] = None) -> ImageDataset:
        """
        Create preprocessed dataset.
        
        Args:
            data: Tuple of (image_paths, labels)
            config: Preprocessing configuration
            
        Returns:
            ImageDataset
        """
        config = config or self.config
        image_paths, labels = data
        
        # Get image size
        image_size = config.get('image_size', [224, 224])
        
        # Build transforms
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize(image_size))
        
        # Augmentation (if enabled)
        augmentation = config.get('augmentation', {})
        if augmentation.get('enabled', False):
            if augmentation.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if augmentation.get('vertical_flip', False):
                transform_list.append(transforms.RandomVerticalFlip())
            
            rotation = augmentation.get('rotation_range', 0)
            if rotation > 0:
                transform_list.append(transforms.RandomRotation(rotation))
            
            zoom = augmentation.get('zoom_range', 0)
            if zoom > 0:
                transform_list.append(transforms.RandomAffine(
                    degrees=0,
                    scale=(1-zoom, 1+zoom)
                ))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if augmentation.get('normalize', True):
            mean = augmentation.get('mean', [0.485, 0.456, 0.406])
            std = augmentation.get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        transform = transforms.Compose(transform_list)
        
        return ImageDataset(image_paths, labels, transform)
    
    def split(
        self,
        data: Tuple[List[str], List[int]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], 
               Tuple[List[str], List[int]]]:
        """
        Split image data into train, validation, and test sets.
        
        Args:
            data: Tuple of (image_paths, labels)
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle
            seed: Random seed
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        image_paths, labels = data
        
        # Set seed
        np.random.seed(seed)
        
        # Shuffle if requested
        indices = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(indices)
        
        # Calculate split points
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Create splits
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        test_paths = [image_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
