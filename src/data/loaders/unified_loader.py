"""Unified data loader for PyTorch."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize tabular dataset.
        
        Args:
            X: Feature array
            y: Optional target array
            device: Device to load tensors on
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.device = device
        
        if self.device:
            self.X = self.X.to(self.device)
            if self.y is not None:
                self.y = self.y.to(self.device)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Features tensor or (features, target) tuple
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ImageDataset(Dataset):
    """PyTorch Dataset for image data."""
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: Optional[List[int]] = None,
        transform: Optional[Any] = None
    ):
        """Initialize image dataset.
        
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels
            transform: Optional transform to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Image tensor or (image, label) tuple
        """
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Return with or without label
        if self.labels is not None:
            return image, self.labels[idx]
        return image


class UnifiedDataLoader:
    """Unified data loader factory for tabular and image data."""
    
    @staticmethod
    def create_tabular_loaders(
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        device: Optional[torch.device] = None
    ) -> Dict[str, DataLoader]:
        """Create data loaders for tabular data.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Optional tuple of (X_val, y_val)
            test_data: Optional tuple of (X_test, y_test)
            batch_size: Batch size
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes
            device: Device to load data on
            
        Returns:
            Dictionary of data loaders
        """
        loaders = {}
        
        # Training loader
        train_dataset = TabularDataset(train_data[0], train_data[1], device)
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        # Validation loader
        if val_data is not None:
            val_dataset = TabularDataset(val_data[0], val_data[1], device)
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        # Test loader
        if test_data is not None:
            test_dataset = TabularDataset(test_data[0], test_data[1], device)
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        
        return loaders
    
    @staticmethod
    def create_image_loaders(
        train_data: Tuple[List[Path], List[int]],
        val_data: Optional[Tuple[List[Path], List[int]]] = None,
        test_data: Optional[Tuple[List[Path], List[int]]] = None,
        train_transform: Optional[Any] = None,
        val_transform: Optional[Any] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> Dict[str, DataLoader]:
        """Create data loaders for image data.
        
        Args:
            train_data: Tuple of (image_paths, labels)
            val_data: Optional tuple of (image_paths, labels)
            test_data: Optional tuple of (image_paths, labels)
            train_transform: Transform for training data
            val_transform: Transform for validation/test data
            batch_size: Batch size
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of data loaders
        """
        loaders = {}
        
        # Training loader
        train_dataset = ImageDataset(
            train_data[0],
            train_data[1],
            train_transform
        )
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Validation loader
        if val_data is not None:
            val_dataset = ImageDataset(
                val_data[0],
                val_data[1],
                val_transform
            )
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        # Test loader
        if test_data is not None:
            test_dataset = ImageDataset(
                test_data[0],
                test_data[1],
                val_transform
            )
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return loaders
