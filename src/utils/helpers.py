"""Utility helper functions."""
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get torch device.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        Torch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_dict_to_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output path
    """
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_dict_from_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        path: Input path
        
    Returns:
        Dictionary
    """
    import json
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_class_weights(labels: Union[List[int], np.ndarray]) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of class weights
    """
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    class_counts = np.bincount(labels)
    
    # Inverse frequency weighting
    weights = len(labels) / (len(unique_classes) * class_counts)
    return torch.FloatTensor(weights)


def train_val_test_split(
    X: Any,
    y: Any,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Dict[str, tuple]:
    """Split data into train, validation, and test sets.
    
    This is a generic utility function for splitting data that avoids
    code duplication across different adapters.
    
    Args:
        X: Features/data to split
        y: Labels/targets to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits as (X, y) tuples
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    from sklearn.model_selection import train_test_split
    
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    
    # Determine stratification
    stratify_arg = None
    if stratify and y is not None:
        try:
            # Only stratify if we have reasonable number of samples per class
            unique_labels, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            # Need at least 2 samples per class for stratified splitting
            if len(unique_labels) < 20 and min_count >= 2:
                stratify_arg = y
        except (TypeError, ValueError):
            pass
    
    # First split: train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=stratify_arg
    )
    
    # Second split: val and test
    if test_ratio > 0:
        val_ratio_adjusted = val_ratio / temp_ratio
        
        # Determine stratification for second split
        stratify_arg_temp = None
        if stratify and y_temp is not None:
            try:
                unique_labels, counts = np.unique(y_temp, return_counts=True)
                min_count = counts.min()
                # Need at least 2 samples per class for stratified splitting
                if len(unique_labels) < 20 and min_count >= 2:
                    stratify_arg_temp = y_temp
            except (TypeError, ValueError):
                pass
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_state,
            stratify=stratify_arg_temp
        )
    else:
        X_val, y_val = X_temp, y_temp
        X_test, y_test = None, None
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

