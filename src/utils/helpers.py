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
