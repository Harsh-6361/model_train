"""Base model abstract class."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models."""
    
    def __init__(self):
        """Initialize base model."""
        super().__init__()
        self.model_name = self.__class__.__name__
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        pass
    
    def count_parameters(self) -> int:
        """Count trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: Union[str, Path], include_config: bool = True) -> None:
        """Save model to file.
        
        Args:
            path: Output path
            include_config: Whether to include configuration in checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_name': self.model_name,
            'state_dict': self.state_dict(),
            'metadata': self.metadata
        }
        
        if include_config:
            checkpoint['config'] = self.get_config()
        
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path], strict: bool = True) -> None:
        """Load model from file.
        
        Args:
            path: Input path
            strict: Whether to strictly enforce state dict keys match
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load state dict
        self.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        # Load metadata if available
        if 'metadata' in checkpoint:
            self.metadata = checkpoint['metadata']
    
    def set_metadata(self, **kwargs: Any) -> None:
        """Set model metadata.
        
        Args:
            **kwargs: Metadata key-value pairs
        """
        self.metadata.update(kwargs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Metadata dictionary
        """
        return self.metadata
    
    def summary(self) -> str:
        """Get model summary.
        
        Returns:
            Model summary string
        """
        lines = [
            f"Model: {self.model_name}",
            f"Total parameters: {self.count_parameters():,}",
            f"Trainable parameters: {self.count_parameters():,}",
        ]
        
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self) -> None:
        """Freeze backbone/encoder parameters (if applicable)."""
        pass  # To be implemented by subclasses if needed
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone/encoder parameters (if applicable)."""
        pass  # To be implemented by subclasses if needed
