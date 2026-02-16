"""Base model class for all models."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or {}
        self.model_type = self.__class__.__name__
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Batch of data (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary containing 'loss' and optionally other metrics
        """
        pass
    
    @abstractmethod
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch of data (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary containing 'loss' and optionally other metrics
        """
        pass
    
    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step (defaults to validation step).
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary containing metrics
        """
        return self.validation_step(batch, batch_idx)
    
    def save(self, path: str, save_config: bool = True) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            save_config: Whether to save config with checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_type': self.model_type,
        }
        
        if save_config:
            checkpoint['config'] = self.config
        
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: Optional[torch.device] = None) -> 'BaseModel':
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model to
            
        Returns:
            Self
        """
        device = device or torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        return self
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            trainable_only: Count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
