"""Tabular model for structured data."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from .base_model import BaseModel


class MLP(BaseModel):
    """Multi-layer perceptron for tabular data."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.3,
        activation: str = 'relu',
        batch_norm: bool = True,
        task: str = 'classification',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MLP.
        
        Args:
            input_size: Input feature dimension
            output_size: Output dimension (num classes or 1 for regression)
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'gelu')
            batch_norm: Whether to use batch normalization
            task: Task type ('classification', 'regression')
            config: Model configuration
        """
        super().__init__(config)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.dropout_p = dropout
        self.activation_name = activation
        self.use_batch_norm = batch_norm
        self.task = task
        
        # Build network
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(self._get_activation())
            
            # Dropout
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Loss function
        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.activation_name == 'relu':
            return nn.ReLU()
        elif self.activation_name == 'tanh':
            return nn.Tanh()
        elif self.activation_name == 'gelu':
            return nn.GELU()
        elif self.activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        return self.network(x)
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Batch of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with 'loss' and metrics
        """
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Compute accuracy for classification
        metrics = {'loss': loss}
        
        if self.task == 'classification':
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            metrics['accuracy'] = acc
        
        return metrics
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        # Same as training step but without gradient computation
        return self.training_step(batch, batch_idx)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            
            if self.task == 'classification':
                # Return class probabilities
                probs = F.softmax(logits, dim=1)
                return probs
            else:
                # Return regression values
                return logits


class TabularModel:
    """Factory for creating tabular models."""
    
    @staticmethod
    def create(
        model_type: str,
        input_size: int,
        output_size: int,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        Create tabular model.
        
        Args:
            model_type: Type of model ('mlp')
            input_size: Input dimension
            output_size: Output dimension
            config: Model configuration
            
        Returns:
            Model instance
        """
        config = config or {}
        
        if model_type == 'mlp':
            return MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=config.get('hidden_layers', [256, 128, 64]),
                dropout=config.get('dropout', 0.3),
                activation=config.get('activation', 'relu'),
                batch_norm=config.get('batch_norm', True),
                task=config.get('task', 'classification'),
                config=config
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
