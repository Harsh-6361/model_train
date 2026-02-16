"""Tabular model implementation."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base_model import BaseModel


class MLPClassifier(BaseModel):
    """Multi-layer perceptron for tabular data classification."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        activation: str = "relu"
    ):
        """Initialize MLP classifier.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes)
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        self.activation_name = activation
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.
        
        Args:
            name: Activation name
            
        Returns:
            Activation module
        """
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_type': 'tabular',
            'architecture': 'mlp',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout_rate,
            'activation': self.activation_name
        }


class TabularModel(BaseModel):
    """Factory for tabular models."""
    
    @staticmethod
    def create(
        architecture: str,
        input_dim: int,
        output_dim: int,
        **kwargs: Any
    ) -> BaseModel:
        """Create a tabular model.
        
        Args:
            architecture: Model architecture ('mlp', etc.)
            input_dim: Input dimension
            output_dim: Output dimension
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if architecture.lower() == 'mlp':
            return MLPClassifier(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=kwargs.get('hidden_layers', [128, 64, 32]),
                dropout=kwargs.get('dropout', 0.3),
                activation=kwargs.get('activation', 'relu')
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
