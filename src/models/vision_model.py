"""Vision model implementation."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models

from .base_model import BaseModel


class ResNetClassifier(BaseModel):
    """ResNet-based image classifier."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before final layer
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.dropout_rate = dropout
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Replace final layer
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self) -> None:
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_type': 'vision',
            'architecture': 'resnet',
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout': self.dropout_rate
        }
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetClassifier(BaseModel):
    """EfficientNet-based image classifier."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """Initialize EfficientNet classifier.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture ('efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before final layer
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.dropout_rate = dropout
        
        # Load backbone
        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Initialize classifier
        self._initialize_classifier()
    
    def _initialize_classifier(self) -> None:
        """Initialize classifier weights."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_type': 'vision',
            'architecture': 'efficientnet',
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout': self.dropout_rate
        }
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class VisionModel(BaseModel):
    """Factory for vision models."""
    
    @staticmethod
    def create(
        architecture: str,
        num_classes: int,
        backbone: Optional[str] = None,
        **kwargs: Any
    ) -> BaseModel:
        """Create a vision model.
        
        Args:
            architecture: Model architecture ('resnet', 'efficientnet')
            num_classes: Number of classes
            backbone: Specific backbone variant
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        architecture = architecture.lower()
        
        if architecture == 'resnet':
            backbone = backbone or 'resnet18'
            return ResNetClassifier(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=kwargs.get('pretrained', True),
                dropout=kwargs.get('dropout', 0.5)
            )
        elif architecture == 'efficientnet':
            backbone = backbone or 'efficientnet_b0'
            return EfficientNetClassifier(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=kwargs.get('pretrained', True),
                dropout=kwargs.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
