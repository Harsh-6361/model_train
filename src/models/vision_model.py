"""Vision model for image classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
import torchvision.models as models

from .base_model import BaseModel


class VisionModel(BaseModel):
    """Vision model for image classification."""
    
    def __init__(
        self,
        model_type: str = 'resnet18',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize vision model.
        
        Args:
            model_type: Type of model ('resnet18', 'resnet34', 'resnet50', etc.)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            dropout: Dropout probability for classifier
            config: Model configuration
        """
        super().__init__(config)
        
        self.model_type_name = model_type
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_p = dropout
        
        # Create backbone
        self.backbone = self._create_backbone()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _create_backbone(self) -> nn.Module:
        """Create model backbone."""
        # Get pretrained model
        if self.model_type_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_type_name == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_type_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_type_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.pretrained)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_type_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=self.pretrained)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(in_features, self.num_classes)
            )
        else:
            # Default to ResNet18
            model = models.resnet18(pretrained=self.pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout_p),
                nn.Linear(in_features, self.num_classes)
            )
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Output tensor [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Batch of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Dictionary with 'loss' and metrics
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': acc
        }
    
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
        return self.training_step(batch, batch_idx)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            return probs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final classifier.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        self.eval()
        with torch.no_grad():
            # Remove final classification layer
            if hasattr(self.backbone, 'fc'):
                # ResNet-like models
                features = nn.Sequential(*list(self.backbone.children())[:-1])
                return features(x).squeeze()
            elif hasattr(self.backbone, 'classifier'):
                # EfficientNet-like models
                features = nn.Sequential(*list(self.backbone.children())[:-1])
                return features(x).squeeze()
            else:
                return self(x)


def create_vision_model(config: Dict[str, Any]) -> VisionModel:
    """
    Create vision model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        VisionModel instance
    """
    return VisionModel(
        model_type=config.get('type', 'resnet18'),
        num_classes=config.get('num_classes', 10),
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        dropout=config.get('dropout', 0.5),
        config=config
    )
