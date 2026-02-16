"""Image data preprocessor."""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torchvision import transforms


class ImagePreprocessor:
    """Preprocessor for image data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.transform = None
        self.augmentation_transform = None
        
        self._build_transforms()
    
    def _build_transforms(self) -> None:
        """Build transformation pipeline."""
        image_size = self.config.get('image_size', [224, 224])
        
        # Base transforms (resize, to tensor, normalize)
        base_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        
        # Normalization
        if self.config.get('normalize', True):
            mean = self.config.get('mean', [0.485, 0.456, 0.406])
            std = self.config.get('std', [0.229, 0.224, 0.225])
            base_transforms.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(base_transforms)
        
        # Augmentation transforms
        aug_config = self.config.get('augmentation', {})
        if aug_config.get('enabled', False):
            aug_transforms = [transforms.Resize(image_size)]
            
            if aug_config.get('horizontal_flip', False):
                aug_transforms.append(transforms.RandomHorizontalFlip())
            
            if aug_config.get('vertical_flip', False):
                aug_transforms.append(transforms.RandomVerticalFlip())
            
            rotation = aug_config.get('rotation_range', 0)
            if rotation > 0:
                aug_transforms.append(transforms.RandomRotation(rotation))
            
            if aug_config.get('color_jitter', False):
                aug_transforms.append(transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ))
            
            aug_transforms.append(transforms.ToTensor())
            
            if self.config.get('normalize', True):
                mean = self.config.get('mean', [0.485, 0.456, 0.406])
                std = self.config.get('std', [0.229, 0.224, 0.225])
                aug_transforms.append(transforms.Normalize(mean=mean, std=std))
            
            self.augmentation_transform = transforms.Compose(aug_transforms)
        else:
            self.augmentation_transform = self.transform
    
    def preprocess(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """
        Preprocess single image.
        
        Args:
            image: PIL Image
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed tensor
        """
        if augment and self.augmentation_transform:
            return self.augmentation_transform(image)
        else:
            return self.transform(image)
    
    def preprocess_batch(
        self,
        images: List[Image.Image],
        augment: bool = False
    ) -> torch.Tensor:
        """
        Preprocess batch of images.
        
        Args:
            images: List of PIL Images
            augment: Whether to apply augmentation
            
        Returns:
            Batch tensor
        """
        processed = [self.preprocess(img, augment) for img in images]
        return torch.stack(processed)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor.
        
        Args:
            tensor: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        if not self.config.get('normalize', True):
            return tensor
        
        mean = torch.tensor(self.config.get('mean', [0.485, 0.456, 0.406]))
        std = torch.tensor(self.config.get('std', [0.229, 0.224, 0.225]))
        
        # Reshape for broadcasting
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        
        return tensor * std + mean
