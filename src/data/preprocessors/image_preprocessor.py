"""Image data preprocessor."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class ImagePreprocessor:
    """Preprocessor for image data."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        augment: bool = False,
        horizontal_flip: bool = False,
        rotation_range: int = 0
    ):
        """Initialize image preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize images
            mean: Mean values for normalization (default: ImageNet)
            std: Standard deviation values for normalization (default: ImageNet)
            augment: Whether to apply data augmentation
            horizontal_flip: Whether to apply random horizontal flips
            rotation_range: Maximum rotation angle in degrees
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.augment = augment
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        
        # Build transforms
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
    
    def _build_train_transform(self) -> transforms.Compose:
        """Build training transforms with augmentation.
        
        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize(self.image_size),
        ]
        
        if self.augment:
            if self.horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            
            if self.rotation_range > 0:
                transform_list.append(
                    transforms.RandomRotation(self.rotation_range)
                )
            
            # Add color jitter for more augmentation
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def _build_val_transform(self) -> transforms.Compose:
        """Build validation/test transforms without augmentation.
        
        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def preprocess_image(
        self,
        image: Image.Image,
        training: bool = False
    ) -> torch.Tensor:
        """Preprocess a single image.
        
        Args:
            image: PIL Image
            training: Whether to use training transforms (with augmentation)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply appropriate transform
        if training:
            return self.train_transform(image)
        else:
            return self.val_transform(image)
    
    def preprocess_batch(
        self,
        images: List[Image.Image],
        training: bool = False
    ) -> torch.Tensor:
        """Preprocess a batch of images.
        
        Args:
            images: List of PIL Images
            training: Whether to use training transforms
            
        Returns:
            Batch of preprocessed image tensors
        """
        processed = [self.preprocess_image(img, training) for img in images]
        return torch.stack(processed)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a tensor for visualization.
        
        Args:
            tensor: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        if not self.normalize:
            return tensor
        
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        return tensor * std + mean
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'image_size': self.image_size,
            'normalize': self.normalize,
            'mean': self.mean,
            'std': self.std,
            'augment': self.augment,
            'horizontal_flip': self.horizontal_flip,
            'rotation_range': self.rotation_range
        }
