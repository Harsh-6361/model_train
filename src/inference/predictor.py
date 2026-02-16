"""Inference engine for model predictions."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ..models.base_model import BaseModel
from ..models.registry import ModelRegistry
from ..utils.helpers import get_device


class Predictor:
    """Unified inference engine."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model ('tabular', 'vision', 'yolo')
            config: Model configuration
            device: Device to run inference on
        """
        self.model_path = model_path
        self.model_type = model_type
        self.config = config or {}
        self.device = get_device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self):
        """Load model from checkpoint."""
        if self.model_type == 'yolo':
            from ..models.yolo_model import YOLOModel
            return YOLOModel(weights=self.model_path)
        else:
            # Load PyTorch model
            model = ModelRegistry.create(self.model_type, self.config)
            model.load(self.model_path, device=self.device)
            return model
    
    def predict(
        self,
        data: Union[np.ndarray, torch.Tensor, Image.Image, str, List],
        **kwargs
    ) -> Any:
        """
        Run inference.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        start_time = time.time()
        
        if self.model_type == 'yolo':
            # YOLO prediction
            results = self.model.predict(data, **kwargs)
            inference_time = time.time() - start_time
            return {
                'predictions': results,
                'inference_time': inference_time
            }
        else:
            # PyTorch model prediction
            # Prepare input
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            elif isinstance(data, Image.Image):
                # Convert image to tensor
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                data = transform(data).unsqueeze(0)
            
            # Move to device
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
            
            # Predict
            with torch.no_grad():
                if hasattr(self.model, 'predict'):
                    output = self.model.predict(data)
                else:
                    output = self.model(data)
            
            inference_time = time.time() - start_time
            
            return {
                'predictions': output.cpu().numpy(),
                'inference_time': inference_time
            }
    
    def predict_batch(
        self,
        data_list: List,
        batch_size: int = 32,
        **kwargs
    ) -> List[Any]:
        """
        Run batch inference.
        
        Args:
            data_list: List of inputs
            batch_size: Batch size
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            if self.model_type == 'yolo':
                batch_results = self.model.predict(batch, **kwargs)
                results.extend(batch_results)
            else:
                # Stack batch
                if isinstance(batch[0], torch.Tensor):
                    batch_tensor = torch.stack(batch)
                elif isinstance(batch[0], np.ndarray):
                    batch_tensor = torch.FloatTensor(np.stack(batch))
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch[0])}")
                
                # Predict
                batch_results = self.predict(batch_tensor, **kwargs)
                results.append(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        info = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'device': str(self.device)
        }
        
        if hasattr(self.model, 'get_num_parameters'):
            info['parameters'] = self.model.get_num_parameters()
        
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        
        return info
