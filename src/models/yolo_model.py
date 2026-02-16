"""
YOLO Model Integration
- Support YOLOv5, YOLOv8, and custom YOLO variants
- Configurable backbone, neck, and head
- Transfer learning support with pretrained weights
- Export to ONNX, TensorRT, CoreML for deployment
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
import yaml
from src.models import BaseModel


class YOLOModel(BaseModel):
    """
    YOLO Object Detection Model
    
    Features:
    - Multiple YOLO versions (v5, v8)
    - Configurable model size (nano, small, medium, large, xlarge)
    - Custom anchor configurations
    - Multi-scale detection
    - NMS (Non-Maximum Suppression) tuning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO model
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__(config)
        self.version = config.get('yolo', {}).get('version', 'v8')
        self.model_size = config.get('yolo', {}).get('model_size', 'medium')
        self.pretrained = config.get('yolo', {}).get('pretrained', True)
        self.pretrained_weights = config.get('yolo', {}).get('pretrained_weights', None)
        self.num_classes = config.get('yolo', {}).get('architecture', {}).get('num_classes', 80)
        
        self.model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model based on configuration"""
        try:
            from ultralytics import YOLO
            
            if self.pretrained and self.pretrained_weights:
                # Load pretrained weights
                self.model = YOLO(self.pretrained_weights)
            else:
                # Load architecture only
                model_name = self._get_model_name()
                self.model = YOLO(model_name)
                
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )
    
    def _get_model_name(self) -> str:
        """Get model name based on version and size"""
        size_map = {
            'nano': 'n',
            'small': 's',
            'medium': 'm',
            'large': 'l',
            'xlarge': 'x'
        }
        size = size_map.get(self.model_size, 'm')
        
        if self.version == 'v5':
            return f'yolov5{size}.pt'
        elif self.version == 'v8':
            return f'yolov8{size}.pt'
        else:
            return f'yolov8{size}.pt'  # Default to v8
    
    def train(
        self, 
        data_config: str,
        epochs: int = None,
        batch_size: int = None,
        img_size: int = None,
        device: Union[str, int] = None,
        project: str = 'artifacts/models',
        name: str = 'yolo',
        **kwargs
    ):
        """
        Train YOLO model
        
        Args:
            data_config: Path to data.yaml configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: Device to use (cuda/cpu)
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional training arguments
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Get training config
        training_config = self.config.get('yolo', {}).get('training', {})
        
        # Override with provided values
        train_args = {
            'data': data_config,
            'epochs': epochs or training_config.get('epochs', 300),
            'batch': batch_size or training_config.get('batch_size', 16),
            'imgsz': img_size or training_config.get('img_size', 640),
            'device': device or self._device,
            'project': project,
            'name': name,
            'patience': training_config.get('patience', 50),
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'lr0': training_config.get('lr0', 0.01),
            'lrf': training_config.get('lrf', 0.01),
            'momentum': training_config.get('momentum', 0.937),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'warmup_epochs': training_config.get('warmup_epochs', 3),
        }
        
        # Add augmentation settings
        aug_config = training_config.get('augmentation', {})
        train_args.update({
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
            'degrees': aug_config.get('degrees', 0.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.5),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'flipud': aug_config.get('flipud', 0.0),
            'fliplr': aug_config.get('fliplr', 0.5),
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0),
        })
        
        # Update with any additional kwargs
        train_args.update(kwargs)
        
        # Train the model
        results = self.model.train(**train_args)
        
        return results
    
    def detect(
        self,
        source: Union[str, List[str]],
        conf_threshold: float = None,
        iou_threshold: float = None,
        max_detections: int = None,
        **kwargs
    ):
        """
        Run detection with configurable thresholds
        
        Args:
            source: Image source (path, URL, or list of paths)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections
            **kwargs: Additional detection arguments
        
        Returns:
            Detection results
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        detection_config = self.config.get('yolo', {}).get('detection', {})
        
        detect_args = {
            'source': source,
            'conf': conf_threshold or detection_config.get('conf_threshold', 0.25),
            'iou': iou_threshold or detection_config.get('iou_threshold', 0.45),
            'max_det': max_detections or detection_config.get('max_detections', 300),
        }
        
        detect_args.update(kwargs)
        
        results = self.model(**detect_args)
        return results
    
    def predict(self, *args, **kwargs):
        """Alias for detect method"""
        return self.detect(*args, **kwargs)
    
    def export(
        self,
        format: str = 'onnx',
        optimize: bool = True,
        half_precision: bool = False,
        dynamic_batch: bool = True,
        **kwargs
    ) -> str:
        """
        Export model for production deployment
        
        Args:
            format: Export format (onnx, tensorrt, coreml, etc.)
            optimize: Apply optimization
            half_precision: Use FP16 precision
            dynamic_batch: Enable dynamic batch size
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        export_config = self.config.get('yolo', {}).get('export', {})
        
        export_args = {
            'format': format,
            'half': half_precision or export_config.get('half_precision', False),
            'dynamic': dynamic_batch or export_config.get('dynamic_batch', True),
            'simplify': optimize or export_config.get('optimize', True),
        }
        
        export_args.update(kwargs)
        
        export_path = self.model.export(**export_args)
        return export_path
    
    def save(self, path: str):
        """
        Save model to file
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
    
    def load(self, path: str):
        """
        Load model from file
        
        Args:
            path: Path to model file
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )
    
    def validate(self, data_config: str = None, **kwargs):
        """
        Validate model on validation dataset
        
        Args:
            data_config: Path to data.yaml configuration
            **kwargs: Additional validation arguments
        
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        val_args = {}
        if data_config:
            val_args['data'] = data_config
        
        val_args.update(kwargs)
        
        results = self.model.val(**val_args)
        return results
    
    @property
    def device(self):
        """Get current device"""
        return self._device
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device"""
        self._device = torch.device(device)
        if self.model is not None:
            self.model.to(self._device)
        return self
