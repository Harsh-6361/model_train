"""YOLO model wrapper for object detection."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOModel:
    """Wrapper for YOLO object detection models."""
    
    def __init__(
        self,
        version: str = 'v8',
        size: str = 'n',
        pretrained: bool = True,
        weights: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize YOLO model.
        
        Args:
            version: YOLO version ('v5', 'v8')
            size: Model size ('n', 's', 'm', 'l', 'x')
            pretrained: Whether to use pretrained weights
            weights: Path to custom weights
            config: Model configuration
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install -r requirements-yolo.txt"
            )
        
        self.version = version
        self.size = size
        self.pretrained = pretrained
        self.config = config or {}
        
        # Initialize model
        if weights:
            self.model = YOLO(weights)
        elif pretrained:
            # Use pretrained weights
            model_name = f'yolo{version}{size}.pt'
            self.model = YOLO(model_name)
        else:
            # Initialize from scratch
            model_name = f'yolo{version}{size}.yaml'
            self.model = YOLO(model_name)
    
    def train(
        self,
        data: Union[str, Dict],
        epochs: int = 300,
        batch_size: int = 16,
        img_size: int = 640,
        **kwargs
    ) -> Any:
        """
        Train YOLO model.
        
        Args:
            data: Path to data.yaml or data configuration
            epochs: Number of epochs
            batch_size: Batch size
            img_size: Image size
            **kwargs: Additional training arguments
            
        Returns:
            Training results
        """
        return self.model.train(
            data=data,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            **kwargs
        )
    
    def val(
        self,
        data: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Validate YOLO model.
        
        Args:
            data: Path to data.yaml
            **kwargs: Additional validation arguments
            
        Returns:
            Validation results
        """
        return self.model.val(data=data, **kwargs)
    
    def predict(
        self,
        source: Union[str, List[str]],
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs
    ) -> Any:
        """
        Run inference with YOLO model.
        
        Args:
            source: Image path, directory, or list of images
            conf: Confidence threshold
            iou: IoU threshold for NMS
            **kwargs: Additional prediction arguments
            
        Returns:
            Prediction results
        """
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            **kwargs
        )
    
    def export(
        self,
        format: str = 'onnx',
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model
        """
        return self.model.export(format=format, **kwargs)
    
    def save(self, path: str) -> None:
        """
        Save model weights.
        
        Args:
            path: Path to save weights
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
    
    def load(self, path: str) -> 'YOLOModel':
        """
        Load model weights.
        
        Args:
            path: Path to weights
            
        Returns:
            Self
        """
        self.model = YOLO(path)
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            'version': self.version,
            'size': self.size,
            'pretrained': self.pretrained,
            'task': self.model.task if hasattr(self.model, 'task') else 'detect'
        }


def create_yolo_model(config: Dict[str, Any]) -> YOLOModel:
    """
    Create YOLO model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        YOLOModel instance
    """
    return YOLOModel(
        version=config.get('version', 'v8'),
        size=config.get('size', 'n'),
        pretrained=config.get('pretrained', True),
        weights=config.get('weights'),
        config=config
    )
