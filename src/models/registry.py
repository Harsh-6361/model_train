"""Model registry for creating and managing models."""
from typing import Any, Dict, Optional

from .base_model import BaseModel
from .tabular_model import TabularModel
from .vision_model import create_vision_model
from .yolo_model import create_yolo_model


class ModelRegistry:
    """Registry for creating different model types."""
    
    _models = {
        'tabular': 'tabular',
        'mlp': 'tabular',
        'vision': 'vision',
        'image': 'vision',
        'cnn': 'vision',
        'yolo': 'yolo',
        'detection': 'yolo'
    }
    
    @staticmethod
    def create(
        model_type: str,
        config: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Create model based on type.
        
        Args:
            model_type: Type of model ('tabular', 'vision', 'yolo')
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        # Normalize model type
        model_type = model_type.lower()
        
        if model_type not in ModelRegistry._models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(ModelRegistry._models.keys())}"
            )
        
        category = ModelRegistry._models[model_type]
        
        if category == 'tabular':
            return ModelRegistry._create_tabular(config, **kwargs)
        elif category == 'vision':
            return ModelRegistry._create_vision(config, **kwargs)
        elif category == 'yolo':
            return ModelRegistry._create_yolo(config, **kwargs)
        else:
            raise ValueError(f"Unknown model category: {category}")
    
    @staticmethod
    def _create_tabular(config: Dict[str, Any], **kwargs) -> BaseModel:
        """Create tabular model."""
        model_config = config.get('models', {}).get('tabular', {})
        
        # Get input/output sizes from kwargs or config
        input_size = kwargs.get('input_size') or model_config.get('input_size')
        output_size = kwargs.get('output_size') or model_config.get('output_size')
        
        if input_size is None:
            raise ValueError("input_size must be specified for tabular model")
        if output_size is None:
            raise ValueError("output_size must be specified for tabular model")
        
        return TabularModel.create(
            model_type=model_config.get('type', 'mlp'),
            input_size=input_size,
            output_size=output_size,
            config=model_config
        )
    
    @staticmethod
    def _create_vision(config: Dict[str, Any], **kwargs) -> BaseModel:
        """Create vision model."""
        model_config = config.get('models', {}).get('vision', {})
        
        # Override with kwargs
        for key, value in kwargs.items():
            if value is not None:
                model_config[key] = value
        
        return create_vision_model(model_config)
    
    @staticmethod
    def _create_yolo(config: Dict[str, Any], **kwargs) -> Any:
        """Create YOLO model."""
        # YOLO config can be in 'models.yolo' or 'yolo'
        if 'yolo' in config:
            model_config = config['yolo']
        else:
            model_config = config.get('models', {}).get('yolo', {})
        
        # Override with kwargs
        for key, value in kwargs.items():
            if value is not None:
                model_config[key] = value
        
        return create_yolo_model(model_config)
    
    @staticmethod
    def list_models() -> Dict[str, list]:
        """
        List available model types.
        
        Returns:
            Dictionary of model categories and types
        """
        return {
            'tabular': ['mlp'],
            'vision': ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenet_v2'],
            'yolo': ['yolov5', 'yolov8']
        }
