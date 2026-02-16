"""Models module"""

from typing import Dict, Any

class ModelRegistry:
    """Registry for trained models"""
    
    _models = {}
    
    @classmethod
    def register(cls, model_path: str, metrics: Dict[str, Any], version: str):
        """Register a trained model"""
        cls._models[version] = {
            'path': model_path,
            'metrics': metrics,
            'version': version
        }
    
    @classmethod
    def get(cls, version: str):
        """Retrieve a registered model"""
        return cls._models.get(version)


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, config):
        self.config = config
    
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError
