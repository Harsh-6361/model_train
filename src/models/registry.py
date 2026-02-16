"""Model registry for tracking model versions and metadata."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_model import BaseModel


class ModelRegistry:
    """Registry for tracking trained models."""
    
    def __init__(self, registry_path: Union[str, Path] = "artifacts/models/registry.json"):
        """Initialize model registry.
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': [],
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_models': 0
                }
            }
        
        # Build index for fast lookups
        self._build_index()
    
    def _build_index(self) -> None:
        """Build index for fast model lookups."""
        self._model_id_index = {}
        for idx, model in enumerate(self.registry['models']):
            self._model_id_index[model['model_id']] = idx
    
    def register_model(
        self,
        model_path: Union[str, Path],
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> str:
        """Register a trained model.
        
        Args:
            model_path: Path to saved model file
            model_name: Name of the model
            model_type: Type of model ('tabular', 'vision')
            metrics: Training metrics
            config: Model configuration
            tags: Optional tags for the model
            notes: Optional notes
            
        Returns:
            Model ID
        """
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        entry = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'model_path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config or {},
            'tags': tags or [],
            'notes': notes or ''
        }
        
        self.registry['models'].append(entry)
        self.registry['metadata']['total_models'] += 1
        
        # Update index
        self._model_id_index[model_id] = len(self.registry['models']) - 1
        
        # Save registry
        self._save()
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model entry by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model entry dictionary or None
        """
        # Use index for O(1) lookup
        idx = self._model_id_index.get(model_id)
        if idx is not None:
            return self.registry['models'][idx]
        return None
    
    def get_latest_model(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get latest model entry.
        
        Args:
            model_name: Optional filter by model name
            model_type: Optional filter by model type
            
        Returns:
            Latest model entry or None
        """
        models = self.registry['models']
        
        # Apply filters
        if model_name:
            models = [m for m in models if m['model_name'] == model_name]
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        
        if not models:
            return None
        
        # Sort by registration time and get latest
        models.sort(key=lambda x: x['registered_at'], reverse=True)
        return models[0]
    
    def get_best_model(
        self,
        metric_name: str,
        higher_is_better: bool = True,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get best model based on a metric.
        
        Args:
            metric_name: Metric to compare
            higher_is_better: Whether higher metric value is better
            model_name: Optional filter by model name
            model_type: Optional filter by model type
            
        Returns:
            Best model entry or None
        """
        models = self.registry['models']
        
        # Apply filters
        if model_name:
            models = [m for m in models if m['model_name'] == model_name]
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        
        # Filter models that have the metric
        models = [m for m in models if metric_name in m.get('metrics', {})]
        
        if not models:
            return None
        
        # Sort by metric
        models.sort(
            key=lambda x: x['metrics'][metric_name],
            reverse=higher_is_better
        )
        return models[0]
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all models with optional filters.
        
        Args:
            model_name: Optional filter by model name
            model_type: Optional filter by model type
            tags: Optional filter by tags
            
        Returns:
            List of model entries
        """
        models = self.registry['models']
        
        # Apply filters
        if model_name:
            models = [m for m in models if m['model_name'] == model_name]
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        if tags:
            models = [
                m for m in models
                if any(tag in m.get('tags', []) for tag in tags)
            ]
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        for i, model in enumerate(self.registry['models']):
            if model['model_id'] == model_id:
                self.registry['models'].pop(i)
                self.registry['metadata']['total_models'] -= 1
                self._save()
                return True
        return False
    
    def _save(self) -> None:
        """Save registry to file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary.
        
        Returns:
            Summary dictionary
        """
        model_types = {}
        for model in self.registry['models']:
            model_type = model['model_type']
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        return {
            'total_models': len(self.registry['models']),
            'model_types': model_types,
            'registry_path': str(self.registry_path)
        }
