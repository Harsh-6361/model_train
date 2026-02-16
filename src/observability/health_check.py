"""Health check utilities."""
import time
from typing import Dict, Optional
import torch


class HealthCheck:
    """Health check for the ML pipeline."""
    
    @staticmethod
    def check_system() -> Dict[str, any]:
        """
        Check system health.
        
        Returns:
            Dictionary with health status
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        # Check PyTorch
        health['checks']['pytorch'] = {
            'available': True,
            'version': torch.__version__
        }
        
        # Check CUDA
        health['checks']['cuda'] = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            health['checks']['cuda']['devices'] = [
                {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i)
                }
                for i in range(torch.cuda.device_count())
            ]
        
        return health
    
    @staticmethod
    def check_model(model) -> Dict[str, any]:
        """
        Check model health.
        
        Args:
            model: Model to check
            
        Returns:
            Dictionary with model status
        """
        status = {
            'loaded': model is not None,
            'parameters': 0
        }
        
        if model is not None:
            status['parameters'] = sum(p.numel() for p in model.parameters())
            status['trainable_parameters'] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        
        return status


def get_health() -> Dict[str, any]:
    """
    Get overall health status.
    
    Returns:
        Health status dictionary
    """
    return HealthCheck.check_system()
