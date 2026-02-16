"""Health check module."""
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch


class HealthCheck:
    """System and model health checks."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize health checker.
        
        Args:
            model_path: Path to model file to check
        """
        self.model_path = model_path
        self.start_time = datetime.now()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics.
        
        Returns:
            Dictionary with system health information
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'system': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'cpu': {
                    'count': psutil.cpu_count(),
                    'percent': cpu_percent,
                    'status': 'healthy' if cpu_percent < 90 else 'warning'
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'percent': memory.percent,
                    'status': 'healthy' if memory.percent < 90 else 'warning'
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': disk.percent,
                    'status': 'healthy' if disk.percent < 90 else 'warning'
                }
            }
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            health['system']['gpu'] = {
                'available': True,
                'count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0)
            }
        else:
            health['system']['gpu'] = {'available': False}
        
        return health
    
    def check_model_health(self) -> Dict[str, Any]:
        """Check model health.
        
        Returns:
            Dictionary with model health information
        """
        from pathlib import Path
        
        health = {
            'model': {
                'loaded': False,
                'path': self.model_path,
                'status': 'unknown'
            }
        }
        
        if self.model_path:
            model_file = Path(self.model_path)
            if model_file.exists():
                health['model']['loaded'] = True
                health['model']['size_mb'] = model_file.stat().st_size / (1024**2)
                health['model']['status'] = 'healthy'
            else:
                health['model']['status'] = 'error'
                health['model']['error'] = 'Model file not found'
        
        return health
    
    def get_full_health(self) -> Dict[str, Any]:
        """Get full health check.
        
        Returns:
            Complete health check information
        """
        system_health = self.get_system_health()
        model_health = self.check_model_health()
        
        # Merge health checks
        system_health.update(model_health)
        
        # Determine overall status
        statuses = [
            system_health['system']['cpu']['status'],
            system_health['system']['memory']['status'],
            system_health['system']['disk']['status'],
            system_health.get('model', {}).get('status', 'unknown')
        ]
        
        if 'error' in statuses:
            system_health['status'] = 'error'
        elif 'warning' in statuses:
            system_health['status'] = 'warning'
        else:
            system_health['status'] = 'healthy'
        
        return system_health
