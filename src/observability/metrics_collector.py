"""Metrics collector for Prometheus."""
from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Dict, Optional


class MetricsCollector:
    """Collect and expose metrics for Prometheus."""
    
    def __init__(self, prefix: str = "ml_pipeline"):
        """
        Initialize metrics collector.
        
        Args:
            prefix: Prefix for metric names
        """
        self.prefix = prefix
        
        # Training metrics
        self.train_loss = Gauge(
            f'{prefix}_train_loss',
            'Training loss'
        )
        self.train_accuracy = Gauge(
            f'{prefix}_train_accuracy',
            'Training accuracy'
        )
        self.val_loss = Gauge(
            f'{prefix}_val_loss',
            'Validation loss'
        )
        self.val_accuracy = Gauge(
            f'{prefix}_val_accuracy',
            'Validation accuracy'
        )
        
        # Inference metrics
        self.inference_latency = Histogram(
            f'{prefix}_inference_latency_seconds',
            'Inference latency in seconds'
        )
        self.inference_requests = Counter(
            f'{prefix}_inference_requests_total',
            'Total number of inference requests'
        )
        
        # System metrics
        self.gpu_utilization = Gauge(
            f'{prefix}_gpu_utilization',
            'GPU utilization percentage'
        )
        self.memory_usage = Gauge(
            f'{prefix}_memory_usage_bytes',
            'Memory usage in bytes'
        )
    
    def update_training_metrics(self, metrics: Dict[str, float]):
        """
        Update training metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        if 'train_loss' in metrics:
            self.train_loss.set(metrics['train_loss'])
        if 'train_accuracy' in metrics:
            self.train_accuracy.set(metrics['train_accuracy'])
        if 'val_loss' in metrics:
            self.val_loss.set(metrics['val_loss'])
        if 'val_accuracy' in metrics:
            self.val_accuracy.set(metrics['val_accuracy'])
    
    def record_inference(self, latency: float):
        """
        Record inference request.
        
        Args:
            latency: Inference latency in seconds
        """
        self.inference_latency.observe(latency)
        self.inference_requests.inc()
    
    def update_system_metrics(self, gpu_util: float, memory: float):
        """
        Update system metrics.
        
        Args:
            gpu_util: GPU utilization (0-100)
            memory: Memory usage in bytes
        """
        self.gpu_utilization.set(gpu_util)
        self.memory_usage.set(memory)


# Global collector instance
_default_collector = None


def get_collector(prefix: str = "ml_pipeline") -> MetricsCollector:
    """
    Get or create metrics collector.
    
    Args:
        prefix: Prefix for metric names
        
    Returns:
        MetricsCollector instance
    """
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector(prefix)
    return _default_collector
