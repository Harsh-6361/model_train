"""Metrics collection module."""
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server


class MetricsCollector:
    """Collect and track metrics."""
    
    def __init__(self, enable_prometheus: bool = False, prometheus_port: int = 9090):
        """Initialize metrics collector.
        
        Args:
            enable_prometheus: Whether to enable Prometheus metrics
            prometheus_port: Port for Prometheus metrics server
        """
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.enable_prometheus = enable_prometheus
        
        if enable_prometheus:
            # Initialize Prometheus metrics
            self.prom_counters: Dict[str, Counter] = {}
            self.prom_gauges: Dict[str, Gauge] = {}
            self.prom_histograms: Dict[str, Histogram] = {}
            
            # Start Prometheus server
            start_http_server(prometheus_port)
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name].append(value)
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        self.counters[name] += amount
        
        if self.enable_prometheus:
            if name not in self.prom_counters:
                self.prom_counters[name] = Counter(
                    name.replace('.', '_'), 
                    f'Counter for {name}'
                )
            self.prom_counters[name].inc(amount)
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value.
        
        Args:
            name: Gauge name
            value: Gauge value
        """
        if self.enable_prometheus:
            if name not in self.prom_gauges:
                self.prom_gauges[name] = Gauge(
                    name.replace('.', '_'),
                    f'Gauge for {name}'
                )
            self.prom_gauges[name].set(value)
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram observation.
        
        Args:
            name: Histogram name
            value: Observation value
        """
        self.metrics[name].append(value)
        
        if self.enable_prometheus:
            if name not in self.prom_histograms:
                self.prom_histograms[name] = Histogram(
                    name.replace('.', '_'),
                    f'Histogram for {name}'
                )
            self.prom_histograms[name].observe(value)
    
    def get_metric(self, name: str) -> List[float]:
        """Get metric values.
        
        Args:
            name: Metric name
            
        Returns:
            List of metric values
        """
        return self.metrics.get(name, [])
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get metric statistics.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with mean, min, max, last values
        """
        values = self.metrics.get(name, [])
        if not values:
            return {}
        
        import numpy as np
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'last': values[-1],
            'count': len(values)
        }
    
    def get_counter(self, name: str) -> int:
        """Get counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Counter value
        """
        return self.counters.get(name, 0)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics.
        
        Returns:
            Dictionary of all metrics and counters
        """
        return {
            'metrics': {name: self.get_metric_stats(name) for name in self.metrics},
            'counters': dict(self.counters)
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None, name: str = ""):
        """Initialize timer.
        
        Args:
            metrics_collector: Metrics collector to record timing
            name: Name for the timing metric
        """
        self.metrics_collector = metrics_collector
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record."""
        self.elapsed = time.time() - self.start_time
        if self.metrics_collector and self.name:
            self.metrics_collector.record_metric(self.name, self.elapsed)
