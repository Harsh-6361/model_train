"""Custom metrics for model evaluation."""
import torch
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    @staticmethod
    def classification_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            num_classes: Number of classes
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            # If logits, convert to class predictions
            if y_pred.dim() > 1:
                y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
        
        # Calculate metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Determine average method based on number of classes
        average = 'binary' if num_classes == 2 else 'macro'
        
        # Precision, recall, F1
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        return metrics
    
    @staticmethod
    def regression_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {}
        
        # MSE, MAE, R2
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            if y_pred.dim() > 1:
                y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
        
        return confusion_matrix(y_true, y_pred)


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: Value to add
            n: Number of items
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def get_average(self) -> float:
        """Get current average."""
        return self.avg


class MetricsTracker:
    """Track metrics across multiple batches."""
    
    def __init__(self, metric_names: Optional[List[str]] = None):
        """
        Initialize tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names or ['loss', 'accuracy']
        self.meters = {name: AverageMeter() for name in self.metric_names}
    
    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric values
            n: Batch size
        """
        for name, value in metrics.items():
            if name in self.meters:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.meters[name].update(value, n)
            else:
                # Add new metric
                self.meters[name] = AverageMeter()
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.meters[name].update(value, n)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get average metrics.
        
        Returns:
            Dictionary of average metrics
        """
        return {name: meter.get_average() for name, meter in self.meters.items()}
    
    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
