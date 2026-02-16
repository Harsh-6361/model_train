"""Custom metrics for model evaluation."""
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Calculate various classification metrics."""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def calculate_precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Calculate precision.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('macro', 'micro', 'weighted')
            
        Returns:
            Precision score
        """
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Calculate recall.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('macro', 'micro', 'weighted')
            
        Returns:
            Recall score
        """
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('macro', 'micro', 'weighted')
            
        Returns:
            F1 score
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_auc(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        multi_class: str = 'ovr'
    ) -> float:
        """Calculate AUC score.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            multi_class: Multi-class strategy ('ovr', 'ovo')
            
        Returns:
            AUC score
        """
        try:
            return float(roc_auc_score(y_true, y_proba, multi_class=multi_class))
        except ValueError:
            # Return 0.0 if AUC cannot be calculated (e.g., only one class)
            return 0.0
    
    @staticmethod
    def calculate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate all specified metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Optional predicted probabilities (for AUC)
            metric_names: List of metric names to calculate
            
        Returns:
            Dictionary of metric values
        """
        if metric_names is None:
            metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        metrics = {}
        
        if 'accuracy' in metric_names:
            metrics['accuracy'] = MetricsCalculator.calculate_accuracy(y_true, y_pred)
        
        if 'precision' in metric_names:
            metrics['precision'] = MetricsCalculator.calculate_precision(y_true, y_pred)
        
        if 'recall' in metric_names:
            metrics['recall'] = MetricsCalculator.calculate_recall(y_true, y_pred)
        
        if 'f1' in metric_names:
            metrics['f1'] = MetricsCalculator.calculate_f1(y_true, y_pred)
        
        if 'auc' in metric_names and y_proba is not None:
            metrics['auc'] = MetricsCalculator.calculate_auc(y_true, y_proba)
        
        return metrics


class MetricsTracker:
    """Track metrics across epochs."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
            prefix: Optional prefix for metric names (e.g., 'train_', 'val_')
        """
        for name, value in metrics.items():
            key = f"{prefix}{name}" if prefix else name
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get history for a specific metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            List of metric values
        """
        return self.history.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Latest metric value or None
        """
        history = self.history.get(metric_name, [])
        return history[-1] if history else None
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Optional[float]:
        """Get best value for a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'
            
        Returns:
            Best metric value or None
        """
        history = self.history.get(metric_name, [])
        if not history:
            return None
        
        return min(history) if mode == 'min' else max(history)
    
    def get_all_latest(self) -> Dict[str, float]:
        """Get latest values for all metrics.
        
        Returns:
            Dictionary of latest metric values
        """
        return {
            name: values[-1]
            for name, values in self.history.items()
            if values
        }
    
    def reset(self) -> None:
        """Reset all metrics history."""
        self.history.clear()
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert history to dictionary.
        
        Returns:
            History dictionary
        """
        return self.history.copy()
