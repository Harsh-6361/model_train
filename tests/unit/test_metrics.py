"""Unit tests for metrics."""
import pytest
import numpy as np
from src.training.metrics import MetricsCalculator, MetricsTracker


def test_calculate_accuracy():
    """Test accuracy calculation."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    
    accuracy = MetricsCalculator.calculate_accuracy(y_true, y_pred)
    
    # 5 out of 6 correct = 0.8333
    assert pytest.approx(accuracy, 0.01) == 0.833


def test_calculate_precision():
    """Test precision calculation."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])
    
    precision = MetricsCalculator.calculate_precision(y_true, y_pred, average='macro')
    
    assert precision >= 0.0
    assert precision <= 1.0


def test_calculate_all_metrics():
    """Test calculating all metrics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    
    metrics = MetricsCalculator.calculate_all_metrics(
        y_true,
        y_pred,
        metric_names=['accuracy', 'precision', 'recall', 'f1']
    )
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_metrics_tracker():
    """Test metrics tracker."""
    tracker = MetricsTracker()
    
    # Update metrics
    tracker.update({'loss': 0.5, 'accuracy': 0.8}, prefix='train_')
    tracker.update({'loss': 0.4, 'accuracy': 0.85}, prefix='train_')
    
    # Check history
    assert len(tracker.get_history('train_loss')) == 2
    assert tracker.get_latest('train_loss') == 0.4
    assert tracker.get_best('train_loss', mode='min') == 0.4


def test_metrics_tracker_get_all_latest():
    """Test getting all latest metrics."""
    tracker = MetricsTracker()
    
    tracker.update({'loss': 0.5, 'accuracy': 0.8})
    tracker.update({'loss': 0.4, 'accuracy': 0.85})
    
    latest = tracker.get_all_latest()
    
    assert latest['loss'] == 0.4
    assert latest['accuracy'] == 0.85
