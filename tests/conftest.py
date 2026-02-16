"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [2.0, 4.0, 6.0, 8.0, 10.0],
        'feature3': ['a', 'b', 'a', 'b', 'a'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_numpy_array():
    """Create sample numpy array for testing."""
    return np.random.randn(10, 5)


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary model path."""
    return tmp_path / "test_model.pth"


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss': 'cross_entropy',
            'device': 'cpu',
            'seed': 42,
            'early_stopping': {
                'enabled': False
            },
            'checkpoint': {
                'enabled': False
            },
            'logging': {
                'enabled': False
            },
            'metrics': ['accuracy']
        }
    }
