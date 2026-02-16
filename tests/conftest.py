"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path


@pytest.fixture
def sample_tabular_data():
    """Generate sample tabular data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    
    return df


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'training': {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'none',
            'early_stopping': {
                'enabled': False
            },
            'checkpoint': {
                'enabled': False
            },
            'device': 'cpu',
            'seed': 42
        },
        'models': {
            'tabular': {
                'type': 'mlp',
                'hidden_layers': [32, 16],
                'dropout': 0.3,
                'activation': 'relu',
                'batch_norm': True,
                'task': 'classification'
            }
        }
    }


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
