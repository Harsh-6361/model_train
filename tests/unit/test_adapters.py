"""Unit tests for data adapters."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.adapters.csv_adapter import CSVAdapter
from src.data.adapters.image_adapter import ImageAdapter


class TestCSVAdapter:
    """Test CSV adapter."""
    
    def test_load(self, tmp_data_dir, sample_tabular_data):
        """Test loading CSV file."""
        # Save sample data
        csv_path = tmp_data_dir / "test.csv"
        sample_tabular_data.to_csv(csv_path, index=False)
        
        # Load with adapter
        adapter = CSVAdapter({'target_column': 'target'})
        df = adapter.load(str(csv_path))
        
        assert len(df) == 100
        assert 'target' in df.columns
    
    def test_validate(self, sample_tabular_data):
        """Test data validation."""
        adapter = CSVAdapter({'target_column': 'target'})
        
        assert adapter.validate(sample_tabular_data) is True
    
    def test_split(self, sample_tabular_data):
        """Test data splitting."""
        adapter = CSVAdapter()
        
        train_df, val_df, test_df = adapter.split(
            sample_tabular_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        assert len(train_df) == 70
        assert len(val_df) == 15
        assert len(test_df) == 15
    
    def test_get_features_and_target(self, sample_tabular_data):
        """Test feature/target separation."""
        adapter = CSVAdapter({'target_column': 'target'})
        
        X, y = adapter.get_features_and_target(sample_tabular_data)
        
        assert X.shape[0] == 100
        assert X.shape[1] == 5  # 5 features
        assert len(y) == 100


class TestImageAdapter:
    """Test image adapter."""
    
    def test_load_empty_dir(self, tmp_data_dir):
        """Test loading from empty directory."""
        adapter = ImageAdapter()
        
        image_paths, labels = adapter.load(str(tmp_data_dir))
        
        assert len(image_paths) == 0
        assert len(labels) == 0
    
    def test_validate(self):
        """Test validation."""
        adapter = ImageAdapter()
        
        # Empty data
        assert adapter.validate(([], [])) is True
