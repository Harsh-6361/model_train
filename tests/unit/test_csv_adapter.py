"""Unit tests for CSV adapter."""
import pytest
import pandas as pd
from src.data.adapters.csv_adapter import CSVAdapter


def test_csv_adapter_load(tmp_path, sample_dataframe):
    """Test loading CSV file."""
    csv_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    
    adapter = CSVAdapter(file_path=csv_path, target_column='target')
    data = adapter.load()
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == len(sample_dataframe)
    assert list(data.columns) == list(sample_dataframe.columns)


def test_csv_adapter_get_features_and_target(tmp_path, sample_dataframe):
    """Test getting features and target."""
    csv_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    
    adapter = CSVAdapter(file_path=csv_path, target_column='target')
    adapter.load()
    
    features, target = adapter.get_features_and_target()
    
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert 'target' not in features.columns
    assert len(features) == len(target)


def test_csv_adapter_split_data(tmp_path, sample_dataframe):
    """Test data splitting."""
    csv_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    
    adapter = CSVAdapter(file_path=csv_path, target_column='target')
    adapter.load()
    
    splits = adapter.split_data(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    
    # Check split sizes (approximately)
    total_size = len(sample_dataframe)
    train_size = len(splits['train'][0])
    val_size = len(splits['val'][0])
    test_size = len(splits['test'][0])
    
    assert train_size + val_size + test_size == total_size


def test_csv_adapter_file_not_found():
    """Test loading non-existent file."""
    adapter = CSVAdapter(file_path="nonexistent.csv")
    
    with pytest.raises(FileNotFoundError):
        adapter.load()
