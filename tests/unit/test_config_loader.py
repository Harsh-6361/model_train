"""Unit tests for config loader."""
import pytest
from pathlib import Path
from src.utils.config_loader import ConfigLoader


def test_load_yaml(tmp_path):
    """Test loading YAML file."""
    yaml_content = """
    key1: value1
    key2:
      nested_key: nested_value
    """
    
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    
    config = ConfigLoader.load_yaml(yaml_file)
    
    assert config['key1'] == 'value1'
    assert config['key2']['nested_key'] == 'nested_value'


def test_load_json(tmp_path):
    """Test loading JSON file."""
    import json
    
    json_data = {
        'key1': 'value1',
        'key2': {'nested_key': 'nested_value'}
    }
    
    json_file = tmp_path / "test.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f)
    
    config = ConfigLoader.load_json(json_file)
    
    assert config['key1'] == 'value1'
    assert config['key2']['nested_key'] == 'nested_value'


def test_load_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load_yaml("nonexistent.yaml")


def test_merge_configs():
    """Test merging configurations."""
    config1 = {'key1': 'value1', 'key2': {'nested1': 'val1'}}
    config2 = {'key2': {'nested2': 'val2'}, 'key3': 'value3'}
    
    merged = ConfigLoader.merge_configs(config1, config2)
    
    assert merged['key1'] == 'value1'
    assert merged['key2']['nested1'] == 'val1'
    assert merged['key2']['nested2'] == 'val2'
    assert merged['key3'] == 'value3'
