"""Configuration loader for YAML/JSON files."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigLoader:
    """Load and parse configuration files."""
    
    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If file is not valid YAML
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config if config else {}
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        return config if config else {}
    
    @staticmethod
    def load(path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration file (auto-detect format).
        
        Args:
            path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            return ConfigLoader.load_yaml(path)
        elif suffix == '.json':
            return ConfigLoader.load_json(path)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def save_json(config: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        result = {}
        for config in configs:
            result = ConfigLoader._deep_merge(result, config)
        return result
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
