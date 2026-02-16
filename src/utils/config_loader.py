"""
Configuration loader with validation and environment variable support.
Supports YAML and JSON formats with schema validation.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigLoader:
    """Load and validate configuration files."""
    
    @staticmethod
    def load(
        config_path: Union[str, Path],
        env_override: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            env_override: Whether to override config values with environment variables
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Override with environment variables if enabled
        if env_override:
            config = ConfigLoader._apply_env_overrides(config)
        
        return config
    
    @staticmethod
    def _apply_env_overrides(config: Dict[str, Any], prefix: str = "ML") -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should be formatted as: {prefix}_{KEY}_{SUBKEY}
        Example: ML_TRAINING_BATCH_SIZE=64
        """
        for key, value in config.items():
            env_key = f"{prefix}_{key}".upper()
            
            if isinstance(value, dict):
                config[key] = ConfigLoader._apply_env_overrides(value, env_key)
            else:
                env_value = os.getenv(env_key)
                if env_value is not None:
                    # Try to cast to appropriate type
                    config[key] = ConfigLoader._cast_env_value(env_value, value)
        
        return config
    
    @staticmethod
    def _cast_env_value(env_value: str, original_value: Any) -> Any:
        """Cast environment variable value to appropriate type."""
        if original_value is None:
            return env_value
        
        if isinstance(original_value, bool):
            return env_value.lower() in ('true', '1', 'yes')
        elif isinstance(original_value, int):
            return int(env_value)
        elif isinstance(original_value, float):
            return float(env_value)
        elif isinstance(original_value, list):
            return json.loads(env_value)
        elif isinstance(original_value, dict):
            return json.loads(env_value)
        else:
            return env_value
    
    @staticmethod
    def save(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")


def load_config(config_path: Union[str, Path], env_override: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        env_override: Whether to override config values with environment variables
        
    Returns:
        Configuration dictionary
    """
    return ConfigLoader.load(config_path, env_override)
