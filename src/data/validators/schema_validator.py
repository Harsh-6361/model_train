"""Schema validation module."""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from PIL import Image


class SchemaValidator:
    """Validate data against JSON schemas."""
    
    def __init__(self, schema_path: Optional[Union[str, Path]] = None):
        """Initialize validator.
        
        Args:
            schema_path: Path to JSON schema file
        """
        self.schema = None
        if schema_path:
            self.load_schema(schema_path)
    
    def load_schema(self, schema_path: Union[str, Path]) -> None:
        """Load schema from file.
        
        Args:
            schema_path: Path to JSON schema file
        """
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
    
    def validate_tabular(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate tabular data.
        
        Args:
            df: Pandas DataFrame to validate
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if self.schema is None:
            result['warnings'].append("No schema provided, skipping validation")
            return result
        
        # Check required columns
        required_cols = self.schema.get('properties', {}).get('required_columns', {}).get('items', [])
        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                result['valid'] = False
                result['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check column types
        column_types = self.schema.get('properties', {}).get('column_types', {})
        if column_types:
            for col in df.columns:
                expected_type = column_types.get(col)
                if expected_type:
                    if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                        result['warnings'].append(f"Column {col} expected to be numeric")
                    elif expected_type == 'categorical' and not pd.api.types.is_object_dtype(df[col]):
                        result['warnings'].append(f"Column {col} expected to be categorical")
        
        # Check for null values
        nullable_cols = self.schema.get('properties', {}).get('nullable_columns', [])
        for col in df.columns:
            if col not in nullable_cols and df[col].isnull().any():
                result['warnings'].append(f"Column {col} contains null values")
        
        return result
    
    def validate_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            img = Image.open(image_path)
            
            if self.schema:
                required = self.schema.get('properties', {}).get('required_structure', {}).get('properties', {})
                
                # Check format
                allowed_formats = required.get('format', {}).get('items', [])
                if allowed_formats:
                    img_format = img.format.lower() if img.format else ''
                    if img_format not in allowed_formats:
                        result['warnings'].append(f"Image format {img_format} not in allowed formats")
                
                # Check size
                min_size = required.get('min_size', {}).get('properties', {})
                if min_size:
                    if img.width < min_size.get('width', 0) or img.height < min_size.get('height', 0):
                        result['warnings'].append(f"Image size {img.size} below minimum")
                
                max_size = required.get('max_size', {}).get('properties', {})
                if max_size:
                    if img.width > max_size.get('width', float('inf')) or img.height > max_size.get('height', float('inf')):
                        result['warnings'].append(f"Image size {img.size} above maximum")
                
                # Check channels
                expected_channels = required.get('channels')
                if expected_channels:
                    actual_channels = len(img.getbands())
                    if actual_channels != expected_channels:
                        result['warnings'].append(
                            f"Image has {actual_channels} channels, expected {expected_channels}"
                        )
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to validate image: {str(e)}")
        
        return result
