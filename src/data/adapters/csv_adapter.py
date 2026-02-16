"""CSV data adapter."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..validators.schema_validator import SchemaValidator
from ...utils.helpers import train_val_test_split


class CSVAdapter:
    """Adapter for loading and validating CSV data."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        schema_path: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """Initialize CSV adapter.
        
        Args:
            file_path: Path to CSV file
            schema_path: Optional path to schema file for validation
            target_column: Name of target column
            feature_columns: List of feature column names (None = all except target)
        """
        self.file_path = Path(file_path)
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.validator = SchemaValidator(schema_path) if schema_path else None
        self.data: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """Load CSV file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def validate(self) -> Dict[str, Any]:
        """Validate loaded data against schema.
        
        Returns:
            Validation result dictionary
            
        Raises:
            ValueError: If data not loaded
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        if self.validator:
            return self.validator.validate_tabular(self.data)
        
        return {
            'valid': True,
            'errors': [],
            'warnings': ['No schema provided, skipping validation']
        }
    
    def get_features_and_target(
        self
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get features and target from data.
        
        Returns:
            Tuple of (features DataFrame, target Series)
            
        Raises:
            ValueError: If data not loaded
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Get feature columns
        if self.feature_columns is not None:
            features = self.data[self.feature_columns]
        elif self.target_column is not None:
            features = self.data.drop(columns=[self.target_column])
        else:
            features = self.data
        
        # Get target column
        target = None
        if self.target_column is not None:
            if self.target_column not in self.data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
            target = self.data[self.target_column]
        
        return features, target
    
    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Split data into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
            
        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        features, target = self.get_features_and_target()
        
        # Use common split utility
        splits = train_val_test_split(
            features, target,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            stratify=True
        )
        
        # Handle empty test set case
        if splits['test'][0] is None:
            splits['test'] = (pd.DataFrame(), None)
        
        return splits
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {'loaded': False}
        
        features, target = self.get_features_and_target()
        
        info = {
            'loaded': True,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isnull().sum().to_dict(),
            'num_features': len(features.columns),
        }
        
        if target is not None:
            info['target_column'] = self.target_column
            info['num_classes'] = len(target.unique())
            info['class_distribution'] = target.value_counts().to_dict()
        
        return info
