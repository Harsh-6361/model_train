"""CSV data adapter for tabular data."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

from .base_adapter import BaseDataAdapter


class CSVAdapter(BaseDataAdapter):
    """Adapter for loading and processing CSV tabular data."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            path: Path to CSV file
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            Pandas DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        # Load CSV with configurable options
        df = pd.read_csv(path, **kwargs)
        return df
    
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate DataFrame.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if len(data) == 0:
            raise ValueError("DataFrame is empty")
        
        # Check for target column if specified in config
        target_col = self.config.get('target_column')
        if target_col and target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        return True
    
    def preprocess(self, data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Preprocess DataFrame.
        
        Args:
            data: DataFrame to preprocess
            config: Preprocessing configuration
            
        Returns:
            Preprocessed DataFrame
        """
        config = config or self.config.get('preprocessing', {})
        df = data.copy()
        
        # Handle missing values
        handle_missing = config.get('handle_missing', 'mean')
        if handle_missing != 'drop':
            for col in df.columns:
                if df[col].isna().any():
                    if df[col].dtype in [np.float64, np.int64]:
                        if handle_missing == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif handle_missing == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            df[col].fillna(handle_missing, inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing', 
                                     inplace=True)
        else:
            df.dropna(inplace=True)
        
        return df
    
    def split(
        self, 
        data: pd.DataFrame, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True, 
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train, validation, and test sets.
        
        Args:
            data: DataFrame to split
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle data
            seed: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            data,
            train_size=train_ratio,
            shuffle=shuffle,
            random_state=seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            shuffle=shuffle,
            random_state=seed
        )
        
        return train_df, val_df, test_df
    
    def get_features_and_target(
        self, 
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target from DataFrame.
        
        Args:
            data: DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names (None = all except target)
            
        Returns:
            Tuple of (features_df, target_series)
        """
        target_col = target_column or self.config.get('target_column')
        if not target_col:
            raise ValueError("Target column must be specified")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Get target
        y = data[target_col]
        
        # Get features
        if feature_columns:
            X = data[feature_columns]
        else:
            X = data.drop(columns=[target_col])
        
        return X, y
