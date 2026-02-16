"""Tabular data preprocessor."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


class TabularPreprocessor:
    """Preprocessor for tabular data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.scaler = None
        self.encoders = {}
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TabularPreprocessor':
        """
        Fit preprocessor to data.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit scaler for numeric columns
        if numeric_cols and self.config.get('normalize', True):
            scaler_type = self.config.get('scaler', 'standard')
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            
            if self.scaler:
                self.scaler.fit(X[numeric_cols])
        
        # Fit encoders for categorical columns
        encoding = self.config.get('categorical_encoding', 'onehot')
        for col in categorical_cols:
            if encoding == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders[col] = encoder
            elif encoding == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed numpy array
        """
        X = X.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformed_parts = []
        
        # Transform numeric columns
        if numeric_cols:
            if self.scaler:
                numeric_transformed = self.scaler.transform(X[numeric_cols])
            else:
                numeric_transformed = X[numeric_cols].values
            transformed_parts.append(numeric_transformed)
        
        # Transform categorical columns
        encoding = self.config.get('categorical_encoding', 'onehot')
        for col in categorical_cols:
            if col in self.encoders:
                if encoding == 'label':
                    encoded = self.encoders[col].transform(X[col]).reshape(-1, 1)
                elif encoding == 'onehot':
                    encoded = self.encoders[col].transform(X[[col]])
                transformed_parts.append(encoded)
        
        # Concatenate all parts
        if transformed_parts:
            return np.hstack(transformed_parts)
        else:
            return X.values
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Transformed numpy array
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """
        Inverse transform data.
        
        Args:
            X: Transformed numpy array
            
        Returns:
            Original DataFrame
        """
        # Simplified inverse transform
        # In production, track column positions for proper reconstruction
        return pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
