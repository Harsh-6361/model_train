"""Tabular data preprocessor."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class TabularPreprocessor:
    """Preprocessor for tabular data."""
    
    def __init__(
        self,
        normalize: bool = True,
        handle_missing: str = "mean",
        categorical_encoding: str = "onehot"
    ):
        """Initialize preprocessor.
        
        Args:
            normalize: Whether to normalize numeric features
            handle_missing: How to handle missing values ('mean', 'median', 'drop')
            categorical_encoding: How to encode categorical features ('onehot', 'label')
        """
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.categorical_encoding = categorical_encoding
        
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TabularPreprocessor':
        """Fit preprocessor to data.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Self
        """
        # Identify column types
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit scalers for numeric columns
        if self.normalize and self.numeric_columns:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numeric_columns])
        
        # Fit encoders for categorical columns
        if self.categorical_columns:
            if self.categorical_encoding == "label":
                for col in self.categorical_columns:
                    encoder = LabelEncoder()
                    encoder.fit(X[col].astype(str))
                    self.label_encoders[col] = encoder
            elif self.categorical_encoding == "onehot":
                self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.onehot_encoder.fit(X[self.categorical_columns].astype(str))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed numpy array
        """
        X = X.copy()
        
        # Handle missing values
        if self.handle_missing == "mean":
            X[self.numeric_columns] = X[self.numeric_columns].fillna(
                X[self.numeric_columns].mean()
            )
            X[self.categorical_columns] = X[self.categorical_columns].fillna(
                X[self.categorical_columns].mode().iloc[0] if len(X[self.categorical_columns]) > 0 else ''
            )
        elif self.handle_missing == "median":
            X[self.numeric_columns] = X[self.numeric_columns].fillna(
                X[self.numeric_columns].median()
            )
            X[self.categorical_columns] = X[self.categorical_columns].fillna(
                X[self.categorical_columns].mode().iloc[0] if len(X[self.categorical_columns]) > 0 else ''
            )
        elif self.handle_missing == "drop":
            X = X.dropna()
        
        # Transform numeric features
        if self.numeric_columns:
            if self.normalize and self.scaler:
                X_numeric = self.scaler.transform(X[self.numeric_columns])
            else:
                X_numeric = X[self.numeric_columns].values
        else:
            X_numeric = np.array([]).reshape(len(X), 0)
        
        # Transform categorical features
        if self.categorical_columns:
            if self.categorical_encoding == "label":
                X_categorical = np.column_stack([
                    self.label_encoders[col].transform(X[col].astype(str))
                    for col in self.categorical_columns
                ])
            elif self.categorical_encoding == "onehot":
                X_categorical = self.onehot_encoder.transform(X[self.categorical_columns].astype(str))
            else:
                X_categorical = np.array([]).reshape(len(X), 0)
        else:
            X_categorical = np.array([]).reshape(len(X), 0)
        
        # Concatenate features
        if X_numeric.size > 0 and X_categorical.size > 0:
            return np.concatenate([X_numeric, X_categorical], axis=1)
        elif X_numeric.size > 0:
            return X_numeric
        else:
            return X_categorical
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform data.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Transformed numpy array
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        names = []
        
        # Add numeric feature names
        names.extend(self.numeric_columns)
        
        # Add categorical feature names
        if self.categorical_columns:
            if self.categorical_encoding == "label":
                names.extend(self.categorical_columns)
            elif self.categorical_encoding == "onehot" and self.onehot_encoder:
                names.extend(self.onehot_encoder.get_feature_names_out(self.categorical_columns))
        
        return names
