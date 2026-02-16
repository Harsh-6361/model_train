"""Base adapter class for all data adapters."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseDataAdapter(ABC):
    """Abstract base class for all data adapters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config or {}
    
    @abstractmethod
    def load(self, path: str, **kwargs) -> Any:
        """
        Load data from source.
        
        Args:
            path: Path to data source
            **kwargs: Additional arguments
            
        Returns:
            Loaded data
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate loaded data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: Any, config: Optional[Dict] = None) -> Any:
        """
        Preprocess data according to config.
        
        Args:
            data: Data to preprocess
            config: Preprocessing configuration
            
        Returns:
            Preprocessed data
        """
        pass
    
    def split(self, data: Any, train_ratio: float = 0.7, val_ratio: float = 0.15, 
              test_ratio: float = 0.15, shuffle: bool = True, seed: int = 42) -> tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Data to split
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle data before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        pass
