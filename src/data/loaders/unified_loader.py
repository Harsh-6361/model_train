"""Unified data loader for all data types."""
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

from ..adapters.csv_adapter import CSVAdapter
from ..adapters.image_adapter import ImageAdapter
from ..adapters.yolo_adapter import YOLOAdapter
from ..preprocessors.tabular_preprocessor import TabularPreprocessor


class UnifiedLoader:
    """Unified data loader for different data types."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize unified loader.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.adapters = {
            'tabular': CSVAdapter(config.get('data', {}).get('tabular', {})),
            'image': ImageAdapter(config.get('data', {}).get('image', {})),
            'yolo': YOLOAdapter(config.get('data', {}).get('yolo', {}))
        }
    
    def get_loaders(
        self,
        model_type: str,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for specified model type.
        
        Args:
            model_type: Type of model ('tabular', 'image', 'yolo')
            batch_size: Batch size (overrides config)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if model_type == 'tabular':
            return self._get_tabular_loaders(batch_size)
        elif model_type == 'image':
            return self._get_image_loaders(batch_size)
        elif model_type == 'yolo':
            return self._get_yolo_loaders(batch_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_tabular_loaders(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get tabular data loaders."""
        adapter = self.adapters['tabular']
        config = self.config.get('data', {}).get('tabular', {})
        
        # Load data
        data_path = config.get('path')
        if not data_path or not Path(data_path).exists():
            raise FileNotFoundError(f"Tabular data not found at: {data_path}")
        
        df = adapter.load(data_path)
        adapter.validate(df)
        
        # Preprocess
        df = adapter.preprocess(df)
        
        # Split data
        train_ratio = config.get('train_split', 0.7)
        val_ratio = config.get('val_split', 0.15)
        test_ratio = config.get('test_split', 0.15)
        
        train_df, val_df, test_df = adapter.split(
            df, train_ratio, val_ratio, test_ratio
        )
        
        # Get features and targets
        X_train, y_train = adapter.get_features_and_target(train_df)
        X_val, y_val = adapter.get_features_and_target(val_df)
        X_test, y_test = adapter.get_features_and_target(test_df)
        
        # Preprocess features
        preprocessor = TabularPreprocessor(config.get('preprocessing', {}))
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train.values)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val.values)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test.values)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Get batch size
        batch_size = batch_size or self.config.get('dataloader', {}).get('batch_size', 32)
        num_workers = self.config.get('dataloader', {}).get('num_workers', 4)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for TensorDataset
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _get_image_loaders(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get image data loaders."""
        adapter = self.adapters['image']
        config = self.config.get('data', {}).get('image', {})
        
        # Load data
        data_path = config.get('path')
        if not data_path or not Path(data_path).exists():
            # Return dummy loaders if no data
            raise FileNotFoundError(f"Image data not found at: {data_path}")
        
        data = adapter.load(data_path)
        adapter.validate(data)
        
        # Split data
        train_ratio = config.get('train_split', 0.7)
        val_ratio = config.get('val_split', 0.15)
        test_ratio = config.get('test_split', 0.15)
        
        train_data, val_data, test_data = adapter.split(
            data, train_ratio, val_ratio, test_ratio
        )
        
        # Create datasets with augmentation for training
        train_config = config.copy()
        train_config['augmentation'] = train_config.get('augmentation', {})
        train_config['augmentation']['enabled'] = True
        
        val_test_config = config.copy()
        val_test_config['augmentation'] = {'enabled': False, 'normalize': True}
        
        train_dataset = adapter.preprocess(train_data, train_config)
        val_dataset = adapter.preprocess(val_data, val_test_config)
        test_dataset = adapter.preprocess(test_data, val_test_config)
        
        # Get batch size
        batch_size = batch_size or self.config.get('dataloader', {}).get('batch_size', 32)
        num_workers = self.config.get('dataloader', {}).get('num_workers', 4)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _get_yolo_loaders(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Get YOLO data loaders.
        
        Note: YOLO uses its own data loading mechanism through ultralytics.
        This returns configuration for YOLO training.
        """
        config = self.config.get('data', {}).get('yolo', {})
        
        # Return data config path for YOLO
        # YOLO handles its own data loading
        return config, None, None
