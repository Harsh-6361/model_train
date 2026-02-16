"""Large-scale data loader with optimization."""
import torch
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader, DistributedSampler

from .unified_loader import UnifiedLoader


class LargeScaleLoader(UnifiedLoader):
    """Data loader optimized for large-scale training."""
    
    def __init__(self, config: Dict[str, Any], distributed: bool = False, rank: int = 0):
        """
        Initialize large-scale loader.
        
        Args:
            config: Data configuration
            distributed: Whether using distributed training
            rank: Process rank for distributed training
        """
        super().__init__(config)
        self.distributed = distributed
        self.rank = rank
        self.large_scale_config = config.get('large_scale', {})
    
    def get_loaders(
        self,
        model_type: str,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get optimized data loaders.
        
        Args:
            model_type: Type of model
            batch_size: Batch size (overrides config)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Get base loaders
        train_loader, val_loader, test_loader = super().get_loaders(model_type, batch_size)
        
        if model_type == 'yolo':
            # YOLO handles its own optimization
            return train_loader, val_loader, test_loader
        
        # Apply optimizations for other types
        pipeline_config = self.large_scale_config.get('data_pipeline', {})
        
        # Get optimized parameters
        num_workers = pipeline_config.get('num_workers', 8)
        prefetch_factor = pipeline_config.get('prefetch_factor', 2)
        persistent_workers = pipeline_config.get('persistent_workers', True)
        pin_memory = pipeline_config.get('pin_memory', True)
        
        # Recreate train loader with optimizations
        if self.distributed:
            # Use DistributedSampler for distributed training
            train_sampler = DistributedSampler(
                train_loader.dataset,
                shuffle=True,
                seed=42
            )
            
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers if num_workers > 0 else False
            )
        else:
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers if num_workers > 0 else False
            )
        
        # Recreate val loader
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )
        
        # Recreate test loader
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )
        
        return train_loader, val_loader, test_loader
