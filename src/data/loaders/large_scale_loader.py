"""
Large-Scale Data Loading

Features:
- Streaming data loading (no full dataset in memory)
- Prefetching and caching
- WebDataset support for sharded data
- On-the-fly augmentation
- Dynamic batching
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import warnings


class LargeScaleDataLoader:
    """
    Memory-efficient data loader for large-scale datasets
    
    Features:
    - Streaming mode for datasets larger than RAM
    - Efficient prefetching
    - Persistent workers
    - Sharded data support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize large-scale data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Data pipeline settings
        data_pipeline = config.get('large_scale_training', {}).get('data_pipeline', {})
        
        self.num_workers = data_pipeline.get('num_workers', 8)
        self.persistent_workers = data_pipeline.get('persistent_workers', True)
        self.prefetch_factor = data_pipeline.get('prefetch_factor', 2)
        self.streaming = data_pipeline.get('streaming', False)
        self.cache_strategy = data_pipeline.get('cache_strategy', 'none')
        self.shard_size = data_pipeline.get('shard_size', 10000)
        
        # Memory optimization
        memory_config = config.get('large_scale_training', {}).get('memory', {})
        self.pin_memory = memory_config.get('pin_memory', True) and torch.cuda.is_available()
    
    def create_loader(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create a standard data loader
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collate function
        
        Returns:
            DataLoader instance
        """
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': drop_last,
        }
        
        # Add prefetch factor if workers > 0
        if self.num_workers > 0:
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['persistent_workers'] = self.persistent_workers
        
        # Shuffle only for map-style datasets
        if not isinstance(dataset, IterableDataset):
            loader_kwargs['shuffle'] = shuffle
        
        if collate_fn is not None:
            loader_kwargs['collate_fn'] = collate_fn
        
        return DataLoader(dataset, **loader_kwargs)
    
    def create_streaming_loader(
        self,
        data_path: Union[str, Path],
        batch_size: int,
        transform: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create memory-efficient streaming loader
        
        Args:
            data_path: Path to data directory
            batch_size: Batch size
            transform: Optional transform function
        
        Returns:
            Streaming DataLoader
        """
        dataset = StreamingDataset(data_path, transform=transform)
        
        return self.create_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Streaming datasets don't support shuffle
            collate_fn=None
        )
    
    def create_sharded_loader(
        self,
        shards_pattern: str,
        batch_size: int,
        transform: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create loader for sharded datasets (WebDataset format)
        
        Args:
            shards_pattern: Pattern for shard files (e.g., 'data/train-{000000..000099}.tar')
            batch_size: Batch size
            transform: Optional transform function
        
        Returns:
            DataLoader for sharded data
        """
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError(
                "WebDataset not installed. Install with: pip install webdataset"
            )
        
        # Create WebDataset
        dataset = wds.WebDataset(shards_pattern)
        
        if transform is not None:
            dataset = dataset.map(transform)
        
        # Batch the dataset
        dataset = dataset.batched(batch_size)
        
        loader_kwargs = {
            'batch_size': None,  # Batching handled by WebDataset
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        
        if self.num_workers > 0:
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['persistent_workers'] = self.persistent_workers
        
        return DataLoader(dataset, **loader_kwargs)
    
    def create_cached_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        cache_dir: Optional[str] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create loader with caching support
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            cache_dir: Directory for cache (if using disk cache)
            shuffle: Whether to shuffle data
        
        Returns:
            DataLoader with caching
        """
        if self.cache_strategy == 'memory':
            # Wrap dataset with memory caching
            dataset = MemoryCachedDataset(dataset)
        elif self.cache_strategy == 'disk' and cache_dir:
            # Wrap dataset with disk caching
            dataset = DiskCachedDataset(dataset, cache_dir)
        
        return self.create_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for memory-efficient loading
    
    Loads data on-the-fly without loading entire dataset into memory
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[Callable] = None
    ):
        """
        Initialize streaming dataset
        
        Args:
            data_path: Path to data directory
            transform: Optional transform function
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Get list of files
        self.file_list = sorted(self.data_path.glob('*'))
    
    def __iter__(self):
        """Iterate over dataset"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker
            file_list = self.file_list
        else:
            # Multiple workers - split files among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_list = self.file_list[worker_id::num_workers]
        
        for file_path in file_list:
            # Load and yield data
            try:
                data = self._load_file(file_path)
                if self.transform:
                    data = self.transform(data)
                yield data
            except Exception as e:
                warnings.warn(f"Error loading {file_path}: {e}")
                continue
    
    def _load_file(self, file_path: Path):
        """
        Load a single file
        
        Args:
            file_path: Path to file
        
        Returns:
            Loaded data
        """
        # This is a placeholder - implement based on your data format
        # For images, you might use PIL or cv2
        # For tensors, you might use torch.load
        import torch
        try:
            return torch.load(file_path)
        except:
            # Fallback for other formats
            return str(file_path)


class MemoryCachedDataset(Dataset):
    """
    Dataset wrapper that caches loaded samples in memory
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize memory-cached dataset
        
        Args:
            dataset: Base dataset to wrap
        """
        self.dataset = dataset
        self.cache = {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


class DiskCachedDataset(Dataset):
    """
    Dataset wrapper that caches loaded samples to disk
    """
    
    def __init__(self, dataset: Dataset, cache_dir: str):
        """
        Initialize disk-cached dataset
        
        Args:
            dataset: Base dataset to wrap
            cache_dir: Directory for cache files
        """
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        cache_path = self.cache_dir / f"sample_{idx}.pt"
        
        if cache_path.exists():
            # Load from cache
            return torch.load(cache_path)
        else:
            # Load from dataset and cache
            sample = self.dataset[idx]
            torch.save(sample, cache_path)
            return sample


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    config: Dict[str, Any],
    batch_size: int,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a data loader
    
    Args:
        dataset: PyTorch dataset
        config: Configuration dictionary
        batch_size: Batch size
        shuffle: Whether to shuffle
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        Configured DataLoader
    """
    loader_factory = LargeScaleDataLoader(config)
    return loader_factory.create_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )
