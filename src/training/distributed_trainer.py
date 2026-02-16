"""Distributed trainer for large-scale training."""
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .trainer import Trainer
from ..models.base_model import BaseModel


class DistributedTrainer(Trainer):
    """Trainer with distributed training support."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        callbacks: Optional[list] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            callbacks: List of callbacks
            rank: Process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        # Initialize parent
        super().__init__(model, config, callbacks)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[rank] if torch.cuda.is_available() else None
            )
    
    @staticmethod
    def setup(rank: int, world_size: int, backend: str = 'nccl'):
        """
        Setup distributed training.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Backend ('nccl', 'gloo')
        """
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    @staticmethod
    def cleanup():
        """Cleanup distributed training."""
        dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
