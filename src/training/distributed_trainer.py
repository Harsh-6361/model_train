"""
Distributed Training for Large-Scale Datasets

Features:
- Multi-GPU training (DataParallel, DistributedDataParallel)
- Gradient accumulation for effective large batch sizes
- Mixed precision training (AMP)
- Checkpoint resume capability
- Memory-efficient data loading
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class DistributedTrainer:
    """
    Distributed trainer for large-scale training
    
    Features:
    - Multi-GPU support
    - Mixed precision training
    - Gradient accumulation
    - Automatic checkpointing
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        world_size: Optional[int] = None
    ):
        """
        Initialize distributed trainer
        
        Args:
            config: Training configuration
            model: Model to train
            world_size: Number of processes (defaults to GPU count)
        """
        self.config = config
        self.model = model
        self.world_size = world_size or torch.cuda.device_count()
        
        # Training settings
        training_config = config.get('large_scale_training', {})
        self.hardware = training_config.get('hardware', {})
        self.memory_config = training_config.get('memory', {})
        self.training_loop = training_config.get('training_loop', {})
        self.fault_tolerance = training_config.get('fault_tolerance', {})
        
        # Setup distributed training
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_distributed = self.world_size > 1
        
        # Device setup
        self.device = self._setup_device()
        
        # Mixed precision
        self.use_amp = self.hardware.get('precision', 'mixed') in ['mixed', 'float16']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.memory_config.get('gradient_accumulation_steps', 1)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.fault_tolerance.get('checkpoint_dir', 'artifacts/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = self.training_loop.get('checkpoint_interval', 1000)
        
        # Model compilation (PyTorch 2.0+)
        if self.hardware.get('compile_model', False):
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Model compilation not supported: {e}")
        
        # Metrics
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')  # For loss (lower is better)
    
    def _setup_device(self) -> torch.device:
        """Setup device for training"""
        if torch.cuda.is_available():
            if self.is_distributed:
                device = torch.device(f'cuda:{self.local_rank}')
            else:
                device = torch.device('cuda:0')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        
        return device
    
    def setup_distributed(self):
        """Initialize distributed process group"""
        if not self.is_distributed:
            return
        
        if not dist.is_initialized():
            # Initialize process group
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
        
        # Wrap model in DDP
        self.model = self.model.to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None
        )
    
    def train_epoch(
        self,
        data_loader,
        optimizer,
        criterion,
        scheduler=None,
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            data_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            callbacks: Optional callbacks dict
        
        Returns:
            Epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar (only on rank 0)
        if self.rank == 0:
            pbar = tqdm(data_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = data_loader
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if isinstance(batch, (tuple, list)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    if isinstance(batch, (tuple, list)):
                        outputs = self.model(*batch[:-1])
                        loss = criterion(outputs, batch[-1])
                    elif isinstance(batch, dict):
                        outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                        loss = criterion(outputs, batch.get('labels'))
                    else:
                        outputs = self.model(batch)
                        loss = criterion(outputs, batch)
                    
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                # Regular forward/backward
                if isinstance(batch, (tuple, list)):
                    outputs = self.model(*batch[:-1])
                    loss = criterion(outputs, batch[-1])
                elif isinstance(batch, dict):
                    outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                    loss = criterion(outputs, batch.get('labels'))
                else:
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({'loss': total_loss / num_batches})
            
            # Checkpointing
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch=self.epoch,
                    metrics={'loss': total_loss / num_batches},
                    optimizer=optimizer,
                    scheduler=scheduler
                )
            
            # Callbacks
            if callbacks and 'on_batch_end' in callbacks:
                callbacks['on_batch_end'](
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    global_step=self.global_step
                )
        
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'global_step': self.global_step
        }
        
        self.epoch += 1
        
        return epoch_metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        optimizer=None,
        scheduler=None,
        is_best: bool = False
    ):
        """
        Save training checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            optimizer: Optimizer state
            scheduler: Scheduler state
            is_best: Whether this is the best model
        """
        if self.rank != 0:
            return  # Only save on rank 0
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
        
        # Keep only last N checkpoints
        keep_last_n = self.fault_tolerance.get('keep_last_n', 5)
        self._cleanup_old_checkpoints(keep_last_n)
    
    def _cleanup_old_checkpoints(self, keep_last_n: int):
        """Remove old checkpoints, keeping only the last N"""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
    
    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        optimizer=None,
        scheduler=None
    ) -> Dict[str, Any]:
        """
        Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint (defaults to latest)
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
        
        Returns:
            Checkpoint data
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Resumed from checkpoint: epoch {self.epoch}, step {self.global_step}")
        
        return checkpoint
    
    def train(
        self,
        train_loader,
        val_loader=None,
        optimizer=None,
        criterion=None,
        scheduler=None,
        num_epochs: int = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs to train
            callbacks: Optional callbacks
        
        Returns:
            Training metrics
        """
        if num_epochs is None:
            num_epochs = self.training_loop.get('epochs', 100)
        
        # Auto-resume if enabled
        if self.fault_tolerance.get('auto_resume', False):
            try:
                self.resume_from_checkpoint(optimizer=optimizer, scheduler=scheduler)
            except FileNotFoundError:
                print("No checkpoint found, starting from scratch")
        
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        for epoch in range(self.epoch, num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler,
                callbacks
            )
            
            metrics_history['train_loss'].append(train_metrics['loss'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, criterion)
                metrics_history['val_loss'].append(val_metrics['loss'])
                
                # Check if best model
                is_best = val_metrics['loss'] < self.best_metric
                if is_best:
                    self.best_metric = val_metrics['loss']
                
                # Save checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    optimizer=optimizer,
                    scheduler=scheduler,
                    is_best=is_best
                )
            
            # Callbacks
            if callbacks and 'on_epoch_end' in callbacks:
                callbacks['on_epoch_end'](
                    epoch=epoch,
                    metrics={**train_metrics, **(val_metrics if val_loader else {})}
                )
        
        return metrics_history
    
    def validate(self, val_loader, criterion) -> Dict[str, float]:
        """
        Validation loop
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(batch, (tuple, list)):
                    outputs = self.model(*batch[:-1])
                    loss = criterion(outputs, batch[-1])
                elif isinstance(batch, dict):
                    outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                    loss = criterion(outputs, batch.get('labels'))
                else:
                    outputs = self.model(batch)
                    loss = criterion(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
