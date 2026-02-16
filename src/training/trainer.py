"""Unified trainer for all model types."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.base_model import BaseModel
from ..utils.helpers import get_device, set_seed
from .callbacks import (
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    MetricsLogger,
    ProgressPrinter
)
from .metrics import MetricsTracker


class Trainer:
    """Unified trainer for all model types."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        callbacks: Optional[List] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            callbacks: List of callbacks
        """
        self.model = model
        self.config = config.get('training', config)  # Handle nested or flat config
        self.callbacks = CallbackList(callbacks or [])
        
        # Training state
        self.current_epoch = 0
        self.stop_training = False
        
        # Set seed for reproducibility
        seed = self.config.get('seed', 42)
        set_seed(seed)
        
        # Get device
        device_str = self.config.get('device', 'auto')
        self.device = get_device(device_str)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup callbacks
        self._setup_default_callbacks()
        
        # Mixed precision
        self.use_amp = self.config.get('mixed_precision', {}).get('enabled', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.config.get('optimizer_params', {}).get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'none').lower()
        
        if scheduler_name == 'none':
            return None
        
        if scheduler_name == 'cosine':
            T_max = self.config.get('scheduler_params', {}).get('T_max', 100)
            eta_min = self.config.get('scheduler_params', {}).get('eta_min', 1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_name == 'step':
            step_size = self.config.get('scheduler_params', {}).get('step_size', 30)
            gamma = self.config.get('scheduler_params', {}).get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'exponential':
            gamma = self.config.get('scheduler_params', {}).get('gamma', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        elif scheduler_name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.1
            )
        else:
            return None
    
    def _setup_default_callbacks(self):
        """Setup default callbacks."""
        # Early stopping
        early_stopping_config = self.config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            self.callbacks.append(EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 10),
                mode=early_stopping_config.get('mode', 'min'),
                min_delta=early_stopping_config.get('min_delta', 0.0001)
            ))
        
        # Model checkpoint
        checkpoint_config = self.config.get('checkpoint', {})
        if checkpoint_config.get('enabled', True):
            save_path = Path(checkpoint_config.get('path', 'artifacts/models'))
            self.callbacks.append(ModelCheckpoint(
                filepath=str(save_path / 'model.pt'),
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                mode=checkpoint_config.get('mode', 'min'),
                save_best=checkpoint_config.get('save_best', True),
                save_last=checkpoint_config.get('save_last', True)
            ))
        
        # Learning rate scheduler
        if self.scheduler is not None:
            self.callbacks.append(LearningRateScheduler(self.scheduler))
        
        # Metrics logger
        self.callbacks.append(MetricsLogger())
        
        # Progress printer
        self.callbacks.append(ProgressPrinter())
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.get('epochs', 100)
        
        # Call on_train_begin
        self.callbacks.on_train_begin(self)
        
        # Training loop
        for epoch in range(epochs):
            if self.stop_training:
                break
            
            self.current_epoch = epoch
            
            # Call on_epoch_begin
            self.callbacks.on_epoch_begin(self, epoch)
            
            # Train epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Combine metrics
            epoch_metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            # Call on_epoch_end
            self.callbacks.on_epoch_end(self, epoch, epoch_metrics)
        
        # Call on_train_end
        self.callbacks.on_train_end(self)
        
        return {}
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricsTracker()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = tuple(x.to(self.device) for x in batch)
            
            # Call on_batch_begin
            self.callbacks.on_batch_begin(self, batch_idx)
            
            # Forward and backward
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    metrics = self.model.training_step(batch, batch_idx)
                    loss = metrics['loss']
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', {}).get('enabled', False):
                    self.scaler.unscale_(self.optimizer)
                    max_norm = self.config['gradient_clip'].get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                metrics = self.model.training_step(batch, batch_idx)
                loss = metrics['loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', {}).get('enabled', False):
                    max_norm = self.config['gradient_clip'].get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
            
            # Update metrics
            tracker.update(metrics, n=len(batch[0]))
            
            # Update progress bar
            pbar.set_postfix(tracker.get_metrics())
            
            # Call on_batch_end
            self.callbacks.on_batch_end(self, batch_idx)
        
        return tracker.get_metrics()
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        tracker = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = tuple(x.to(self.device) for x in batch)
                
                # Forward pass
                metrics = self.model.validation_step(batch, batch_idx)
                
                # Update metrics
                tracker.update(metrics, n=len(batch[0]))
                
                # Update progress bar
                pbar.set_postfix(tracker.get_metrics())
        
        return tracker.get_metrics()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        self.model.eval()
        tracker = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = tuple(x.to(self.device) for x in batch)
                
                # Forward pass
                metrics = self.model.test_step(batch, batch_idx)
                
                # Update metrics
                tracker.update(metrics, n=len(batch[0]))
                
                # Update progress bar
                pbar.set_postfix(tracker.get_metrics())
        
        return tracker.get_metrics()
