"""Unified trainer for model training."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.base_model import BaseModel
from ..observability.logger import get_logger
from ..utils.helpers import get_device, set_seed
from .callbacks import CallbackList
from .metrics import MetricsCalculator, MetricsTracker

logger = get_logger(__name__)


class Trainer:
    """Unified trainer for tabular and vision models."""
    
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Any]] = None,
        metric_names: Optional[List[str]] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            optimizer: Optimizer (if None, uses Adam)
            criterion: Loss function (if None, uses CrossEntropyLoss)
            device: Device to train on
            callbacks: List of callbacks
            metric_names: List of metrics to track
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        self.device = device if device else get_device()
        self.model = self.model.to(self.device)
        
        # Set optimizer
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters())
        
        # Set criterion
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        
        # Set callbacks
        self.callbacks = CallbackList(callbacks) if callbacks else CallbackList([])
        
        # Set metrics
        self.metric_names = metric_names or ['accuracy', 'precision', 'recall', 'f1']
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # Callback: batch begin
            self.callbacks.on_batch_begin(batch_idx, self)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                raise ValueError("Training requires targets")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': batch_loss})
            
            # Callback: batch end
            self.callbacks.on_batch_end(batch_idx, batch_loss, self)
        
        return total_loss / num_batches
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                # Get data
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Get predictions
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    # Store for metrics - use tensors for efficiency
                    all_preds.append(preds)
                    all_targets.append(targets)
                    all_probs.append(probs)
        
        # Calculate metrics
        metrics = {'val_loss': total_loss / len(loader)}
        
        if all_targets:
            # Concatenate tensors on GPU, then convert to numpy once
            all_preds = torch.cat(all_preds).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()
            all_probs = torch.cat(all_probs).cpu().numpy()
            
            # Calculate classification metrics
            calc_metrics = MetricsCalculator.calculate_all_metrics(
                all_targets,
                all_preds,
                all_probs,
                self.metric_names
            )
            
            # Add 'val_' prefix
            for key, value in calc_metrics.items():
                metrics[f'val_{key}'] = value
        
        return metrics
    
    def train(
        self,
        epochs: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            seed: Random seed for reproducibility
            
        Returns:
            Training history
        """
        # Set seed
        if seed is not None:
            set_seed(seed)
        
        # Callback: training begin
        self.callbacks.on_train_begin(self)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        try:
            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                
                # Callback: epoch begin
                self.callbacks.on_epoch_begin(epoch, self)
                
                # Train epoch
                train_loss = self.train_epoch()
                
                # Validation
                metrics = {'train_loss': train_loss}
                if self.val_loader is not None:
                    val_metrics = self.validate(self.val_loader)
                    metrics.update(val_metrics)
                
                # Track metrics
                self.metrics_tracker.update(metrics)
                
                # Callback: epoch end
                self.callbacks.on_epoch_end(epoch, metrics, self)
                
                # Check early stopping
                if self.callbacks.should_stop_training():
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Callback: training end
            self.callbacks.on_train_end(self)
        
        logger.info("Training completed")
        
        return {
            'history': self.metrics_tracker.to_dict(),
            'final_metrics': self.metrics_tracker.get_all_latest()
        }
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            loader: Data loader
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        metrics = self.validate(loader)
        
        # Remove 'val_' prefix for clarity
        eval_metrics = {
            key.replace('val_', ''): value
            for key, value in metrics.items()
        }
        
        logger.info(f"Evaluation metrics: {eval_metrics}")
        return eval_metrics


def create_trainer_from_config(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any]
) -> Trainer:
    """Create trainer from configuration.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Optional validation data loader
        config: Training configuration dictionary
        
    Returns:
        Configured trainer
    """
    from .callbacks import EarlyStopping, MetricsLogger, ModelCheckpoint
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    
    # Create optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create criterion
    loss_name = config.get('loss', 'cross_entropy').lower()
    if loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', {}).get('enabled', True):
        es_config = config['early_stopping']
        callbacks.append(EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            mode=es_config.get('mode', 'min')
        ))
    
    # Model checkpoint
    if config.get('checkpoint', {}).get('enabled', True):
        ckpt_config = config['checkpoint']
        callbacks.append(ModelCheckpoint(
            filepath=ckpt_config.get('path', 'artifacts/models/'),
            monitor=ckpt_config.get('monitor', 'val_loss'),
            mode=ckpt_config.get('mode', 'min'),
            save_best_only=ckpt_config.get('save_best', True),
            save_last=ckpt_config.get('save_last', True)
        ))
    
    # Metrics logger
    if config.get('logging', {}).get('enabled', True):
        log_config = config['logging']
        callbacks.append(MetricsLogger(
            log_dir=log_config.get('log_dir', 'artifacts/logs/'),
            log_interval=log_config.get('log_interval', 10)
        ))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        metric_names=config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
    )
    
    return trainer
