"""Training callbacks."""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from ..observability.logger import get_logger

logger = get_logger(__name__)


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0
    ):
        """Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value: Optional[float] = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_train_begin(self, trainer: Any) -> None:
        """Reset state at training start."""
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Check if training should stop."""
        if self.monitor not in metrics:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        current_value = metrics[self.monitor]
        
        if self.best_value is None:
            self.best_value = current_value
            return
        
        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor}: {self.best_value:.4f}"
                )


class ModelCheckpoint(Callback):
    """Model checkpoint callback."""
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_last: bool = True
    ):
        """Initialize model checkpoint.
        
        Args:
            filepath: Path to save model
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to save only the best model
            save_last: Whether to save the last model
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        
        self.best_value: Optional[float] = None
        
        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, trainer: Any) -> None:
        """Reset state at training start."""
        self.best_value = None
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Save model if needed."""
        if self.monitor not in metrics:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        current_value = metrics[self.monitor]
        
        # Determine if this is the best model
        is_best = False
        if self.best_value is None:
            is_best = True
        elif self.mode == 'min':
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value
        
        # Save best model
        if is_best:
            self.best_value = current_value
            if self.save_best_only or not self.save_last:
                self._save_model(trainer, "best")
                logger.info(
                    f"Saved best model at epoch {epoch}. "
                    f"{self.monitor}: {current_value:.4f}"
                )
        
        # Save last model
        if self.save_last and not self.save_best_only:
            self._save_model(trainer, "last")
    
    def _save_model(self, trainer: Any, prefix: str) -> None:
        """Save model to file.
        
        Args:
            trainer: Trainer instance
            prefix: Filename prefix ('best' or 'last')
        """
        save_path = self.filepath.parent / f"{prefix}_model.pth"
        trainer.model.save(save_path)


class MetricsLogger(Callback):
    """Metrics logging callback."""
    
    def __init__(self, log_dir: Union[str, Path], log_interval: int = 10, save_interval: int = 5):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to save metrics
            log_interval: Interval for logging batches
            save_interval: Interval for saving metrics to disk (epochs)
        """
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "metrics.json"
        self.epoch_metrics: Dict[str, Any] = {}
        self._epochs_since_save = 0
    
    def on_train_begin(self, trainer: Any) -> None:
        """Initialize metrics file."""
        self.epoch_metrics = {'epochs': []}
        self._epochs_since_save = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Log epoch metrics."""
        epoch_data = {
            'epoch': epoch,
            'metrics': metrics
        }
        self.epoch_metrics['epochs'].append(epoch_data)
        self._epochs_since_save += 1
        
        # Save to file periodically or on final epoch
        if self._epochs_since_save >= self.save_interval or epoch == trainer.epochs - 1:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.epoch_metrics, f, indent=2)
            self._epochs_since_save = 0
        
        # Log to console
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch}: {metrics_str}")
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Log batch loss."""
        if batch_idx % self.log_interval == 0:
            logger.debug(f"Batch {batch_idx}: loss = {loss:.4f}")
    
    def on_train_end(self, trainer: Any) -> None:
        """Save final metrics on training end."""
        if self._epochs_since_save > 0:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.epoch_metrics, f, indent=2)


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: list):
        """Initialize callback list.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks
    
    def on_train_begin(self, trainer: Any) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)
    
    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, trainer)
    
    def should_stop_training(self) -> bool:
        """Check if any callback requests training to stop."""
        return any(
            getattr(callback, 'should_stop', False)
            for callback in self.callbacks
        )
