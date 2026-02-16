"""Training callbacks for monitoring and control."""
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Check for early stopping."""
        if logs is None or self.monitor not in logs:
            return
        
        current = logs[self.monitor]
        
        if self.best_value is None:
            self.best_value = current
        elif self.monitor_op(current, self.best_value):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}")


class ModelCheckpoint(Callback):
    """Model checkpoint callback."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best: Whether to save best model
            save_last: Whether to save last model
            verbose: Whether to print messages
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_value = None
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
        else:
            self.monitor_op = lambda a, b: a > b
        
        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint if improved."""
        if logs is None:
            return
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.filepath.parent / 'last.pt'
            trainer.model.save(str(last_path))
        
        # Save best checkpoint
        if self.save_best and self.monitor in logs:
            current = logs[self.monitor]
            
            if self.best_value is None or self.monitor_op(current, self.best_value):
                self.best_value = current
                best_path = self.filepath.parent / 'best.pt'
                trainer.model.save(str(best_path))
                
                if self.verbose:
                    print(f"\nSaved best model to {best_path} ({self.monitor}={current:.4f})")


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""
    
    def __init__(self, scheduler):
        """
        Initialize scheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate."""
        if hasattr(self.scheduler, 'step'):
            # For ReduceLROnPlateau, pass metric value
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                if logs and 'val_loss' in logs:
                    self.scheduler.step(logs['val_loss'])
            else:
                self.scheduler.step()


class MetricsLogger(Callback):
    """Log metrics to file."""
    
    def __init__(self, log_dir: str = 'artifacts/logs', log_file: str = 'metrics.json'):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            log_file: Name of log file
        """
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.history = []
        
        # Ensure directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Log metrics."""
        if logs is None:
            return
        
        # Add epoch number
        log_entry = {'epoch': epoch, **logs}
        self.history.append(log_entry)
        
        # Save to file
        log_path = self.log_dir / self.log_file
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def on_train_end(self, trainer, logs: Optional[Dict] = None):
        """Save final history."""
        log_path = self.log_dir / self.log_file
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class ProgressPrinter(Callback):
    """Print training progress."""
    
    def __init__(self, print_every: int = 1):
        """
        Initialize progress printer.
        
        Args:
            print_every: Print every N epochs
        """
        self.print_every = print_every
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Print progress."""
        if epoch % self.print_every == 0:
            if logs:
                metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in logs.items()])
                print(f"Epoch {epoch:03d}: {metrics_str}")


class CallbackList:
    """Container for callbacks."""
    
    def __init__(self, callbacks: Optional[list] = None):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """Add callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer, logs: Optional[Dict] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer, logs)
    
    def on_train_end(self, trainer, logs: Optional[Dict] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer, logs)
    
    def on_epoch_begin(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch, logs)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)
    
    def on_batch_begin(self, trainer, batch_idx: int, logs: Optional[Dict] = None):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx, logs)
    
    def on_batch_end(self, trainer, batch_idx: int, logs: Optional[Dict] = None):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, logs)
