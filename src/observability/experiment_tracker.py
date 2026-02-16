"""
Experiment Tracking Integration

Support for:
- MLflow
- Weights & Biases
- TensorBoard
- Custom logging
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json


class ExperimentTracker:
    """
    Unified experiment tracking interface
    
    Supports multiple backends:
    - MLflow
    - Weights & Biases (wandb)
    - TensorBoard
    - Custom JSON logging
    """
    
    def __init__(
        self,
        experiment_name: str,
        backend: str = 'wandb',
        project: str = 'model_train',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            backend: Tracking backend ('wandb', 'mlflow', 'tensorboard', 'custom')
            project: Project name
            config: Optional configuration to log
        """
        self.experiment_name = experiment_name
        self.backend = backend.lower()
        self.project = project
        self.config = config or {}
        
        self._run = None
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup the specified tracking backend"""
        if self.backend == 'wandb':
            self._setup_wandb()
        elif self.backend == 'mlflow':
            self._setup_mlflow()
        elif self.backend == 'tensorboard':
            self._setup_tensorboard()
        elif self.backend == 'custom':
            self._setup_custom()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases"""
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                name=self.experiment_name,
                config=self.config,
                resume='allow'
            )
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")
    
    def _setup_mlflow(self):
        """Setup MLflow"""
        try:
            import mlflow
            mlflow.set_experiment(self.project)
            self._run = mlflow.start_run(run_name=self.experiment_name)
            
            # Log config
            if self.config:
                mlflow.log_params(self.config)
        except ImportError:
            raise ImportError("mlflow not installed. Install with: pip install mlflow")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path('artifacts/tensorboard') / self.project / self.experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
            self._run = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            raise ImportError("tensorboard not installed. Install with: pip install tensorboard")
    
    def _setup_custom(self):
        """Setup custom JSON logging"""
        log_dir = Path('artifacts/logs') / self.project
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / f"{self.experiment_name}.jsonl"
        self._run = {'experiment': self.experiment_name, 'project': self.project}
        
        # Log config
        if self.config:
            self._write_log({'type': 'config', 'data': self.config})
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            params: Dictionary of parameters
        """
        if self.backend == 'wandb':
            import wandb
            wandb.config.update(params)
        elif self.backend == 'mlflow':
            import mlflow
            mlflow.log_params(params)
        elif self.backend == 'tensorboard':
            # TensorBoard doesn't have direct param logging
            # Store as text or hparams
            pass
        elif self.backend == 'custom':
            self._write_log({'type': 'params', 'data': params})
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/iteration number
        """
        if self.backend == 'wandb':
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == 'mlflow':
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        elif self.backend == 'tensorboard':
            if step is None:
                step = 0
            for key, value in metrics.items():
                self._run.add_scalar(key, value, step)
        elif self.backend == 'custom':
            self._write_log({'type': 'metrics', 'step': step, 'data': metrics})
    
    def log_artifact(self, artifact_path: str, artifact_type: str = 'file'):
        """
        Log an artifact (file, model, etc.)
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
        """
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        if self.backend == 'wandb':
            import wandb
            wandb.save(str(artifact_path))
        elif self.backend == 'mlflow':
            import mlflow
            if artifact_path.is_dir():
                mlflow.log_artifacts(str(artifact_path))
            else:
                mlflow.log_artifact(str(artifact_path))
        elif self.backend == 'tensorboard':
            # TensorBoard doesn't support artifacts directly
            pass
        elif self.backend == 'custom':
            self._write_log({
                'type': 'artifact',
                'artifact_type': artifact_type,
                'path': str(artifact_path)
            })
    
    def log_model(self, model_path: str, signature=None, metadata: Optional[Dict] = None):
        """
        Log a trained model
        
        Args:
            model_path: Path to model file
            signature: Model signature (for MLflow)
            metadata: Optional metadata
        """
        model_path = Path(model_path)
        
        if self.backend == 'wandb':
            import wandb
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}-model",
                type='model',
                metadata=metadata
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
        elif self.backend == 'mlflow':
            import mlflow
            if signature:
                mlflow.log_model(model_path, "model", signature=signature)
            else:
                mlflow.log_artifact(str(model_path))
        elif self.backend == 'tensorboard':
            pass
        elif self.backend == 'custom':
            self._write_log({
                'type': 'model',
                'path': str(model_path),
                'metadata': metadata
            })
    
    def get_best_metric(self, metric_name: str) -> Optional[float]:
        """
        Get the best value for a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Best metric value or None
        """
        if self.backend == 'wandb':
            import wandb
            if wandb.run:
                return wandb.run.summary.get(f"best_{metric_name}")
        elif self.backend == 'mlflow':
            import mlflow
            # This would require querying the MLflow tracking server
            # For simplicity, return None
            return None
        elif self.backend == 'custom':
            # Read from log file
            return self._get_best_from_logs(metric_name)
        
        return None
    
    def _get_best_from_logs(self, metric_name: str) -> Optional[float]:
        """Get best metric from custom logs"""
        if not hasattr(self, '_log_file') or not self._log_file.exists():
            return None
        
        best_value = None
        with open(self._log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if log_entry.get('type') == 'metrics':
                        metrics = log_entry.get('data', {})
                        if metric_name in metrics:
                            value = metrics[metric_name]
                            if best_value is None or value > best_value:
                                best_value = value
                except json.JSONDecodeError:
                    continue
        
        return best_value
    
    def _write_log(self, log_entry: Dict[str, Any]):
        """Write log entry to custom log file"""
        if not hasattr(self, '_log_file'):
            return
        
        with open(self._log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def finish(self):
        """Finish the experiment run"""
        if self.backend == 'wandb':
            import wandb
            wandb.finish()
        elif self.backend == 'mlflow':
            import mlflow
            mlflow.end_run()
        elif self.backend == 'tensorboard':
            if self._run:
                self._run.close()
        elif self.backend == 'custom':
            self._write_log({'type': 'finish', 'experiment': self.experiment_name})
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()


def create_tracker(
    experiment_name: str,
    backend: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> ExperimentTracker:
    """
    Convenience function to create an experiment tracker
    
    Args:
        experiment_name: Name of the experiment
        backend: Tracking backend (auto-detect if None)
        config: Optional configuration
    
    Returns:
        ExperimentTracker instance
    """
    if backend is None:
        # Auto-detect backend based on environment
        if os.environ.get('WANDB_API_KEY'):
            backend = 'wandb'
        elif os.environ.get('MLFLOW_TRACKING_URI'):
            backend = 'mlflow'
        else:
            backend = 'custom'
    
    return ExperimentTracker(
        experiment_name=experiment_name,
        backend=backend,
        config=config
    )
