"""Experiment tracking with MLflow."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Track experiments with MLflow."""
    
    def __init__(
        self,
        experiment_name: str = "ml_pipeline",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking URI
        """
        if not MLFLOW_AVAILABLE:
            print("Warning: MLflow not available. Install with: pip install mlflow")
            self.enabled = False
            return
        
        self.enabled = True
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new run.
        
        Args:
            run_name: Optional run name
        """
        if not self.enabled:
            return
        
        mlflow.start_run(run_name=run_name)
    
    def end_run(self):
        """End current run."""
        if not self.enabled:
            return
        
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return
        
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.enabled:
            return
        
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, artifact_path: str):
        """
        Log artifact.
        
        Args:
            artifact_path: Path to artifact
        """
        if not self.enabled:
            return
        
        mlflow.log_artifact(artifact_path)
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log model.
        
        Args:
            model: Model to log
            artifact_path: Artifact path
        """
        if not self.enabled:
            return
        
        mlflow.pytorch.log_model(model, artifact_path)


def get_tracker(
    experiment_name: str = "ml_pipeline",
    tracking_uri: Optional[str] = None
) -> ExperimentTracker:
    """
    Get experiment tracker.
    
    Args:
        experiment_name: Name of experiment
        tracking_uri: MLflow tracking URI
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(experiment_name, tracking_uri)
