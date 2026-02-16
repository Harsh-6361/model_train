"""FastAPI serving for model inference."""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from ..inference.predictor import Predictor
from ..observability.health_check import HealthCheck
from ..observability.logger import get_logger, setup_logging
from ..observability.metrics_collector import MetricsCollector, Timer

logger = get_logger(__name__)


# Pydantic models for API
class TabularPredictionRequest(BaseModel):
    """Request model for tabular prediction."""
    features: List[List[float]]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Dict[str, Any]]
    model_type: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool


# Global state
app_state = {
    'predictor': None,
    'model_type': None,
    'metrics_collector': None,
    'health_checker': None
}


# Create FastAPI app
app = FastAPI(
    title="Model Training API",
    description="Multi-modal ML model serving API",
    version="0.1.0"
)


def create_app(
    model_path: str,
    model: Any,
    model_type: str,
    preprocessor: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
    enable_metrics: bool = True
) -> FastAPI:
    """Create and configure FastAPI app.
    
    Args:
        model_path: Path to model checkpoint
        model: Model instance
        model_type: Type of model ('tabular' or 'vision')
        preprocessor: Data preprocessor
        class_names: List of class names
        enable_metrics: Whether to enable Prometheus metrics
        
    Returns:
        Configured FastAPI app
    """
    # Create predictor
    predictor = Predictor.from_checkpoint(
        checkpoint_path=model_path,
        model=model,
        preprocessor=preprocessor,
        class_names=class_names
    )
    
    # Create metrics collector
    metrics_collector = MetricsCollector(enable_prometheus=enable_metrics)
    
    # Create health checker
    health_checker = HealthCheck(model_path=model_path)
    
    # Update app state
    app_state['predictor'] = predictor
    app_state['model_type'] = model_type
    app_state['metrics_collector'] = metrics_collector
    app_state['health_checker'] = health_checker
    
    logger.info(f"FastAPI app initialized with {model_type} model")
    
    return app


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "message": "Model Training API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict_tabular": "/predict/tabular (POST)",
            "predict_image": "/predict/image (POST)",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if app_state['health_checker'] is None:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    health = app_state['health_checker'].get_full_health()
    
    return HealthResponse(
        status=health['status'],
        timestamp=health['timestamp'],
        model_loaded=app_state['predictor'] is not None
    )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    if app_state['metrics_collector'] is None:
        return {"message": "Metrics collection not enabled"}
    
    return app_state['metrics_collector'].get_all_metrics()


@app.post("/predict/tabular", response_model=PredictionResponse)
async def predict_tabular(request: TabularPredictionRequest):
    """Predict on tabular data.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response
    """
    if app_state['predictor'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if app_state['model_type'] != 'tabular':
        raise HTTPException(
            status_code=400,
            detail=f"Model type is {app_state['model_type']}, not tabular"
        )
    
    try:
        # Track request
        if app_state['metrics_collector']:
            app_state['metrics_collector'].increment_counter('predict_tabular_requests')
        
        # Convert features to numpy array
        X = np.array(request.features)
        
        # Predict with timing
        timer = Timer(app_state['metrics_collector'], 'predict_tabular_latency')
        with timer:
            results = app_state['predictor'].predict_with_labels(X, model_type='tabular')
        
        inference_time_ms = timer.elapsed * 1000
        
        return PredictionResponse(
            predictions=results,
            model_type='tabular',
            inference_time_ms=inference_time_ms
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        if app_state['metrics_collector']:
            app_state['metrics_collector'].increment_counter('predict_tabular_errors')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict on image data.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction response
    """
    if app_state['predictor'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if app_state['model_type'] != 'vision':
        raise HTTPException(
            status_code=400,
            detail=f"Model type is {app_state['model_type']}, not vision"
        )
    
    try:
        # Track request
        if app_state['metrics_collector']:
            app_state['metrics_collector'].increment_counter('predict_image_requests')
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Predict with timing
        timer = Timer(app_state['metrics_collector'], 'predict_image_latency')
        with timer:
            results = app_state['predictor'].predict_with_labels(image, model_type='vision')
        
        inference_time_ms = timer.elapsed * 1000
        
        return PredictionResponse(
            predictions=results,
            model_type='vision',
            inference_time_ms=inference_time_ms
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        if app_state['metrics_collector']:
            app_state['metrics_collector'].increment_counter('predict_image_errors')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch/tabular")
async def predict_batch_tabular(request: TabularPredictionRequest):
    """Batch predict on tabular data.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Batch prediction response
    """
    if app_state['predictor'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if app_state['model_type'] != 'tabular':
        raise HTTPException(status_code=400, detail="Model is not tabular")
    
    try:
        X = np.array(request.features)
        
        timer = Timer()
        with timer:
            predictions = app_state['predictor'].batch_predict(
                X,
                batch_size=32,
                model_type='tabular'
            )
        
        return {
            'predictions': predictions.tolist(),
            'num_samples': len(predictions),
            'inference_time_ms': timer.elapsed * 1000
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def serve(
    model_path: str,
    model: Any,
    model_type: str,
    preprocessor: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    enable_metrics: bool = True
):
    """Start serving the model.
    
    Args:
        model_path: Path to model checkpoint
        model: Model instance
        model_type: Type of model
        preprocessor: Data preprocessor
        class_names: List of class names
        host: Host address
        port: Port number
        log_level: Logging level
        enable_metrics: Whether to enable metrics
    """
    # Setup logging
    setup_logging(log_level=log_level.upper())
    
    # Initialize app
    create_app(
        model_path=model_path,
        model=model,
        model_type=model_type,
        preprocessor=preprocessor,
        class_names=class_names,
        enable_metrics=enable_metrics
    )
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run server
    uvicorn.run(app, host=host, port=port, log_level=log_level)
