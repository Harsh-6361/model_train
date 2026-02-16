"""FastAPI serving for ML models."""
import io
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from .predictor import Predictor
from ..observability.health_check import get_health
from ..observability.metrics_collector import get_collector


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for tabular predictions."""
    data: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List
    inference_time: float
    model_type: str


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    model_path: str
    device: str
    parameters: Optional[int] = None


# Global predictor instance
_predictor: Optional[Predictor] = None


def create_app(
    model_path: str,
    model_type: str,
    config: Optional[Dict[str, Any]] = None
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        config: Model configuration
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="ML Pipeline API",
        description="Multi-modal ML prediction API",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize predictor
    global _predictor
    _predictor = Predictor(model_path, model_type, config)
    
    # Initialize metrics collector
    metrics_collector = get_collector()
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "ML Pipeline API",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "predict": "/predict",
                "predict_image": "/predict/image",
                "model_info": "/model/info"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        health = get_health()
        
        # Check model
        if _predictor:
            health['model'] = {
                'loaded': True,
                'type': _predictor.model_type
            }
        else:
            health['model'] = {
                'loaded': False
            }
        
        return health
    
    @app.get("/metrics")
    async def get_metrics():
        """Get Prometheus metrics."""
        from prometheus_client import generate_latest
        return generate_latest()
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Predict endpoint for tabular data.
        
        Args:
            request: Prediction request
            
        Returns:
            Predictions
        """
        if not _predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Convert to numpy array
            data = np.array(request.data)
            
            # Run prediction
            start_time = time.time()
            result = _predictor.predict(data)
            inference_time = time.time() - start_time
            
            # Record metrics
            metrics_collector.record_inference(inference_time)
            
            return PredictionResponse(
                predictions=result['predictions'].tolist(),
                inference_time=inference_time,
                model_type=_predictor.model_type
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/image")
    async def predict_image(file: UploadFile = File(...)):
        """
        Predict endpoint for image data.
        
        Args:
            file: Uploaded image file
            
        Returns:
            Predictions
        """
        if not _predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if _predictor.model_type not in ['vision', 'yolo']:
            raise HTTPException(
                status_code=400,
                detail=f"Model type {_predictor.model_type} does not support image predictions"
            )
        
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Run prediction
            start_time = time.time()
            result = _predictor.predict(image)
            inference_time = time.time() - start_time
            
            # Record metrics
            metrics_collector.record_inference(inference_time)
            
            return {
                "predictions": result.get('predictions'),
                "inference_time": inference_time,
                "model_type": _predictor.model_type,
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/model/info", response_model=ModelInfo)
    async def model_info():
        """
        Get model information.
        
        Returns:
            Model info
        """
        if not _predictor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        info = _predictor.get_model_info()
        return ModelInfo(**info)
    
    return app
