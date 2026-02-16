"""Prediction engine for inference."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ..data.preprocessors.image_preprocessor import ImagePreprocessor
from ..data.preprocessors.tabular_preprocessor import TabularPreprocessor
from ..models.base_model import BaseModel
from ..observability.logger import get_logger
from ..utils.helpers import get_device

logger = get_logger(__name__)


class Predictor:
    """Inference engine for trained models."""
    
    def __init__(
        self,
        model: BaseModel,
        preprocessor: Optional[Union[TabularPreprocessor, ImagePreprocessor]] = None,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None
    ):
        """Initialize predictor.
        
        Args:
            model: Trained model
            preprocessor: Data preprocessor
            device: Device for inference
            class_names: Optional list of class names
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device if device else get_device()
        self.class_names = class_names
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Predictor initialized on device: {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model: BaseModel,
        preprocessor: Optional[Any] = None,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None
    ) -> 'Predictor':
        """Create predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model: Model instance (with correct architecture)
            preprocessor: Data preprocessor
            device: Device for inference
            class_names: Optional list of class names
            
        Returns:
            Predictor instance
        """
        # Load model
        model.load(checkpoint_path)
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return cls(model, preprocessor, device, class_names)
    
    def predict_tabular(
        self,
        X: Union[np.ndarray, List[List[float]]],
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict on tabular data.
        
        Args:
            X: Input features
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or dict with predictions and probabilities
        """
        # Convert to numpy if needed
        if isinstance(X, list):
            X = np.array(X)
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        # Convert to numpy
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        if return_proba:
            return {
                'predictions': preds_np,
                'probabilities': probs_np
            }
        return preds_np
    
    def predict_image(
        self,
        images: Union[Image.Image, List[Image.Image], str, Path, List[Union[str, Path]]],
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict on image data.
        
        Args:
            images: Single image, list of images, or paths
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or dict with predictions and probabilities
        """
        # Load images if paths provided
        if isinstance(images, (str, Path)):
            images = [Image.open(images).convert('RGB')]
        elif isinstance(images, list) and isinstance(images[0], (str, Path)):
            images = [Image.open(img).convert('RGB') for img in images]
        elif isinstance(images, Image.Image):
            images = [images]
        
        # Preprocess images
        if self.preprocessor is None:
            raise ValueError("Image preprocessor required for image prediction")
        
        # Process images
        processed_images = []
        for img in images:
            processed = self.preprocessor.preprocess_image(img, training=False)
            processed_images.append(processed)
        
        # Stack into batch
        X_tensor = torch.stack(processed_images).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        # Convert to numpy
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        if return_proba:
            return {
                'predictions': preds_np,
                'probabilities': probs_np
            }
        return preds_np
    
    def predict_with_labels(
        self,
        X: Any,
        model_type: str = 'tabular'
    ) -> List[Dict[str, Any]]:
        """Predict and return results with class labels.
        
        Args:
            X: Input data
            model_type: Type of model ('tabular' or 'vision')
            
        Returns:
            List of prediction dictionaries
        """
        # Get predictions and probabilities
        if model_type == 'tabular':
            result = self.predict_tabular(X, return_proba=True)
        else:
            result = self.predict_image(X, return_proba=True)
        
        preds = result['predictions']
        probs = result['probabilities']
        
        # Format results
        results = []
        for i in range(len(preds)):
            pred_class = int(preds[i])
            pred_probs = probs[i].tolist()
            
            result_dict = {
                'predicted_class': pred_class,
                'predicted_label': self.class_names[pred_class] if self.class_names else str(pred_class),
                'confidence': float(pred_probs[pred_class]),
                'probabilities': pred_probs
            }
            
            # Add top-k predictions
            top_k_indices = np.argsort(pred_probs)[::-1][:3]
            result_dict['top_3'] = [
                {
                    'class': int(idx),
                    'label': self.class_names[idx] if self.class_names else str(idx),
                    'probability': float(pred_probs[idx])
                }
                for idx in top_k_indices
            ]
            
            results.append(result_dict)
        
        return results
    
    def batch_predict(
        self,
        X: Any,
        batch_size: int = 32,
        model_type: str = 'tabular'
    ) -> np.ndarray:
        """Predict on large dataset in batches.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
            model_type: Type of model
            
        Returns:
            Array of predictions
        """
        all_preds = []
        
        # Process in batches
        num_samples = len(X)
        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            
            if model_type == 'tabular':
                preds = self.predict_tabular(batch)
            else:
                preds = self.predict_image(batch)
            
            all_preds.append(preds)
        
        return np.concatenate(all_preds)
