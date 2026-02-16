"""Prediction script entry point."""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.preprocessors.image_preprocessor import ImagePreprocessor
from src.data.preprocessors.tabular_preprocessor import TabularPreprocessor
from src.inference.predictor import Predictor
from src.models.tabular_model import MLPClassifier
from src.models.vision_model import ResNetClassifier
from src.observability.logger import get_logger, setup_logging


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Run predictions on new data")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data (CSV file or image directory)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['tabular', 'vision'],
        required=True,
        help='Model type'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Path to output predictions file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO')
    logger = get_logger(__name__)
    
    logger.info("=== Model Prediction ===")
    
    try:
        # Create model instance
        if args.model_type == 'tabular':
            model = MLPClassifier(input_dim=4, output_dim=3)
            preprocessor = None
        elif args.model_type == 'vision':
            model = ResNetClassifier(num_classes=10, backbone='resnet18', pretrained=False)
            preprocessor = ImagePreprocessor(image_size=(224, 224), normalize=True)
        else:
            logger.error(f"Unknown model type: {args.model_type}")
            sys.exit(1)
        
        # Create predictor
        predictor = Predictor.from_checkpoint(
            checkpoint_path=args.model_path,
            model=model,
            preprocessor=preprocessor
        )
        
        # Load input data
        if args.model_type == 'tabular':
            import pandas as pd
            df = pd.read_csv(args.input)
            X = df.values
            
            # Predict
            results = predictor.predict_tabular(X, return_proba=True)
            
            predictions = {
                'predictions': results['predictions'].tolist(),
                'probabilities': results['probabilities'].tolist()
            }
        
        elif args.model_type == 'vision':
            from PIL import Image
            
            input_path = Path(args.input)
            
            if input_path.is_file():
                # Single image
                image = Image.open(input_path)
                results = predictor.predict_image(image, return_proba=True)
                
                predictions = {
                    'predictions': results['predictions'].tolist(),
                    'probabilities': results['probabilities'].tolist()
                }
            
            elif input_path.is_dir():
                # Directory of images
                image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
                images = [Image.open(f) for f in image_files]
                
                results = predictor.predict_image(images, return_proba=True)
                
                predictions = {
                    'files': [str(f) for f in image_files],
                    'predictions': results['predictions'].tolist(),
                    'probabilities': results['probabilities'].tolist()
                }
            else:
                logger.error(f"Invalid input path: {input_path}")
                sys.exit(1)
        
        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Predictions saved to {output_path}")
        logger.info(f"Number of predictions: {len(predictions['predictions'])}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
