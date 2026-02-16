#!/usr/bin/env python3
"""
Prediction entrypoint.
Usage: python scripts/predict.py --model tabular --checkpoint artifacts/models/best.pt --input data.csv
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import Predictor
from src.observability.logger import get_logger
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Run predictions')
    parser.add_argument('--model', choices=['tabular', 'vision', 'yolo'], required=True,
                       help='Model type')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', required=True,
                       help='Input file or directory')
    parser.add_argument('--output', default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--config', default='configs/model_config.yaml',
                       help='Model configuration file')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(log_file='artifacts/logs/predict.log')
    logger.info(f"Starting prediction with {args.model} model")
    
    try:
        # Load config
        from src.utils.config_loader import load_config
        config = load_config(args.config)
        
        # Create predictor
        logger.info("Loading predictor...")
        predictor = Predictor(args.checkpoint, args.model, config)
        
        # Load input data
        logger.info(f"Loading input from {args.input}")
        
        if args.model == 'tabular':
            # Load CSV
            df = pd.read_csv(args.input)
            data = df.values
            
            # Predict
            results = predictor.predict(data)
            predictions = results['predictions'].tolist()
            
            # Add predictions to dataframe
            df['prediction'] = predictions
            
            # Save
            output_path = Path(args.output)
            if output_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            else:
                output_dict = {
                    'predictions': predictions,
                    'inference_time': results['inference_time']
                }
                with open(output_path, 'w') as f:
                    json.dump(output_dict, f, indent=2)
                    
        elif args.model in ['vision', 'yolo']:
            # Predict on image(s)
            input_path = Path(args.input)
            
            if input_path.is_file():
                # Single image
                results = predictor.predict(str(input_path))
            else:
                # Directory of images
                image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
                results = predictor.predict_batch([str(f) for f in image_files])
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(str(results), f, indent=2)
        
        logger.info(f"Predictions saved to {args.output}")
        logger.info(f"Inference time: {results.get('inference_time', 'N/A')}s")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
