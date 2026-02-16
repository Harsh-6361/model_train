#!/usr/bin/env python3
"""
Evaluation entrypoint.
Usage: python scripts/evaluate.py --model tabular --checkpoint artifacts/models/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.models.registry import ModelRegistry
from src.data.loaders.unified_loader import UnifiedLoader
from src.observability.logger import get_logger
import torch


def main():
    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--model', choices=['tabular', 'vision', 'yolo'], required=True,
                       help='Model type')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-config', default='configs/data_config.yaml',
                       help='Path to data config')
    parser.add_argument('--model-config', default='configs/model_config.yaml',
                       help='Path to model config')
    parser.add_argument('--output', default='artifacts/metrics/eval_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(log_file='artifacts/logs/evaluate.log')
    logger.info(f"Starting evaluation for {args.model} model")
    
    try:
        # Load configs
        logger.info("Loading configurations...")
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config)
        
        # Handle YOLO separately
        if args.model == 'yolo':
            logger.info("Evaluating YOLO model...")
            from src.models.yolo_model import YOLOModel
            
            model = YOLOModel(weights=args.checkpoint)
            
            # Get data path
            data_path = data_config.get('data', {}).get('yolo', {}).get('path')
            data_yaml = Path(data_path) / 'data.yaml'
            
            # Validate
            results = model.val(data=str(data_yaml))
            
            logger.info(f"Evaluation results: {results}")
            return
        
        # Create data loaders
        logger.info("Creating data loaders...")
        loader = UnifiedLoader(data_config)
        _, _, test_loader = loader.get_loaders(args.model)
        
        # Load model
        logger.info("Loading model...")
        if args.model == 'tabular':
            # Get dimensions from data
            sample_batch = next(iter(test_loader))
            input_size = sample_batch[0].shape[1]
            output_size = len(set([b[1].item() for b in test_loader]))
            
            model = ModelRegistry.create(
                args.model,
                model_config,
                input_size=input_size,
                output_size=output_size
            )
        else:
            model = ModelRegistry.create(args.model, model_config)
        
        model.load(args.checkpoint)
        model.eval()
        
        # Evaluate
        logger.info("Running evaluation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        from src.training.metrics import MetricsTracker
        tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = tuple(x.to(device) for x in batch)
                metrics = model.validation_step(batch, batch_idx)
                tracker.update(metrics, n=len(batch[0]))
        
        results = tracker.get_metrics()
        logger.info(f"Evaluation results: {results}")
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
