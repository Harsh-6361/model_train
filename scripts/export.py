#!/usr/bin/env python3
"""
Model export script.
Usage: python scripts/export.py --model artifacts/models/best.pt --format onnx
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.registry import ModelRegistry
from src.utils.config_loader import load_config
from src.observability.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description='Export model to different formats')
    parser.add_argument('--model', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', required=True,
                       choices=['tabular', 'vision', 'yolo'],
                       help='Type of model')
    parser.add_argument('--format', default='onnx',
                       choices=['onnx', 'torchscript', 'coreml'],
                       help='Export format')
    parser.add_argument('--output', default=None,
                       help='Output path')
    parser.add_argument('--config', default='configs/model_config.yaml',
                       help='Model configuration file')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(log_file='artifacts/logs/export.log')
    logger.info(f"Exporting {args.model_type} model to {args.format}")
    
    try:
        # Set output path
        if args.output is None:
            model_path = Path(args.model)
            args.output = str(model_path.with_suffix(f'.{args.format}'))
        
        # Handle YOLO separately
        if args.model_type == 'yolo':
            from src.models.yolo_model import YOLOModel
            
            logger.info("Loading YOLO model...")
            model = YOLOModel(weights=args.model)
            
            logger.info(f"Exporting to {args.format}...")
            exported_path = model.export(format=args.format)
            
            logger.info(f"Model exported to {exported_path}")
            return
        
        # Load config
        config = load_config(args.config)
        
        # Load model
        logger.info("Loading model...")
        # Note: For tabular/vision models, we need dimensions
        # This is a simplified version
        if args.model_type == 'tabular':
            # Default dimensions for export
            model = ModelRegistry.create(
                args.model_type,
                config,
                input_size=10,
                output_size=2
            )
        else:
            model = ModelRegistry.create(args.model_type, config)
        
        model.load(args.model)
        model.eval()
        
        # Export
        logger.info(f"Exporting to {args.format}...")
        
        if args.format == 'onnx':
            # Export to ONNX
            if args.model_type == 'tabular':
                dummy_input = torch.randn(1, 10)
            else:
                dummy_input = torch.randn(1, 3, 224, 224)
            
            torch.onnx.export(
                model,
                dummy_input,
                args.output,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
        elif args.format == 'torchscript':
            # Export to TorchScript
            if args.model_type == 'tabular':
                dummy_input = torch.randn(1, 10)
            else:
                dummy_input = torch.randn(1, 3, 224, 224)
            
            scripted_model = torch.jit.trace(model, dummy_input)
            scripted_model.save(args.output)
        
        logger.info(f"Model exported to {args.output}")
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
