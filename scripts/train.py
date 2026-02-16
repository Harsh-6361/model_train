#!/usr/bin/env python3
"""
Training entrypoint.
Usage: python scripts/train.py --model tabular --config configs/training_config.yaml
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.models.registry import ModelRegistry
from src.training.trainer import Trainer
from src.data.loaders.unified_loader import UnifiedLoader
from src.observability.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--model', choices=['tabular', 'vision', 'yolo'], required=True,
                       help='Model type to train')
    parser.add_argument('--config', default='configs/training_config.yaml',
                       help='Path to training config')
    parser.add_argument('--data-config', default='configs/data_config.yaml',
                       help='Path to data config')
    parser.add_argument('--model-config', default='configs/model_config.yaml',
                       help='Path to model config')
    parser.add_argument('--experiment-name', default=None,
                       help='Experiment name for tracking')
    parser.add_argument('--output-dir', default='artifacts/models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(log_file='artifacts/logs/train.log')
    logger.info(f"Starting training for {args.model} model")
    
    try:
        # Load configs
        logger.info("Loading configurations...")
        training_config = load_config(args.config)
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config)
        
        # Handle YOLO separately as it has its own training mechanism
        if args.model == 'yolo':
            logger.info("Training YOLO model...")
            yolo_config = load_config('configs/yolo_config.yaml')
            
            # Create YOLO model
            model = ModelRegistry.create('yolo', yolo_config)
            
            # Get data path
            data_path = data_config.get('data', {}).get('yolo', {}).get('path')
            if not data_path:
                raise ValueError("YOLO data path not specified in data_config.yaml")
            
            # Check for data.yaml
            data_yaml = Path(data_path) / 'data.yaml'
            if not data_yaml.exists():
                raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
            
            # Train
            results = model.train(
                data=str(data_yaml),
                epochs=yolo_config['yolo']['training'].get('epochs', 300),
                batch_size=yolo_config['yolo']['training'].get('batch_size', 16),
                img_size=yolo_config['yolo']['training'].get('img_size', 640)
            )
            
            logger.info("YOLO training completed")
            return
        
        # Create data loaders for non-YOLO models
        logger.info("Creating data loaders...")
        loader = UnifiedLoader(data_config)
        train_loader, val_loader, test_loader = loader.get_loaders(args.model)
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Get data dimensions for model creation
        if args.model == 'tabular':
            # Get input/output sizes from first batch
            sample_batch = next(iter(train_loader))
            input_size = sample_batch[0].shape[1]
            
            # Get number of classes
            all_labels = []
            for batch in train_loader:
                all_labels.extend(batch[1].tolist())
            output_size = len(set(all_labels))
            
            logger.info(f"Input size: {input_size}, Output size: {output_size}")
            
            # Create model
            logger.info("Creating model...")
            model = ModelRegistry.create(
                args.model,
                model_config,
                input_size=input_size,
                output_size=output_size
            )
        elif args.model == 'vision':
            # Create model
            logger.info("Creating model...")
            model = ModelRegistry.create(args.model, model_config)
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        logger.info(f"Model parameters: {model.get_num_parameters():,}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(model, training_config)
        
        # Train
        logger.info("Starting training...")
        trainer.fit(train_loader, val_loader)
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(test_loader)
        logger.info(f"Test metrics: {metrics}")
        
        # Save final model
        output_path = Path(args.output_dir) / f'{args.model}_final.pt'
        model.save(str(output_path))
        logger.info(f"Model saved to {output_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
