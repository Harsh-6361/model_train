#!/usr/bin/env python3
"""
Automated training orchestrator.
Usage: python scripts/auto_train.py --config configs/training_config.yaml
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability.logger import get_logger
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Automated training orchestrator')
    parser.add_argument('--model', choices=['tabular', 'vision', 'yolo'], default='tabular',
                       help='Model type to train')
    parser.add_argument('--config', default='configs/training_config.yaml',
                       help='Training configuration')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Prepare sample data before training')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger(log_file='artifacts/logs/auto_train.log')
    logger.info(f"Starting automated training for {args.model}")
    
    try:
        # Prepare data if requested
        if args.prepare_data:
            logger.info("Preparing sample data...")
            result = subprocess.run(
                ['python', 'scripts/prepare_data.py', '--type', args.model],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(result.stdout)
        
        # Train model
        logger.info("Starting training...")
        train_cmd = [
            'python', 'scripts/train.py',
            '--model', args.model,
            '--config', args.config,
            '--experiment-name', f'{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        ]
        
        result = subprocess.run(
            train_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Training completed")
        logger.info(result.stdout)
        
        # Evaluate if requested
        if args.evaluate:
            logger.info("Evaluating model...")
            eval_cmd = [
                'python', 'scripts/evaluate.py',
                '--model', args.model,
                '--checkpoint', 'artifacts/models/best.pt'
            ]
            
            result = subprocess.run(
                eval_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Evaluation completed")
            logger.info(result.stdout)
        
        logger.info("Automated training pipeline completed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Automated training failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
