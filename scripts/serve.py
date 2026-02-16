"""Serving script entry point."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessors.image_preprocessor import ImagePreprocessor
from src.data.preprocessors.tabular_preprocessor import TabularPreprocessor
from src.inference.serving import serve
from src.models.tabular_model import MLPClassifier
from src.models.vision_model import ResNetClassifier
from src.observability.logger import get_logger, setup_logging
from src.utils.config_loader import ConfigLoader


def main():
    """Main serving function."""
    parser = argparse.ArgumentParser(description="Serve ML model via API")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/deployment_config.yaml',
        help='Path to deployment config file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['tabular', 'vision'],
        help='Model type (overrides config)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Host address (overrides config)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port number (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO')
    logger = get_logger(__name__)
    
    logger.info("=== Model Serving ===")
    
    try:
        # Load config
        config = ConfigLoader.load(args.config)
        
        # Get model configuration
        model_path = args.model_path or config['deployment']['model']['path']
        model_type = args.model_type or config['deployment']['model']['type']
        host = args.host or config['deployment']['api'].get('host', '0.0.0.0')
        port = args.port or config['deployment']['api'].get('port', 8000)
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model type: {model_type}")
        
        # Check if model file exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            logger.info("Please train a model first using: python scripts/train.py")
            sys.exit(1)
        
        # Create model instance (architecture must match)
        if model_type == 'tabular':
            # For tabular, we need to know the architecture
            # In a real scenario, this should be stored with the model
            model = MLPClassifier(
                input_dim=4,  # This should be loaded from model metadata
                output_dim=3,
                hidden_layers=[128, 64, 32],
                dropout=0.3
            )
            preprocessor = None
            class_names = None
        elif model_type == 'vision':
            model = ResNetClassifier(
                num_classes=10,  # This should be loaded from model metadata
                backbone='resnet18',
                pretrained=False
            )
            preprocessor = ImagePreprocessor(
                image_size=(224, 224),
                normalize=True
            )
            class_names = None
        else:
            logger.error(f"Unknown model type: {model_type}")
            sys.exit(1)
        
        # Start serving
        logger.info(f"Starting API server on {host}:{port}")
        serve(
            model_path=model_path,
            model=model,
            model_type=model_type,
            preprocessor=preprocessor,
            class_names=class_names,
            host=host,
            port=port,
            log_level='info',
            enable_metrics=config['deployment']['monitoring'].get('enabled', True)
        )
        
    except Exception as e:
        logger.error(f"Serving failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
