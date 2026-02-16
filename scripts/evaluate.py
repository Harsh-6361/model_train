"""Evaluation script entry point."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.adapters.csv_adapter import CSVAdapter
from src.data.adapters.image_adapter import ImageAdapter
from src.data.loaders.unified_loader import UnifiedDataLoader
from src.data.preprocessors.image_preprocessor import ImagePreprocessor
from src.data.preprocessors.tabular_preprocessor import TabularPreprocessor
from src.models.tabular_model import MLPClassifier
from src.models.vision_model import ResNetClassifier
from src.observability.logger import get_logger, setup_logging
from src.training.metrics import MetricsCalculator
from src.utils.config_loader import ConfigLoader


def evaluate_tabular_model(model_path: str, data_config: dict):
    """Evaluate tabular model."""
    logger = get_logger(__name__)
    
    # Load data
    tabular_config = data_config['data']['tabular']
    adapter = CSVAdapter(
        file_path=tabular_config['path'],
        target_column=tabular_config.get('target_column')
    )
    
    adapter.load()
    splits = adapter.split_data(
        train_ratio=tabular_config.get('train_split', 0.7),
        val_ratio=tabular_config.get('validation_split', 0.15),
        test_ratio=tabular_config.get('test_split', 0.15)
    )
    
    # Preprocess
    preprocessing_config = tabular_config.get('preprocessing', {})
    preprocessor = TabularPreprocessor(
        normalize=preprocessing_config.get('normalize', True),
        handle_missing=preprocessing_config.get('handle_missing', 'mean'),
        categorical_encoding=preprocessing_config.get('categorical_encoding', 'onehot')
    )
    
    X_train, y_train = splits['train']
    X_test, y_test = splits['test']
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Load model
    input_dim = X_train_processed.shape[1]
    output_dim = len(y_train.unique())
    
    model = MLPClassifier(input_dim=input_dim, output_dim=output_dim)
    model.load(model_path)
    model.eval()
    
    # Create loader
    loaders = UnifiedDataLoader.create_tabular_loaders(
        train_data=(X_train_processed, y_train.values),
        test_data=(X_test_processed, y_test.values),
        batch_size=32,
        shuffle=False
    )
    
    # Evaluate
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_all_metrics(
        np.array(all_targets),
        np.array(all_preds)
    )
    
    logger.info("Evaluation Results:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data config file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['tabular', 'vision'],
        default='tabular',
        help='Model type'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO')
    logger = get_logger(__name__)
    
    logger.info("=== Model Evaluation ===")
    
    try:
        # Load config
        data_config = ConfigLoader.load(args.data_config)
        
        # Evaluate based on model type
        if args.model_type == 'tabular':
            metrics = evaluate_tabular_model(args.model_path, data_config)
        else:
            logger.error("Vision model evaluation not yet implemented")
            sys.exit(1)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
