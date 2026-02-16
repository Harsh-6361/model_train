"""Training script entry point."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.adapters.csv_adapter import CSVAdapter
from src.data.adapters.image_adapter import ImageAdapter
from src.data.loaders.unified_loader import UnifiedDataLoader
from src.data.preprocessors.image_preprocessor import ImagePreprocessor
from src.data.preprocessors.tabular_preprocessor import TabularPreprocessor
from src.models.registry import ModelRegistry
from src.models.tabular_model import TabularModel
from src.models.vision_model import VisionModel
from src.observability.logger import get_logger, setup_logging
from src.training.trainer import create_trainer_from_config
from src.utils.config_loader import ConfigLoader


def train_tabular_model(
    data_config: dict,
    model_config: dict,
    training_config: dict
):
    """Train tabular model.
    
    Args:
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
    """
    logger = get_logger(__name__)
    logger.info("Training tabular model")
    
    # Load data
    tabular_config = data_config['data']['tabular']
    adapter = CSVAdapter(
        file_path=tabular_config['path'],
        schema_path=tabular_config.get('schema'),
        target_column=tabular_config.get('target_column'),
        feature_columns=tabular_config.get('feature_columns')
    )
    
    logger.info(f"Loading data from {tabular_config['path']}")
    adapter.load()
    
    # Validate data
    validation_result = adapter.validate()
    if not validation_result['valid']:
        logger.error(f"Data validation failed: {validation_result['errors']}")
        return
    
    # Split data
    splits = adapter.split_data(
        train_ratio=tabular_config.get('train_split', 0.7),
        val_ratio=tabular_config.get('validation_split', 0.15),
        test_ratio=tabular_config.get('test_split', 0.15),
        random_state=training_config['training'].get('seed', 42)
    )
    
    logger.info(f"Data info: {adapter.get_info()}")
    
    # Preprocess data
    preprocessing_config = tabular_config.get('preprocessing', {})
    preprocessor = TabularPreprocessor(
        normalize=preprocessing_config.get('normalize', True),
        handle_missing=preprocessing_config.get('handle_missing', 'mean'),
        categorical_encoding=preprocessing_config.get('categorical_encoding', 'onehot')
    )
    
    # Fit preprocessor on training data
    X_train, y_train = splits['train']
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform validation data
    X_val, y_val = splits['val']
    X_val_processed = preprocessor.transform(X_val)
    
    logger.info(f"Processed feature shape: {X_train_processed.shape}")
    
    # Create model
    model_arch = model_config['model']['architecture']
    input_dim = X_train_processed.shape[1]
    output_dim = len(y_train.unique())
    
    model = TabularModel.create(
        architecture='mlp',
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=model_arch.get('hidden_layers', [128, 64, 32]),
        dropout=model_arch.get('dropout', 0.3),
        activation=model_arch.get('activation', 'relu')
    )
    
    logger.info(f"Model created: {model.model_name}")
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    # Create data loaders
    loaders = UnifiedDataLoader.create_tabular_loaders(
        train_data=(X_train_processed, y_train.values),
        val_data=(X_val_processed, y_val.values),
        batch_size=training_config['training'].get('batch_size', 32),
        shuffle=True
    )
    
    # Create trainer
    trainer = create_trainer_from_config(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=training_config['training']
    )
    
    # Train model
    history = trainer.train(
        epochs=training_config['training'].get('epochs', 50),
        seed=training_config['training'].get('seed', 42)
    )
    
    logger.info("Training completed")
    logger.info(f"Final metrics: {history['final_metrics']}")
    
    # Register model
    registry = ModelRegistry()
    model_id = registry.register_model(
        model_path="artifacts/models/best_model.pth",
        model_name=model_config['model'].get('name', 'mlp_classifier'),
        model_type='tabular',
        metrics=history['final_metrics'],
        config=model.get_config(),
        notes="Trained via train.py script"
    )
    
    logger.info(f"Model registered with ID: {model_id}")
    
    return history


def train_vision_model(
    data_config: dict,
    model_config: dict,
    training_config: dict
):
    """Train vision model.
    
    Args:
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
    """
    logger = get_logger(__name__)
    logger.info("Training vision model")
    
    # Load data
    image_config = data_config['data']['image']
    adapter = ImageAdapter(
        data_dir=image_config['path'],
        schema_path=image_config.get('schema')
    )
    
    logger.info(f"Loading images from {image_config['path']}")
    adapter.load()
    
    # Validate data
    validation_result = adapter.validate(sample_size=10)
    if not validation_result['valid']:
        logger.error(f"Data validation failed: {validation_result['errors']}")
    
    # Split data
    splits = adapter.split_data(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=training_config['training'].get('seed', 42)
    )
    
    logger.info(f"Data info: {adapter.get_info()}")
    
    # Create preprocessor
    augmentation_config = image_config.get('augmentation', {})
    preprocessor = ImagePreprocessor(
        image_size=tuple(image_config.get('image_size', [224, 224])),
        normalize=augmentation_config.get('normalize', True),
        mean=augmentation_config.get('mean'),
        std=augmentation_config.get('std'),
        augment=True,
        horizontal_flip=augmentation_config.get('horizontal_flip', False),
        rotation_range=augmentation_config.get('rotation_range', 0)
    )
    
    # Create data loaders
    loaders = UnifiedDataLoader.create_image_loaders(
        train_data=splits['train'],
        val_data=splits['val'],
        train_transform=preprocessor.train_transform,
        val_transform=preprocessor.val_transform,
        batch_size=training_config['training'].get('batch_size', 32),
        shuffle=True
    )
    
    # Create model
    vision_config = model_config.get('vision_model', model_config['model'])
    num_classes = len(adapter.class_to_idx)
    
    model = VisionModel.create(
        architecture='resnet',
        num_classes=num_classes,
        backbone=vision_config['architecture'].get('backbone', 'resnet18'),
        pretrained=vision_config['architecture'].get('pretrained', True),
        dropout=vision_config['architecture'].get('dropout', 0.5)
    )
    
    logger.info(f"Model created: {model.model_name}")
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = create_trainer_from_config(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=training_config['training']
    )
    
    # Train model
    history = trainer.train(
        epochs=training_config['training'].get('epochs', 50),
        seed=training_config['training'].get('seed', 42)
    )
    
    logger.info("Training completed")
    logger.info(f"Final metrics: {history['final_metrics']}")
    
    # Register model
    registry = ModelRegistry()
    model_id = registry.register_model(
        model_path="artifacts/models/best_model.pth",
        model_name=vision_config.get('name', 'resnet_classifier'),
        model_type='vision',
        metrics=history['final_metrics'],
        config=model.get_config(),
        notes="Trained via train.py script"
    )
    
    logger.info(f"Model registered with ID: {model_id}")
    
    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data config file'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to model config file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['tabular', 'vision'],
        default=None,
        help='Model type (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO')
    logger = get_logger(__name__)
    
    logger.info("=== Model Training ===")
    
    try:
        # Load configs
        training_config = ConfigLoader.load(args.config)
        data_config = ConfigLoader.load(args.data_config)
        model_config = ConfigLoader.load(args.model_config)
        
        # Determine model type
        model_type = args.model_type or model_config['model'].get('type', 'tabular')
        
        logger.info(f"Model type: {model_type}")
        
        # Train based on model type
        if model_type == 'tabular':
            train_tabular_model(data_config, model_config, training_config)
        elif model_type == 'vision':
            train_vision_model(data_config, model_config, training_config)
        else:
            logger.error(f"Unknown model type: {model_type}")
            sys.exit(1)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
