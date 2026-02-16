"""Unit tests for models."""
import pytest
import torch

from src.models.tabular_model import MLP
from src.models.vision_model import VisionModel
from src.models.registry import ModelRegistry


class TestMLPModel:
    """Test MLP model."""
    
    def test_create_model(self):
        """Test model creation."""
        model = MLP(
            input_size=10,
            output_size=2,
            hidden_layers=[32, 16],
            dropout=0.3
        )
        
        assert model.input_size == 10
        assert model.output_size == 2
        assert len(model.hidden_layers) == 2
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = MLP(input_size=10, output_size=2)
        
        x = torch.randn(4, 10)
        output = model(x)
        
        assert output.shape == (4, 2)
    
    def test_training_step(self):
        """Test training step."""
        model = MLP(input_size=10, output_size=2, task='classification')
        
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        
        metrics = model.training_step((x, y), 0)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics


class TestVisionModel:
    """Test vision model."""
    
    def test_create_model(self):
        """Test model creation."""
        model = VisionModel(
            model_type='resnet18',
            num_classes=10,
            pretrained=False
        )
        
        assert model.num_classes == 10
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = VisionModel(
            model_type='resnet18',
            num_classes=10,
            pretrained=False
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 10)


class TestModelRegistry:
    """Test model registry."""
    
    def test_create_tabular_model(self, sample_config):
        """Test creating tabular model."""
        model = ModelRegistry.create(
            'tabular',
            sample_config,
            input_size=10,
            output_size=2
        )
        
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_list_models(self):
        """Test listing available models."""
        models = ModelRegistry.list_models()
        
        assert 'tabular' in models
        assert 'vision' in models
        assert 'yolo' in models
