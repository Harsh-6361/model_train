"""Unit tests for tabular model."""
import pytest
import torch
from src.models.tabular_model import MLPClassifier, TabularModel


def test_mlp_classifier_creation():
    """Test MLP classifier creation."""
    model = MLPClassifier(
        input_dim=10,
        output_dim=3,
        hidden_layers=[64, 32],
        dropout=0.3
    )
    
    assert model.input_dim == 10
    assert model.output_dim == 3
    assert model.hidden_layers == [64, 32]
    assert model.dropout_rate == 0.3


def test_mlp_classifier_forward():
    """Test forward pass."""
    model = MLPClassifier(input_dim=10, output_dim=3)
    
    # Create random input
    x = torch.randn(5, 10)
    
    # Forward pass
    output = model(x)
    
    assert output.shape == (5, 3)


def test_mlp_classifier_save_load(temp_model_path):
    """Test model save and load."""
    model = MLPClassifier(input_dim=10, output_dim=3)
    
    # Save model
    model.save(temp_model_path)
    
    # Create new model and load
    new_model = MLPClassifier(input_dim=10, output_dim=3)
    new_model.load(temp_model_path)
    
    # Check that weights are the same
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_mlp_classifier_count_parameters():
    """Test parameter counting."""
    model = MLPClassifier(input_dim=10, output_dim=3, hidden_layers=[64])
    
    params = model.count_parameters()
    
    # Should have: (10*64 + 64) + (64*3 + 3) = 640 + 64 + 192 + 3 = 899
    expected = (10 * 64 + 64) + (64 * 3 + 3)
    assert params == expected


def test_tabular_model_factory():
    """Test model factory."""
    model = TabularModel.create(
        architecture='mlp',
        input_dim=10,
        output_dim=3
    )
    
    assert isinstance(model, MLPClassifier)
    assert model.input_dim == 10
    assert model.output_dim == 3
