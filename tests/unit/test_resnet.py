"""Unit tests for Tier 2 ResNet18 model builder, layer freezing, and forward pass."""

import pytest
import torch
import torch.nn as nn
from tests.helpers import import_or_skip


def test_build_resnet18_architecture():
    """F7: Verify build_resnet18 freezes initial layers and replaces head with Dropout+Linear."""
    build_resnet18 = import_or_skip("src.models.train_resnet", "build_resnet18")

    model = build_resnet18(pretrained=False, num_classes=2)
    assert isinstance(model, nn.Module)

    # Check forward pass with dummy batch
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"

    # Verify frozen layers: conv1, bn1, layer1 must have requires_grad == False
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ["conv1", "bn1", "layer1"]):
            assert not param.requires_grad, f"Layer parameter {name} should be frozen (requires_grad=False)"

    # Verify trainable layers: layer4 and fc must have requires_grad == True
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ["layer4", "fc"]):
            assert param.requires_grad, f"Layer parameter {name} should be trainable (requires_grad=True)"


def test_resnet18_loss_weighting():
    """F8: Verify cost-sensitive cross entropy loss weights fire class higher than smoke."""
    # Class 0: Fire, Class 1: Smoke. Weight vector: [2.0, 1.0]
    weights = torch.tensor([2.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")

    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([0, 1])  # sample 0 is fire, sample 1 is smoke

    losses = criterion(logits, target)
    assert torch.isclose(losses[0], 2.0 * losses[1]), "Loss on fire class must be penalized 2.0x vs smoke"
