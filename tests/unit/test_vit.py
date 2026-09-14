"""Unit tests for Tier 3 DeiT-Tiny Vision Transformer builder, freezing scheme, and forward pass."""

import pytest
import torch
import torch.nn as nn
from tests.helpers import import_or_skip


def test_build_deit_tiny_architecture():
    """F9: Verify build_deit_tiny freezes first 6 transformer blocks and outputs 2 logits."""
    build_deit_tiny = import_or_skip("src.models.train_vit", "build_deit_tiny")

    model = build_deit_tiny(pretrained=False, num_classes=2)
    assert isinstance(model, nn.Module)

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"

    # Verify frozen layers: patch_embed and blocks 0..5
    for name, param in model.named_parameters():
        if "patch_embed" in name:
            assert not param.requires_grad, f"Parameter {name} in patch_embed should be frozen"
        for b_idx in range(6):
            if f"blocks.{b_idx}." in name:
                assert not param.requires_grad, f"Parameter {name} in block {b_idx} should be frozen"

    # Verify trainable layers: blocks 6..11 and head
    for name, param in model.named_parameters():
        if "head" in name or any(f"blocks.{b_idx}." in name for b_idx in range(6, 12)):
            assert param.requires_grad, f"Parameter {name} should be trainable"
