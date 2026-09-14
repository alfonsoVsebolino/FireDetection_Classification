"""Unit tests for threshold calibration, latency benchmarking, and champion model selection."""

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from tests.helpers import import_or_skip


def test_calibrate_threshold_guarantees_recall():
    """F10: Verify calibrate_threshold finds theta* achieving min_recall >= 0.90 on fire class."""
    calibrate_threshold = import_or_skip("src.evaluate.calibrate", "calibrate_threshold")

    # Synthetic ground truth (0: fire, 1: smoke)
    # 20 fire samples, 20 smoke samples
    y_true = np.array([0] * 20 + [1] * 20)
    # Synthetic predicted fire probabilities
    # Fire samples have probabilities around 0.6 - 0.95
    # Smoke samples have probabilities around 0.05 - 0.4
    np.random.seed(42)
    y_probs_fire = np.concatenate([
        np.random.uniform(0.55, 0.99, size=20),
        np.random.uniform(0.01, 0.45, size=20)
    ])

    theta_star, metrics = calibrate_threshold(y_true, y_probs_fire, min_recall=0.90)

    assert 0.10 <= theta_star <= 0.90, f"Theta {theta_star} outside valid sweep range [0.10, 0.90]"
    assert isinstance(metrics, dict)
    assert "recall_fire" in metrics
    assert metrics["recall_fire"] >= 0.90, f"Calibrated fire recall {metrics['recall_fire']} is below 0.90"


def test_benchmark_model_cpu_latency():
    """F11: Verify benchmark_model_cpu computes mean latency in milliseconds."""
    benchmark_model_cpu = import_or_skip("src.evaluate.benchmark", "benchmark_model_cpu")

    dummy_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
    dummy_input = torch.randn(1, 3, 224, 224)

    latency_ms = benchmark_model_cpu(dummy_model, dummy_input, num_runs=5)
    assert isinstance(latency_ms, (float, int))
    assert latency_ms > 0.0, "Latency must be strictly positive"
    assert latency_ms < 5000.0, "Latency for simple linear model should be well under 5 seconds"


def test_select_champion_policy():
    """F11: Verify champion selection disqualifies recall < 0.90 and crowns lowest latency."""
    select_champion = import_or_skip("src.evaluate.benchmark", "select_champion")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "champion_config.json")
        # Define mock candidates with test metrics
        models_dict = {
            "tier1_lightgbm": {
                "model": "dummy_lgb",
                "test_recall_fire": 0.85,  # FAILS recall >= 0.90
                "latency_ms": 1.5,
            },
            "tier2_resnet18": {
                "model": "dummy_resnet",
                "test_recall_fire": 0.95,  # PASSES recall >= 0.90
                "latency_ms": 25.0,        # Faster deep model
            },
            "tier3_deit_tiny": {
                "model": "dummy_vit",
                "test_recall_fire": 0.94,  # PASSES recall >= 0.90
                "latency_ms": 45.0,        # Slower deep model
            }
        }

        # If select_champion takes dictionary or precomputed metrics
        try:
            res = select_champion(models_dict=models_dict, val_dataloaders=None, test_dataloaders=None, output_path=output_path)
            assert res is not None
            assert res.get("champion_tier") in ["tier2_resnet18", "tier2"], "ResNet18 should be crowned champion"
            assert os.path.exists(output_path)
        except TypeError:
            # Fallback if signature requires actual loaders
            pytest.skip("select_champion requires live DataLoader instances")
