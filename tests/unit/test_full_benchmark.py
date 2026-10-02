"""Unit tests for full_benchmark module."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.evaluate.full_benchmark import (
    _calc_metrics,
    benchmark_pytorch_cpu,
    benchmark_lightgbm_cpu,
    evaluate_all_tiers,
)


def test_calc_metrics():
    # 0 = fire, 1 = smoke
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 1 TP fire, 1 FN fire, 2 TN smoke, 0 FP
    metrics = _calc_metrics(y_true, y_pred)
    assert metrics["fire_recall"] == 0.5
    assert metrics["fire_precision"] == 1.0
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_benchmark_pytorch_cpu():
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.zeros((x.size(0), 2))

    dummy_tensor = torch.zeros(3, 224, 224)
    mock_dataset = [(dummy_tensor, 0, "path1"), (dummy_tensor, 1, "path2")]
    res = benchmark_pytorch_cpu(DummyModel(), mock_dataset, max_samples=2)
    assert "mean_ms" in res
    assert "median_ms" in res
    assert "p95_ms" in res
    assert len(res["latencies"]) == 2


def test_benchmark_lightgbm_cpu():
    mock_lgbm = MagicMock()
    mock_lgbm.predict_proba.return_value = np.array([[0.8, 0.2]])

    dummy_tensor = torch.zeros(3, 224, 224)
    mock_dataset = [(dummy_tensor, 0, "dummy.jpg"), (dummy_tensor, 1, "dummy2.jpg")]
    X_test = np.zeros((2, 10))

    with patch("src.evaluate.full_benchmark.extract_all_features", return_value=np.zeros(10)):
        res = benchmark_lightgbm_cpu(mock_lgbm, mock_dataset, X_test, max_samples=2)
    assert "raw_model" in res
    assert "end_to_end" in res
    assert res["raw_model"]["mean_ms"] >= 0.0
    assert res["end_to_end"]["mean_ms"] >= 0.0


def test_evaluate_all_tiers_device_sync(tmp_path):
    mock_lgbm = MagicMock()
    mock_lgbm.predict_proba.return_value = np.array([[0.8, 0.2]])

    mock_resnet = MagicMock()
    mock_resnet.to.return_value = mock_resnet
    mock_resnet.eval.return_value = mock_resnet
    mock_resnet.parameters.return_value = iter([torch.zeros(1)])
    mock_resnet.return_value = torch.tensor([[1.0, 0.0]])

    mock_vit = MagicMock()
    mock_vit.to.return_value = mock_vit
    mock_vit.eval.return_value = mock_vit
    mock_vit.parameters.return_value = iter([torch.zeros(1)])
    mock_vit.return_value = torch.tensor([[1.0, 0.0]])

    dummy_tensor = torch.zeros(1, 3, 224, 224)
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(dummy_tensor, torch.tensor([0]))]
    mock_loader.dataset = [(dummy_tensor[0], 0, "dummy.jpg")]
    X_test = np.zeros((1, 10))
    y_test = np.array([0])
    target_dev = torch.device("cpu")

    out_file = str(tmp_path / "tiers.json")
    with patch("src.evaluate.full_benchmark.extract_all_features", return_value=np.zeros(10)):
        evaluate_all_tiers(
            mock_lgbm, mock_resnet, mock_vit,
            mock_loader, X_test, y_test,
            device=target_dev,
            output_path=out_file
        )

    mock_resnet.to.assert_any_call(target_dev)
    mock_vit.to.assert_any_call(target_dev)

