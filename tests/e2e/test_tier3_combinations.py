"""Tier 3: Pairwise Combinations & Subsystem Interactions Test Suite.
Verifies cross-module coupling: pHash dedup + DataLoader, features + LightGBM, calibration + inference.
"""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from tests.helpers import import_or_skip


class TestTier3Combinations:

    def test_phash_deduplication_coupled_with_dataloader(self, mock_dataset_dir):
        """Coupling: pHash filtering drops duplicate images and DataLoader collates the deduplicated dataset."""
        FireSmokeDataset = import_or_skip("src.data.pipeline", "FireSmokeDataset")
        get_eval_transforms = import_or_skip("src.data.pipeline", "get_eval_transforms")

        # Test split contains 3 fire + 3 smoke + 1 identical duplicate fire
        dataset = FireSmokeDataset(
            root_dir=mock_dataset_dir,
            split="test",
            transform=get_eval_transforms(224),
            filter_duplicates=True
        )

        # Ensure index access works seamlessly without IndexError after pruning
        assert len(dataset) == 6
        for idx in range(len(dataset)):
            img, label, path = dataset[idx]
            assert img.shape == (3, 224, 224)
            assert label in (0, 1)

        loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2  # 6 items / 3 per batch = 2 batches
        assert batches[0][0].shape == (3, 3, 224, 224)
        assert batches[1][0].shape == (3, 3, 224, 224)

    def test_feature_extraction_pipeline_to_lightgbm_training(self, synthetic_fire_image, synthetic_smoke_image):
        """Coupling: Extract features across synthetic fire and smoke images and train LightGBM."""
        extract_all_features = import_or_skip("src.features.extract", "extract_all_features")
        train_lightgbm = import_or_skip("src.models.train_lightgbm", "train_lightgbm")

        # Create small dataset of 6 fire and 6 smoke images
        imgs = []
        labels = []
        for _ in range(6):
            imgs.append(synthetic_fire_image)
            labels.append(0)  # Fire
            imgs.append(synthetic_smoke_image)
            labels.append(1)  # Smoke

        feats = [extract_all_features(img) for img in imgs]
        X = np.vstack(feats)
        y = np.array(labels)

        # Train/val split
        X_train, y_train = X[:8], y[:8]
        X_val, y_val = X[8:], y[8:]

        model = train_lightgbm(X_train, y_train, X_val, y_val, config={"n_estimators": 5, "min_child_samples": 1})
        assert hasattr(model, "predict_proba")

        probs = model.predict_proba(X_val)
        assert probs.shape == (4, 2)

    def test_calibration_coupled_with_inference_dual_gating(self):
        """Coupling: Calibrate operational threshold theta* and feed into FireClassifier."""
        calibrate_threshold = import_or_skip("src.evaluate.calibrate", "calibrate_threshold")
        FireClassifier = import_or_skip("src.inference", "FireClassifier")

        # Mock validation dataset
        y_val_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Fire model probabilities
        y_val_probs = np.array([0.95, 0.90, 0.88, 0.85, 0.72, 0.30, 0.25, 0.15, 0.10, 0.05])

        theta_star, metrics = calibrate_threshold(y_val_true, y_val_probs, min_recall=0.90)
        assert metrics["recall_fire"] >= 0.90

        # Inject calibrated theta* into classifier
        classifier = FireClassifier(model_type="mock", model_path="", threshold=theta_star, ambient_tau=0.70)

        class MockModel:
            def __init__(self, p0, p1):
                self.p0 = p0
                self.p1 = p1
            def predict_proba(self, x):
                return np.array([[self.p0, self.p1]])
            def __call__(self, x):
                import torch
                return torch.tensor([[np.log(self.p0 + 1e-6), np.log(self.p1 + 1e-6)]])

        classifier.model = MockModel(0.80, 0.20)
        predict_fn = classifier.predict if hasattr(classifier, "predict") else classifier.predict_image
        dummy = Image.new("RGB", (224, 224), color=(255, 100, 0))

        res = predict_fn(dummy)
        assert res["operational_threshold"] == pytest.approx(theta_star, abs=1e-3)
        assert res["hazard_detected"] is True

    def test_resnet_freezing_gradient_flow_during_backward(self):
        """Coupling: Ensure backpropagation updates ONLY layer2..4 and fc, keeping frozen layers untouched."""
        build_resnet18 = import_or_skip("src.models.train_resnet", "build_resnet18")
        model = build_resnet18(pretrained=False, num_classes=2)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]))

        dummy_x = torch.randn(2, 3, 224, 224)
        dummy_y = torch.tensor([0, 1])

        out = model(dummy_x)
        loss = loss_fn(out, dummy_y)
        loss.backward()

        # Frozen layers must have grad is None
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in ["conv1", "bn1", "layer1"]):
                assert param.grad is None, f"Frozen parameter {name} received gradient!"
            elif "fc" in name:
                assert param.grad is not None, f"Trainable head parameter {name} has no gradient!"
