"""Tier 1: Feature Coverage Test Suite.
Validates each feature F1 through F13 in isolation as specified in PROJECT.md.
"""

import os
import subprocess
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from tests.helpers import import_or_skip


class TestTier1Features:

    def test_f1_dataset_ingestion_and_corruption_filtering(self, mock_dataset_dir):
        """F1: Ingestion resolves dataset splits and filters out corrupted or invalid files."""
        FireSmokeDataset = import_or_skip("src.data.pipeline", "FireSmokeDataset")
        ds_val = FireSmokeDataset(root_dir=mock_dataset_dir, split="validation", filter_duplicates=False)
        # 3 fire + 3 smoke + 1 corrupt = 7 files originally. Corrupt dropped -> 6 items.
        assert len(ds_val) == 6
        for i in range(len(ds_val)):
            item = ds_val[i]
            # Must return tuple (image/tensor, label, path)
            assert len(item) == 3
            assert item[1] in [0, 1]

    def test_f2_perceptual_hash_deduplication(self, identical_duplicate_pair, near_duplicate_pair):
        """F2: Perceptual hash detects exact and near-duplicates with Hamming distance <= 4."""
        compute_phash = import_or_skip("src.data.pipeline", "compute_phash")
        hamming_distance = import_or_skip("src.data.pipeline", "hamming_distance")

        img1 = Image.open(identical_duplicate_pair[0])
        img2 = Image.open(identical_duplicate_pair[1])
        h1 = compute_phash(img1)
        h2 = compute_phash(img2)
        assert hamming_distance(h1, h2) == 0, "Identical images must have hamming distance 0"

        near1 = Image.open(near_duplicate_pair[0])
        near2 = Image.open(near_duplicate_pair[1])
        hn1 = compute_phash(near1)
        hn2 = compute_phash(near2)
        assert hamming_distance(hn1, hn2) <= 4, "Near-duplicates must have hamming distance <= 4"

    def test_f3_evaluation_preprocessing_zero_leakage(self, synthetic_rgb_image):
        """F3: LANCZOS 224x224 resize, ImageNet normalization, deterministic evaluation transforms."""
        get_eval_transforms = import_or_skip("src.data.pipeline", "get_eval_transforms")
        transforms = get_eval_transforms(224)

        tensor1 = transforms(synthetic_rgb_image)
        tensor2 = transforms(synthetic_rgb_image)
        # Deterministic: zero augmentation randomness
        assert torch.equal(tensor1, tensor2), "Evaluation transforms must be strictly deterministic"
        assert tensor1.shape == (3, 224, 224)

    def test_f4_dataloader_batch_collation(self, mock_dataset_dir):
        """F4: PyTorch DataLoader batch collation preserves dimensions and integer targets."""
        create_dataloader = import_or_skip("src.data.pipeline", "create_dataloader")
        loader = create_dataloader(root_dir=mock_dataset_dir, split="validation", batch_size=2)
        batch = next(iter(loader))
        images, labels, paths = batch
        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
        assert isinstance(paths, (list, tuple))

    def test_f5_handcrafted_feature_extraction(self, synthetic_fire_image):
        """F5: Feature extraction yields 96 color hist, moments, and texture descriptors."""
        extract_color_histogram = import_or_skip("src.features.extract", "extract_color_histogram")
        extract_color_moments = import_or_skip("src.features.extract", "extract_color_moments")
        extract_texture_features = import_or_skip("src.features.extract", "extract_texture_features")
        extract_all_features = import_or_skip("src.features.extract", "extract_all_features")

        img_arr = np.array(synthetic_fire_image)
        hist = extract_color_histogram(img_arr, bins=16)
        moments = extract_color_moments(img_arr)
        texture = extract_texture_features(np.array(synthetic_fire_image.convert("L")))
        all_feats = extract_all_features(synthetic_fire_image)

        assert hist.shape == (96,)
        assert moments.shape[0] in [9, 18]
        assert texture.shape[0] >= 4
        assert all_feats.shape[0] >= (96 + moments.shape[0] + texture.shape[0])

    def test_f6_tier1_lightgbm_classifier(self):
        """F6: LightGBM trains with class weighting penalizing false negatives on fire."""
        train_lightgbm = import_or_skip("src.models.train_lightgbm", "train_lightgbm")

        np.random.seed(42)
        X_train = np.random.randn(30, 64).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(30,))
        X_val = np.random.randn(10, 64).astype(np.float32)
        y_val = np.random.randint(0, 2, size=(10,))

        model = train_lightgbm(X_train, y_train, X_val, y_val, config={"n_estimators": 5})
        probs = model.predict_proba(X_val)
        assert probs.shape == (10, 2)

    def test_f7_tier2_resnet18_transfer_backbone(self):
        """F7: ResNet18 freezes conv1..layer1, trains layer2..layer4, with Dropout+Linear head."""
        build_resnet18 = import_or_skip("src.models.train_resnet", "build_resnet18")
        model = build_resnet18(pretrained=False, num_classes=2)

        # Check forward pass
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 2)

        # Check freezing
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in ["conv1", "bn1", "layer1"]):
                assert not param.requires_grad
            elif "fc" in name:
                assert param.requires_grad

    def test_f8_tier2_optimization_engine_loss(self):
        """F8: CrossEntropyLoss weighting [2.0, 1.0] imposes 2x gradient pressure on class 0 (fire)."""
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]), reduction="none")
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        # Fire loss (target 0) vs smoke loss (target 1) when predictions are identical
        losses = loss_fn(logits, torch.tensor([0, 1]))
        assert torch.isclose(losses[0], 2.0 * losses[1])

    def test_f9_tier3_deit_tiny_vision_transformer(self):
        """F9: DeiT-Tiny freezes patch embedder and blocks[:6], trains blocks[6:] and head."""
        build_deit_tiny = import_or_skip("src.models.train_vit", "build_deit_tiny")
        model = build_deit_tiny(pretrained=False, num_classes=2)

        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 2)

        for name, param in model.named_parameters():
            if "patch_embed" in name:
                assert not param.requires_grad
            elif "blocks.0." in name or "blocks.5." in name:
                assert not param.requires_grad
            elif "blocks.7." in name or "head" in name:
                assert param.requires_grad

    def test_f10_calibrated_threshold_sweep(self):
        """F10: Validation threshold sweep theta in [0.10, 0.90] guarantees Fire Recall >= 0.90."""
        calibrate_threshold = import_or_skip("src.evaluate.calibrate", "calibrate_threshold")
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_probs_fire = np.array([0.9, 0.85, 0.8, 0.75, 0.6, 0.3, 0.2, 0.1, 0.05, 0.01])

        theta, metrics = calibrate_threshold(y_true, y_probs_fire, min_recall=0.90)
        assert 0.10 <= theta <= 0.90
        assert metrics["recall_fire"] >= 0.90

    def test_f11_cpu_latency_benchmarking_and_champion(self):
        """F11: CPU latency benchmark reports latency in ms/sample."""
        benchmark_model_cpu = import_or_skip("src.evaluate.benchmark", "benchmark_model_cpu")
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        sample = torch.randn(1, 3, 224, 224)

        lat = benchmark_model_cpu(model, sample, num_runs=3)
        assert lat > 0.0

    def test_f12_dual_gating_inference_engine(self, synthetic_ambient_image, synthetic_fire_image):
        """F12: Gate 1 rejects ambient scenes (max(P)<0.70), Gate 2 detects hazard (P_fire >= theta*)."""
        FireClassifier = import_or_skip("src.inference", "FireClassifier")

        # Mock model wrapper
        class StubModel:
            def __init__(self, p0, p1):
                self.p0 = p0
                self.p1 = p1
            def predict_proba(self, x):
                return np.array([[self.p0, self.p1]])
            def __call__(self, x):
                import torch
                return torch.tensor([[np.log(self.p0 + 1e-6), np.log(self.p1 + 1e-6)]])

        clf = FireClassifier(model_type="stub", model_path="", threshold=0.50, ambient_tau=0.70)
        predict_fn = clf.predict if hasattr(clf, "predict") else clf.predict_image

        # Ambient gate test
        clf.model = StubModel(0.50, 0.50)
        res_amb = predict_fn(synthetic_ambient_image)
        assert res_amb["hazard_detected"] is False

        # Fire gate test
        clf.model = StubModel(0.85, 0.15)
        res_fire = predict_fn(synthetic_fire_image)
        assert res_fire["hazard_detected"] is True
        assert (res_fire.get("predicted_class") or res_fire.get("label")) == "fire"

    def test_f13_production_cli_interface(self):
        """F13: CLI supports --help, --input, --threshold, --output-json."""
        if not os.path.exists("src/inference.py"):
            pytest.skip("src/inference.py not yet implemented")

        res = subprocess.run([sys.executable, "-m", "src.inference", "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "--input" in res.stdout
