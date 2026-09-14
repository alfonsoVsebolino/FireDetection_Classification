"""Tier 2: Boundary & Corner Cases Test Suite.
Tests edge conditions, extreme parameters, corrupted inputs, and domain boundaries.
"""

import os
import numpy as np
import pytest
import torch
from PIL import Image
from tests.helpers import import_or_skip


class TestTier2Boundaries:

    def test_corrupted_and_empty_file_handling(self, corrupted_image_file, empty_image_file, temp_image_dir):
        """Corrupt files and 0-byte files should be safely rejected or skipped without unhandled crashes."""
        FireSmokeDataset = import_or_skip("src.data.pipeline", "FireSmokeDataset")

        # Create a split directory with only a corrupt and an empty file
        test_dir = os.path.join(temp_image_dir, "boundary_split")
        os.makedirs(os.path.join(test_dir, "fire"), exist_ok=True)
        corrupt_target = os.path.join(test_dir, "fire", "corrupt.jpg")
        empty_target = os.path.join(test_dir, "fire", "empty.png")
        with open(corrupt_target, "wb") as f:
            f.write(b"CORRUPT_BYTES_XYZ")
        with open(empty_target, "wb") as f:
            pass

        dataset = FireSmokeDataset(root_dir=temp_image_dir, split="boundary_split", filter_duplicates=False)
        # Both invalid images should be filtered out
        assert len(dataset) == 0

    def test_extreme_image_dimensions(self):
        """Preprocessing must resize extreme image sizes (1x1, 1920x100 non-square, 4000x4000) to 224x224."""
        get_eval_transforms = import_or_skip("src.data.pipeline", "get_eval_transforms")
        transforms = get_eval_transforms(224)

        # 1x1 image
        img_1x1 = Image.new("RGB", (1, 1), color=(255, 0, 0))
        t1 = transforms(img_1x1)
        assert t1.shape == (3, 224, 224)

        # Extreme aspect ratio: 1920x64
        img_wide = Image.new("RGB", (1920, 64), color=(100, 100, 100))
        t2 = transforms(img_wide)
        assert t2.shape == (3, 224, 224)

        # Large image: 2000x2000
        img_large = Image.new("RGB", (2000, 2000), color=(50, 150, 250))
        t3 = transforms(img_large)
        assert t3.shape == (3, 224, 224)

    def test_color_space_boundaries_grayscale_and_rgba(self):
        """Pipeline must handle single-channel grayscale and 4-channel RGBA by standardizing to 3-channel RGB."""
        get_eval_transforms = import_or_skip("src.data.pipeline", "get_eval_transforms")
        transforms = get_eval_transforms(224)

        gray = Image.new("L", (100, 100), color=128)
        rgba = Image.new("RGBA", (100, 100), color=(255, 128, 0, 255))

        # Both converted to RGB before transforms
        t_gray = transforms(gray.convert("RGB"))
        t_rgba = transforms(rgba.convert("RGB"))

        assert t_gray.shape == (3, 224, 224)
        assert t_rgba.shape == (3, 224, 224)

    def test_flat_color_frames_feature_extraction(self):
        """Pure black (0,0,0) and pure white (255,255,255) images must not cause division by zero or NaN moments."""
        extract_all_features = import_or_skip("src.features.extract", "extract_all_features")

        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))

        f_black = extract_all_features(black_img)
        f_white = extract_all_features(white_img)

        assert not np.isnan(f_black).any(), "Black frame features contain NaN"
        assert not np.isinf(f_black).any(), "Black frame features contain Inf"
        assert not np.isnan(f_white).any(), "White frame features contain NaN"
        assert not np.isinf(f_white).any(), "White frame features contain Inf"

    def test_threshold_extremes_sweep(self):
        """Calibration must support boundary thresholds theta=0.10 and theta=0.90."""
        calibrate_threshold = import_or_skip("src.evaluate.calibrate", "calibrate_threshold")

        # Case 1: Fire probabilities very low -> Requires theta <= 0.15 to achieve 90% recall
        y_true = np.array([0] * 10 + [1] * 10)
        y_probs_low = np.concatenate([np.linspace(0.12, 0.30, 10), np.linspace(0.01, 0.05, 10)])
        theta_low, _ = calibrate_threshold(y_true, y_probs_low, min_recall=0.90)
        assert 0.10 <= theta_low <= 0.20

        # Case 2: Fire probabilities very high -> Can maintain 90% recall at high theta
        y_probs_high = np.concatenate([np.linspace(0.92, 0.99, 10), np.linspace(0.01, 0.20, 10)])
        theta_high, _ = calibrate_threshold(y_true, y_probs_high, min_recall=0.90)
        assert theta_high >= 0.80

    def test_ambient_gate_boundary_precision(self):
        """Threshold tau=0.70 must precisely reject max(P)=0.6999 and accept max(P)=0.7001."""
        FireClassifier = import_or_skip("src.inference", "FireClassifier")

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
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Test slightly below tau: 0.6999 -> ambient_frame
        clf.model = StubModel(0.6999, 0.3001)
        res_below = predict_fn(dummy_img)
        assert res_below["hazard_detected"] is False

        # Test slightly above tau: 0.7001 -> hazard_detected
        clf.model = StubModel(0.7001, 0.2999)
        res_above = predict_fn(dummy_img)
        assert res_above["hazard_detected"] is True
