"""Adversarial stress-testing suite for Milestone 2 (ISSUE-02).
Tests feature extraction and LightGBM baseline against extreme dimensions,
edge cases, numerical stability (NaN/Inf), and PIL vs NumPy parity.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest
from PIL import Image

from src.features.extract import (
    extract_color_histogram,
    extract_color_moments,
    extract_texture_features,
    extract_all_features,
    _to_rgb_array,
)
from src.models.train_lightgbm import (
    train_lightgbm,
    extract_features_from_dataset,
    save_model,
    load_model,
)


class TestExtremeDimensions:
    """Stress test boundary and extreme spatial dimensions."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 3),      # 1x1 RGB
            (1, 1, 4),      # 1x1 RGBA
            (1, 1),         # 1x1 2D grayscale
            (1, 1, 1),      # 1x1 3D grayscale
            (1, 2, 3),      # 1x2 horizontal line
            (2, 1, 3),      # 2x1 vertical line
            (2, 2, 3),      # 2x2 sub-LBP kernel
            (3, 3, 3),      # 3x3 exact minimal LBP kernel
            (1, 5000, 3),   # Extreme wide aspect ratio
            (5000, 1, 3),   # Extreme tall aspect ratio
            (15, 73, 3),    # Odd prime non-square dimensions
            (224, 100, 3),  # Typical non-square frame
        ],
    )
    def test_extreme_dimensions_yield_valid_features(self, shape: tuple[int, ...]):
        arr = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        feats = extract_all_features(arr)

        assert isinstance(feats, np.ndarray)
        assert feats.ndim == 1
        assert feats.shape[0] == 134, f"Feature vector dimension mismatch: got {feats.shape[0]}, expected 134"
        assert not np.isnan(feats).any(), f"NaN encountered for input shape {shape}"
        assert not np.isinf(feats).any(), f"Inf encountered for input shape {shape}"

    def test_pil_image_1x1_all_modes(self):
        for mode in ("RGB", "RGBA", "L"):
            img = Image.new(mode, (1, 1), color=0)
            feats = extract_all_features(img)
            assert feats.shape[0] == 134
            assert np.all(np.isfinite(feats))


class TestNumericalStabilityAndFlatImages:
    """Stress test flat, uniform, and single-pixel outlier images."""

    @pytest.mark.parametrize(
        "val,name",
        [
            (0, "flat_black"),
            (255, "flat_white"),
            (128, "flat_gray"),
            (1, "near_black"),
            (254, "near_white"),
        ],
    )
    def test_flat_uniform_images_no_nans(self, val: int, name: str):
        arr = np.full((224, 224, 3), val, dtype=np.uint8)
        feats = extract_all_features(arr)

        assert feats.shape == (134,)
        assert not np.isnan(feats).any(), f"NaN found in {name}"
        assert not np.isinf(feats).any(), f"Inf found in {name}"

        # In flat images, standard deviations should be 0.0 and skewness 0.0
        moments = extract_color_moments(arr)
        # Moments: 6 channels (RGB, HSV) * 3 stats (mean, std, skew)
        # std indices: 1, 4, 7, 10, 13, 16
        # skew indices: 2, 5, 8, 11, 14, 17
        for std_idx in (1, 4, 7, 10, 13, 16):
            assert moments[std_idx] == 0.0, f"Std at {std_idx} should be 0 for {name}"
        for skew_idx in (2, 5, 8, 11, 14, 17):
            assert moments[skew_idx] == 0.0, f"Skew at {skew_idx} should be 0 for {name}"

    def test_single_pixel_noise_in_black(self):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[112, 112] = [255, 255, 255]
        feats = extract_all_features(arr)

        assert feats.shape == (134,)
        assert np.all(np.isfinite(feats))

    def test_single_pixel_noise_in_white(self):
        arr = np.full((224, 224, 3), 255, dtype=np.uint8)
        arr[112, 112] = [0, 0, 0]
        feats = extract_all_features(arr)

        assert feats.shape == (134,)
        assert np.all(np.isfinite(feats))


class TestFormatParityPILAndNumPy:
    """Validate strict parity between PIL Image and NumPy array inputs."""

    def test_rgb_parity(self):
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")

        feat_arr = extract_all_features(arr)
        feat_pil = extract_all_features(pil_img)

        assert np.allclose(feat_arr, feat_pil, atol=1e-5), "RGB PIL and NumPy features must be identical"

    def test_rgba_parity(self):
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (128, 128, 4), dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode="RGBA")

        feat_arr = extract_all_features(arr)
        feat_pil = extract_all_features(pil_img)

        assert np.allclose(feat_arr, feat_pil, atol=1e-5), "RGBA PIL and NumPy features must be identical"

    def test_grayscale_parity(self):
        rng = np.random.RandomState(42)
        arr_2d = rng.randint(0, 256, (128, 128), dtype=np.uint8)
        arr_3d = arr_2d[:, :, None]
        pil_l = Image.fromarray(arr_2d, mode="L")

        feat_2d = extract_all_features(arr_2d)
        feat_3d = extract_all_features(arr_3d)
        feat_pil = extract_all_features(pil_l)

        assert np.allclose(feat_2d, feat_pil, atol=1e-5), "2D Grayscale vs PIL L parity failed"
        assert np.allclose(feat_3d, feat_pil, atol=1e-5), "3D Grayscale vs PIL L parity failed"

    def test_float_input_scaling_parity(self):
        rng = np.random.RandomState(42)
        arr_u8 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        arr_f32 = (arr_u8.astype(np.float32) / 255.0)

        feat_u8 = extract_all_features(arr_u8)
        feat_f32 = extract_all_features(arr_f32)

        # Due to float to uint8 rounding (f32 * 255.0), max absolute error should be small
        assert np.allclose(feat_u8, feat_f32, atol=0.05), "Float [0, 1] scaled array must closely match uint8"


class TestFeatureContractIntegrity:
    """Validate specific mathematical invariants of descriptors."""

    def test_color_histogram_sums(self):
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        hist = extract_color_histogram(arr, bins=16)

        assert hist.shape == (96,)
        # 6 individual histograms of 16 bins each: R, G, B, H, S, V
        for i in range(6):
            channel_hist = hist[i * 16 : (i + 1) * 16]
            assert np.isclose(np.sum(channel_hist), 1.0, atol=1e-4), f"Histogram channel {i} does not sum to 1.0"

    def test_texture_features_sub_3x3(self):
        # Images smaller than 3x3 cannot compute 3x3 LBP, so LBP histogram must be zeros
        arr = np.random.randint(0, 256, (2, 2), dtype=np.uint8)
        texture = extract_texture_features(arr)

        assert texture.shape == (20,)
        # First 4 are Sobel stats, next 16 are LBP histogram
        lbp_hist = texture[4:]
        assert np.all(lbp_hist == 0.0), "LBP histogram must be zeros for < 3x3 images"
        assert np.all(np.isfinite(texture))


class TestLightGBMModelIntegrity:
    """Validate LightGBM training, class weights, serialization, and latency."""

    def test_train_lightgbm_class_weight_and_predictions(self):
        rng = np.random.RandomState(42)
        n_samples, n_features = 100, 134
        X_train = rng.randn(n_samples, n_features).astype(np.float32)
        y_train = rng.choice([0, 1], size=n_samples, p=[0.3, 0.7])  # Imbalanced

        X_val = rng.randn(20, n_features).astype(np.float32)
        y_val = rng.choice([0, 1], size=20)

        t0 = time.perf_counter()
        model = train_lightgbm(X_train, y_train, X_val, y_val, config={"n_estimators": 50})
        train_elapsed = time.perf_counter() - t0

        assert train_elapsed < 60.0, f"Training took too long: {train_elapsed:.2f}s (must be < 60s)"

        probs = model.predict_proba(X_val)
        assert probs.shape == (20, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_model_serialization_joblib_roundtrip(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 134).astype(np.float32)
        y = rng.choice([0, 1], size=40)

        model = train_lightgbm(X, y, config={"n_estimators": 20})
        orig_preds = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "tier1_lightgbm.joblib"
            save_model(model, model_path)
            loaded_model = load_model(model_path)

            loaded_preds = loaded_model.predict_proba(X)
            assert np.allclose(orig_preds, loaded_preds, atol=1e-6)

    def test_cpu_inference_latency_benchmark(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 134).astype(np.float32)
        y = rng.choice([0, 1], size=50)

        # Single-sample inference with n_jobs=1 avoids OpenMP thread pool creation overhead
        model = train_lightgbm(X, y, config={"n_estimators": 50, "n_jobs": 1})
        sample = X[0:1]

        # Warmup
        for _ in range(5):
            model.predict_proba(sample)

        # Benchmark 100 single-sample inferences
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict_proba(sample)
        avg_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

        # Tier 1 baseline latency with n_jobs=1 should be ultra-fast (< 5ms/sample on CPU)
        assert avg_ms < 5.0, f"Single sample latency too high: {avg_ms:.3f} ms/sample"

    def test_extract_features_from_empty_dataset(self):
        empty_dataset = []
        feats, labels = extract_features_from_dataset(empty_dataset)
        assert feats.shape == (0, 0)
        assert labels.shape == (0,)
