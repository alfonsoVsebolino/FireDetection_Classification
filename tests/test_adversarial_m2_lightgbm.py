"""Adversarial stress test suite for Milestone 2: Handcrafted Features & LightGBM Baseline."""

import os
import time
import numpy as np
import pytest
from PIL import Image

from src.models.train_lightgbm import (
    train_lightgbm,
    extract_features_from_dataset,
    save_model,
    load_model,
)
from src.features.extract import (
    extract_color_histogram,
    extract_color_moments,
    extract_texture_features,
    extract_all_features,
)


class TestClassWeightPenalty:
    """Adversarial tests for class weight penalty {0: 2.0, 1: 1.0}."""

    def test_class_0_weight_penalizes_false_negatives(self):
        """Verify that giving class 0 a 2.0 weight reduces class 0 false negatives vs unweighted."""
        np.random.seed(42)
        # Create overlapping Gaussian clusters where classification is non-trivial
        n_samples = 300
        n_features = 30
        # Class 0 centered around 0.2, Class 1 centered around -0.2 (high overlap)
        X0 = np.random.normal(loc=0.2, scale=1.0, size=(n_samples // 2, n_features))
        X1 = np.random.normal(loc=-0.2, scale=1.0, size=(n_samples // 2, n_features))
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)

        # Test set
        X_test0 = np.random.normal(loc=0.2, scale=1.0, size=(100, n_features)).astype(np.float32)
        X_test1 = np.random.normal(loc=-0.2, scale=1.0, size=(100, n_features)).astype(np.float32)
        X_test = np.vstack([X_test0, X_test1])
        y_test = np.array([0] * 100 + [1] * 100, dtype=np.int64)

        # 1. Unweighted model
        model_unweighted = train_lightgbm(
            X, y, config={"class_weight": {0: 1.0, 1: 1.0}, "n_estimators": 50, "random_state": 42}
        )
        preds_unweighted = model_unweighted.predict(X_test)
        fn_unweighted = np.sum((y_test == 0) & (preds_unweighted == 1))

        # 2. Weighted model (default: {0: 2.0, 1: 1.0})
        model_weighted = train_lightgbm(
            X, y, config={"class_weight": {0: 2.0, 1: 1.0}, "n_estimators": 50, "random_state": 42}
        )
        preds_weighted = model_weighted.predict(X_test)
        fn_weighted = np.sum((y_test == 0) & (preds_weighted == 1))

        # 3. Heavily weighted model ({0: 5.0, 1: 1.0})
        model_heavy = train_lightgbm(
            X, y, config={"class_weight": {0: 5.0, 1: 1.0}, "n_estimators": 50, "random_state": 42}
        )
        preds_heavy = model_heavy.predict(X_test)
        fn_heavy = np.sum((y_test == 0) & (preds_heavy == 1))

        # Probabilities for class 0 should shift higher with higher class 0 weight
        p0_unweighted = model_unweighted.predict_proba(X_test)[:, 0].mean()
        p0_weighted = model_weighted.predict_proba(X_test)[:, 0].mean()
        p0_heavy = model_heavy.predict_proba(X_test)[:, 0].mean()

        print(f"\n[Class Weight] FN unweighted: {fn_unweighted}, weighted: {fn_weighted}, heavy: {fn_heavy}")
        print(f"[Class Weight] Mean P(class 0): unweighted={p0_unweighted:.4f}, weighted={p0_weighted:.4f}, heavy={p0_heavy:.4f}")

        assert p0_weighted > p0_unweighted, "Weighted model must produce higher average probability for class 0"
        assert p0_heavy > p0_weighted, "Heavily weighted model must produce even higher average probability for class 0"
        assert fn_weighted <= fn_unweighted, f"Weighted model FN ({fn_weighted}) should be <= unweighted FN ({fn_unweighted})"
        assert fn_heavy <= fn_weighted, f"Heavily weighted model FN ({fn_heavy}) should be <= weighted FN ({fn_weighted})"


class TestEdgeCasesAndImbalance:
    """Stress tests with single-class, extreme imbalance, and boundary data."""

    def test_single_class_training_handling(self):
        """Test behavior when dataset contains only one class."""
        X = np.random.randn(50, 20).astype(np.float32)
        y_all_zero = np.zeros(50, dtype=np.int64)

        # Expect exception or graceful failure
        with pytest.raises(Exception) as exc_info:
            train_lightgbm(X, y_all_zero, config={"n_estimators": 10})
        print(f"\n[Single Class] Raised expected exception: {exc_info.value}")

    def test_severe_imbalance_100_to_1(self):
        """Test model stability and outputs under 100:1 class imbalance."""
        np.random.seed(42)
        n_majority, n_minority = 500, 5
        X_maj = np.random.randn(n_majority, 20).astype(np.float32)
        X_min = np.random.randn(n_minority, 20).astype(np.float32) + 1.0
        X = np.vstack([X_maj, X_min])
        y = np.array([1] * n_majority + [0] * n_minority, dtype=np.int64)

        model = train_lightgbm(X, y, config={"n_estimators": 20})
        probs = model.predict_proba(X[:10])

        assert probs.shape == (10, 2), f"Expected (10, 2), got {probs.shape}"
        assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be within [0, 1]"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1.0"

    def test_predict_proba_edge_shapes(self):
        """Test predict_proba with single sample, empty array, and calibration."""
        np.random.seed(42)
        X_train = np.random.randn(40, 20).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(40,))
        model = train_lightgbm(X_train, y_train, config={"n_estimators": 10})

        # Single sample (1, 20)
        single_x = np.random.randn(1, 20).astype(np.float32)
        probs_single = model.predict_proba(single_x)
        assert probs_single.shape == (1, 2)
        assert np.isclose(probs_single.sum(), 1.0, atol=1e-5)

        # 0 samples (0, 20) — LightGBM rejects empty arrays; confirm ValueError raised
        empty_x = np.empty((0, 20), dtype=np.float32)
        with pytest.raises(ValueError):
            model.predict_proba(empty_x)


class TestModelSerialization:
    """Test save_model and load_model for both joblib and txt formats."""

    def test_joblib_save_and_load(self, tmp_path):
        """Verify model round-trip via joblib."""
        np.random.seed(42)
        X = np.random.randn(30, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=(30,))
        model = train_lightgbm(X, y, config={"n_estimators": 10})

        pkl_path = tmp_path / "model.joblib"
        save_model(model, pkl_path)
        assert pkl_path.exists()

        loaded_model = load_model(pkl_path)
        p_orig = model.predict_proba(X[:5])
        p_loaded = loaded_model.predict_proba(X[:5])
        assert np.allclose(p_orig, p_loaded, atol=1e-5)

    def test_txt_save_and_load(self, tmp_path):
        """Verify model round-trip via txt format."""
        np.random.seed(42)
        X = np.random.randn(30, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=(30,))
        model = train_lightgbm(X, y, config={"n_estimators": 10})

        txt_path = tmp_path / "model.txt"
        save_model(model, txt_path)
        assert txt_path.exists()

        loaded_model = load_model(txt_path)
        p_orig = model.predict_proba(X[:5])
        p_loaded = loaded_model.predict_proba(X[:5])
        assert np.allclose(p_orig, p_loaded, atol=1e-4)


class TestCPULatencyBenchmark:
    """Measure actual CPU training execution latency (< 60s limit)."""

    def test_training_latency_on_large_synthetic_data(self):
        """Verify CPU training latency on N=2000, D=134 is well under 60 seconds."""
        np.random.seed(42)
        n_samples = 2000
        n_features = 134  # Matching extract_all_features dimension

        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(n_samples,))
        X_val = np.random.randn(400, n_features).astype(np.float32)
        y_val = np.random.randint(0, 2, size=(400,))

        start_time = time.perf_counter()
        model = train_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            config={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
            },
        )
        elapsed_s = time.perf_counter() - start_time
        print(f"\n[Latency] CPU training on {n_samples} samples with 200 trees: {elapsed_s:.3f} seconds")

        assert elapsed_s < 60.0, f"Training latency {elapsed_s:.2f}s exceeded 60s limit"
        assert hasattr(model, "predict_proba")
