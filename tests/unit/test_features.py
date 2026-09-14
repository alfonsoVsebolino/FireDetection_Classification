"""Unit tests for handcrafted feature extraction (color histograms, moments, texture) and LightGBM."""

import numpy as np
import pytest
from PIL import Image
from tests.helpers import import_or_skip


def test_extract_color_histogram(synthetic_rgb_image):
    """F5: Verify extract_color_histogram returns 96-element normalized feature vector."""
    extract_color_histogram = import_or_skip("src.features.extract", "extract_color_histogram")
    img_arr = np.array(synthetic_rgb_image)
    hist = extract_color_histogram(img_arr, bins=16)

    assert isinstance(hist, np.ndarray)
    assert hist.ndim == 1
    assert hist.shape[0] == 96, f"Expected 96 features (16 bins * 3 channels * 2 spaces), got {hist.shape[0]}"
    assert np.all(np.isfinite(hist)), "Feature vector must contain only finite numbers"
    assert np.all(hist >= 0.0), "Histogram values must be non-negative"


def test_extract_color_moments(synthetic_rgb_image):
    """F5: Verify extract_color_moments returns mean, std, skewness per channel."""
    extract_color_moments = import_or_skip("src.features.extract", "extract_color_moments")
    img_arr = np.array(synthetic_rgb_image)
    moments = extract_color_moments(img_arr)

    assert isinstance(moments, np.ndarray)
    assert moments.ndim == 1
    # RGB 3x3=9 or RGB+HSV 6x3=18 moments
    assert moments.shape[0] in (9, 18), f"Expected 9 or 18 moments, got {moments.shape[0]}"
    assert np.all(np.isfinite(moments))


def test_extract_texture_features(synthetic_rgb_image):
    """F5: Verify extract_texture_features handles grayscale array and outputs finite vector."""
    extract_texture_features = import_or_skip("src.features.extract", "extract_texture_features")
    img_gray = np.array(synthetic_rgb_image.convert("L"))
    texture = extract_texture_features(img_gray)

    assert isinstance(texture, np.ndarray)
    assert texture.ndim == 1
    assert texture.shape[0] >= 4, "Texture descriptor should contain at least Sobel/LBP stats"
    assert np.all(np.isfinite(texture))


def test_extract_all_features_concatenation(synthetic_rgb_image):
    """F5: Verify extract_all_features accepts both PIL Image and ndarray and returns 1D vector."""
    extract_all_features = import_or_skip("src.features.extract", "extract_all_features")

    vec_pil = extract_all_features(synthetic_rgb_image)
    vec_arr = extract_all_features(np.array(synthetic_rgb_image))

    assert isinstance(vec_pil, np.ndarray)
    assert isinstance(vec_arr, np.ndarray)
    assert vec_pil.ndim == 1
    assert vec_pil.shape == vec_arr.shape
    assert np.allclose(vec_pil, vec_arr, atol=1e-5)


def test_train_lightgbm_smoke():
    """F6: Verify LightGBM training function trains and outputs binary probabilities."""
    train_lightgbm = import_or_skip("src.models.train_lightgbm", "train_lightgbm")

    # Generate synthetic feature matrices
    np.random.seed(42)
    n_samples, n_features = 40, 120
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(n_samples,))
    X_val = np.random.randn(10, n_features).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(10,))

    model = train_lightgbm(X_train, y_train, X_val, y_val, config={"n_estimators": 5, "learning_rate": 0.1})
    assert hasattr(model, "predict_proba"), "Trained model must have predict_proba method"

    probs = model.predict_proba(X_val)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
