from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


def _to_rgb_array(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
        arr = arr * 255.0
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def extract_color_histogram(img: np.ndarray, bins: int = 16) -> np.ndarray:
    rgb = _to_rgb_array(img)
    total_pixels = float(rgb.shape[0] * rgb.shape[1])
    norm = max(total_pixels, 1.0)

    hists = []
    for c in range(3):
        h, _ = np.histogram(rgb[:, :, c], bins=bins, range=(0, 256))
        hists.append((h.astype(np.float32) / norm))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))
    hists.append((h_hist.astype(np.float32) / norm))
    for c in (1, 2):
        h, _ = np.histogram(hsv[:, :, c], bins=bins, range=(0, 256))
        hists.append((h.astype(np.float32) / norm))

    return np.concatenate(hists).astype(np.float32)


def extract_color_moments(img: np.ndarray) -> np.ndarray:
    rgb = _to_rgb_array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    moments = []
    for arr in (rgb, hsv):
        for c in range(3):
            ch = arr[:, :, c].astype(np.float64)
            mean_val = float(np.mean(ch))
            std_val = float(np.std(ch))
            skew_val = 0.0
            if std_val > 1e-7:
                skew_val = float(np.mean(((ch - mean_val) / std_val) ** 3))
            moments.extend([mean_val, std_val, skew_val])

    return np.array(moments, dtype=np.float32)


def extract_texture_features(img_gray: np.ndarray) -> np.ndarray:
    if img_gray.ndim == 3:
        if img_gray.shape[2] == 1:
            gray = img_gray[:, :, 0]
        elif img_gray.shape[2] == 3:
            gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        elif img_gray.shape[2] == 4:
            gray = cv2.cvtColor(img_gray[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = img_gray[:, :, 0]
    else:
        gray = img_gray

    gray_f = gray.astype(np.float64)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    sobel_stats = np.array([
        float(np.mean(grad_mag)),
        float(np.std(grad_mag)),
        float(np.var(grad_mag)),
        float(np.max(grad_mag)),
    ], dtype=np.float32)

    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        u8 = np.clip(gray, 0, 255).astype(np.uint8)
        c = u8[1:-1, 1:-1]
        neighbors = (
            u8[0:-2, 0:-2], u8[0:-2, 1:-1], u8[0:-2, 2:],
            u8[1:-1, 2:], u8[2:, 2:], u8[2:, 1:-1],
            u8[2:, 0:-2], u8[1:-1, 0:-2],
        )
        lbp = np.zeros_like(c, dtype=np.uint8)
        for i, n in enumerate(neighbors):
            lbp |= ((n >= c).astype(np.uint8) << i)
        lbp_hist, _ = np.histogram(lbp, bins=16, range=(0, 256))
        total = max(float(c.size), 1.0)
        lbp_norm = lbp_hist.astype(np.float32) / total
    else:
        lbp_norm = np.zeros(16, dtype=np.float32)

    return np.concatenate([sobel_stats, lbp_norm]).astype(np.float32)


def extract_all_features(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    rgb = _to_rgb_array(img)
    hist = extract_color_histogram(rgb, bins=16)
    moments = extract_color_moments(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    texture = extract_texture_features(gray)
    return np.concatenate([hist, moments, texture]).astype(np.float32)


def extract_features_dataset(
    dataset_or_loader: Any,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = getattr(dataset_or_loader, "dataset", dataset_or_loader)
    from src.models.train_lightgbm import extract_features_from_dataset

    return extract_features_from_dataset(dataset, max_samples=max_samples)

