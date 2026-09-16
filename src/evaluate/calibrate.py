"""Validation threshold calibration for fire recall safety constraint."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def calibrate_threshold(
    y_true: np.ndarray,
    y_probs_fire: np.ndarray,
    min_recall: float = 0.90,
    sweep_lo: float = 0.10,
    sweep_hi: float = 0.90,
    sweep_step: float = 0.02,
) -> Tuple[float, Dict[str, float]]:
    """Sweep θ ∈ [sweep_lo, sweep_hi] and return the θ* that satisfies
    Recall_fire(val) >= min_recall with the highest precision.

    Parameters
    ----------
    y_true:        Ground-truth labels (0=fire, 1=smoke).
    y_probs_fire:  Predicted probability for class 0 (fire).
    min_recall:    Minimum acceptable fire recall (default 0.90).

    Returns
    -------
    (theta_star, metrics_dict)
        theta_star  – calibrated decision threshold
        metrics_dict – recall_fire, precision_fire, f1, accuracy at theta_star
    """
    thresholds = np.arange(sweep_lo, sweep_hi + sweep_step / 2, sweep_step)
    y_true = np.asarray(y_true)
    y_probs_fire = np.asarray(y_probs_fire)

    best_theta: float = sweep_lo
    best_precision: float = -1.0
    best_metrics: Dict[str, float] = {}

    for theta in thresholds:
        preds_binary = np.where(y_probs_fire >= theta, 0, 1)

        fire_mask = y_true == 0
        tp = int(((preds_binary == 0) & fire_mask).sum())
        fn = int(((preds_binary == 1) & fire_mask).sum())
        fp = int(((preds_binary == 0) & (y_true == 1)).sum())
        tn = int(((preds_binary == 1) & (y_true == 1)).sum())

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(len(y_true), 1)

        if recall >= min_recall and precision >= best_precision:
            best_precision = precision
            best_theta = float(theta)
            best_metrics = {
                "recall_fire": recall,
                "precision_fire": precision,
                "f1": f1,
                "accuracy": accuracy,
                "threshold": best_theta,
            }

    # Fallback: if no threshold satisfies min_recall, return sweep_lo with its metrics
    if not best_metrics:
        theta = sweep_lo
        preds_binary = np.where(y_probs_fire >= theta, 0, 1)
        fire_mask = y_true == 0
        tp = int(((preds_binary == 0) & fire_mask).sum())
        fn = int(((preds_binary == 1) & fire_mask).sum())
        fp = int(((preds_binary == 0) & (y_true == 1)).sum())
        tn = int(((preds_binary == 1) & (y_true == 1)).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(len(y_true), 1)
        best_theta = float(theta)
        best_metrics = {
            "recall_fire": recall,
            "precision_fire": precision,
            "f1": f1,
            "accuracy": accuracy,
            "threshold": best_theta,
        }

    return best_theta, best_metrics
