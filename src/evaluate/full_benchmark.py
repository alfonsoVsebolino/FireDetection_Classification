"""Full test split tier benchmarking and visualization suite."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.evaluate.calibrate import calibrate_threshold
from src.features.extract import extract_all_features


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics for Class 0 (Fire) & Class 1 (Smoke)."""
    # pos_label=0 for fire
    fire_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    fire_precision = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "fire_recall": fire_recall,
        "fire_precision": fire_precision,
        "accuracy": acc,
        "fire_f1": f1,
        "confusion_matrix": cm,
    }


def benchmark_pytorch_cpu(
    model: nn.Module,
    dataset: Any,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Profile single-sample (batch_size=1) CPU inference latency on dataset samples."""
    try:
        orig_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        orig_device = torch.device("cpu")

    model_cpu = model.to(torch.device("cpu")).eval()
    latencies: list[float] = []
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    with torch.no_grad():
        for i in range(total):
            item = dataset[i]
            tensor = item[0].unsqueeze(0).to(torch.device("cpu"))  # (1, 3, 224, 224)
            t0 = time.perf_counter()
            _ = model_cpu(tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    # Restore original device so model is not mutated for subsequent steps
    model.to(orig_device)

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "latencies": latencies,
    }


def benchmark_lightgbm_cpu(
    model_lgbm: Any,
    dataset: Any,
    X_test: np.ndarray,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Profile LightGBM single-sample CPU latency: both raw model and end-to-end."""
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    # 1. Raw tabular model inference latency
    raw_latencies: list[float] = []
    for i in range(total):
        row = X_test[i : i + 1]
        t0 = time.perf_counter()
        _ = model_lgbm.predict_proba(row)
        raw_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. End-to-end latency (image feature extraction + predict)
    e2e_latencies: list[float] = []
    for i in range(total):
        item = dataset[i]
        path = item[2] if len(item) >= 3 else None
        t0 = time.perf_counter()
        if path and os.path.isfile(str(path)):
            with Image.open(str(path)) as im:
                feat = extract_all_features(im)
        else:
            feat = extract_all_features(item[0])
        _ = model_lgbm.predict_proba(feat.reshape(1, -1))
        e2e_latencies.append((time.perf_counter() - t0) * 1000.0)

    raw_arr = np.array(raw_latencies)
    e2e_arr = np.array(e2e_latencies)

    return {
        "raw_model": {
            "mean_ms": float(np.mean(raw_arr)),
            "median_ms": float(np.median(raw_arr)),
            "p95_ms": float(np.percentile(raw_arr, 95)),
        },
        "end_to_end": {
            "mean_ms": float(np.mean(e2e_arr)),
            "median_ms": float(np.median(e2e_arr)),
            "p95_ms": float(np.percentile(e2e_arr, 95)),
        },
    }


def evaluate_all_tiers(
    model_lgbm: Any,
    model_resnet: nn.Module,
    model_vit: nn.Module,
    test_loader: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    theta_star_lgbm: float = 0.50,
    theta_star_resnet: float = 0.50,
    theta_star_vit: float = 0.50,
    device: Optional[torch.device] = None,
    output_path: str = "models/test_benchmark_tiers.json",
) -> Dict[str, Any]:
    """Evaluate 3 tiers across 100% test split with calibrated theta* and single-sample CPU latency."""
    dataset = test_loader.dataset
    y_true = np.asarray(y_test)

    # -------------------------------------------------------------
    # 1. Predictions & Probabilities
    # -------------------------------------------------------------
    # Tier 1: LightGBM (class 0 is fire)
    lgbm_probs = model_lgbm.predict_proba(X_test)[:, 0]
    lgbm_preds_default = np.where(lgbm_probs >= 0.50, 0, 1)
    lgbm_preds_calib = np.where(lgbm_probs >= theta_star_lgbm, 0, 1)

    # Tier 2: ResNet18
    dev = device or torch.device("cpu")
    model_resnet = model_resnet.to(dev).eval()
    all_resnet_probs: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(dev)
            p = torch.softmax(model_resnet(imgs), dim=1)[:, 0].cpu().numpy()
            all_resnet_probs.extend(p)
    resnet_probs = np.array(all_resnet_probs)
    resnet_preds_default = np.where(resnet_probs >= 0.50, 0, 1)
    resnet_preds_calib = np.where(resnet_probs >= theta_star_resnet, 0, 1)

    # Tier 3: DeiT-Tiny
    model_vit = model_vit.to(dev).eval()
    all_vit_probs: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(dev)
            p = torch.softmax(model_vit(imgs), dim=1)[:, 0].cpu().numpy()
            all_vit_probs.extend(p)
    vit_probs = np.array(all_vit_probs)
    vit_preds_default = np.where(vit_probs >= 0.50, 0, 1)
    vit_preds_calib = np.where(vit_probs >= theta_star_vit, 0, 1)

    # -------------------------------------------------------------
    # 2. Single-Sample CPU Latency Benchmarking
    # -------------------------------------------------------------
    print("Profiling single-sample CPU latency across full test split...")
    lgbm_latency = benchmark_lightgbm_cpu(model_lgbm, dataset, X_test)
    resnet_latency = benchmark_pytorch_cpu(model_resnet, dataset)
    vit_latency = benchmark_pytorch_cpu(model_vit, dataset)

    # Restore models to target device
    model_resnet.to(dev)
    model_vit.to(dev)

    # -------------------------------------------------------------
    # 3. Assemble Results Dictionary
    # -------------------------------------------------------------
    results: Dict[str, Any] = {
        "Tier1_LightGBM": {
            "theta_calibrated": theta_star_lgbm,
            "metrics_default_050": _calc_metrics(y_true, lgbm_preds_default),
            "metrics_calibrated": _calc_metrics(y_true, lgbm_preds_calib),
            "latency_cpu_ms": lgbm_latency["raw_model"],
            "latency_cpu_e2e_ms": lgbm_latency["end_to_end"],
        },
        "Tier2_ResNet18": {
            "theta_calibrated": theta_star_resnet,
            "metrics_default_050": _calc_metrics(y_true, resnet_preds_default),
            "metrics_calibrated": _calc_metrics(y_true, resnet_preds_calib),
            "latency_cpu_ms": {
                "mean_ms": resnet_latency["mean_ms"],
                "median_ms": resnet_latency["median_ms"],
                "p95_ms": resnet_latency["p95_ms"],
            },
        },
        "Tier3_DeiT_Tiny": {
            "theta_calibrated": theta_star_vit,
            "metrics_default_050": _calc_metrics(y_true, vit_preds_default),
            "metrics_calibrated": _calc_metrics(y_true, vit_preds_calib),
            "latency_cpu_ms": {
                "mean_ms": vit_latency["mean_ms"],
                "median_ms": vit_latency["median_ms"],
                "p95_ms": vit_latency["p95_ms"],
            },
        },
    }

    # Save artifact
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved full test benchmark to {output_path}")

    return results


def plot_benchmark_dashboard(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Render 4-panel dashboard with metrics bar charts, latencies, trade-off scatter, and confusion matrices."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    tiers = ["Tier 1\n(LightGBM)", "Tier 2\n(ResNet18)", "Tier 3\n(DeiT-Tiny)"]
    keys = ["Tier1_LightGBM", "Tier2_ResNet18", "Tier3_DeiT_Tiny"]

    # -------------------------------------------------------------
    # Panel 1: Fire Recall & Accuracy with 90% Safety Line
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    recalls = [results[k]["metrics_calibrated"]["fire_recall"] for k in keys]
    accuracies = [results[k]["metrics_calibrated"]["accuracy"] for k in keys]

    x = np.arange(len(tiers))
    width = 0.35
    b1 = ax1.bar(x - width / 2, recalls, width, label="Fire Recall (θ*)", color="#e74c3c")
    b2 = ax1.bar(x + width / 2, accuracies, width, label="Accuracy (θ*)", color="#3498db")
    ax1.axhline(0.90, color="darkred", linestyle="--", linewidth=1.8, label="90% Safety Floor")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Test Safety Metrics (Calibrated θ*)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers)
    ax1.legend(loc="lower right")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.annotate(f"{h:.1%}", (bar.get_x() + bar.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # -------------------------------------------------------------
    # Panel 2: CPU Latency Comparison (Mean, Median, P95)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    latency_labels = [
        "LGBM\n(Model)",
        "LGBM\n(End-to-End)",
        "ResNet18\n(CPU)",
        "DeiT-Tiny\n(CPU)",
    ]
    means = [
        results["Tier1_LightGBM"]["latency_cpu_ms"]["mean_ms"],
        results["Tier1_LightGBM"]["latency_cpu_e2e_ms"]["mean_ms"],
        results["Tier2_ResNet18"]["latency_cpu_ms"]["mean_ms"],
        results["Tier3_DeiT_Tiny"]["latency_cpu_ms"]["mean_ms"],
    ]
    p95s = [
        results["Tier1_LightGBM"]["latency_cpu_ms"]["p95_ms"],
        results["Tier1_LightGBM"]["latency_cpu_e2e_ms"]["p95_ms"],
        results["Tier2_ResNet18"]["latency_cpu_ms"]["p95_ms"],
        results["Tier3_DeiT_Tiny"]["latency_cpu_ms"]["p95_ms"],
    ]

    xl = np.arange(len(latency_labels))
    b_mean = ax2.bar(xl - 0.18, means, 0.35, label="Mean Latency (ms)", color="#2ecc71")
    b_p95 = ax2.bar(xl + 0.18, p95s, 0.35, label="P95 Latency (ms)", color="#f39c12")
    ax2.set_ylabel("Latency (ms/sample)")
    ax2.set_title("Single-Sample CPU Latency Distribution", fontweight="bold")
    ax2.set_xticks(xl)
    ax2.set_xticklabels(latency_labels)
    ax2.legend(loc="upper left")

    for bar in list(b_mean) + list(b_p95):
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

    # -------------------------------------------------------------
    # Panel 3: Latency vs Fire Recall Trade-Off
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    scatter_pts = [
        ("Tier 1 (LGBM E2E)", results["Tier1_LightGBM"]["latency_cpu_e2e_ms"]["mean_ms"], recalls[0], "#e67e22", "s"),
        ("Tier 2 (ResNet18)", results["Tier2_ResNet18"]["latency_cpu_ms"]["mean_ms"], recalls[1], "#2980b9", "o"),
        ("Tier 3 (DeiT-Tiny)", results["Tier3_DeiT_Tiny"]["latency_cpu_ms"]["mean_ms"], recalls[2], "#8e44ad", "^"),
    ]
    ax3.axhspan(0.90, 1.05, color="green", alpha=0.10, label="Safety Zone (≥90% Recall)")
    ax3.axhline(0.90, color="darkred", linestyle="--", linewidth=1.5)

    for name, lat, rec, c, marker in scatter_pts:
        ax3.scatter(lat, rec, s=160, color=c, marker=marker, label=name, edgecolors="black", zorder=5)
        ax3.annotate(f"{name}\n({lat:.1f}ms, {rec:.1%})", (lat, rec),
                     xytext=(10, -5), textcoords="offset points", fontsize=9, fontweight="semibold")

    ax3.set_xlabel("Mean CPU Latency (ms/sample)")
    ax3.set_ylabel("Test Fire Recall")
    ax3.set_ylim(min(recalls) - 0.05, 1.03)
    ax3.set_title("Efficiency vs Safety Trade-Off (CPU)", fontweight="bold")
    ax3.legend(loc="lower right")

    # -------------------------------------------------------------
    # Panel 4: 3-Tier Confusion Matrices
    # -------------------------------------------------------------
    # Subdivide Panel 4 into 3 mini heatmap axes
    sub_gs = gs[1, 1].subgridspec(1, 3, wspace=0.40)
    cm_titles = ["Tier 1: LightGBM", "Tier 2: ResNet18", "Tier 3: DeiT-Tiny"]
    cm_palettes = ["Oranges", "Blues", "Purples"]

    for idx, (k, title, pal) in enumerate(zip(keys, cm_titles, cm_palettes)):
        ax_cm = fig.add_subplot(sub_gs[0, idx])
        cm = np.array(results[k]["metrics_calibrated"]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=pal,
            cbar=False,
            ax=ax_cm,
            xticklabels=["Fire", "Smoke"],
            yticklabels=["Fire", "Smoke"] if idx == 0 else False,
        )
        ax_cm.set_title(title, fontsize=10, fontweight="bold")
        ax_cm.set_xlabel("Predicted")
        if idx == 0:
            ax_cm.set_ylabel("Actual")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Dashboard figure saved to {save_path}")

    plt.show()
