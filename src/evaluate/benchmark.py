"""CPU latency benchmarking and champion model selection."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def benchmark_model_cpu(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 100,
) -> float:
    """Measure mean inference latency (ms/sample) over num_runs forward passes on CPU.

    Parameters
    ----------
    model:        Any nn.Module in eval mode.
    sample_input: Input tensor with shape (1, C, H, W) or compatible.
    num_runs:     Number of timed iterations.

    Returns
    -------
    Mean latency in milliseconds.
    """
    model = model.cpu().eval()
    sample_input = sample_input.cpu()

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(sum(latencies) / len(latencies))


def select_champion(
    models_dict: Dict[str, Dict[str, Any]],
    val_dataloaders: Optional[Any] = None,
    test_dataloaders: Optional[Any] = None,
    output_path: Optional[str | Path] = None,
    recall_threshold: float = 0.90,
) -> Dict[str, Any]:
    """Apply safety-first champion selection policy.

    Policy
    ------
    1. Disqualify tiers with test_recall_fire < recall_threshold.
    2. Among qualifiers, crown the tier with lowest latency_ms.
    3. Export champion_config.json to output_path.

    Parameters
    ----------
    models_dict:  Dict keyed by tier name → dict with keys:
                  model, test_recall_fire, latency_ms.
    output_path:  Where to write champion_config.json.

    Returns
    -------
    Dict with champion_tier, champion_recall, champion_latency_ms.
    """
    qualifying = {
        name: meta
        for name, meta in models_dict.items()
        if meta.get("test_recall_fire", 0.0) >= recall_threshold
    }

    if not qualifying:
        raise ValueError(
            f"No model meets Fire Recall >= {recall_threshold}. "
            f"Recalls: { {n: m.get('test_recall_fire') for n, m in models_dict.items()} }"
        )

    champion_name = min(qualifying, key=lambda n: qualifying[n].get("latency_ms", float("inf")))
    champion_meta = qualifying[champion_name]

    result: Dict[str, Any] = {
        "champion_tier": champion_name,
        "champion_recall_fire": champion_meta.get("test_recall_fire"),
        "champion_latency_ms": champion_meta.get("latency_ms"),
        "all_tiers": {
            name: {
                "test_recall_fire": meta.get("test_recall_fire"),
                "latency_ms": meta.get("latency_ms"),
                "qualified": name in qualifying,
            }
            for name, meta in models_dict.items()
        },
    }

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result, f, indent=2)

    return result
