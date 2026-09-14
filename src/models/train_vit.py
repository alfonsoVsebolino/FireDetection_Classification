"""Tier 3 Transformer: DeiT-Tiny Vision Transformer for binary fire/smoke classification."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("timm is required for DeiT-Tiny: pip install timm") from e


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_deit_tiny(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Return DeiT-Tiny with patch_embed + blocks[0:6] frozen.

    Fine-tuned: blocks[6:], head (replaced to num_classes).
    """
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Freeze patch embedder
    for name, param in model.named_parameters():
        if "patch_embed" in name:
            param.requires_grad = False

    # Freeze first 6 transformer blocks
    for i in range(6):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_criterion(fire_weight: float = 2.0, device: torch.device | None = None) -> nn.CrossEntropyLoss:
    w = torch.tensor([fire_weight, 1.0])
    if device is not None:
        w = w.to(device)
    return nn.CrossEntropyLoss(weight=w)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fire_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    fire_mask = labels == 0
    if fire_mask.sum() == 0:
        return 1.0
    return (preds[fire_mask] == 0).sum().item() / fire_mask.sum().item()


# ---------------------------------------------------------------------------
# Training loop (with gradient accumulation)
# ---------------------------------------------------------------------------

def train_vit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train DeiT-Tiny with gradient accumulation (accum_steps=2, effective BS=32).

    Early stopping patience on validation Fire Recall.

    Returns
    -------
    dict: model, history, best_fire_recall, train_time_s
    """
    cfg: Dict[str, Any] = {
        "epochs": 8,
        "lr": 5e-5,
        "weight_decay": 1e-2,
        "patience": 3,
        "fire_weight": 2.0,
        "accum_steps": 2,
        **(config or {}),
    }

    device = device or torch.device("cpu")
    model = model.to(device)
    criterion = build_criterion(cfg["fire_weight"], device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    history: Dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_fire_recall": [],
    }

    best_fire_recall = -1.0
    patience_counter = 0
    best_state: Dict[str, Any] = {}
    accum = cfg["accum_steps"]
    t0 = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(imgs)
            loss = criterion(logits, labels) / accum
            loss.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        history["train_loss"].append(running_loss / max(total, 1))
        history["train_acc"].append(correct / max(total, 1))

        # ---- Validate ----
        val_fire_recall = 0.0
        if val_loader is not None:
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs, labels = batch[0].to(device), batch[1].to(device)
                    logits = model(imgs)
                    v_loss += criterion(logits, labels).item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += imgs.size(0)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_fire_recall = _fire_recall(torch.cat(all_preds), torch.cat(all_labels))
            history["val_loss"].append(v_loss / max(v_total, 1))
            history["val_acc"].append(v_correct / max(v_total, 1))
            history["val_fire_recall"].append(val_fire_recall)

            if val_fire_recall > best_fire_recall:
                best_fire_recall = val_fire_recall
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg["patience"]:
                break

    train_time_s = time.perf_counter() - t0

    if best_state:
        model.load_state_dict(best_state)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(p))

    return {
        "model": model,
        "history": history,
        "best_fire_recall": best_fire_recall,
        "train_time_s": train_time_s,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_vit(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate DeiT model; returns accuracy, fire recall, avg latency ms/sample."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    all_preds, all_labels, latencies = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            t0 = time.perf_counter()
            logits = model(imgs)
            latencies.append((time.perf_counter() - t0) / imgs.size(0) * 1000)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    return {
        "accuracy": (preds == labels).float().mean().item(),
        "fire_recall": _fire_recall(preds, labels),
        "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
        "preds": preds,
        "labels": labels,
    }
