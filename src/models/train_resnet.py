"""Tier 2 CNN Anchor: ResNet18 transfer learning for binary fire/smoke classification."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_resnet18(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Return ResNet18 with frozen early layers and a custom 2-class head.

    Frozen: conv1, bn1, layer1.
    Trainable: layer2, layer3, layer4, fc.
    Head: Dropout(0.3) → Linear(512, num_classes).
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze conv1, bn1, layer1
    for name, param in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1")):
            param.requires_grad = False

    # Replace classification head
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_criterion(fire_weight: float = 2.0, device: torch.device | None = None) -> nn.CrossEntropyLoss:
    """CrossEntropyLoss with fire (class 0) weighted 2× smoke (class 1)."""
    w = torch.tensor([fire_weight, 1.0])
    if device is not None:
        w = w.to(device)
    return nn.CrossEntropyLoss(weight=w)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _fire_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Recall on class 0 (fire)."""
    fire_mask = labels == 0
    if fire_mask.sum() == 0:
        return 1.0
    correct = (preds[fire_mask] == 0).sum().item()
    return correct / fire_mask.sum().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_resnet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train ResNet18 on CPU with early stopping on Fire Recall.

    Returns
    -------
    dict with keys: model, history, best_fire_recall, train_time_s
    """
    cfg: Dict[str, Any] = {
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "patience": 3,
        "fire_weight": 2.0,
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
    t0 = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ---- Validate ----
        val_loss_epoch = 0.0
        val_fire_recall = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs, labels = batch[0].to(device), batch[1].to(device)
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    v_loss += loss.item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += imgs.size(0)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
            val_loss_epoch = v_loss / max(v_total, 1)
            val_acc = v_correct / max(v_total, 1)
            all_preds_t = torch.cat(all_preds)
            all_labels_t = torch.cat(all_labels)
            val_fire_recall = _fire_recall(all_preds_t, all_labels_t)

            history["val_loss"].append(val_loss_epoch)
            history["val_acc"].append(val_acc)
            history["val_fire_recall"].append(val_fire_recall)

            # Early stopping on Fire Recall
            if val_fire_recall > best_fire_recall:
                best_fire_recall = val_fire_recall
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg["patience"]:
                break

    train_time_s = time.perf_counter() - t0

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Save checkpoint
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

def evaluate_resnet(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate model; returns accuracy, fire recall, latency (ms/sample)."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    all_preds, all_labels = [], []
    latencies: list[float] = []

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
    accuracy = (preds == labels).float().mean().item()
    fire_recall = _fire_recall(preds, labels)
    avg_latency_ms = sum(latencies) / max(len(latencies), 1)

    return {
        "accuracy": accuracy,
        "fire_recall": fire_recall,
        "avg_latency_ms": avg_latency_ms,
        "preds": preds,
        "labels": labels,
    }
