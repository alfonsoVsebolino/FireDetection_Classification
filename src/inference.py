"""Production inference module: FireClassifier with dual-gate ambient rejection.

CLI usage:
    python -m src.inference --input <path/to/image_or_dir> \\
        --threshold 0.70 --output-json results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(img: Image.Image) -> torch.Tensor:
    """PIL → (1, 3, 224, 224) normalised tensor."""
    img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor


def _load_image(src: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, np.ndarray):
        return Image.fromarray(src)
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).copy()


# ---------------------------------------------------------------------------
# FireClassifier
# ---------------------------------------------------------------------------

class FireClassifier:
    """Dual-gate fire/smoke classifier.

    Gate 1 (Ambient Rejection): max(P_fire, P_smoke) < ambient_tau → ambient_frame
    Gate 2 (Calibrated Detection): P_fire >= threshold → fire, else smoke
    """

    def __init__(
        self,
        model_path: str = "",
        config_path: str = "",
        threshold: float = 0.50,
        ambient_tau: float = 0.70,
        device: str = "cpu",
        model_type: str = "pytorch",
    ) -> None:
        self.threshold = threshold
        self.ambient_tau = ambient_tau
        self.device = torch.device(device)
        self.model_type = model_type
        self.model: Any = None

        if model_path and Path(model_path).exists():
            self._load_model(model_path, config_path)

    def _load_model(self, model_path: str, config_path: str) -> None:
        p = Path(model_path)
        if p.suffix in (".pt", ".pth"):
            # Lazy import to avoid hard dep when testing with mock
            from torchvision import models
            from torchvision.models import ResNet18_Weights
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
            m.load_state_dict(torch.load(str(p), map_location=self.device))
            m.to(self.device).eval()
            self.model = m
            self.model_type = "pytorch"
        elif p.suffix in (".pkl", ".joblib"):
            import joblib
            self.model = joblib.load(str(p))
            self.model_type = "sklearn"
        elif p.suffix == ".txt":
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(p))
            self.model_type = "lgbm_booster"

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.threshold = cfg.get("threshold", self.threshold)
            self.ambient_tau = cfg.get("ambient_tau", self.ambient_tau)

    def _get_probs(self, img: Image.Image) -> tuple[float, float]:
        """Return (p_fire, p_smoke)."""
        if self.model_type in ("pytorch",):
            tensor = _preprocess(img).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            return float(probs[0]), float(probs[1])

        if self.model_type in ("mock", "stub"):
            # model has __call__ returning logits tensor
            tensor = _preprocess(img)
            with torch.no_grad():
                logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            return float(probs[0]), float(probs[1])

        if self.model_type in ("sklearn", "lgbm"):
            from src.features.extract import extract_all_features
            feat = extract_all_features(img).reshape(1, -1)
            probs = self.model.predict_proba(feat)[0]
            return float(probs[0]), float(probs[1])

        if self.model_type == "lgbm_booster":
            from src.features.extract import extract_all_features
            feat = extract_all_features(img).reshape(1, -1)
            p_fire = float(self.model.predict(feat)[0])
            return p_fire, 1.0 - p_fire

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """Ingest arbitrary image input → structured detection dict."""
        t0 = time.perf_counter()
        img = _load_image(image_input)
        p_fire, p_smoke = self._get_probs(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        max_conf = max(p_fire, p_smoke)

        # Gate 1: Ambient rejection
        if max_conf < self.ambient_tau:
            return {
                "hazard_detected": False,
                "label": "ambient_frame",
                "status": "ambient_frame",
                "predicted_class": "ambient_frame",
                "confidence": max_conf,
                "probabilities": {"fire": p_fire, "smoke": p_smoke},
                "latency_ms": round(latency_ms, 3),
                "operational_threshold": self.threshold,
            }

        # Gate 2: Calibrated detection
        if p_fire >= self.threshold:
            label = "fire"
        else:
            label = "smoke"

        return {
            "hazard_detected": True,
            "label": label,
            "status": "detected",
            "predicted_class": label,
            "confidence": round(max_conf, 6),
            "probabilities": {"fire": round(p_fire, 6), "smoke": round(p_smoke, 6)},
            "latency_ms": round(latency_ms, 3),
            "operational_threshold": self.threshold,
        }

    # Alias for test compatibility
    predict = predict_image

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
    ) -> List[Dict[str, Any]]:
        results = []
        for p in image_paths:
            try:
                res = self.predict_image(p)
                res["file"] = str(p)
            except Exception as exc:
                res = {"file": str(p), "error": str(exc), "hazard_detected": False}
            results.append(res)
        return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _iter_images(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        for ext in _SUPPORTED_EXTS:
            yield from path.rglob(f"*{ext}")
            yield from path.rglob(f"*{ext.upper()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FireClassifier inference CLI")
    parser.add_argument("--input", required=True, help="Path to image file or directory")
    parser.add_argument("--model-path", default="", help="Path to champion model artifact")
    parser.add_argument("--config-path", default="", help="Path to champion_config.json")
    parser.add_argument("--threshold", type=float, default=0.50, help="Fire detection threshold θ*")
    parser.add_argument("--ambient-tau", type=float, default=0.70, help="Ambient rejection gate τ")
    parser.add_argument("--output-json", default=None, help="Write results to JSON file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    classifier = FireClassifier(
        model_path=args.model_path,
        config_path=args.config_path,
        threshold=args.threshold,
        ambient_tau=args.ambient_tau,
        device=args.device,
    )

    input_path = Path(args.input)
    image_files = list(_iter_images(input_path))

    if not image_files:
        print(f"No supported images found at: {input_path}")
        return

    results = classifier.predict_batch(image_files)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"Results written to {out}")
    else:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
