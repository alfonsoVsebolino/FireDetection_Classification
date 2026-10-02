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
        threshold: Optional[float] = None,
        ambient_tau: Optional[float] = None,
        device: str = "cpu",
        model_type: str = "pytorch",
    ) -> None:
        self.threshold = 0.50 if threshold is None else threshold
        self.ambient_tau = 0.70 if ambient_tau is None else ambient_tau
        self.device = torch.device(device)
        self.model_type = model_type
        self.model: Any = None

        if model_type not in ("mock", "stub"):
            if not model_path and not config_path:
                for cand in [
                    Path("src/models/models_reproduce/champion_config.json"),
                    Path("models/champion_config.json"),
                ]:
                    if cand.exists():
                        config_path = str(cand)
                        break

            if model_path or config_path:
                self._load_model(model_path, config_path)

        if threshold is not None:
            self.threshold = threshold
        if ambient_tau is not None:
            self.ambient_tau = ambient_tau

    def _load_model(self, model_path: str, config_path: str) -> None:
        architecture = None
        if config_path:
            cp = Path(config_path)
            if not cp.exists():
                for cdir in [Path("src/models/models_reproduce"), Path("models")]:
                    if (cdir / cp.name).exists():
                        cp = cdir / cp.name
                        break
                    if (cdir / cp).exists():
                        cp = cdir / cp
                        break
            if cp.exists():
                with open(cp) as f:
                    cfg = json.load(f)
                if "calibrated_threshold" in cfg:
                    self.threshold = float(cfg["calibrated_threshold"])
                elif "threshold" in cfg:
                    self.threshold = float(cfg["threshold"])
                if "ambient_tau" in cfg:
                    self.ambient_tau = float(cfg["ambient_tau"])
                architecture = cfg.get("architecture")
                if not model_path:
                    model_path = cfg.get("model_path", "")

        if not model_path:
            return

        p = Path(model_path)
        if not p.exists():
            for cdir in [Path("src/models/models_reproduce"), Path("models")]:
                if (cdir / p.name).exists():
                    p = cdir / p.name
                    break
                if (cdir / p).exists():
                    p = cdir / p
                    break

        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if p.suffix in (".pt", ".pth"):
            state_dict = torch.load(str(p), map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            if architecture:
                arch_lower = architecture.lower()
                if "deit" in arch_lower or "vit" in arch_lower:
                    from src.models.train_vit import build_deit_tiny
                    m = build_deit_tiny(pretrained=False, num_classes=2)
                else:
                    from src.models.train_resnet import build_resnet18
                    m = build_resnet18(pretrained=False, num_classes=2)
            else:
                keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
                if any("patch_embed" in k or "blocks." in k for k in keys):
                    from src.models.train_vit import build_deit_tiny
                    m = build_deit_tiny(pretrained=False, num_classes=2)
                else:
                    from src.models.train_resnet import build_resnet18
                    m = build_resnet18(pretrained=False, num_classes=2)

            m.load_state_dict(state_dict)
            m.to(self.device).eval()
            self.model = m
            self.model_type = "pytorch"

        elif p.suffix == ".txt":
            from src.models.train_lightgbm import load_model
            self.model = load_model(str(p))
            self.model_type = "lgbm"

        elif p.suffix in (".pkl", ".joblib"):
            import joblib
            self.model = joblib.load(str(p))
            self.model_type = "sklearn"

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
            p_smoke = float(self.model.predict(feat)[0])
            return 1.0 - p_smoke, p_smoke

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
                "ambient_tau": self.ambient_tau,
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
            "ambient_tau": self.ambient_tau,
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
    parser.add_argument("--threshold", type=float, default=None, help="Fire detection threshold θ*")
    parser.add_argument("--ambient-tau", type=float, default=None, help="Ambient rejection gate τ")
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

    payload = results[0] if len(results) == 1 else results
    
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent = 2)
        print(f"Results written to {out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
