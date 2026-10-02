"""Dedicated Gradio Inference UI Window for real-time fire and smoke classification.

Dual-gate architecture:
  Gate 1 (Ambient Rejection): max(p_fire, p_smoke) < tau -> ambient_frame
  Gate 2 (Hazard Thresholding): p_fire >= theta -> fire, else smoke
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import urllib.request

import gradio as gr
import numpy as np
from PIL import Image

from src.inference import FireClassifier

THEMES: Dict[str, Dict[str, str]] = {
    "fire": {"bg": "#cf1322", "fg": "#ffffff", "icon": "🔥", "label": "FIRE DETECTED"},
    "smoke": {"bg": "#d46b08", "fg": "#ffffff", "icon": "💨", "label": "SMOKE DETECTED"},
    "ambient_frame": {"bg": "#595959", "fg": "#ffffff", "icon": "🛡️", "label": "AMBIENT FRAME (REJECTED)"},
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "Champion (Tier3_DeiT_Tiny)": 0.46,
    "Tier1_LightGBM": 0.30,
    "Tier2_ResNet18": 0.54,
    "Tier3_DeiT_Tiny": 0.46,
}


def load_tier_thresholds(
    models_dir: Union[str, Path] = "src/models/models_reproduce",
) -> Tuple[str, Dict[str, float]]:
    """Read champion config and tier benchmark to retrieve calibrated thresholds."""
    mdir = Path(models_dir)
    thresholds = dict(DEFAULT_THRESHOLDS)
    champ_tier = "Tier3_DeiT_Tiny"

    cfg_path = mdir / "champion_config.json"
    if not cfg_path.exists():
        for cand in [Path("models/champion_config.json"), Path("src/models/models_reproduce/champion_config.json")]:
            if cand.exists():
                cfg_path = cand
                break

    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            champ_tier = cfg.get("champion_tier", champ_tier)
            if "calibrated_threshold" in cfg:
                c_val = round(float(cfg["calibrated_threshold"]), 2)
                thresholds[champ_tier] = c_val
                thresholds[f"Champion ({champ_tier})"] = c_val
        except Exception:
            pass

    bench_path = mdir / "test_benchmark_tiers.json"
    if not bench_path.exists():
        for cand in [Path("models/test_benchmark_tiers.json"), Path("src/models/models_reproduce/test_benchmark_tiers.json")]:
            if cand.exists():
                bench_path = cand
                break

    if bench_path.exists():
        try:
            with open(bench_path) as f:
                bench = json.load(f)
            for tier_k, tier_v in bench.items():
                if isinstance(tier_v, dict) and "theta_calibrated" in tier_v:
                    thresholds[tier_k] = round(float(tier_v["theta_calibrated"]), 2)
        except Exception:
            pass

    champ_key = f"Champion ({champ_tier})"
    if champ_key not in thresholds and champ_tier in thresholds:
        thresholds[champ_key] = thresholds[champ_tier]

    return champ_key, thresholds


def get_model_calibrated_threshold(
    model_name: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> float:
    """Retrieve calibrated threshold theta* for a specified model tier."""
    if thresholds is None:
        _, thresholds = load_tier_thresholds()
    if model_name in thresholds:
        return thresholds[model_name]
    # Check if alias or tier suffix matches
    for k, v in thresholds.items():
        if model_name in k or k in model_name:
            return v
    return DEFAULT_THRESHOLDS.get(model_name, 0.50)


def apply_dual_gating(
    p_fire: float,
    p_smoke: float,
    theta: float,
    tau: float,
    latency_ms: float = 0.0,
) -> Dict[str, Any]:
    """Execute dual-gating logic:
    Gate 1 (Ambient Rejection): max(p_fire, p_smoke) < tau -> ambient_frame
    Gate 2 (Hazard Thresholding): p_fire >= theta -> fire, else smoke
    """
    conf = max(p_fire, p_smoke)

    if conf < tau:
        label = "ambient_frame"
        hazard_detected = False
        status = "ambient_frame"
    else:
        hazard_detected = True
        status = "detected"
        label = "fire" if p_fire >= theta else "smoke"

    return {
        "hazard_detected": hazard_detected,
        "label": label,
        "status": status,
        "predicted_class": label,
        "confidence": round(float(conf), 4),
        "probabilities": {"fire": round(float(p_fire), 4), "smoke": round(float(p_smoke), 4)},
        "latency_ms": round(float(latency_ms), 2),
        "operational_threshold": round(float(theta), 4),
        "ambient_tau": round(float(tau), 4),
    }


def render_verdict_badge(verdict: str) -> str:
    """Render HTML badge with designated styling and color."""
    theme = THEMES.get(verdict, THEMES["ambient_frame"])
    return (
        f'<div style="background-color: {theme["bg"]}; color: {theme["fg"]}; '
        f'padding: 14px 20px; border-radius: 8px; font-weight: 700; '
        f'text-align: center; font-size: 20px; letter-spacing: 0.5px; '
        f'box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
        f'{theme["icon"]} {theme["label"]}'
        f'</div>'
    )


def load_image_from_url(url: str) -> Optional[Image.Image]:
    """Fetch remote image from URL and return as RGB PIL Image."""
    if not url or not url.strip():
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


class ModelRegistry:
    """Lazy loader and cache for tier classifier instances."""

    def __init__(
        self,
        models_dir: Union[str, Path] = "src/models/models_reproduce",
        device: str = "cpu",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.device = device
        self._classifiers: Dict[str, FireClassifier] = {}
        self.champ_key, self.thresholds = load_tier_thresholds(self.models_dir)

    def get_classifier(self, tier_name: str) -> FireClassifier:
        if tier_name in self._classifiers:
            return self._classifiers[tier_name]

        mdir = self.models_dir
        if not mdir.exists() and Path("src/models/models_reproduce").exists():
            mdir = Path("src/models/models_reproduce")

        cfg_p = mdir / "champion_config.json"

        clf: Optional[FireClassifier] = None

        if "Champion" in tier_name:
            m_path = mdir / "champion_model.pt"
            if not m_path.exists() and (mdir / "tier3_deit_tiny_best.pt").exists():
                m_path = mdir / "tier3_deit_tiny_best.pt"
            if m_path.exists() or cfg_p.exists():
                clf = FireClassifier(
                    model_path=str(m_path) if m_path.exists() else "",
                    config_path=str(cfg_p) if cfg_p.exists() else "",
                    device=self.device,
                )
        elif "LightGBM" in tier_name:
            for p_lgb in [
                mdir / "tier1_lightgbm.txt",
                mdir / "tier1_lightgbm.pkl",
                mdir / "tier1_lightgbm.joblib",
            ]:
                if p_lgb.exists():
                    clf = FireClassifier(model_path=str(p_lgb), device=self.device)
                    break
        elif "ResNet18" in tier_name:
            m_path = mdir / "tier2_resnet18_best.pt"
            if m_path.exists():
                clf = FireClassifier(model_path=str(m_path), device=self.device)
        elif "DeiT" in tier_name or "Tier3" in tier_name:
            m_path = mdir / "tier3_deit_tiny_best.pt"
            if not m_path.exists() and (mdir / "champion_model.pt").exists():
                m_path = mdir / "champion_model.pt"
            if m_path.exists() or cfg_p.exists():
                clf = FireClassifier(
                    model_path=str(m_path) if m_path.exists() else "",
                    config_path=str(cfg_p) if cfg_p.exists() else "",
                    device=self.device,
                )

        if clf is None or clf.model is None:
            # Fallback to zero-arg FireClassifier (discovers default champion)
            try:
                clf = FireClassifier(device=self.device)
            except Exception:
                clf = FireClassifier(device=self.device, model_type="mock")

        self._classifiers[tier_name] = clf
        return clf


def on_slider_change(
    cached_state: Optional[Dict[str, Any]],
    theta: float,
    tau: float,
) -> Tuple[str, float, float, float, float, Dict[str, float], Dict[str, Any]]:
    """Instant zero-latency re-gating using cached backbone probabilities."""
    if not cached_state or "p_fire" not in cached_state:
        badge = render_verdict_badge("ambient_frame")
        return badge, 0.0, 0.0, float(theta), float(tau), {}, {"status": "no_image_analyzed"}

    p_fire = float(cached_state["p_fire"])
    p_smoke = float(cached_state["p_smoke"])
    latency_ms = float(cached_state.get("latency_ms", 0.0))

    res = apply_dual_gating(p_fire, p_smoke, theta, tau, latency_ms=latency_ms)
    badge = render_verdict_badge(res["predicted_class"])
    probs = {"Fire": res["probabilities"]["fire"], "Smoke": res["probabilities"]["smoke"]}
    return (
        badge,
        res["confidence"],
        res["latency_ms"],
        res["operational_threshold"],
        res["ambient_tau"],
        probs,
        res,
    )


def build_app(
    models_dir: str = "src/models/models_reproduce",
    device: str = "cpu",
) -> gr.Blocks:
    """Build and wire the dedicated Gradio Blocks UI."""
    registry = ModelRegistry(models_dir=models_dir, device=device)
    champ_key, thresholds = registry.champ_key, registry.thresholds

    model_choices = [
        champ_key,
        "Tier1_LightGBM",
        "Tier2_ResNet18",
        "Tier3_DeiT_Tiny",
    ]
    # Remove duplicates preserving order
    seen = set()
    unique_choices = [c for c in model_choices if not (c in seen or seen.add(c))]

    default_model = unique_choices[0]
    default_theta = get_model_calibrated_threshold(default_model, thresholds)
    default_tau = 0.70

    with gr.Blocks(title="Fire & Smoke Dual-Gate Inference Dashboard") as demo:
        gr.HTML(
            "<style>"
            ".metric-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; text-align: center; }"
            "</style>"
        )
        gr.Markdown(
            "# 🔥 Fire & Smoke Hazard Detection System\n"
            "### Dual-Gate Calibrated Operational Inference Window\n"
            "Gate 1: Ambient Rejection ($\\tau$) | Gate 2: Calibrated Hazard Detection ($\\theta^*$)"
        )

        cached_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### ⚙️ Model Selection & Gating Thresholds")
                model_dropdown = gr.Dropdown(
                    choices=unique_choices,
                    value=default_model,
                    label="Model Tier Selector",
                    interactive=True,
                )

                with gr.Row():
                    theta_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        step=0.01,
                        value=default_theta,
                        label="θ Slider (Fire Detection Threshold)",
                        interactive=True,
                    )
                    tau_slider = gr.Slider(
                        minimum=0.50,
                        maximum=0.95,
                        step=0.01,
                        value=default_tau,
                        label="τ Slider (Ambient Rejection Gate)",
                        interactive=True,
                    )

                gr.Markdown("#### 📥 Tri-Modal Input Source")
                with gr.Tabs() as tabs:
                    with gr.Tab("📁 File Upload", id="tab_file"):
                        input_file = gr.Image(type="pil", label="Upload Local Image File")
                        btn_file = gr.Button("Analyze Uploaded Image", variant="secondary")

                    with gr.Tab("📷 Live Webcam", id="tab_webcam"):
                        input_webcam = gr.Image(sources=["webcam"], type="pil", label="Webcam Snapshot")
                        btn_webcam = gr.Button("Analyze Webcam Snapshot", variant="secondary")

                    with gr.Tab("🌐 Image URL", id="tab_url"):
                        input_url = gr.Textbox(
                            placeholder="https://example.com/fire_sample.jpg",
                            label="Remote Image URL",
                        )
                        btn_url = gr.Button("Fetch & Analyze Image URL", variant="secondary")

                btn_main = gr.Button("🚀 Run Full Inference", variant="primary", size="lg")

            with gr.Column(scale=5):
                gr.Markdown("#### 📊 Real-Time Detection Dashboard")
                verdict_badge = gr.HTML(value=render_verdict_badge("ambient_frame"))

                with gr.Row():
                    metric_conf = gr.Number(label="Confidence", value=0.0, interactive=False)
                    metric_lat = gr.Number(label="Latency (ms)", value=0.0, interactive=False)
                    metric_theta = gr.Number(label="Active θ*", value=default_theta, interactive=False)
                    metric_tau = gr.Number(label="Active τ", value=default_tau, interactive=False)

                prob_label = gr.Label(label="Probability Distribution (Fire % vs Smoke %)")

                with gr.Accordion("Raw Inference Dictionary (JSON)", open=False):
                    json_output = gr.JSON(label="Inference Payload")

        # -------------------------------------------------------------------
        # Event Callbacks
        # -------------------------------------------------------------------

        def on_model_change_callback(selected_tier: str) -> float:
            return get_model_calibrated_threshold(selected_tier, thresholds)

        model_dropdown.change(
            fn=on_model_change_callback,
            inputs=[model_dropdown],
            outputs=[theta_slider],
        )

        def infer_from_source(
            file_img: Optional[Image.Image],
            webcam_img: Optional[Image.Image],
            url_str: str,
            model_name: str,
            theta: float,
            tau: float,
        ) -> Tuple[str, float, float, float, float, Dict[str, float], Dict[str, Any], Optional[Dict[str, Any]]]:
            target_img = None
            if file_img is not None:
                target_img = file_img
            elif webcam_img is not None:
                target_img = webcam_img
            elif url_str and url_str.strip():
                try:
                    target_img = load_image_from_url(url_str)
                except Exception as exc:
                    badge = (
                        '<div style="background-color: #595959; color: #fff; padding: 14px 20px; '
                        'border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px;">'
                        f'⚠️ URL FETCH FAILED: {exc}</div>'
                    )
                    payload = {"error": str(exc), "status": "fetch_error"}
                    return badge, 0.0, 0.0, float(theta), float(tau), {}, payload, None

            if target_img is None:
                badge = (
                    '<div style="background-color: #595959; color: #fff; padding: 14px 20px; '
                    'border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px;">'
                    '⚠️ PLEASE PROVIDE AN IMAGE INPUT</div>'
                )
                payload = {"error": "No image provided", "status": "awaiting_input"}
                return badge, 0.0, 0.0, float(theta), float(tau), {}, payload, None

            target_img = target_img.convert("RGB")
            t0 = time.perf_counter()
            clf = registry.get_classifier(model_name)
            p_fire, p_smoke = clf._get_probs(target_img)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            res = apply_dual_gating(p_fire, p_smoke, theta, tau, latency_ms=latency_ms)
            badge = render_verdict_badge(res["predicted_class"])
            probs = {"Fire": res["probabilities"]["fire"], "Smoke": res["probabilities"]["smoke"]}
            new_cache = {
                "p_fire": float(p_fire),
                "p_smoke": float(p_smoke),
                "latency_ms": round(float(latency_ms), 2),
                "model_name": model_name,
            }
            return (
                badge,
                res["confidence"],
                res["latency_ms"],
                res["operational_threshold"],
                res["ambient_tau"],
                probs,
                res,
                new_cache,
            )

        output_targets = [
            verdict_badge,
            metric_conf,
            metric_lat,
            metric_theta,
            metric_tau,
            prob_label,
            json_output,
            cached_state,
        ]

        infer_inputs = [
            input_file,
            input_webcam,
            input_url,
            model_dropdown,
            theta_slider,
            tau_slider,
        ]

        btn_main.click(fn=infer_from_source, inputs=infer_inputs, outputs=output_targets)
        btn_file.click(fn=infer_from_source, inputs=infer_inputs, outputs=output_targets)
        btn_webcam.click(fn=infer_from_source, inputs=infer_inputs, outputs=output_targets)
        btn_url.click(fn=infer_from_source, inputs=infer_inputs, outputs=output_targets)

        # Zero-latency slider re-gating callbacks
        slider_outputs = [
            verdict_badge,
            metric_conf,
            metric_lat,
            metric_theta,
            metric_tau,
            prob_label,
            json_output,
        ]

        theta_slider.change(
            fn=on_slider_change,
            inputs=[cached_state, theta_slider, tau_slider],
            outputs=slider_outputs,
        )

        tau_slider.change(
            fn=on_slider_change,
            inputs=[cached_state, theta_slider, tau_slider],
            outputs=slider_outputs,
        )

    return demo


def launch_ui(
    models_dir: str = "src/models/models_reproduce",
    device: str = "cpu",
    port: int = 7860,
    host: str = "127.0.0.1",
    share: bool = False,
    in_notebook: bool = False,
) -> Any:
    """Launch the Gradio interface."""
    demo = build_app(models_dir=models_dir, device=device)
    if in_notebook:
        return demo.launch(inline=True, share=share)
    return demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedicated Gradio Inference UI Window")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio link")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="src/models/models_reproduce",
        help="Model directory",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    args = parser.parse_args()

    launch_ui(
        models_dir=args.models_dir,
        device=args.device,
        port=args.port,
        host=args.host,
        share=args.share,
    )
