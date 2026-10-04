import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import ipywidgets as widgets
from IPython.display import display
from PIL import Image

from src.inference import FireClassifier

THEMES = {
    "fire": {"bg": "#cf1322", "fg": "#fff", "icon": "🔥", "label": "FIRE DETECTED"},
    "smoke": {"bg": "#d46b08", "fg": "#fff", "icon": "💨", "label": "SMOKE DETECTED"},
    "ambient_frame": {"bg": "#595959", "fg": "#fff", "icon": "🛡️", "label": "AMBIENT FRAME (REJECTED)"},
}


def _load_input_image(upload_val: Any, path_val: str) -> Optional[Image.Image]:
    if upload_val:
        if isinstance(upload_val, (tuple, list)):
            item = upload_val[0]
        else: 
            item = list(upload_val.values())[0]
        
        if isinstance(item, dict):
            content = item.get("content")
        else: 
            content = getattr(item, "content", None)

        if content:
            return Image.open(io.BytesIO(content)).convert("RGB")
    if path_val.strip():
        p = path_val.strip()
        if p.startswith(("http://", "https://")):
            req = urllib.request.Request(p, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        if Path(p).exists():
            return Image.open(p).convert("RGB")
    return None


def _render_dashboard(img: Image.Image, res: Dict[str, Any]) -> widgets.VBox:
    verdict = res.get("predicted_class", res.get("label", "ambient_frame"))
    theme = THEMES.get(verdict, THEMES["ambient_frame"])
    probs = res.get("probabilities", {})
    p_fire = probs.get("fire", 0.0)
    p_smoke = probs.get("smoke", 0.0)

    buf = io.BytesIO()
    thumb = img.copy()
    thumb.thumbnail((260, 260))
    thumb.save(buf, format="JPEG")
    img_widget = widgets.Image(value=buf.getvalue(), format="jpeg", width=260)

    theta = res.get("operational_threshold", 0.50)
    tau = res.get("ambient_tau", 0.70)
    conf = res.get("confidence", max(p_fire, p_smoke))
    lat = res.get("latency_ms", 0.0)

    html_str = f"""
    <div style="font-family: sans-serif; min-width: 320px; padding: 12px; border: 1px solid #d9d9d9; border-radius: 8px;">
      <div style="background-color: {theme['bg']}; color: {theme['fg']}; padding: 10px; border-radius: 6px; font-weight: bold; text-align: center; font-size: 14px;">
        {theme['icon']} {theme['label']}
      </div>
      <div style="margin-top: 12px; font-size: 13px;">
        <div><b>Confidence:</b> {conf:.3f} | <b>Latency:</b> {lat:.2f} ms</div>
        <div><b>θ (Hazard):</b> {theta:.2f} | <b>τ (Ambient):</b> {tau:.2f}</div>
      </div>
      <div style="margin-top: 12px;">
        <div style="font-size: 12px; margin-bottom: 2px;"><b>Fire Probability:</b> {p_fire * 100:.1f}%</div>
        <div style="background: #f0f0f0; border-radius: 4px; height: 14px; width: 100%;">
          <div style="background: #cf1322; width: {p_fire * 100:.1f}%; height: 100%; border-radius: 4px;"></div>
        </div>
        <div style="font-size: 12px; margin-top: 8px; margin-bottom: 2px;"><b>Smoke Probability:</b> {p_smoke * 100:.1f}%</div>
        <div style="background: #f0f0f0; border-radius: 4px; height: 14px; width: 100%;">
          <div style="background: #d46b08; width: {p_smoke * 100:.1f}%; height: 100%; border-radius: 4px;"></div>
        </div>
      </div>
    </div>
    """
    card = widgets.HTML(value=html_str)
    return widgets.HBox([img_widget, card], layout=widgets.Layout(gap="16px", align_items="center"))


def load_saved_tier_models(
    models_dir: str = "models",
    device: str = "cpu",
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """Scan disk/Drive checkpoints and reconstruct evaluation-ready model dictionary."""
    mdir = Path(models_dir)
    reproduce_dir = Path("src/models/models_reproduce")
    drive_dir = Path("/content/drive/MyDrive/FireDetection_Classification_Outputs/models")
    if not mdir.exists():
        if reproduce_dir.exists():
            mdir = reproduce_dir
        elif drive_dir.exists():
            mdir = drive_dir

    tier_models: Dict[str, Any] = {}
    models_dict: Dict[str, Any] = {}
    champion_tier = "Tier2_ResNet18"

    bench_path = mdir / "test_benchmark_tiers.json"
    if bench_path.exists():
        try:
            with open(bench_path) as f:
                bench = json.load(f)
            for tier_name, tier_info in bench.items():
                if tier_name not in models_dict:
                    models_dict[tier_name] = {}
                if "theta_calibrated" in tier_info:
                    models_dict[tier_name]["calibrated_threshold"] = tier_info["theta_calibrated"]
        except Exception:
            pass

    cfg_path = mdir / "champion_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        champion_tier = cfg.get("champion_tier", champion_tier)
        cfg_models_dict = cfg.get("models_dict", {})
        if cfg_models_dict:
            models_dict.update(cfg_models_dict)
        if champion_tier not in models_dict:
            models_dict[champion_tier] = {}
        if "calibrated_threshold" in cfg:
            models_dict[champion_tier]["calibrated_threshold"] = cfg["calibrated_threshold"]
        if "ambient_tau" in cfg:
            models_dict[champion_tier]["ambient_tau"] = cfg.get("ambient_tau", 0.70)

    import torch

    p_resnet = mdir / "tier2_resnet18_best.pt"
    if not p_resnet.exists() and (mdir / "champion_model.pt").exists() and champion_tier == "Tier2_ResNet18":
        p_resnet = mdir / "champion_model.pt"
    if p_resnet.exists():
        from src.models.train_resnet import build_resnet18
        m_res = build_resnet18(pretrained=False)
        m_res.load_state_dict(torch.load(str(p_resnet), map_location=device))
        tier_models["Tier2_ResNet18"] = m_res.to(device).eval()

    p_vit = mdir / "tier3_deit_tiny_best.pt"
    if not p_vit.exists() and (mdir / "champion_model.pt").exists() and champion_tier == "Tier3_DeiT_Tiny":
        p_vit = mdir / "champion_model.pt"
    if p_vit.exists():
        from src.models.train_vit import build_deit_tiny
        m_vit = build_deit_tiny(pretrained=False)
        m_vit.load_state_dict(torch.load(str(p_vit), map_location=device))
        tier_models["Tier3_DeiT_Tiny"] = m_vit.to(device).eval()

    for p_lgb in [mdir / "tier1_lightgbm.txt", mdir / "tier1_lightgbm.pkl", mdir / "tier1_lightgbm.joblib"]:
        if p_lgb.exists():
            if p_lgb.suffix == ".txt":
                try:
                    from src.models.train_lightgbm import load_model
                    tier_models["Tier1_LightGBM"] = load_model(str(p_lgb))
                except Exception:
                    import lightgbm as lgb
                    tier_models["Tier1_LightGBM"] = lgb.Booster(model_file=str(p_lgb))
            else:
                import joblib
                tier_models["Tier1_LightGBM"] = joblib.load(str(p_lgb))
            break

    return tier_models, models_dict, champion_tier


def render_inference_widget(
    tier_models: Optional[Dict[str, Any]] = None,
    champion_tier: Optional[str] = None,
    models_dict: Optional[Dict[str, Any]] = None,
    models_dir: str = "models",
    device: str = "cpu",
) -> widgets.VBox:
    disk_models, disk_dict, disk_champ = load_saved_tier_models(models_dir=models_dir, device=device)
    tier_models = {**disk_models, **(tier_models or {})}
    models_dict = {**disk_dict, **(models_dict or {})}
    champion_tier = champion_tier or disk_champ

    opts = list(tier_models.keys())
    if not opts and Path("models/champion_config.json").exists():
        with open("models/champion_config.json") as f:
            cfg = json.load(f)
            champ_tier = cfg.get("champion_tier", "Champion")
            opts = [champ_tier]

    if champion_tier in opts:
        init_model = champion_tier
    elif opts:
        init_model = opts[0]
    else:
        init_model = "Tier2_ResNet18"
    init_theta = models_dict.get(init_model, {}).get("calibrated_threshold", 0.50)

    w_model = widgets.Dropdown(options=opts, value=init_model, description="Model:")
    w_theta = widgets.FloatSlider(value=init_theta, min=0.10, max=0.90, step=0.01, description="θ (Fire):")
    w_tau = widgets.FloatSlider(value=0.70, min=0.50, max=0.95, step=0.01, description="τ (Ambient):")

    def _on_model_change(change):
        m_name = change["new"]
        if m_name in models_dict and "calibrated_threshold" in models_dict[m_name]:
            w_theta.value = models_dict[m_name]["calibrated_threshold"]

    w_model.observe(_on_model_change, names="value")

    w_upload = widgets.FileUpload(accept="image/*", multiple=False, description="Upload Image")
    w_path = widgets.Text(placeholder="Local file path or image URL...", description="Path/URL:", layout=widgets.Layout(width="400px"))
    w_btn = widgets.Button(description="Test Inference", button_style="primary", icon="play")
    w_out = widgets.Output()

    def _run_inference(_):
        with w_out:
            w_out.clear_output()
            img = _load_input_image(w_upload.value, w_path.value)
            if img is None:
                print("⚠️ Please upload an image or provide a valid file path / URL.")
                return

            clf = FireClassifier(threshold=w_theta.value, ambient_tau=w_tau.value, device=device)
            selected_tier = w_model.value

            if selected_tier in tier_models:
                m = tier_models[selected_tier]
                if hasattr(m, "to"):
                    clf.model = m.to(device).eval()
                    clf.model_type = "pytorch"
                elif hasattr(m, "predict_proba"):
                    clf.model = m
                    clf.model_type = "sklearn"
                else:
                    clf.model = m
                    clf.model_type = "lgbm_booster"
            elif Path("models/champion_config.json").exists():
                clf = FireClassifier(
                    model_path="models/champion_model.pt",
                    config_path="models/champion_config.json",
                    threshold=w_theta.value,
                    ambient_tau=w_tau.value,
                    device=device,
                )

            res = clf.predict_image(img)
            dashboard = _render_dashboard(img, res)
            display(dashboard)

    w_btn.on_click(_run_inference)

    controls_top = widgets.HBox([w_model, w_theta, w_tau])
    inputs_box = widgets.HBox([w_upload, w_path, w_btn])
    ui = widgets.VBox([controls_top, inputs_box, w_out], layout=widgets.Layout(padding="12px", border="1px solid #ccc", border_radius="8px"))
    display(ui)
    return ui
