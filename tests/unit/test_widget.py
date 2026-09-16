"""Unit tests for interactive inference widget module."""

import io
import json
from pathlib import Path
from PIL import Image
import pytest
from tests.helpers import import_or_skip


def test_widget_load_input_image_local_and_upload(tmp_path):
    widget = import_or_skip("src.ui.widget")
    _load_input_image = getattr(widget, "_load_input_image")

    # 1. Local path loading
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_path)
    loaded = _load_input_image(None, str(img_path))
    assert loaded is not None
    assert loaded.size == (64, 64)

    # 2. FileUpload mock loading
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(buf, format="PNG")
    mock_upload = {"upload": {"content": buf.getvalue()}}
    loaded_upload = _load_input_image(mock_upload, "")
    assert loaded_upload is not None
    assert loaded_upload.size == (32, 32)

    # 3. None on empty input
    assert _load_input_image(None, "") is None


def test_widget_render_dashboard():
    widget = import_or_skip("src.ui.widget")
    _render_dashboard = getattr(widget, "_render_dashboard")

    test_img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    res = {
        "hazard_detected": False,
        "predicted_class": "ambient_frame",
        "confidence": 0.52,
        "probabilities": {"fire": 0.48, "smoke": 0.52},
        "latency_ms": 15.2,
        "operational_threshold": 0.60,
        "ambient_tau": 0.70,
    }

    dashboard = _render_dashboard(test_img, res)
    assert dashboard is not None
    assert len(dashboard.children) == 2

    # Regression test for missing ambient_tau key
    res_no_tau = dict(res)
    del res_no_tau["ambient_tau"]
    dashboard_fallback = _render_dashboard(test_img, res_no_tau)
    assert dashboard_fallback is not None


def test_render_inference_widget_initialization():
    widget = import_or_skip("src.ui.widget")
    render_inference_widget = getattr(widget, "render_inference_widget")

    ui = render_inference_widget(
        tier_models={"Tier2_ResNet18": object()},
        champion_tier="Tier2_ResNet18",
        models_dict={"Tier2_ResNet18": {"calibrated_threshold": 0.62}},
        device="cpu",
    )
    assert ui is not None
    assert len(ui.children) == 3


def test_load_saved_tier_models_empty_dir(tmp_path):
    widget = import_or_skip("src.ui.widget")
    load_saved_tier_models = getattr(widget, "load_saved_tier_models")

    tier_models, models_dict, champion_tier = load_saved_tier_models(models_dir=str(tmp_path), device="cpu")
    assert isinstance(tier_models, dict)
    assert isinstance(models_dict, dict)
    assert champion_tier == "Tier2_ResNet18"


def test_load_saved_tier_models_with_config(tmp_path):
    widget = import_or_skip("src.ui.widget")
    load_saved_tier_models = getattr(widget, "load_saved_tier_models")

    cfg_file = tmp_path / "champion_config.json"
    cfg_file.write_text(json.dumps({
        "champion_tier": "Tier3_DeiT_Tiny",
        "calibrated_threshold": 0.65,
        "ambient_tau": 0.72
    }))

    tier_models, models_dict, champion_tier = load_saved_tier_models(models_dir=str(tmp_path), device="cpu")
    assert champion_tier == "Tier3_DeiT_Tiny"
    assert models_dict["Tier3_DeiT_Tiny"]["calibrated_threshold"] == 0.65
