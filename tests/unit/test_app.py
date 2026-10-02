"""Unit tests for dedicated Gradio inference app module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import gradio as gr
import numpy as np
from PIL import Image
import pytest

from tests.helpers import import_or_skip


def test_build_app_construction():
    """Verify build_app constructs and returns a valid gr.Blocks instance."""
    app_mod = import_or_skip("src.ui.app")
    build_app = getattr(app_mod, "build_app")

    demo = build_app(models_dir="src/models/models_reproduce", device="cpu")
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Fire & Smoke Dual-Gate Inference Dashboard"


def test_build_app_construction_empty_dir(tmp_path):
    """Verify build_app constructs even if models_dir is empty (graceful fallback)."""
    app_mod = import_or_skip("src.ui.app")
    build_app = getattr(app_mod, "build_app")

    demo = build_app(models_dir=str(tmp_path), device="cpu")
    assert isinstance(demo, gr.Blocks)


def test_apply_dual_gating_ambient_rejection():
    """Gate 1: max(p_fire, p_smoke) < tau -> ambient_frame (rejected)."""
    app_mod = import_or_skip("src.ui.app")
    apply_dual_gating = getattr(app_mod, "apply_dual_gating")

    # Max prob is 0.55 < 0.70 -> ambient
    res = apply_dual_gating(p_fire=0.55, p_smoke=0.45, theta=0.50, tau=0.70, latency_ms=12.3)
    assert res["hazard_detected"] is False
    assert res["label"] == "ambient_frame"
    assert res["predicted_class"] == "ambient_frame"
    assert res["status"] == "ambient_frame"
    assert res["confidence"] == 0.55
    assert res["ambient_tau"] == 0.70
    assert res["latency_ms"] == 12.3


def test_apply_dual_gating_fire_detection():
    """Gate 2: max(p_fire, p_smoke) >= tau and p_fire >= theta -> fire."""
    app_mod = import_or_skip("src.ui.app")
    apply_dual_gating = getattr(app_mod, "apply_dual_gating")

    res = apply_dual_gating(p_fire=0.85, p_smoke=0.15, theta=0.46, tau=0.70, latency_ms=25.0)
    assert res["hazard_detected"] is True
    assert res["label"] == "fire"
    assert res["predicted_class"] == "fire"
    assert res["status"] == "detected"
    assert res["confidence"] == 0.85
    assert res["operational_threshold"] == 0.46


def test_apply_dual_gating_smoke_detection():
    """Gate 2: max(p_fire, p_smoke) >= tau and p_fire < theta -> smoke."""
    app_mod = import_or_skip("src.ui.app")
    apply_dual_gating = getattr(app_mod, "apply_dual_gating")

    res = apply_dual_gating(p_fire=0.30, p_smoke=0.88, theta=0.50, tau=0.70, latency_ms=18.5)
    assert res["hazard_detected"] is True
    assert res["label"] == "smoke"
    assert res["predicted_class"] == "smoke"
    assert res["status"] == "detected"
    assert res["confidence"] == 0.88


def test_apply_dual_gating_boundary_conditions():
    """Verify exact boundary threshold evaluations."""
    app_mod = import_or_skip("src.ui.app")
    apply_dual_gating = getattr(app_mod, "apply_dual_gating")

    # Exact threshold: p_fire == theta and conf >= tau -> fire
    res_exact = apply_dual_gating(p_fire=0.70, p_smoke=0.30, theta=0.70, tau=0.70)
    assert res_exact["hazard_detected"] is True
    assert res_exact["predicted_class"] == "fire"

    # Just below tau: max_conf = 0.6999 < 0.70 -> ambient
    res_sub_tau = apply_dual_gating(p_fire=0.6999, p_smoke=0.30, theta=0.50, tau=0.70)
    assert res_sub_tau["hazard_detected"] is False
    assert res_sub_tau["predicted_class"] == "ambient_frame"


def test_model_dropdown_change_callback():
    """Verify callback correctly updates theta slider value when switching tiers."""
    app_mod = import_or_skip("src.ui.app")
    get_model_calibrated_threshold = getattr(app_mod, "get_model_calibrated_threshold")

    thresholds_map = {
        "Champion (Tier3_DeiT_Tiny)": 0.46,
        "Tier1_LightGBM": 0.30,
        "Tier2_ResNet18": 0.54,
        "Tier3_DeiT_Tiny": 0.46,
    }

    # Test calibrated theta retrieval for all tiers
    assert get_model_calibrated_threshold("Tier1_LightGBM", thresholds_map) == 0.30
    assert get_model_calibrated_threshold("Tier2_ResNet18", thresholds_map) == 0.54
    assert get_model_calibrated_threshold("Tier3_DeiT_Tiny", thresholds_map) == 0.46
    assert get_model_calibrated_threshold("Champion (Tier3_DeiT_Tiny)", thresholds_map) == 0.46

    # Test fallback
    assert get_model_calibrated_threshold("Unknown_Tier", thresholds_map) == 0.50


def test_render_verdict_badge():
    """Verify HTML verdict badge renders correct icon, color, and label."""
    app_mod = import_or_skip("src.ui.app")
    render_verdict_badge = getattr(app_mod, "render_verdict_badge")

    badge_fire = render_verdict_badge("fire")
    assert "#cf1322" in badge_fire
    assert "FIRE DETECTED" in badge_fire
    assert "🔥" in badge_fire

    badge_smoke = render_verdict_badge("smoke")
    assert "#d46b08" in badge_smoke
    assert "SMOKE DETECTED" in badge_smoke
    assert "💨" in badge_smoke

    badge_ambient = render_verdict_badge("ambient_frame")
    assert "#595959" in badge_ambient
    assert "AMBIENT FRAME (REJECTED)" in badge_ambient
    assert "🛡️" in badge_ambient


def test_zero_latency_slider_regating():
    """Verify on_slider_change re-runs dual gating without backbone model inference."""
    app_mod = import_or_skip("src.ui.app")
    on_slider_change = getattr(app_mod, "on_slider_change")

    cached_state = {
        "p_fire": 0.85,
        "p_smoke": 0.15,
        "latency_ms": 32.5,
        "model_name": "Tier3_DeiT_Tiny",
    }

    # 1. At theta=0.50, tau=0.70 -> Fire
    badge, conf, lat, act_theta, act_tau, probs, res = on_slider_change(cached_state, theta=0.50, tau=0.70)
    assert res["predicted_class"] == "fire"
    assert "FIRE DETECTED" in badge
    assert conf == 0.85
    assert lat == 32.5
    assert act_theta == 0.50
    assert act_tau == 0.70
    assert probs == {"Fire": 0.85, "Smoke": 0.15}

    # 2. Adjust theta slider to 0.90 (above p_fire 0.85) -> Smoke
    badge, conf, lat, act_theta, act_tau, probs, res = on_slider_change(cached_state, theta=0.90, tau=0.70)
    assert res["predicted_class"] == "smoke"
    assert "SMOKE DETECTED" in badge
    assert act_theta == 0.90

    # 3. Adjust tau slider to 0.90 (above max prob 0.85) -> Ambient Frame
    badge, conf, lat, act_theta, act_tau, probs, res = on_slider_change(cached_state, theta=0.50, tau=0.90)
    assert res["predicted_class"] == "ambient_frame"
    assert "AMBIENT FRAME (REJECTED)" in badge
    assert act_tau == 0.90

    # 4. Handle None state gracefully
    badge_none, conf_none, lat_none, _, _, probs_none, res_none = on_slider_change(None, theta=0.50, tau=0.70)
    assert conf_none == 0.0
    assert "ambient_frame" in res_none.get("status", "") or "no_image_analyzed" in res_none.get("status", "")


def test_load_tier_thresholds_with_fixture(tmp_path):
    """Verify reading thresholds from custom champion config and benchmark json."""
    app_mod = import_or_skip("src.ui.app")
    load_tier_thresholds = getattr(app_mod, "load_tier_thresholds")

    cfg_file = tmp_path / "champion_config.json"
    cfg_file.write_text(json.dumps({
        "champion_tier": "Tier2_ResNet18",
        "calibrated_threshold": 0.58,
    }))

    bench_file = tmp_path / "test_benchmark_tiers.json"
    bench_file.write_text(json.dumps({
        "Tier1_LightGBM": {"theta_calibrated": 0.28},
        "Tier2_ResNet18": {"theta_calibrated": 0.58},
        "Tier3_DeiT_Tiny": {"theta_calibrated": 0.44},
    }))

    champ_key, thresholds = load_tier_thresholds(models_dir=str(tmp_path))
    assert champ_key == "Champion (Tier2_ResNet18)"
    assert thresholds["Tier1_LightGBM"] == 0.28
    assert thresholds["Tier2_ResNet18"] == 0.58
    assert thresholds["Tier3_DeiT_Tiny"] == 0.44
    assert thresholds["Champion (Tier2_ResNet18)"] == 0.58


def test_load_image_from_url_invalid():
    """Verify load_image_from_url returns None for invalid or empty inputs."""
    app_mod = import_or_skip("src.ui.app")
    load_image_from_url = getattr(app_mod, "load_image_from_url")

    assert load_image_from_url("") is None
    assert load_image_from_url("not_a_valid_url") is None
    assert load_image_from_url("ftp://unsupported.url") is None


def test_src_ui_exports():
    """Verify build_app and launch_ui are cleanly exported in src.ui."""
    import src.ui as ui_pkg

    assert hasattr(ui_pkg, "build_app")
    assert hasattr(ui_pkg, "launch_ui")
    assert hasattr(ui_pkg, "load_saved_tier_models")
    assert hasattr(ui_pkg, "render_inference_widget")
