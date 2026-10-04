"""Unit tests for hotstart and architecture-aware model factory loading."""

import pytest
from src.inference import FireClassifier
from src.ui.widget import load_saved_tier_models


def test_zero_args_hotstart_loads_champion(synthetic_fire_image):
    """Test 1: FireClassifier() zero-args auto-discovers champion_config.json and loads champion_model.pt."""
    clf = FireClassifier()
    assert clf.model is not None
    assert clf.model_type == "pytorch"
    assert hasattr(clf.model, "blocks")
    assert hasattr(clf.model, "patch_embed")
    assert pytest.approx(clf.threshold, abs=1e-4) == 0.46
    assert clf.ambient_tau == 0.70

    res = clf.predict_image(synthetic_fire_image)
    assert "hazard_detected" in res
    assert "probabilities" in res
    assert "fire" in res["probabilities"]


def test_load_deit_tiny_by_model_path(synthetic_fire_image):
    """Test 2: FireClassifier(model_path=...) loads DeiT-Tiny."""
    clf = FireClassifier(model_path="src/models/models_reproduce/tier3_deit_tiny_best.pt")
    assert clf.model is not None
    assert clf.model_type == "pytorch"
    assert hasattr(clf.model, "patch_embed")
    assert hasattr(clf.model, "blocks")

    res = clf.predict_image(synthetic_fire_image)
    assert "hazard_detected" in res


def test_load_resnet18_by_model_path(synthetic_fire_image):
    """Test 3: FireClassifier(model_path=...) loads ResNet18."""
    clf = FireClassifier(model_path="src/models/models_reproduce/tier2_resnet18_best.pt")
    assert clf.model is not None
    assert clf.model_type == "pytorch"
    assert hasattr(clf.model, "conv1")
    assert hasattr(clf.model, "fc")
    assert not hasattr(clf.model, "patch_embed")

    res = clf.predict_image(synthetic_fire_image)
    assert "hazard_detected" in res


def test_load_lightgbm_by_model_path(synthetic_fire_image):
    """Test 4: FireClassifier(model_path=...) loads LightGBM."""
    clf = FireClassifier(model_path="src/models/models_reproduce/tier1_lightgbm.txt")
    assert clf.model is not None
    assert clf.model_type in ("lgbm", "lgbm_booster", "sklearn")

    res = clf.predict_image(synthetic_fire_image)
    assert "hazard_detected" in res
    assert "probabilities" in res


def test_load_saved_tier_models_returns_evaluation_ready_dict():
    """Test 5: load_saved_tier_models() returns evaluation-ready dictionary from src/models/models_reproduce/."""
    tier_models, models_dict, champion_tier = load_saved_tier_models()
    assert champion_tier == "Tier3_DeiT_Tiny"
    assert "Tier1_LightGBM" in tier_models
    assert "Tier2_ResNet18" in tier_models
    assert "Tier3_DeiT_Tiny" in tier_models

    assert tier_models["Tier1_LightGBM"] is not None
    assert tier_models["Tier2_ResNet18"] is not None
    assert tier_models["Tier3_DeiT_Tiny"] is not None

    assert "Tier3_DeiT_Tiny" in models_dict
    assert pytest.approx(models_dict["Tier3_DeiT_Tiny"]["calibrated_threshold"], abs=1e-4) == 0.46
