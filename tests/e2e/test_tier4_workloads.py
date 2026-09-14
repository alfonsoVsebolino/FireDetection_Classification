"""Tier 4: Real-World Application Scenarios Test Suite.
Simulates end-to-end surveillance feeds, batch directory ingestion, and JSON schema compliance.
"""

import json
import os
import time
import numpy as np
import pytest
from PIL import Image
from tests.helpers import import_or_skip


class TestTier4Workloads:

    def test_surveillance_feed_temporal_stream_simulation(
        self,
        synthetic_ambient_image,
        synthetic_smoke_image,
        synthetic_fire_image
    ):
        """Simulate real-time surveillance feed transitioning from ambient -> smoke -> fire -> ambient."""
        FireClassifier = import_or_skip("src.inference", "FireClassifier")

        # Stream sequence:
        # Frame 0: Ambient (normal day) -> max(P) < 0.70 -> ambient_frame
        # Frame 1: Early Smoke plume   -> smoke detected
        # Frame 2: Active Fire breakout -> fire detected
        # Frame 3: Extinguished / Ambient -> ambient_frame

        stream = [
            ("frame_001_daylight.jpg", synthetic_ambient_image, [0.52, 0.48], False, "ambient"),
            ("frame_002_plume.jpg", synthetic_smoke_image, [0.15, 0.85], True, "smoke"),
            ("frame_003_flame.jpg", synthetic_fire_image, [0.92, 0.08], True, "fire"),
            ("frame_004_extinguished.jpg", synthetic_ambient_image, [0.50, 0.50], False, "ambient"),
        ]

        class DynamicMockModel:
            def __init__(self):
                self.probs = [0.5, 0.5]
            def predict_proba(self, x):
                return np.array([self.probs])
            def __call__(self, x):
                import torch
                p0, p1 = self.probs
                return torch.tensor([[np.log(max(p0, 1e-6)), np.log(max(p1, 1e-6))]])

        mock_backend = DynamicMockModel()
        clf = FireClassifier(model_type="mock", model_path="", threshold=0.50, ambient_tau=0.70)
        clf.model = mock_backend
        predict_fn = clf.predict if hasattr(clf, "predict") else clf.predict_image

        results = []
        for name, img, expected_probs, expected_hazard, expected_subclass in stream:
            mock_backend.probs = expected_probs
            t0 = time.perf_counter()
            res = predict_fn(img)
            latency = (time.perf_counter() - t0) * 1000.0

            assert res["hazard_detected"] == expected_hazard, f"Failed hazard detection for {name}"
            pred_class = (res.get("predicted_class") or res.get("label") or "").lower()
            assert expected_subclass in pred_class, f"Expected {expected_subclass} in {pred_class} for {name}"
            # Single sample CPU latency budget
            assert latency < 500.0, f"Latency {latency}ms exceeds 500ms budget"
            results.append(res)

        assert len(results) == 4

    def test_batch_directory_ingestion_workload(
        self,
        temp_image_dir,
        synthetic_ambient_image,
        synthetic_smoke_image,
        synthetic_fire_image
    ):
        """Simulate batch ingestion of mixed format images (.jpg, .png) from a surveillance storage drop."""
        FireClassifier = import_or_skip("src.inference", "FireClassifier")

        # Save multiple files with varying extensions
        p_jpg = os.path.join(temp_image_dir, "feed_1.jpg")
        p_png = os.path.join(temp_image_dir, "feed_2.png")
        p_jpeg = os.path.join(temp_image_dir, "feed_3.jpeg")

        synthetic_fire_image.save(p_jpg, format="JPEG")
        synthetic_smoke_image.save(p_png, format="PNG")
        synthetic_ambient_image.save(p_jpeg, format="JPEG")

        file_list = [p_jpg, p_png, p_jpeg]

        class MockBatchModel:
            def predict_proba(self, x):
                return np.array([[0.8, 0.2]] * len(x))
            def __call__(self, x):
                import torch
                n = x.shape[0]
                return torch.tensor([[1.5, 0.0]] * n)

        clf = FireClassifier(model_type="mock", model_path="", threshold=0.50, ambient_tau=0.70)
        clf.model = MockBatchModel()

        if hasattr(clf, "predict_batch"):
            batch_res = clf.predict_batch(file_list)
            assert len(batch_res) == 3
            for r in batch_res:
                assert "hazard_detected" in r
        else:
            # Fallback to serial predict
            predict_fn = clf.predict if hasattr(clf, "predict") else clf.predict_image
            for fpath in file_list:
                r = predict_fn(fpath)
                assert "hazard_detected" in r

    def test_json_output_schema_conformance(self, temp_image_dir, synthetic_fire_image):
        """Verify production output JSON conform precisely to ISSUE-06 schema."""
        FireClassifier = import_or_skip("src.inference", "FireClassifier")

        img_path = os.path.join(temp_image_dir, "schema_test.jpg")
        synthetic_fire_image.save(img_path)

        class MockFireModel:
            def predict_proba(self, x):
                return np.array([[0.942, 0.058]])
            def __call__(self, x):
                import torch
                return torch.tensor([[np.log(0.942), np.log(0.058)]])

        clf = FireClassifier(model_type="mock", model_path="", threshold=0.50, ambient_tau=0.70)
        clf.model = MockFireModel()
        predict_fn = clf.predict if hasattr(clf, "predict") else clf.predict_image

        res = predict_fn(img_path)

        # Expected Schema:
        # {
        #   "hazard_detected": bool,
        #   "label" or "predicted_class": str,
        #   "probabilities": {"fire": float, "smoke": float} or float confidence
        # }
        assert isinstance(res["hazard_detected"], bool)
        assert isinstance(res.get("label") or res.get("predicted_class"), str)
        if "probabilities" in res:
            assert "fire" in res["probabilities"]
            assert "smoke" in res["probabilities"]
            assert 0.0 <= res["probabilities"]["fire"] <= 1.0
            assert 0.0 <= res["probabilities"]["smoke"] <= 1.0
