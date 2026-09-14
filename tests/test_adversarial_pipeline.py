"""Adversarial stress testing harness for src.data.pipeline."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FireSmokeDataset,
    _DATASET_CACHE,
    _parse_label,
    compute_phash,
    create_dataloader,
    detect_corrupted,
    get_eval_transforms,
    hamming_distance,
    is_corrupt,
    resolve_dataset_dir,
)


class TestPipelineAdversarial(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)
        _DATASET_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        _DATASET_CACHE.clear()

    # ==========================================
    # 1. BOUNDARY SIZES & EXTREME DIMENSIONS
    # ==========================================
    def test_1x1_pixel_image_phash(self):
        """1x1 image must not crash DCT or LANCZOS resize in pHash."""
        for mode in ("RGB", "L", "RGBA", "1"):
            img = Image.new(mode, (1, 1), color=1 if mode == "1" else (255 if mode == "L" else (255, 0, 0)))
            h = compute_phash(img)
            self.assertIsInstance(h, int)
            self.assertGreaterEqual(h, 0)

    def test_extreme_aspect_ratios_phash(self):
        """Extreme aspect ratios (10000x1, 1x10000) must compute pHash cleanly."""
        img_h = Image.new("RGB", (10000, 1), color=(120, 50, 200))
        img_v = Image.new("RGB", (1, 10000), color=(120, 50, 200))
        h_h = compute_phash(img_h)
        h_v = compute_phash(img_v)
        self.assertIsInstance(h_h, int)
        self.assertIsInstance(h_v, int)

    def test_flat_color_frames(self):
        """Uniform frames (all-black, all-white, mid-gray) must not yield NaN DCT."""
        black = Image.new("RGB", (64, 64), color=(0, 0, 0))
        white = Image.new("RGB", (64, 64), color=(255, 255, 255))
        gray = Image.new("RGB", (64, 64), color=(128, 128, 128))

        h_black = compute_phash(black)
        h_white = compute_phash(white)
        h_gray = compute_phash(gray)

        self.assertIsInstance(h_black, int)
        self.assertIsInstance(h_white, int)
        self.assertIsInstance(h_gray, int)

    # ==========================================
    # 2. IMAGE MODES & COLOR CHANNELS
    # ==========================================
    def test_uncommon_image_modes_phash(self):
        """Palette (P), CMYK, Grayscale+Alpha (LA) modes in compute_phash."""
        modes = ["P", "CMYK", "LA"]
        for m in modes:
            base = Image.new("RGB", (64, 64), color=(100, 150, 200))
            converted = base.convert(m)
            h = compute_phash(converted)
            self.assertIsInstance(h, int)

    def test_eval_transforms_channels_contract(self):
        """get_eval_transforms contract on RGB vs non-RGB PIL images."""
        tf = get_eval_transforms(224)
        rgb = Image.new("RGB", (300, 200), color=(255, 100, 50))
        out_rgb = tf(rgb)
        self.assertEqual(out_rgb.shape, (3, 224, 224))
        self.assertEqual(out_rgb.dtype, torch.float32)

    def test_eval_transforms_direct_rgba_or_grayscale_behavior(self):
        """Verify behavior when non-RGB PIL image is passed directly to get_eval_transforms."""
        tf = get_eval_transforms(224)
        gray = Image.new("L", (100, 100), color=128)
        rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        # Check if direct call fails or works
        # If transforms.ToTensor() creates 1-channel tensor, Normalize([3 elements]) will fail
        try:
            res_gray = tf(gray)
            gray_failed = False
        except Exception as e:
            gray_failed = True
            gray_err = str(e)

        try:
            res_rgba = tf(rgba)
            rgba_failed = False
        except Exception as e:
            rgba_failed = True
            rgba_err = str(e)

        self.assertFalse(gray_failed, f"Direct gray transform failed: {gray_err if gray_failed else ''}")
        self.assertFalse(rgba_failed, f"Direct rgba transform failed: {rgba_err if rgba_failed else ''}")
        self.assertEqual(res_gray.shape, (3, 224, 224))
        self.assertEqual(res_rgba.shape, (3, 224, 224))

    # ==========================================
    # 3. CORRUPT FILES & MALFORMED INPUTS
    # ==========================================
    def test_detect_corrupted_scenarios(self):
        """detect_corrupted must catch 0-byte, truncated, garbage, and invalid types."""
        # 0-byte file
        p_empty = self.root / "empty.jpg"
        p_empty.touch()
        self.assertTrue(detect_corrupted(p_empty))

        # Garbage bytes
        p_garbage = self.root / "garbage.png"
        p_garbage.write_bytes(b"RANDOM_NON_IMAGE_BYTES_1234567890")
        self.assertTrue(detect_corrupted(p_garbage))

        # Truncated JPEG
        p_trunc_jpg = self.root / "trunc.jpg"
        p_trunc_jpg.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
        self.assertTrue(detect_corrupted(p_trunc_jpg))

        # Truncated PNG
        p_trunc_png = self.root / "trunc.png"
        p_trunc_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        self.assertTrue(detect_corrupted(p_trunc_png))

        # Non-existent file
        self.assertTrue(detect_corrupted(self.root / "does_not_exist.jpg"))

        # Invalid object types
        self.assertTrue(detect_corrupted(None))  # type: ignore
        self.assertTrue(detect_corrupted(12345))  # type: ignore
        self.assertTrue(detect_corrupted({"key": "val"}))  # type: ignore

    # ==========================================
    # 4. DUPLICATE THRESHOLDS (dist=4 vs dist=5)
    # ==========================================
    def test_hamming_distance_exact_threshold_dist4_vs_dist5(self):
        """Strict verification: Hamming distance <= 4 is dropped, == 5 is KEPT."""
        # Base hash: 0
        h_base = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000
        # Hash with 4 bits flipped (dist=4) -> should be dropped
        h_dist4 = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111
        # Hash with 5 bits flipped (dist=5) -> should be kept
        h_dist5 = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00011111

        self.assertEqual(hamming_distance(h_base, h_dist4), 4)
        self.assertEqual(hamming_distance(h_base, h_dist5), 5)

        # Now test inside FireSmokeDataset
        val_img_dir = self.root / "valid" / "images"
        val_lbl_dir = self.root / "valid" / "labels"
        val_img_dir.mkdir(parents=True)
        val_lbl_dir.mkdir(parents=True)

        # Create 3 images
        # Image 0: Base image
        img0 = Image.new("RGB", (64, 64), color=(50, 50, 50))
        img0.save(val_img_dir / "img0.jpg")
        (val_lbl_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2")

        # Mock compute_phash so we control exact hashes
        import src.data.pipeline as pipeline_mod
        orig_phash = pipeline_mod.compute_phash

        hashes_map = {
            "img0.jpg": h_base,
            "img_dist4.jpg": h_dist4,
            "img_dist5.jpg": h_dist5,
        }

        # Save dist4 and dist5 images
        img1 = Image.new("RGB", (64, 64), color=(60, 60, 60))
        img1.save(val_img_dir / "img_dist4.jpg")
        (val_lbl_dir / "img_dist4.txt").write_text("0 0.5 0.5 0.2 0.2")

        img2 = Image.new("RGB", (64, 64), color=(70, 70, 70))
        img2.save(val_img_dir / "img_dist5.jpg")
        (val_lbl_dir / "img_dist5.txt").write_text("1 0.5 0.5 0.2 0.2")

        def mock_phash(im):
            if isinstance(im, (str, Path)):
                name = Path(im).name
            elif hasattr(im, "filename") and im.filename:
                name = Path(im.filename).name
            else:
                return orig_phash(im)
            return hashes_map.get(name, orig_phash(im))

        pipeline_mod.compute_phash = mock_phash
        try:
            ds = FireSmokeDataset(self.root, split="valid", filter_duplicates=True, use_cache=False)
            kept_names = [p.name for p, _ in ds.samples]

            # img0 must be kept
            self.assertIn("img0.jpg", kept_names)
            # img_dist4 (dist=4 <= 4) must be dropped
            self.assertNotIn("img_dist4.jpg", kept_names, "img_dist4 with Hamming distance 4 was NOT dropped!")
            # img_dist5 (dist=5 > 4) must be kept
            self.assertIn("img_dist5.jpg", kept_names, "img_dist5 with Hamming distance 5 was mistakenly dropped!")
            self.assertEqual(len(ds), 2)
        finally:
            pipeline_mod.compute_phash = orig_phash

    # ==========================================
    # 5. CROSS-SPLIT DEDUPLICATION & LEAKAGE
    # ==========================================
    def test_cross_split_deduplication_audit(self):
        """Audit whether FireSmokeDataset filters images in eval split that duplicate train images."""
        train_img_dir = self.root / "train" / "images"
        train_lbl_dir = self.root / "train" / "labels"
        val_img_dir = self.root / "valid" / "images"
        val_lbl_dir = self.root / "valid" / "labels"

        train_img_dir.mkdir(parents=True)
        train_lbl_dir.mkdir(parents=True)
        val_img_dir.mkdir(parents=True)
        val_lbl_dir.mkdir(parents=True)

        # Identical image in train and validation
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        img.save(train_img_dir / "train_01.jpg")
        (train_lbl_dir / "train_01.txt").write_text("0 0.5 0.5 0.2 0.2")

        img.save(val_img_dir / "val_dup_of_train.jpg")
        (val_lbl_dir / "val_dup_of_train.txt").write_text("0 0.5 0.5 0.2 0.2")

        # Load val dataset
        val_ds = FireSmokeDataset(self.root, split="valid", filter_duplicates=True, use_cache=False)

        # Check if val_dup_of_train is present or filtered
        val_names = [p.name for p, _ in val_ds.samples]
        cross_split_filtered = ("val_dup_of_train.jpg" not in val_names)
        self.assertTrue(cross_split_filtered, "Cross-split duplicate from train was not filtered from validation split")

    # ==========================================
    # 6. LABEL PARSING CORNER CASES
    # ==========================================
    def test_label_parsing_anomalies(self):
        """Test malformed, blank, multi-line, comment, or negative label values."""
        lbl_dir = self.root / "valid" / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Missing file
        self.assertEqual(_parse_label(None), 0)
        self.assertEqual(_parse_label(lbl_dir / "nonexistent.txt"), 0)

        # Empty file
        p_empty = lbl_dir / "empty.txt"
        p_empty.write_text("")
        self.assertEqual(_parse_label(p_empty), 0)

        # Whitespace file
        p_ws = lbl_dir / "whitespace.txt"
        p_ws.write_text("   \n\t  \n  ")
        self.assertEqual(_parse_label(p_ws), 0)

        # Smoke label
        p_smoke = lbl_dir / "smoke.txt"
        p_smoke.write_text("1 0.3 0.3 0.1 0.1\n")
        self.assertEqual(_parse_label(p_smoke), 1)

        # Fire label
        p_fire = lbl_dir / "fire.txt"
        p_fire.write_text("0 0.5 0.5 0.2 0.2\n")
        self.assertEqual(_parse_label(p_fire), 0)

        # Multi-box (first is 1, second is 0) -> Fire (0) must take precedence
        p_multi = lbl_dir / "multi.txt"
        p_multi.write_text("1 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.1 0.1\n")
        self.assertEqual(_parse_label(p_multi), 0)

        # Corrupt text in label
        p_corrupt_lbl = lbl_dir / "corrupt.txt"
        p_corrupt_lbl.write_text("NOT_AN_INTEGER 0.5 0.5 0.2 0.2\n")
        self.assertEqual(_parse_label(p_corrupt_lbl), 0)

    def test_validation_dir_resolution(self):
        """Directory named 'validation' must resolve properly even when SPLIT_MAP maps to 'valid'."""
        val_dir = self.root / "validation" / "images"
        val_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir = self.root / "validation" / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (64, 64), color=(100, 150, 200))
        im.save(val_dir / "val_sample.jpg")
        (lbl_dir / "val_sample.txt").write_text("0 0.5 0.5 0.2 0.2")

        ds = FireSmokeDataset(self.root, split="validation", use_cache=False)
        self.assertEqual(len(ds), 1)

    # ==========================================
    # 7. DATALOADER STRESS & MULTI-WORKER
    # ==========================================
    def test_dataloader_batching_and_multiworker(self):
        """Test create_dataloader with batch sizes and num_workers."""
        img_dir = self.root / "valid" / "images"
        lbl_dir = self.root / "valid" / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            im = Image.new("RGB", (64, 64), color=(i * 20, (i * 30) % 255, 100))
            im.save(img_dir / f"test_{i:02d}.jpg")
            (lbl_dir / f"test_{i:02d}.txt").write_text(f"{i % 2} 0.5 0.5 0.2 0.2")

        # Test batch_size > len(dataset)
        loader_large_batch = create_dataloader(self.root, split="valid", batch_size=32, num_workers=0, filter_duplicates=False)
        for batch_x, batch_y, paths in loader_large_batch:
            self.assertEqual(batch_x.shape, (10, 3, 224, 224))
            self.assertEqual(batch_y.shape, (10,))
            self.assertEqual(len(paths), 10)

        # Test num_workers=2
        loader_workers = create_dataloader(self.root, split="valid", batch_size=4, num_workers=2, filter_duplicates=False)
        total_samples = 0
        for batch_x, batch_y, paths in loader_workers:
            total_samples += len(batch_y)
        self.assertEqual(total_samples, 10)


if __name__ == "__main__":
    unittest.main()
