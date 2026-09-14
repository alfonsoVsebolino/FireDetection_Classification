"""Comprehensive Empirical Adversarial Stress Testing for M1 Iteration 2."""

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import (
    FireSmokeDataset,
    HammingIndex,
    _DATASET_CACHE,
    _TRAIN_INDEX_CACHE,
    _ensure_rgb,
    _get_dir_fingerprint,
    _get_train_hash_index,
    _parse_label,
    _resolve_split_dir,
    compute_phash,
    create_dataloader,
    detect_corrupted,
    get_eval_transforms,
    hamming_distance,
)


class TestChallengerM1Iter2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        _DATASET_CACHE.clear()
        _TRAIN_INDEX_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        _DATASET_CACHE.clear()
        _TRAIN_INDEX_CACHE.clear()

    # -------------------------------------------------------------
    # 1. HammingIndex Mathematical Soundness & Exhaustive Stress
    # -------------------------------------------------------------
    def test_hamming_index_pigeonhole_guarantee(self):
        """Verify Pigeonhole Principle for 64-bit hashes partitioned into 5 chunks (13+13+13+13+12).
        For any distance d in [0, 4], at least one chunk must have 0 bit flips.
        """
        rng = np.random.RandomState(42)
        idx = HammingIndex()
        base_hashes = [int(h) for h in rng.randint(0, 2**63 - 1, size=200, dtype=np.int64)]
        for h in base_hashes:
            idx.add(h)

        # Test exact match (d=0)
        for h in base_hashes:
            self.assertTrue(idx.has_near_duplicate(h, 4))

        # Test d in {1, 2, 3, 4}
        for d in (1, 2, 3, 4):
            for base in base_hashes[:50]:
                bit_positions = rng.choice(64, size=d, replace=False)
                mask = 0
                for pos in bit_positions:
                    mask |= (1 << int(pos))
                mutated = base ^ mask
                self.assertTrue(
                    idx.has_near_duplicate(mutated, 4),
                    f"HammingIndex missed near-duplicate at distance {d}"
                )

    def test_hamming_index_exact_cutoff_at_5(self):
        """HammingIndex with a single hash should reject query with d_H=5 unless matched."""
        idx = HammingIndex()
        base = 0xAAAAAAAAAAAAAAAA  # 101010...1010
        idx.add(base)

        # Flip 4 bits: bits 0, 1, 2, 3
        mut_4 = base ^ 0b1111
        self.assertEqual(hamming_distance(base, mut_4), 4)
        self.assertTrue(idx.has_near_duplicate(mut_4, 4))

        # Flip 5 bits: bits 0, 1, 2, 3, 4
        mut_5 = base ^ 0b11111
        self.assertEqual(hamming_distance(base, mut_5), 5)
        self.assertFalse(idx.has_near_duplicate(mut_5, 4))

    # -------------------------------------------------------------
    # 2. Cross-Split Deduplication Against Train
    # -------------------------------------------------------------
    def test_cross_split_deduplication_eval_against_train(self):
        """Verify that validation and test splits drop near-duplicates (d <= 4) of train."""
        for split_name in ("valid", "validation", "test"):
            d_dir = self.temp_dir / f"test_{split_name}"
            d_dir.mkdir(parents=True, exist_ok=True)
            (d_dir / "train" / "images").mkdir(parents=True)
            (d_dir / "train" / "labels").mkdir(parents=True)
            (d_dir / split_name / "images").mkdir(parents=True)
            (d_dir / split_name / "labels").mkdir(parents=True)

            # Create base train image
            train_im = Image.new("RGB", (64, 64), color=(200, 100, 50))
            train_im.save(d_dir / "train" / "images" / "train_0.jpg")
            (d_dir / "train" / "labels" / "train_0.txt").write_text("0 0.5 0.5 0.2 0.2")

            # Create identical image in eval split (should be dropped)
            train_im.save(d_dir / split_name / "images" / "eval_dup.jpg")
            (d_dir / split_name / "labels" / "eval_dup.txt").write_text("0 0.5 0.5 0.2 0.2")

            # Create distinct image in eval split (should be kept)
            # Make distinct pattern
            distinct_im = Image.new("RGB", (64, 64), color=(10, 20, 30))
            for x in range(30):
                distinct_im.putpixel((x, x), (255, 255, 255))
            distinct_im.save(d_dir / split_name / "images" / "eval_unique.jpg")
            (d_dir / split_name / "labels" / "eval_unique.txt").write_text("1 0.5 0.5 0.2 0.2")

            ds = FireSmokeDataset(d_dir, split=split_name, filter_duplicates=True, use_cache=False)
            kept_stems = [p.stem for p, _ in ds.samples]

            self.assertNotIn(
                "eval_dup",
                kept_stems,
                f"Split '{split_name}' leaked duplicate of train!"
            )
            self.assertIn(
                "eval_unique",
                kept_stems,
                f"Split '{split_name}' dropped unique image!"
            )
            self.assertEqual(len(ds), 1)

    def test_train_split_does_not_filter_against_itself_improperly(self):
        """Train split must deduplicate internally, keeping 1 copy of identical images."""
        d_dir = self.temp_dir / "train_test"
        (d_dir / "train" / "images").mkdir(parents=True)
        (d_dir / "train" / "labels").mkdir(parents=True)

        im = Image.new("RGB", (64, 64), color=(123, 45, 67))
        im.save(d_dir / "train" / "images" / "t1.jpg")
        (d_dir / "train" / "labels" / "t1.txt").write_text("0 0.5 0.5 0.2 0.2")
        im.save(d_dir / "train" / "images" / "t2.jpg")
        (d_dir / "train" / "labels" / "t2.txt").write_text("0 0.5 0.5 0.2 0.2")

        ds = FireSmokeDataset(d_dir, split="train", filter_duplicates=True, use_cache=False)
        self.assertEqual(len(ds), 1)

    # -------------------------------------------------------------
    # 3. Split Directory Resolution Robustness
    # -------------------------------------------------------------
    def test_split_dir_resolution_variants(self):
        """Check all directory variations for validation/train/test resolution."""
        variants = [
            ("valid", "valid"),
            ("validation", "validation"),
            ("val", "val"),
            ("VALID", "valid"),
            ("Validation", "validation"),
            ("train", "train"),
            ("training", "training"),
            ("test", "test"),
            ("testing", "testing"),
        ]
        for query_split, dir_name in variants:
            root = self.temp_dir / f"res_{query_split}"
            (root / dir_name / "images").mkdir(parents=True)
            (root / dir_name / "labels").mkdir(parents=True)

            im = Image.new("RGB", (64, 64), color=(50, 100, 150))
            im.save(root / dir_name / "images" / "sample.jpg")
            (root / dir_name / "labels" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2")

            resolved = _resolve_split_dir(root, query_split)
            self.assertEqual(
                resolved.resolve(),
                (root / dir_name).resolve(),
                f"Failed to resolve {query_split} to {dir_name}"
            )

            ds = FireSmokeDataset(root, split=query_split, filter_duplicates=False, use_cache=False)
            self.assertEqual(
                len(ds),
                1,
                f"Dataset for split='{query_split}' resolved to 0 samples when disk dir is '{dir_name}'"
            )

    # -------------------------------------------------------------
    # 4. Cache Invalidation & Fingerprinting
    # -------------------------------------------------------------
    def test_cache_disabled_by_default(self):
        """Verify use_cache defaults to False in FireSmokeDataset and create_dataloader."""
        ds = FireSmokeDataset.__init__.__defaults__
        # Look at use_cache default
        import inspect
        sig = inspect.signature(FireSmokeDataset.__init__)
        self.assertFalse(sig.parameters["use_cache"].default)

        sig_loader = inspect.signature(create_dataloader)
        self.assertFalse(sig_loader.parameters["use_cache"].default)

    def test_cache_invalidation_on_file_addition(self):
        """When use_cache=True, adding a new file modifies fingerprint and busts cache."""
        d_dir = self.temp_dir / "cache_test"
        img_dir = d_dir / "valid" / "images"
        lbl_dir = d_dir / "valid" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        im1 = Image.new("RGB", (64, 64), color=(10, 20, 30))
        im1.save(img_dir / "im1.jpg")
        (lbl_dir / "im1.txt").write_text("0 0.5 0.5 0.2 0.2")

        ds1 = FireSmokeDataset(d_dir, split="valid", filter_duplicates=False, use_cache=True)
        self.assertEqual(len(ds1), 1)

        # Add second image
        time.sleep(0.01)  # Ensure filesystem mtime changes
        im2 = Image.new("RGB", (64, 64), color=(100, 200, 250))
        im2.save(img_dir / "im2.jpg")
        (lbl_dir / "im2.txt").write_text("1 0.5 0.5 0.2 0.2")

        ds2 = FireSmokeDataset(d_dir, split="valid", filter_duplicates=False, use_cache=True)
        self.assertEqual(len(ds2), 2, "Cache was NOT invalidated when new image was added!")

    # -------------------------------------------------------------
    # 5. Non-RGB Image Handling & Multi-Worker Picklability
    # -------------------------------------------------------------
    def test_ensure_rgb_and_transforms_all_modes(self):
        """get_eval_transforms must process RGB, L, RGBA, CMYK, P, 1 without error."""
        tf = get_eval_transforms(224)
        modes = ["RGB", "L", "RGBA", "CMYK", "P", "1"]
        for m in modes:
            im = Image.new(m, (80, 80), color=1 if m == "1" else (100 if m == "L" else (100, 150, 200)))
            out = tf(im)
            self.assertEqual(out.shape, (3, 224, 224), f"Failed for mode {m}")
            self.assertEqual(out.dtype, torch.float32)

    def test_ensure_rgb_tensor_inputs(self):
        """_ensure_rgb handles 1-channel, 2-channel, 4-channel, and 2D tensors."""
        # 1-channel: (1, 32, 32)
        t1 = torch.rand(1, 32, 32)
        self.assertEqual(_ensure_rgb(t1).shape, (3, 32, 32))

        # 2D: (32, 32)
        t2d = torch.rand(32, 32)
        self.assertEqual(_ensure_rgb(t2d).shape, (3, 32, 32))

        # 4-channel RGBA tensor: (4, 32, 32)
        t4 = torch.rand(4, 32, 32)
        self.assertEqual(_ensure_rgb(t4).shape, (3, 32, 32))

        # 3-channel: should pass untouched
        t3 = torch.rand(3, 32, 32)
        self.assertEqual(_ensure_rgb(t3).shape, (3, 32, 32))

    def test_transforms_picklability_and_multiprocessing(self):
        """Transforms must be fully picklable (no local lambda) for multiprocessing DataLoader."""
        tf = get_eval_transforms(224)
        dumped = pickle.dumps(tf)
        loaded = pickle.loads(dumped)
        self.assertIsNotNone(loaded)

        im = Image.new("L", (100, 100), color=50)
        out = loaded(im)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_dataloader_num_workers_with_non_rgb(self):
        """DataLoader with num_workers=2 must successfully load dataset containing Grayscale and RGBA images."""
        d_dir = self.temp_dir / "loader_test"
        img_dir = d_dir / "valid" / "images"
        lbl_dir = d_dir / "valid" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(8):
            mode = "RGBA" if i % 2 == 0 else "L"
            im = Image.new(mode, (64, 64), color=(i * 20, 100, 150) if mode == "RGBA" else i * 30)
            im.save(img_dir / f"im_{i}.png")
            (lbl_dir / f"im_{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.2 0.2")

        loader = create_dataloader(
            d_dir,
            split="valid",
            batch_size=4,
            num_workers=2,
            filter_duplicates=False,
            use_cache=False
        )

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        for x, y, paths in batches:
            self.assertEqual(x.shape, (4, 3, 224, 224))
            self.assertEqual(y.shape, (4,))

    # -------------------------------------------------------------
    # 6. Safety-First Label Parsing
    # -------------------------------------------------------------
    def test_parse_label_safety_prioritization(self):
        """Prioritize class 0 (Fire) over class 1 (Smoke) when both exist."""
        lbl_dir = self.temp_dir / "labels"
        lbl_dir.mkdir(parents=True)

        # Smoke first, Fire second -> Fire (0)
        f1 = lbl_dir / "smoke_then_fire.txt"
        f1.write_text("1 0.1 0.1 0.2 0.2\n0 0.5 0.5 0.3 0.3\n")
        self.assertEqual(_parse_label(f1), 0)

        # Fire first, Smoke second -> Fire (0)
        f2 = lbl_dir / "fire_then_smoke.txt"
        f2.write_text("0 0.5 0.5 0.3 0.3\n1 0.1 0.1 0.2 0.2\n")
        self.assertEqual(_parse_label(f2), 0)

        # Multiple smoke only -> Smoke (1)
        f3 = lbl_dir / "only_smoke.txt"
        f3.write_text("1 0.1 0.1 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")
        self.assertEqual(_parse_label(f3), 1)

        # Corrupt and empty cases
        f4 = lbl_dir / "corrupted_lines.txt"
        f4.write_text("invalid_label\n1 0.2 0.2 0.1 0.1\n")
        self.assertEqual(_parse_label(f4), 1)


if __name__ == "__main__":
    unittest.main()
