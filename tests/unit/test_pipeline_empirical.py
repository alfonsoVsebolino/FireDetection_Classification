"""Empirical stress test harness for DataLoader batching, multiprocessing, shapes, and memory efficiency."""

import gc
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FireSmokeDataset,
    create_dataloader,
    get_dataloaders,
    get_eval_transforms,
    resolve_dataset_dir,
)


class TestPipelineEmpirical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root_path = Path(cls.temp_dir.name)

        # Create mock synthetic dataset with edge cases: RGBA, Grayscale, CMYK, Palette, 1x1, large image
        cls.splits = ["train", "valid", "test"]
        cls.image_counts = {"train": 15, "valid": 9, "test": 7}

        for split in cls.splits:
            img_dir = cls.root_path / split / "images"
            lbl_dir = cls.root_path / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            count = cls.image_counts[split]
            for i in range(count):
                lbl_val = i % 2
                # Generate varied image types
                if i == 0:
                    # 1x1 tiny image
                    im = Image.new("RGB", (1, 1), color=(255, 0, 0))
                elif i == 1:
                    # RGBA image
                    im = Image.new("RGBA", (120, 80), color=(0, 255, 0, 128))
                elif i == 2:
                    # Grayscale (L) image
                    im = Image.new("L", (80, 120), color=200)
                elif i == 3:
                    # CMYK image
                    im = Image.new("CMYK", (90, 90), color=(100, 50, 0, 20))
                elif i == 4:
                    # Palette (P) image
                    im = Image.new("P", (60, 60))
                    im.putpalette([i % 256 for i in range(768)])
                elif i == 5:
                    # Uneven aspect ratio (10 x 500)
                    im = Image.new("RGB", (10, 500), color=(50, 100, 150))
                else:
                    # Standard RGB with random noise
                    arr = np.random.randint(0, 255, (100 + i * 5, 100 + i * 5, 3), dtype=np.uint8)
                    im = Image.fromarray(arr)

                if i == 3:
                    img_path = img_dir / f"img_{i:03d}.jpg"
                else:
                    img_path = img_dir / f"img_{i:03d}.png"
                im.save(img_path)

                # Label files: test normal, multi-line, empty, whitespace
                lbl_path = lbl_dir / f"img_{i:03d}.txt"
                if i == 3:
                    # Empty label file
                    lbl_path.write_text("")
                elif i == 4:
                    # Whitespace label file
                    lbl_path.write_text("   \n\t\n")
                elif i == 5:
                    # Multi-line YOLO box
                    lbl_path.write_text(f"{lbl_val} 0.5 0.5 0.2 0.2\n{lbl_val} 0.2 0.2 0.1 0.1\n")
                else:
                    lbl_path.write_text(f"{lbl_val} 0.5 0.5 0.4 0.4\n")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_tensor_shapes_and_types_all_batch_sizes(self):
        """Verify tensor shapes (B, 3, 224, 224) and labels (B,) across batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            loader = create_dataloader(
                root_dir=self.root_path,
                split="valid",
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                filter_duplicates=False,
            )
            total_samples = 0
            for imgs, labels, paths in loader:
                b = imgs.shape[0]
                self.assertLessEqual(b, batch_size)
                self.assertEqual(imgs.shape, (b, 3, 224, 224))
                self.assertEqual(imgs.dtype, torch.float32)
                self.assertEqual(labels.shape, (b,))
                self.assertEqual(labels.dtype, torch.int64)
                self.assertEqual(len(paths), b)
                for lbl in labels:
                    self.assertIn(lbl.item(), [0, 1])
                total_samples += b
            self.assertEqual(total_samples, self.image_counts["valid"])

    def test_multiprocessing_workers(self):
        """Verify DataLoader runs reliably with num_workers in [0, 1, 2, 4]."""
        for workers in [0, 1, 2, 4]:
            loader = create_dataloader(
                root_dir=self.root_path,
                split="train",
                batch_size=4,
                shuffle=True,
                num_workers=workers,
                filter_duplicates=False,
            )
            batch_count = 0
            sample_count = 0
            for imgs, labels, paths in loader:
                b = imgs.shape[0]
                self.assertEqual(imgs.shape, (b, 3, 224, 224))
                self.assertEqual(labels.shape, (b,))
                batch_count += 1
                sample_count += b
            self.assertEqual(sample_count, self.image_counts["train"])
            self.assertGreater(batch_count, 0)

    def test_memory_stability_and_file_descriptors(self):
        """Verify memory stability and absence of file descriptor leaks during iteration."""
        # Get baseline FDs
        fd_dir = "/proc/self/fd"
        initial_fds = len(os.listdir(fd_dir)) if os.path.exists(fd_dir) else 0

        loader = create_dataloader(
            root_dir=self.root_path,
            split="train",
            batch_size=2,
            shuffle=False,
            num_workers=0,
            filter_duplicates=False,
        )

        gc.collect()
        rss_start = 0
        try:
            import psutil
            process = psutil.Process()
            rss_start = process.memory_info().rss
        except ImportError:
            pass

        # Iterate multiple epochs through dataset
        for _ in range(5):
            for imgs, labels, paths in loader:
                _ = imgs.sum().item()

        gc.collect()
        if os.path.exists(fd_dir):
            final_fds = len(os.listdir(fd_dir))
            # FD difference should be minimal (no leaking open image handles)
            self.assertLessEqual(final_fds - initial_fds, 5, f"Leaked FDs: {final_fds - initial_fds}")

        if rss_start > 0:
            import psutil
            rss_end = psutil.Process().memory_info().rss
            delta_mb = (rss_end - rss_start) / (1024 * 1024)
            # Memory delta across 5 epochs on tiny dataset should be negligible (< 50MB)
            self.assertLess(delta_mb, 50.0, f"Excessive memory accumulation: {delta_mb:.2f} MB")

    def test_get_dataloaders_dict_contract(self):
        """Verify get_dataloaders returns dictionary with train, validation, test."""
        loaders = get_dataloaders(root_dir=self.root_path, batch_size=4, num_workers=0)
        self.assertIn("train", loaders)
        self.assertIn("validation", loaders)
        self.assertIn("test", loaders)
        for split_name, ldr in loaders.items():
            self.assertIsInstance(ldr, DataLoader)
            imgs, lbls, paths = next(iter(ldr))
            self.assertEqual(imgs.shape[1:], (3, 224, 224))
            self.assertEqual(lbls.ndim, 1)

    def test_real_hf_dataset_snapshot(self):
        """Verify actual HuggingFace cached dataset if available."""
        try:
            real_root = resolve_dataset_dir()
        except FileNotFoundError:
            self.skipTest("HuggingFace dataset cache not found")

        # Test valid split on real data with num_workers=2
        real_loader = create_dataloader(
            root_dir=real_root,
            split="valid",
            batch_size=32,
            shuffle=False,
            num_workers=2,
            filter_duplicates=True,
        )
        total_batches = 0
        total_items = 0
        for imgs, lbls, paths in real_loader:
            b = imgs.shape[0]
            self.assertEqual(imgs.shape, (b, 3, 224, 224))
            self.assertEqual(lbls.shape, (b,))
            self.assertEqual(imgs.dtype, torch.float32)
            self.assertEqual(lbls.dtype, torch.int64)
            total_batches += 1
            total_items += b
            if total_batches >= 5:  # Sample 5 batches
                break

        self.assertGreaterEqual(total_batches, 5)
        self.assertEqual(total_items, 5 * 32)


if __name__ == "__main__":
    unittest.main()
