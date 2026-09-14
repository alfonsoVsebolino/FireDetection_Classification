"""Unit tests for data pipeline, deduplication, and ingestion."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch

from src.data.pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FireSmokeDataset,
    compute_phash,
    create_dataloader,
    detect_corrupted,
    get_eval_transforms,
    hamming_distance,
    resolve_dataset_dir,
)


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.root_path = Path(self.temp_dir)
        self.valid_images_dir = self.root_path / "valid" / "images"
        self.valid_labels_dir = self.root_path / "valid" / "labels"
        self.valid_images_dir.mkdir(parents=True, exist_ok=True)
        self.valid_labels_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_phash_and_hamming_distance(self) -> None:
        img_a = Image.new("RGB", (100, 100), color=(255, 50, 50))
        img_a_clone = Image.new("RGB", (100, 100), color=(255, 50, 50))
        img_b = Image.new("RGB", (100, 100), color=(20, 200, 40))

        hash_a = compute_phash(img_a)
        hash_clone = compute_phash(img_a_clone)
        hash_b = compute_phash(img_b)

        self.assertIsInstance(hash_a, int)
        self.assertEqual(hash_a, hash_clone)
        self.assertEqual(hamming_distance(hash_a, hash_clone), 0)

        dist_diff = hamming_distance(hash_a, hash_b)
        self.assertGreaterEqual(dist_diff, 0)
        self.assertLessEqual(dist_diff, 64)

    def test_hamming_distance_bitwise(self) -> None:
        self.assertEqual(hamming_distance(0b1010, 0b1010), 0)
        self.assertEqual(hamming_distance(0b1010, 0b1011), 1)
        self.assertEqual(hamming_distance(0b0000, 0b1111), 4)

    def test_detect_corrupted(self) -> None:
        valid_img_path = self.valid_images_dir / "valid_sample.jpg"
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(valid_img_path)
        self.assertFalse(detect_corrupted(valid_img_path))

        corrupt_img_path = self.valid_images_dir / "corrupt_sample.jpg"
        with open(corrupt_img_path, "wb") as f:
            f.write(b"NOT_AN_IMAGE_HEADER_CORRUPTED_BYTES")
        self.assertTrue(detect_corrupted(corrupt_img_path))

    def test_get_eval_transforms(self) -> None:
        transform = get_eval_transforms(img_size=224)
        img = Image.new("RGB", (640, 480), color=(200, 100, 50))
        tensor = transform(img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))
        self.assertEqual(tensor.dtype, torch.float32)

        # Determinism check (zero augmentations)
        tensor2 = transform(img)
        self.assertTrue(torch.allclose(tensor, tensor2))

    def test_dataset_ingestion_and_deduplication(self) -> None:
        # Sample 1: Fire (label 0, horizontal stripes)
        arr1 = np.zeros((64, 64, 3), dtype=np.uint8)
        arr1[::8, :] = 255
        img1 = Image.fromarray(arr1)
        img1_path = self.valid_images_dir / "fire_01.jpg"
        img1.save(img1_path)
        with open(self.valid_labels_dir / "fire_01.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Sample 2: Duplicate of Sample 1 (should be filtered out)
        img2_path = self.valid_images_dir / "fire_01_dup.jpg"
        img1.save(img2_path)
        with open(self.valid_labels_dir / "fire_01_dup.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Sample 3: Smoke (label 1, vertical stripes)
        arr3 = np.zeros((64, 64, 3), dtype=np.uint8)
        arr3[:, ::8] = 255
        img3 = Image.fromarray(arr3)
        img3_path = self.valid_images_dir / "smoke_01.jpg"
        img3.save(img3_path)
        with open(self.valid_labels_dir / "smoke_01.txt", "w") as f:
            f.write("1 0.4 0.4 0.3 0.3\n")

        # Sample 4: Empty label file (random noise pattern, should safely default to 0)
        np.random.seed(123)
        arr4 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img4 = Image.fromarray(arr4)
        img4_path = self.valid_images_dir / "empty_lbl.jpg"
        img4.save(img4_path)
        with open(self.valid_labels_dir / "empty_lbl.txt", "w") as f:
            f.write("")

        # Sample 5: Corrupt image (should be dropped)
        corrupt_path = self.valid_images_dir / "bad.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPT")

        # Instantiate with split="validation" to test split mapping ('validation' -> 'valid')
        ds_dedup = FireSmokeDataset(
            root_dir=self.root_path,
            split="validation",
            filter_duplicates=True,
            use_cache=False,
        )

        # Expected: fire_01 kept, fire_01_dup dropped, smoke_01 kept, empty_lbl kept, bad dropped -> 3 samples
        self.assertEqual(len(ds_dedup), 3)

        labels = [item[1] for item in ds_dedup]
        self.assertIn(0, labels)
        self.assertIn(1, labels)

        # Test __getitem__
        tensor, label, path_str = ds_dedup[0]
        self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))
        self.assertIsInstance(label, int)
        self.assertTrue(os.path.exists(path_str))

        # Dataset without deduplication: should include duplicate sample (4 samples)
        ds_raw = FireSmokeDataset(
            root_dir=self.root_path,
            split="validation",
            filter_duplicates=False,
            use_cache=False,
        )
        self.assertEqual(len(ds_raw), 4)

    def test_create_dataloader(self) -> None:
        # Create minimal dataset
        for i in range(4):
            img = Image.new("RGB", (64, 64), color=(i * 40, i * 40, i * 40))
            img_path = self.valid_images_dir / f"img_{i}.jpg"
            img.save(img_path)
            with open(self.valid_labels_dir / f"img_{i}.txt", "w") as f:
                f.write(f"{i % 2} 0.5 0.5 0.2 0.2\n")

        loader = create_dataloader(
            root_dir=self.root_path,
            split="validation",
            batch_size=2,
            shuffle=False,
            num_workers=0,
            filter_duplicates=False,
        )

        batch_count = 0
        for batch_imgs, batch_lbls, batch_paths in loader:
            self.assertEqual(batch_imgs.shape, torch.Size([2, 3, 224, 224]))
            self.assertEqual(batch_lbls.shape, torch.Size([2]))
            self.assertEqual(len(batch_paths), 2)
            batch_count += 1

        self.assertEqual(batch_count, 2)


if __name__ == "__main__":
    unittest.main()
