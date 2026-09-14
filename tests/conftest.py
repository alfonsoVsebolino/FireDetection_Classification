"""Pytest test configuration and synthetic fixtures for FireDetection_Classification."""

import os
import sys
import shutil
import tempfile
import numpy as np
import pytest
from PIL import Image

# Guarantee project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.helpers import import_or_skip  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_rgb_image():
    """Generates a deterministic 224x224 RGB image array."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)[:, None]
    arr[:, :, 1] = np.linspace(255, 0, 224, dtype=np.uint8)[None, :]
    arr[:, :, 2] = 128
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="session")
def synthetic_fire_image():
    """Generates a synthetic fire image (predominantly high red/orange tones)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Fire signature: High Red (200-255), Medium Green (80-160), Low Blue (0-40)
    arr[:, :, 0] = np.random.RandomState(42).randint(200, 256, size=(224, 224), dtype=np.uint8)
    arr[:, :, 1] = np.random.RandomState(43).randint(80, 160, size=(224, 224), dtype=np.uint8)
    arr[:, :, 2] = np.random.RandomState(44).randint(0, 40, size=(224, 224), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="session")
def synthetic_smoke_image():
    """Generates a synthetic smoke image (predominantly gray/whitish diffuse values)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Smoke signature: R ~ G ~ B roughly equal (120-180)
    gray = np.random.RandomState(101).randint(120, 180, size=(224, 224), dtype=np.uint8)
    arr[:, :, 0] = gray
    arr[:, :, 1] = gray
    arr[:, :, 2] = gray
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="session")
def synthetic_ambient_image():
    """Generates a synthetic ambient frame (e.g. green forest / blue sky without flame/smoke)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Sky upper half (blue), grass lower half (green)
    arr[:112, :, 2] = 200
    arr[:112, :, 0] = 50
    arr[:112, :, 1] = 120
    arr[112:, :, 1] = 180
    arr[112:, :, 0] = 40
    arr[112:, :, 2] = 40
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def temp_image_dir():
    """Creates a temporary directory with image files and cleans up afterwards."""
    temp_dir = tempfile.mkdtemp(prefix="fire_smoke_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def corrupted_image_file(temp_image_dir):
    """Creates an invalid/corrupt image file with arbitrary non-image bytes."""
    corrupt_path = os.path.join(temp_image_dir, "corrupt.jpg")
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES_1234567890")
    return corrupt_path


@pytest.fixture
def empty_image_file(temp_image_dir):
    """Creates a 0-byte file."""
    empty_path = os.path.join(temp_image_dir, "empty.png")
    with open(empty_path, "wb") as f:
        pass
    return empty_path


@pytest.fixture
def identical_duplicate_pair(temp_image_dir, synthetic_fire_image):
    """Creates two identical images saved to distinct file paths."""
    p1 = os.path.join(temp_image_dir, "orig.jpg")
    p2 = os.path.join(temp_image_dir, "duplicate.jpg")
    synthetic_fire_image.save(p1, format="JPEG")
    synthetic_fire_image.save(p2, format="JPEG")
    return p1, p2


@pytest.fixture
def near_duplicate_pair(temp_image_dir, synthetic_smoke_image):
    """Creates two images with perceptual hash hamming distance <= 4."""
    p1 = os.path.join(temp_image_dir, "smoke1.jpg")
    p2 = os.path.join(temp_image_dir, "smoke2.jpg")
    synthetic_smoke_image.save(p1, format="JPEG")

    # Perturb only 2 pixels slightly
    perturbed = synthetic_smoke_image.copy()
    perturbed.putpixel((10, 10), (125, 125, 125))
    perturbed.putpixel((11, 11), (126, 126, 126))
    perturbed.save(p2, format="JPEG")
    return p1, p2


@pytest.fixture
def mock_dataset_dir(temp_image_dir):
    """Creates a mock dataset structure with train, valid, validation, and test splits."""
    splits = ["train", "valid", "test"]
    
    def make_distinct_img(base_color: tuple, idx: int):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :] = base_color
        # Insert distinct spatial patterns across 4x4 grid to guarantee pHash distance > 4 between indices
        bx = 10 + (idx % 4) * 45
        by = 10 + ((idx // 4) % 4) * 45
        arr[by:by+40, bx:bx+40] = np.random.RandomState(idx + 100).randint(0, 255, (40, 40, 3))
        return Image.fromarray(arr, mode="RGB")

    for split_idx, split in enumerate(splits):
        img_dir = os.path.join(temp_image_dir, split, "images")
        lbl_dir = os.path.join(temp_image_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Populate 3 distinct fire (label 0) and 3 distinct smoke (label 1) samples
        for i in range(3):
            f_img = make_distinct_img((220, 50, 10), i + split_idx * 10)
            f_img_path = os.path.join(img_dir, f"fire_{i}.jpg")
            f_lbl_path = os.path.join(lbl_dir, f"fire_{i}.txt")
            f_img.save(f_img_path, format="JPEG")
            with open(f_lbl_path, "w") as f:
                f.write(f"0 0.5 0.5 0.2 0.2\n")

            s_img = make_distinct_img((140, 140, 140), i + 100 + split_idx * 10)
            s_img_path = os.path.join(img_dir, f"smoke_{i}.jpg")
            s_lbl_path = os.path.join(lbl_dir, f"smoke_{i}.txt")
            s_img.save(s_img_path, format="JPEG")
            with open(s_lbl_path, "w") as f:
                f.write(f"1 0.4 0.4 0.3 0.3\n")

    # Add a corrupt sample to valid split
    corrupt_path = os.path.join(temp_image_dir, "valid", "images", "corrupt_val.jpg")
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPTED_STREAM")

    # Add an identical duplicate of test fire_0 (idx = 20) to test split
    dup_img = make_distinct_img((220, 50, 10), 20)
    dup_img_path = os.path.join(temp_image_dir, "test", "images", "duplicate_fire.jpg")
    dup_lbl_path = os.path.join(temp_image_dir, "test", "labels", "duplicate_fire.txt")
    dup_img.save(dup_img_path, format="JPEG")
    with open(dup_lbl_path, "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    return temp_image_dir
