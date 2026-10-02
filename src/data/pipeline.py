"""Data ingestion, deduplication, and preprocessing pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image
import scipy.fftpack
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_IMAGE_SIZE = (224, 224)

SPLIT_MAP: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
    "testing": "test",
}

_DATASET_CACHE: Dict[Tuple[str, str, bool, Tuple[int, int]], List[Tuple[Path, int]]] = {}
_TRAIN_INDEX_CACHE: Dict[Tuple[str, Tuple[int, int]], HammingIndex] = {}


class HammingIndex:
    """Multi-Index Hash for O(1) Hamming distance <= 4 lookups on 64-bit pHashes."""
    __slots__ = ("_tables", "_hashes")

    def __init__(self, hashes: Optional[Iterable[int]] = None) -> None:
        self._tables: Tuple[Dict[int, List[int]], ...] = tuple({} for _ in range(5))
        self._hashes: Set[int] = set()
        if hashes:
            for h in hashes:
                self.add(h)

    @staticmethod
    def _slices(h: int) -> Tuple[int, int, int, int, int]:
        return (
            h & 0x1FFF,
            (h >> 13) & 0x1FFF,
            (h >> 26) & 0x1FFF,
            (h >> 39) & 0x1FFF,
            (h >> 52) & 0xFFF,
        )

    def add(self, h: int) -> None:
        h = int(h)
        if h in self._hashes:
            return
        self._hashes.add(h)
        for i, s in enumerate(self._slices(h)):
            self._tables[i].setdefault(s, []).append(h)

    def has_near_duplicate(self, q: int, max_dist: int = 4) -> bool:
        q = int(q)
        if q in self._hashes:
            return True
        seen: Set[int] = set()
        for i, s in enumerate(self._slices(q)):
            bucket = self._tables[i].get(s)
            if bucket:
                for candidate in bucket:
                    if candidate not in seen:
                        seen.add(candidate)
                        if (q ^ candidate).bit_count() <= max_dist:
                            return True
        return False

    def __len__(self) -> int:
        return len(self._hashes)


def _ensure_rgb(img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
    """Ensure input image or tensor is 3-channel RGB before tensor transforms."""
    if hasattr(img, "convert"):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        if img.ndim == 3 and img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        if img.ndim == 2:
            return img.unsqueeze(0).repeat(3, 1, 1)
        if img.ndim == 3 and img.shape[0] == 4:
            return img[:3, :, :]
    return img


def compute_phash(img: Union[Image.Image, str, Path]) -> int:
    """Compute 64-bit DCT perceptual hash (pHash) on image."""
    if not isinstance(img, Image.Image):
        with Image.open(img) as opened_img:
            return compute_phash(opened_img)
    
    gray = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64)
    dct = scipy.fftpack.dct(
        scipy.fftpack.dct(arr, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )
    low_freq = dct[:8, :8]
    med = np.median(low_freq)
    diff = low_freq > med
    
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    return (int(h1) ^ int(h2)).bit_count()


def detect_corrupted(img: Union[Image.Image, str, Path]) -> bool:
    """Check if image is corrupted or cannot be decoded as RGB."""
    try:
        if isinstance(img, (str, Path)):
            with Image.open(img) as im:
                im.convert("RGB").load()
        elif hasattr(img, "convert"):
            img.convert("RGB").load()
        else:
            return True
        return False
    except Exception:
        return True


def is_corrupt(img: Union[Image.Image, str, Path]) -> bool:
    """Alias for detect_corrupted."""
    return detect_corrupted(img)


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """Zero-leakage evaluation transforms with LANCZOS resize and ImageNet norm."""
    try:
        interp = getattr(InterpolationMode, "BILANCZOS", InterpolationMode.LANCZOS)
    except Exception:
        interp = Image.Resampling.LANCZOS

    return transforms.Compose([
        transforms.Lambda(_ensure_rgb),
        transforms.Resize((img_size, img_size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def resolve_dataset_dir(root_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory containing dataset splits from local, Google Drive, or zip archive."""
    import zipfile

    def _find_data_root(candidate: Path) -> Optional[Path]:
        if not candidate.exists():
            if candidate.parent.is_dir():
                matched = None
                for p in candidate.parent.iterdir():
                    if p.name.lower() == candidate.name.lower():
                        matched = p
                        break
                if matched is None: 
                    return None
                candidate = matched
            else:
                return None

        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            extract_base = Path("/content")
            if not extract_base.is_dir():
                extract_base = candidate.parent
            
            has_train = False
            for d in extract_base.iterdir():
                if d.is_dir() and "data_preprocessed" in d.name.lower():
                    if (d / "train").is_dir() or next(d.glob("**/train"), None):
                        has_train = True
                        break
                    
            if not has_train:
                with zipfile.ZipFile(candidate, "r") as zf:
                    zf.extractall(extract_base)
            candidate = extract_base

        if not candidate.is_dir():
            return None

        for s in ("train", "training", "valid", "validation", "val", "test", "testing"):
            if (candidate / s).is_dir():
                return candidate

        for sub in ("Data_Preprocessed", "Data_PreProcessed", "Data"):
            subpath = candidate / sub
            if subpath.is_dir() and any((subpath / s).is_dir() for s in ("train", "valid", "val", "test")):
                return subpath

        for p in candidate.glob("**/train"):
            if p.is_dir():
                return p.parent

        if (candidate / "images").is_dir() or (candidate / "labels").is_dir():
            return candidate

        return None

    if root_dir:
        r_path = Path(root_dir).expanduser().resolve()
        resolved = _find_data_root(r_path)
        if resolved:
            return resolved
        if r_path.is_dir():
            return r_path

    candidate_paths = [
        Path("/content/Data_Preprocessed"),
        Path("/content/Data_PreProcessed"),
        Path("/content/drive/MyDrive/FireDetection_Classification/Data_Preprocessed.zip"),
        Path("/content/drive/MyDrive/FireDetection_Classification/Data_PreProcessed.zip"),
        Path("/content/drive/MyDrive/FireDetection_Classification/Data_Preprocessed"),
        Path("/content/drive/MyDrive/FireDetection_Classification/Data_PreProcessed"),
        Path("/content/drive/MyDrive/Data_Preprocessed.zip"),
        Path("/content/drive/MyDrive/Data_PreProcessed.zip"),
        Path("/content"),
        Path("Data_Preprocessed").resolve(),
        Path("Data_PreProcessed").resolve(),
        Path("Data_Preprocessed.zip").resolve(),
        Path("Data_PreProcessed.zip").resolve(),
        Path("Data").resolve(),
    ]

    for cand in candidate_paths:
        resolved = _find_data_root(cand)
        if resolved and (resolved / "train").is_dir():
            return resolved

    raise FileNotFoundError(
        "Could not resolve dataset directory containing 'train' split. "
        "Expected Google Drive archive at '/content/drive/MyDrive/FireDetection_Classification/Data_Preprocessed.zip' "
        "or extracted folder at '/content/Data_Preprocessed'."
    )


def download_dataset(repo_id: str = "Vertex-Test/FireSmokeDataset") -> str:
    """Return resolved dataset path."""
    return str(resolve_dataset_dir())


def _parse_label(lbl_path: Optional[Path]) -> int:
    """Parse Roboflow YOLO label file (0=fire, 1=smoke), prioritizing fire hazard (0)."""
    if lbl_path is None or not lbl_path.is_file():
        return 0
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return 0
            labels: List[int] = []
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        labels.append(int(parts[0]))
                    except ValueError:
                        continue
            if not labels:
                return 0
            return 0 if 0 in labels else labels[0]
    except Exception:
        return 0


def _get_dir_fingerprint(dir_path: Path) -> Tuple[int, int]:
    """Return (mtime_ns, file_count) for directory contents to detect modifications."""
    if not dir_path.is_dir():
        return (0, 0)
    try:
        stat = dir_path.stat()
        file_count = len(os.listdir(dir_path))
        return (stat.st_mtime_ns, file_count)
    except Exception:
        return (0, 0)


def _resolve_split_dir(root_dir: Path, split: str) -> Path:
    """Resolve split directory with robust candidate search across valid/validation/val."""
    split_clean = split.strip().lower()
    mapped = SPLIT_MAP.get(split_clean, split_clean)
    if split_clean in ("valid", "validation", "val"):
        candidates = [root_dir / split, root_dir / mapped, root_dir / "valid", root_dir / "validation", root_dir / "val"]
    elif split_clean in ("train", "training"):
        candidates = [root_dir / split, root_dir / mapped, root_dir / "train", root_dir / "training"]
    elif split_clean in ("test", "testing"):
        candidates = [root_dir / split, root_dir / mapped, root_dir / "test", root_dir / "testing"]
    else:
        candidates = [root_dir / split, root_dir / mapped]

    seen: Set[str] = set()
    for cand in candidates:
        cand_str = str(cand)
        if cand_str not in seen:
            seen.add(cand_str)
            if cand.is_dir():
                return cand

    if (root_dir / "images").is_dir() or (root_dir / "labels").is_dir():
        return root_dir
    return root_dir / mapped


def _get_train_hash_index(root_dir: Path, split_dir: Path) -> Optional[HammingIndex]:
    """Retrieve or compute HammingIndex for train split to enable fast cross-split deduplication."""
    train_candidates = [
        root_dir / "train",
        root_dir / "training",
        split_dir.parent / "train",
        split_dir.parent / "training",
    ]
    train_dir = None
    for c in train_candidates:
        if c.is_dir() and c != split_dir:
            train_dir = c
            break
        
    if train_dir is None:
        return None

    train_img = train_dir / "images"
    if not train_img.is_dir():
        train_img = train_dir
    fp = _get_dir_fingerprint(train_img)
    ckey = (str(train_dir.resolve()), fp)
    if ckey in _TRAIN_INDEX_CACHE:
        return _TRAIN_INDEX_CACHE[ckey]

    cache_dir = Path(os.path.expanduser("~/.cache/fire_smoke_phash_cache"))
    train_hash = hashlib.md5(str(train_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_file = cache_dir / f"train_{train_hash}_{fp[0]}_{fp[1]}.json"
    if cache_file.is_file():
        try:
            with open(cache_file, "r") as f:
                cached_hashes = json.load(f)
            t_idx = HammingIndex(cached_hashes)
            _TRAIN_INDEX_CACHE[ckey] = t_idx
            return t_idx
        except Exception:
            pass

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(train_img.glob(ext))
    paths = sorted(set(paths))
    if not paths:
        return None

    t_idx = HammingIndex()
    if len(paths) > 50:
        def _hash_worker(p: Path) -> Optional[int]:
            if detect_corrupted(p):
                return None
            try:
                return compute_phash(p)
            except Exception:
                return None

        workers = min(16, (os.cpu_count() or 1) * 2)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for h in executor.map(_hash_worker, paths):
                if h is not None:
                    t_idx.add(h)
        
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(list(t_idx._hashes), f)
        except Exception:
            pass
    else:
        for p in paths:
            if not detect_corrupted(p):
                try:
                    t_idx.add(compute_phash(p))
                except Exception:
                    pass

    _TRAIN_INDEX_CACHE[ckey] = t_idx
    return t_idx


class FireSmokeDataset(Dataset):
    """PyTorch Dataset for Fire and Smoke classification."""

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        filter_duplicates: bool = True,
        use_cache: bool = False,
        cross_split_dedup: bool = True,
    ) -> None:
        self.root_dir = resolve_dataset_dir(root_dir)
        self.split_name = split
        self.split = SPLIT_MAP.get(split.lower(), split.lower())
        self.transform = transform or get_eval_transforms()
        self.filter_duplicates = filter_duplicates
        self.use_cache = use_cache
        self.cross_split_dedup = cross_split_dedup

        split_dir = _resolve_split_dir(self.root_dir, self.split_name)
        img_dir = split_dir / "images"
        if not img_dir.is_dir():
            img_dir = split_dir
        fp = _get_dir_fingerprint(img_dir)

        cache_key = (str(split_dir.resolve()), self.split, filter_duplicates, fp)
        if use_cache and cache_key in _DATASET_CACHE:
            cached_samples = _DATASET_CACHE[cache_key]
            if cached_samples and all(p.is_file() for p, _ in cached_samples):
                self.samples = list(cached_samples)
                return

        self.samples = self._load_samples(split_dir)
        if use_cache:
            _DATASET_CACHE[cache_key] = list(self.samples)

    def _load_samples(self, split_dir: Path) -> List[Tuple[Path, int]]:
        img_dir = split_dir / "images"
        if not img_dir.is_dir():
            img_dir = split_dir
        
        lbl_dir = split_dir / "labels"
        if not lbl_dir.is_dir():
            lbl_dir = split_dir

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
        image_paths: List[Path] = []
        for ext in extensions:
            image_paths.extend(img_dir.glob(ext))
        image_paths = sorted(set(image_paths))

        kept_samples: List[Tuple[Path, int]] = []
        local_index = HammingIndex()

        is_eval = self.split != "train"
        train_index = None
        if is_eval and self.filter_duplicates and self.cross_split_dedup:
            train_index = _get_train_hash_index(self.root_dir, split_dir)
            

        for p in image_paths:
            if detect_corrupted(p):
                continue

            base_name = p.stem
            lbl_file = lbl_dir / f"{base_name}.txt"
            label = _parse_label(lbl_file if lbl_file.is_file() else None)

            if self.filter_duplicates:
                try:
                    h = compute_phash(p)
                except Exception:
                    continue

                if train_index is not None and train_index.has_near_duplicate(h, 4):
                    continue

                if local_index.has_near_duplicate(h, 4):
                    continue

                local_index.add(h)

            kept_samples.append((p, label))

        if self.split == "train" and self.filter_duplicates:
            fp = _get_dir_fingerprint(img_dir)
            _TRAIN_INDEX_CACHE[(str(split_dir.resolve()), fp)] = local_index

        return kept_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as im:
            img_rgb = im.convert("RGB")
            tensor = self.transform(img_rgb)
        return tensor, label, str(img_path)


def create_dataloader(
    root_dir: Optional[Union[str, Path]] = None,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    filter_duplicates: bool = True,
    use_cache: bool = False,
    cross_split_dedup: bool = True,
) -> DataLoader:
    """Create a standard PyTorch DataLoader for the specified split."""
    dataset = FireSmokeDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        filter_duplicates=filter_duplicates,
        use_cache=use_cache,
        cross_split_dedup=cross_split_dedup,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_dataloaders(
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for all three standard splits."""
    return {
        "train": create_dataloader(
            root_dir=root_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            filter_duplicates=True,
        ),
        "validation": create_dataloader(
            root_dir=root_dir,
            split="validation",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            filter_duplicates=True,
        ),
        "test": create_dataloader(
            root_dir=root_dir,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            filter_duplicates=True,
        ),
    }


def load_raw_split(repo_path: str, split_name: str) -> List[Dict[str, Union[str, int]]]:
    """Load raw image records and labels for a split as list of dicts."""
    resolved_root = resolve_dataset_dir(repo_path)
    split_dir = _resolve_split_dir(resolved_root, split_name)

    img_dir = split_dir / "images"
    if not img_dir.is_dir():
        img_dir = split_dir
    
    lbl_dir = split_dir / "labels"
    if not split_dir.is_dir():
        lbl_dir = split_dir

    records: List[Dict[str, Union[str, int]]] = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            lbl_file = lbl_dir / f"{img_path.stem}.txt"
            label = _parse_label(lbl_file)
            records.append({"image": str(img_path), "label": label})
    return records
