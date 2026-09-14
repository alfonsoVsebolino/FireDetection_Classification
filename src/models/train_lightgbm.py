from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import joblib
from lightgbm import LGBMClassifier
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from src.features.extract import extract_all_features


def train_lightgbm(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: Optional[np.ndarray] = None,
    labels_val: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> LGBMClassifier:
    default_params: Dict[str, Any] = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "class_weight": {0: 2.0, 1: 1.0},
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    params = {**default_params, **(config or {})}
    model = LGBMClassifier(**params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if (
            features_val is not None
            and labels_val is not None
            and len(features_val) > 0
        ):
            model.fit(features_train, labels_train, eval_set=[(features_val, labels_val)])
        else:
            model.fit(features_train, labels_train)

    return model


def extract_features_from_dataset(
    dataset: Any,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    features: list[np.ndarray] = []
    labels: list[int] = []

    for i in range(total):
        item = dataset[i]
        if isinstance(item, (tuple, list)):
            data_elem = item[0]
            label = int(item[1])
            if len(item) >= 3 and isinstance(item[2], (str, Path)) and os.path.isfile(str(item[2])):
                with Image.open(str(item[2])) as im:
                    feat = extract_all_features(im)
            elif isinstance(data_elem, (Image.Image, np.ndarray)):
                feat = extract_all_features(data_elem)
            elif hasattr(data_elem, "numpy"):
                t = data_elem.detach().cpu()
                if t.ndim == 3 and t.shape[0] in (1, 3):
                    arr = t.permute(1, 2, 0).numpy()
                    arr = np.clip((arr - arr.min()) / (arr.max() - arr.min() + 1e-7) * 255.0, 0, 255).astype(np.uint8)
                    feat = extract_all_features(arr)
                else:
                    feat = extract_all_features(t.numpy())
            else:
                feat = extract_all_features(data_elem)
        else:
            feat = extract_all_features(item)
            label = 0

        features.append(feat)
        labels.append(label)

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.vstack(features).astype(np.float32), np.array(labels, dtype=np.int64)


def save_model(model: LGBMClassifier, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".txt":
        model.booster_.save_model(str(p))
    else:
        joblib.dump(model, str(p))


def load_model(path: Union[str, Path]) -> LGBMClassifier:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    if p.suffix == ".txt":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(p))
        model = LGBMClassifier()
        model._Booster = booster
        model._n_features = booster.num_feature()
        model._n_features_in = booster.num_feature()
        model._classes = np.array([0, 1])
        model._n_classes = 2
        le = LabelEncoder()
        le.classes_ = np.array([0, 1])
        model._le = le
        model.fitted_ = True
        return model

    return joblib.load(str(p))
