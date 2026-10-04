# Project: FireDetection_Classification

## Architecture
Modular, production-ready PyTorch & Scikit-Learn/LightGBM classification system for fire and smoke detection, adhering to zero-leakage evaluation, cost-sensitive fire hazard recall ($\ge 90\%$), dual-gate ambient rejection, and champion selection governed by CPU inference latency.

```
                    Surveillance Image / Frame
                               │
                               ▼
                    [src/data/pipeline.py]
             (Corrupt Drop, pHash Dedup, 224x224 LANCZOS, ImageNet Norm)
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
 [Tier 1: LightGBM]      [Tier 2: ResNet18]      [Tier 3: DeiT-Tiny]
src/features/extract.py  src/models/train_resnet.py src/models/train_vit.py
src/models/train_lightgbm.py
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                               ▼
                   [src/evaluate/calibrate.py]
             Validation Threshold Sweep θ ∈ [0.10, 0.90]
             Enforce: Validation Fire Recall ≥ 0.90
                               │
                               ▼
                   [src/evaluate/benchmark.py]
             CPU Single-Sample Latency Benchmark
             Champion Selection & Export champion_config.json
                               │
                               ▼
                      [src/inference.py]
             Dual-Gated Real-World Inference Engine:
             Gate 1: max(P_fire, P_smoke) < 0.70 → ambient_frame
             Gate 2: P_fire ≥ θ* → fire, else smoke
             CLI: --input, --threshold, --output-json, --save-overlay
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dataset Snapshot & Ingestion | Resolve HF snapshot paths, map 'valid' ↔ 'validation', filter corrupt/empty images | M1 | ISSUE-01 |
| 2 | Perceptual Hash Deduplication | DCT-based pHash Hamming distance $\le 4$ intra- and cross-split filtering | M1 | ISSUE-01 |
| 3 | Evaluation Preprocessing & Zero Leakage | LANCZOS 224x224 resize, ImageNet normalization, strict zero augmentation on eval | M1 | ISSUE-01 |
| 4 | PyTorch Dataset & DataLoader Collation | Custom Dataset and DataLoader classes with reproducible collation and batching | M1 | ISSUE-01 |
| 5 | Handcrafted Feature Extraction | 96-bin RGB+HSV histograms, 9 color moments, Sobel/LBP texture stats | M2 | ISSUE-02 |
| 6 | Tier 1 LightGBM Classifier | LightGBM with fire class penalty ($w=2.0$), CPU train $<60$s, feature pipeline integration | M2 | ISSUE-02 |
| 7 | Tier 2 ResNet18 Transfer Backbone | Freeze conv1..layer1, trainable layer2..layer4 + Dropout(0.3)+Linear(512, 2) head | M3 | ISSUE-03 |
| 8 | Tier 2 Weighted Optimization Engine | CrossEntropyLoss weight=[2.0, 1.0], AdamW lr=1e-4, checkpoint saving | M3 | ISSUE-03 |
| 9 | Tier 3 DeiT-Tiny Vision Transformer | timm deit_tiny_patch16_224, freeze patch_embed+blocks[:6], train blocks[6:]+head, accum=2 | M4 | ISSUE-04 |
| 10 | Calibrated Threshold Sweep | Sweep $\theta \in [0.10, 0.90]$ (step 0.02) on val $P_{\text{fire}}$, enforce Fire Recall $\ge 0.90$ | M5 | ISSUE-05 |
| 11 | CPU Latency Benchmarking & Champion Governance | Measure single-sample CPU latency, select fastest compliant model, export config | M5 | ISSUE-05 |
| 12 | Dual-Gating Inference Engine | Gate 1 ($\max(P) < 0.70 \rightarrow \text{ambient\_frame}$) + Gate 2 ($P_{\text{fire}} \ge \theta^* \rightarrow \text{fire}$) | M6 | ISSUE-06 |
| 13 | Production CLI & Visual Overlay | CLI with `--input`, `--threshold`, `--output-json`, `--save-overlay` | M6 | ISSUE-06 |
| 14 | Dedicated Gradio Inference UI | Gradio Blocks interactive window with dynamic multi-tier model selection, dual-gating (θ, τ), zero-latency slider re-gating, tri-modal inputs (file, webcam, URL) | M6 | ISSUE-09 |
| 15 | E2E Testing Suite & Harness | Tiers 1-4 opaque-box verification test harness, synthetic fixtures, runner | E2E-TEST | Survey |
| 16 | Hot-Start & Checkpoint Ingestion | Architecture-aware model factory auto-discovering repo-committed checkpoints (`src/models/models_reproduce/`) with SHA-256 integrity validation and notebook retraining bypass | M6 | ISSUE-07 / ISSUE-08 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Harness, synthetic test fixtures, Tiers 1-4 test suites (`tests/`) | none | DONE (TEST_READY.md) |
| M1 | Dataset & DataLoader Pipeline | `src/data/pipeline.py`, pHash dedup, PyTorch Dataset/DataLoader | none | DONE |
| M2 | Handcrafted Features & LightGBM | `src/features/extract.py`, `src/models/train_lightgbm.py` | M1 | DONE |
| M3 | ResNet18 Transfer Learning Anchor | `src/models/train_resnet.py`, model builder, training/eval loop | M1 | DONE |
| M4 | DeiT-Tiny Vision Transformer | `src/models/train_vit.py`, model builder, training/eval loop | M1 | DONE |
| M5 | Evaluation & Champion Selection | `src/evaluate/calibrate.py`, `src/evaluate/benchmark.py` | M2, M3, M4 | DONE |
| M6 | Dual-Gate Inference API, Gradio UI & Hotstart | `src/inference.py`, `FireClassifier`, `src/ui/app.py`, `src/models/models_reproduce/` | M5 | DONE |
| M7 | Final Acceptance & Adversarial Hardening | Pass 100% E2E tests (Tiers 1-4) + Tier 5 adversarial stress testing | M1-M6, E2E | DONE |

## Interface Contracts

### 1. `src/data/pipeline.py`
- `compute_phash(img: PIL.Image.Image) -> int`
- `hamming_distance(h1: int, h2: int) -> int`
- `get_eval_transforms(img_size: int = 224) -> torchvision.transforms.Compose`:
  - `Resize((224, 224), interpolation=InterpolationMode.BILANCZOS)`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- `class FireSmokeDataset(torch.utils.data.Dataset)`:
  - `__init__(self, root_dir: str, split: str, transform=None, filter_duplicates: bool = True)`
  - `__len__(self) -> int`
  - `__getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]` (tensor, label, img_path)
- `create_dataloader(root_dir: str, split: str, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0) -> DataLoader`

### 2. `src/features/extract.py`
- `extract_color_histogram(img: np.ndarray, bins: int = 16) -> np.ndarray` (shape: `(96,)`)
- `extract_color_moments(img: np.ndarray) -> np.ndarray` (mean, std, skewness per channel in RGB & HSV: shape `(9,)` or `(18,)`)
- `extract_texture_features(img_gray: np.ndarray) -> np.ndarray` (Sobel gradients & LBP stats: shape `(16,)`+)
- `extract_all_features(img: PIL.Image.Image | np.ndarray) -> np.ndarray` (combined 1D feature vector)

### 3. `src/models/train_lightgbm.py`
- `train_lightgbm(features_train: np.ndarray, labels_train: np.ndarray, features_val: np.ndarray, labels_val: np.ndarray, config: dict = None) -> LGBMClassifier`
  - Class weights: `{0: 2.0, 1: 1.0}`
  - Params: `n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31`

### 4. `src/models/train_resnet.py`
- `build_resnet18(pretrained: bool = True, num_classes: int = 2) -> nn.Module`
  - Freezes `conv1, bn1, layer1`
  - Head: `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))`
- `train_resnet(dataloader_train, dataloader_val, epochs: int = 5, lr: float = 1e-4, device: str = "cpu") -> Tuple[nn.Module, dict]`
  - Loss: `nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]))`

### 5. `src/models/train_vit.py`
- `build_deit_tiny(pretrained: bool = True, num_classes: int = 2) -> nn.Module`
  - Freezes `patch_embed, blocks[:6]`
  - Trains `blocks[6:], head`
- `train_vit(dataloader_train, dataloader_val, epochs: int = 5, lr: float = 5e-5, accum_steps: int = 2, device: str = "cpu") -> Tuple[nn.Module, dict]`
  - Loss: `nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]))`

### 6. `src/evaluate/calibrate.py` & `src/evaluate/benchmark.py`
- `calibrate_threshold(y_true: np.ndarray, y_probs_fire: np.ndarray, min_recall: float = 0.90) -> Tuple[float, dict]`
  - Sweeps $\theta \in [0.10, 0.90]$ with step $0.02$
  - Selects operational threshold $\theta^*$ satisfying $\text{Recall}_{\text{fire}} \ge 0.90$
- `benchmark_model_cpu(model, sample_input: torch.Tensor | np.ndarray, num_runs: int = 100) -> float` (latency in ms)
- `select_champion(models_dict: dict, val_dataloaders: dict, test_dataloaders: dict, output_path: str = "models/champion_config.json") -> dict`

### 7. `src/inference.py`
- `class FireClassifier`:
  - `__init__(self, model_type: str = "auto", model_path: str = "", config_path: str = "", threshold: float = None, ambient_tau: float = None, device: str = "cpu")`
  - Zero-arg discovery: searches `src/models/models_reproduce/champion_config.json`, then `models/champion_config.json`.
  - Architecture factory: dynamically loads weights matching `architecture` config (`"deit_tiny_patch16_224"`, `"resnet18"`, `"lightgbm"`) with state-dict sniffing fallback.
  - Integrity validation: validates weights SHA-256 against `champion_config.json` hash contract.
  - `predict(self, image: PIL.Image.Image | str | np.ndarray) -> dict`:
    - Return dict:
      ```python
      {
          "hazard_detected": bool,         # False if ambient_frame, True if fire or smoke
          "predicted_class": str,          # "fire" | "smoke" | "ambient_frame"
          "probabilities": {
              "fire": float,
              "smoke": float
          },
          "operational_threshold": float,
          "confidence": float
      }
      ```
- CLI: `python3 -m src.inference --input <path> [--model-path <path>] [--config-path <path>] [--threshold <val>] [--ambient-tau <val>] [--output-json <path>]`

### 8. `src/ui/app.py` (Dedicated Gradio UI)
- `build_app(models_dir: str = "src/models/models_reproduce", device: str = "cpu") -> gr.Blocks`:
  - Dynamic multi-tier dropdown: auto-populates calibrated $\theta^*$ and $\tau$ per model.
  - Tri-modal inputs: drag-and-drop file upload, webcam snapshot, remote URL ingestion.
  - Zero-latency slider re-gating (<1ms): cached probabilities in `gr.State()` evaluated against dynamic $\theta, \tau$ sliders without repeating backbone forward passes.
- `launch_ui(models_dir: str = "src/models/models_reproduce", device: str = "cpu", port: int = 7860, host: str = "127.0.0.1", share: bool = False, in_notebook: bool = False)`:
  - Launches local server, Colab public share URL (`share=True`), or inline notebook iframe.
- CLI: `python -m src.ui.app [--port 7860] [--host 127.0.0.1] [--models-dir src/models/models_reproduce] [--share] [--device cpu]`

### 9. Repo-Tracked Checkpoints (`src/models/models_reproduce/`)
- `champion_config.json`: Champion metadata (`architecture: "deit_tiny_patch16_224"`, $\theta^* = 0.46$, $\tau = 0.70$, SHA-256: `10050e26de1cf71e...`).
- `champion_model.pt`: Canonical DeiT-Tiny weights matching config hash.
- `tier1_lightgbm.txt`: Exported LightGBM tree booster.
- `tier2_resnet18_best.pt`: PyTorch ResNet18 transfer learning weights.
- `tier3_deit_tiny_best.pt`: PyTorch DeiT-Tiny Vision Transformer weights.
- `test_benchmark_tiers.json`: Calibrated operational threshold evaluation dictionary.

## Code Layout
```
/home/alfonsovsebolino/Documents/CSELEC1-ML/EDA/FireDetection_Classification/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── extract.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models_reproduce/
│   │   │   ├── champion_config.json
│   │   │   ├── champion_model.pt
│   │   │   ├── tier1_lightgbm.txt
│   │   │   ├── tier2_resnet18_best.pt
│   │   │   ├── tier3_deit_tiny_best.pt
│   │   │   └── test_benchmark_tiers.json
│   │   ├── train_lightgbm.py
│   │   ├── train_resnet.py
│   │   └── train_vit.py
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── calibrate.py
│   │   └── benchmark.py
│   ├── inference.py
│   └── ui/
│       ├── __init__.py
│       ├── app.py
│       └── widget.py
├── tests/
│   ├── conftest.py
│   ├── test_runner.py
│   ├── unit/
│   │   ├── test_app.py
│   │   ├── test_hotstart.py
│   │   ├── test_pipeline.py
│   │   ├── test_features.py
│   │   ├── test_resnet.py
│   │   ├── test_vit.py
│   │   ├── test_calibrate.py
│   │   ├── test_inference.py
│   │   ├── test_full_benchmark.py
│   │   ├── test_pipeline_empirical.py
│   │   └── test_widget.py
│   └── e2e/
│       ├── test_tier1_features.py
│       ├── test_tier2_boundaries.py
│       ├── test_tier3_combinations.py
│       └── test_tier4_workloads.py
├── models/
│   └── champion_config.json
├── PROJECT.md
└── ORIGINAL_REQUEST.md
```
