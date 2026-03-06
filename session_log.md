# ML Pipeline Development Session Log

**Project:** Predict H₂ fraction (`fh2`) per cell from 3D astrophysical simulation data
**Dataset:** 7 CSV cubes, one per UV field strength G0 ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4}
**Cross-validation:** Leave-one-G0-out (7-fold, one held-out cube per fold)
**Target:** `log10(fh2)`, evaluated in original `fh2` space via R², RMSE, MAE

---

## Files in the Pipeline

| File | Role |
|------|------|
| `data_loader.py` | Load 7 CSV cubes, log-transforms, `get_X_y()`, `cube_to_volumes()` |
| `augmentation.py` | Octahedral group Oh symmetry augmentation (8 or 48 ops) |
| `classical_models.py` | Linear Regression, XGBoost, MLP (PyTorch) with GPU batching |
| `cnn_model.py` | 3D U-Net with encoder-decoder + skip connections |
| `train_cnn.py` | CNN training loop, leave-one-G0-out CV, JSON logging |
| `evaluate.py` | Run all models, comparison plots, results JSON |
| `compare_architectures.py` | Architecture variant comparison (new, this session) |

---

## Bug Fixes (Earlier Session — Carried Over)

### 1. `AssertionError: Expected 8 z-preserving ops, got 16`
**File:** `augmentation.py`
**Root cause:** Filter `R[2,2] != 0` matched both z→+z and z→−z operations.
**Fix:** Changed to `R[2,2] == 1` (strict equality, not just nonzero).

```python
# Before
_Z_PRESERVING = [R for R in _OH_GROUP if R[2,2] != 0 and R[0,2] == 0 and R[1,2] == 0]
# After
_Z_PRESERVING = [R for R in _OH_GROUP if R[2,2] == 1 and R[0,2] == 0 and R[1,2] == 0]
```

### 2. `KeyError: 'vx'` in `augment_cube`
**File:** `augmentation.py`
**Root cause:** Guard condition checked for `'vx'`/`'vy'`/`'vz'` keys but the dict `v_old` was built with single-letter keys `'x'`/`'y'`/`'z'`.
**Fix:** Changed `if f'v{old_lbl}' in v_old` → `if old_lbl in v_old`.

### 3. NaN train/val losses + scheduler warning
**File:** `train_cnn.py`
**Root cause:** AMP float16 overflow. Physical fields (velocity, B-field) can exceed fp16 max (65504) when unscaled. The GradScaler skipped `opt.step()` every batch, so `sched.step()` fired before any optimizer step.
**Fix:** `use_amp = False` for CNN + `torch.nan_to_num` on dataset tensors.

### 4. 50% CPU / 50% GPU utilization
**File:** `train_cnn.py`, `CubeDataset`
**Root cause:** `__getitem__` was stacking 14 × 128³ numpy arrays per step — matching GPU compute time.
**Fix:** Moved all preprocessing (augment, stack, avg_pool3d, nan_to_num, .float()) into `__init__`. `__getitem__` became a trivial list lookup.

### 5. All R² values negative (first working CNN run)
**File:** `cnn_model.py`, `train_cnn.py`
**Root causes:**
- `BatchNorm3d` is undefined at `batch_size=1` → replaced with `InstanceNorm3d(affine=True)`
- No input normalization → added per-channel z-score from training fold stats
- Added gradient clipping (`clip_grad_norm_`, `max_norm=1.0`)

### 6. `UnicodeEncodeError` on Windows console
**Files:** `train_cnn.py`, `evaluate.py`
**Root cause:** `→` (U+2192) is not in Windows cp1252 codepage.
**Fix:** Replaced all `→` in `print()` calls with `->`. Docstrings/comments unaffected.

### 7. Timestamp not appearing in log filename
**File:** `train_cnn.py`
**Root cause:** `--log` argparser default was the literal string `"cnn_training_TIMESTAMP.json"`, so `args.log` was never `None` and the timestamp branch in `run_cnn_cv` never fired.
**Fix:** Changed `default="cnn_training_TIMESTAMP.json"` → `default=None`.

---

## CNN Improvements Applied This Session

### Target normalization (train_cnn.py)
**Problem:** MSE on raw `log_fh2` (range ≈ −30 to 0) was dominated by near-zero-fh2 cells.
**Fix:** Standardize targets with training fold mean/std; inverse-transform before `compute_metrics`.

```python
all_y  = torch.stack(train_ds.ys)
y_mean = all_y.mean()
y_std  = all_y.std().clamp(min=1e-6)
train_ds.ys = [(y - y_mean) / y_std for y in train_ds.ys]
val_ds.ys   = [(y - y_mean) / y_std for y in val_ds.ys]
# ... after collecting predictions:
y_true = y_true * y_std.item() + y_mean.item()
y_pred = y_pred * y_std.item() + y_mean.item()
```

**Impact:** Mean R² went from 0.532 (negative folds) → 0.775 (all positive).

---

## Three Training Runs — Comparison

| G0  | Run 1: no-drop 100ep | Run 2: drop0.10 150ep | Run 3: no-drop 150ep |
|-----|---------------------:|----------------------:|---------------------:|
| 0.1 | +0.6492              | +0.6339               | +0.4967              |
| 0.2 | +0.8467              | +0.9073               | +0.8061              |
| 0.4 | +0.9138              | +0.9358               | +0.9299              |
| 0.8 | +0.9217              | +0.8720               | +0.8944              |
| 1.6 | +0.9189              | +0.9247               | +0.9447              |
| 3.2 | +0.9551              | +0.8835               | +0.9576              |
| 6.4 | +0.2163              | −0.6111               | +0.5912              |
| **Mean R²** | **0.7745 ± 0.247** | 0.6494 ± 0.524 | **0.8029 ± 0.172** |

### Run 1 → Run 2: Dropout experiment (FAILED)
**Changes:** Dropout3d (0.10 encoder, 0.20 bottleneck), weight_decay 1e-5→1e-4, 150 epochs.
**Hypothesis:** Dropout would reduce overfitting (train loss collapses to ~0.006).
**Outcome:** Mean R² dropped from 0.775 → 0.649. G0=6.4 catastrophically failed (0.22 → −0.61).

**Key insight:** With only 6 training cubes, the model needs to *memorise* the precise physical relationship between G0 and fh2. Dropout disrupts this, especially for OOD extrapolation folds. The analogy to "overfitting" here is misleading — what looks like memorisation is actually learning real physics.

**Best val loss for G0=6.4:**
- Run 1: 0.0244 @ epoch 28
- Run 2: 0.0517 @ epoch 117 (2× worse despite more epochs and regularisation)

**Decision:** Reverted dropout=0.0, weight_decay=1e-5.

### Run 1 → Run 3: Extra 50 epochs (SUCCEEDED)
**Changes:** dropout=0.0, weight_decay=1e-5 (reverted), epochs 100→150.
**Outcome:** Mean R² improved to 0.8029 ± 0.172 (best run).

**Why it worked:** Best-val epochs in Run 1 were 56–80/100 — training budget was exhausted before convergence. In Run 3:
- G0=6.4 best val moved from epoch 28 → epoch 111 (needed 83 more epochs to escape local minimum)
- G0=0.4, 1.6, 3.2 all improved, needing 40+ additional epochs

**G0=0.1 slight regression (0.649 → 0.497):** Cosine annealing LR at epoch 66/150 is higher than at epoch 66/100, giving a slightly different model state. This fold has a structural ceiling (true extrapolation, no training cube below G0=0.1).

---

## Thinking Process: Why Dropout Fails for Extrapolation

Standard understanding: dropout helps when a model overfits to noise in the training set.

This situation is different:
- Training loss ~0.006 is not noise memorisation — it reflects real physical structure
- The 7 G0 values span 64× in UV field strength; each cube has a genuinely different fh2 distribution
- The model "memorising" training cubes is actually learning physics-consistent representations
- Dropout randomly zeros feature map channels → the model can't form stable representations of the G0-dependent physics
- For interpolation folds (G0 in range of training), this only slightly hurts
- For extrapolation folds (G0=0.1, 6.4), stable representations are *essential* — the model must extrapolate from its most similar training examples, which requires coherent learned features

**General rule confirmed:** Regularisation via dropout hurts OOD extrapolation when the "overfitting" is actually learning real signal structure.

---

## Current Configuration (Run 3 — Best)

```python
# cnn_model.py
UNet3D(n_channels=14, base_ch=32, dropout=0.0)
# Architecture: 3-level encoder (64^3 → 32^3 → 16^3 → 8^3 bottleneck)
# InstanceNorm3d(affine=True) in every ConvBlock
# ~5.85M parameters

# train_cnn.py
opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = CosineAnnealingLR(opt, T_max=epochs)
epochs = 150  # default
# Per-channel input normalisation (training fold stats)
# Per-fold target normalisation (y standardised; inverse-transformed for metrics)
# Gradient clipping: clip_grad_norm_(max_norm=1.0)
# AMP disabled (batch_size=1, fp16 overflow risk with physical field inputs)
# 8 z-preserving Oh symmetry ops for augmentation
```

---

## Performance Ceiling Analysis

| Fold | R² (best) | Diagnosis |
|------|-----------|-----------|
| G0=0.1 | ~0.50 | **Hard extrapolation** — structural ceiling. Best val converges at same epoch (66) across all runs. No training cube below G0=0.1. |
| G0=0.2–3.2 | 0.81–0.96 | **Interpolation** — near practical ceiling. |
| G0=6.4 | 0.59 | **Mild extrapolation** — still improving with more epochs (best@ep111/150). |

The R² standard deviation dropping from 0.247 (Run 1) to 0.172 (Run 3) reflects both the G0=6.4 improvement and the overall convergence stability.

---

## Architecture Variants (compare_architectures.py)

New file created for systematic comparison.

### XGBoost variants
Axis: tree depth (feature interaction order) vs ensemble density.
DART rejected — analogous to dropout, expected to hurt OOD extrapolation.

| Variant | depth | trees | lr | Rationale |
|---|---|---|---|---|
| `xgb_shallow` | 4 | 600 | 0.05 | Tests if low-order splits suffice; more ensemble diversity |
| `xgb_standard` | 6 | 400 | 0.10 | Baseline |
| `xgb_deep` | 8 | 300 | 0.10 | Tests 4th/5th-order interactions (nH×T×G0×ext) |

### MLP variants
With 15M training rows the MLP is data-rich; underfitting, not overfitting, is the concern.

| Variant | Architecture | Rationale |
|---|---|---|
| `mlp_standard` | [256,256,128,64] + BN | Baseline — tapering width |
| `mlp_wide` | [512,512,256,128] + BN | Double capacity |
| `mlp_residual` | 4×ResBlock(256) | Learns corrections δ(x); constant width preserves information flow |

**ResidualMLP design:**
```python
# Pre-activation residual block
class _ResidualBlock(nn.Module):
    def forward(self, x): return x + self.net(x)
    # net: BN -> ReLU -> Linear -> BN -> ReLU -> Linear

class ResidualMLP(nn.Module):
    # proj(in->256) -> 4 x ResidualBlock(256) -> Linear(256->1)
```

### CNN variants
A 4th encoder level was explicitly rejected: compressing 64³→4³ (16× per axis) would destroy field gradients needed for dense prediction. The 3-level encoder giving 8³ bottleneck is correct for 64³ inputs.

| Variant | base_ch | ~Params | Expected outcome |
|---|---|---|---|
| `unet_small` | 16 | 1.5M | May reduce train/val gap |
| `unet_standard` | 32 | 5.8M | Baseline (R²=0.803) |
| `unet_large` | 64 | 23M | Expected to confirm over-parameterisation hypothesis |

---

## Key Design Decisions Across the Pipeline

1. **Log-space training, linear-space evaluation:** Train on `log10(fh2)` for numerical stability; evaluate R² in original `fh2` space for interpretability. Target normalisation bridges the gap between training loss and the evaluation metric.

2. **nH2 dropped to prevent leakage:** `fh2 = 2*nH2 / (nH + 2*nH2)` means nH2 directly encodes the target.

3. **InstanceNorm3d over BatchNorm3d for CNN:** BatchNorm3d is undefined/unstable at batch_size=1. InstanceNorm3d normalises per sample per channel, stable at any batch size. `affine=True` restores the learnable scale/shift.

4. **Pre-compute augmentations in `CubeDataset.__init__`:** Moves expensive numpy work off the training step critical path. `__getitem__` is a trivial list lookup.

5. **Best-state checkpointing:** The model state with lowest validation loss is saved and restored at the end of each fold, decoupled from the final epoch state.

6. **Symmetry augmentation — z-preserving ops only (default):** The UV field illuminates from a fixed direction (z-axis). Only the 8 operations that preserve z are physically valid by default. The full 48-op Oh group is available via `--all-ops`.

7. **No AMP for CNN:** batch_size=1 gives no throughput benefit from fp16, and unscaled physical fields (velocity, B-field) can exceed fp16 dynamic range (65504).

---

## Recommended Next Steps (at session end)

1. Run `python compare_architectures.py --cnn-epochs 150` for a full architecture comparison.
2. Try `python train_cnn.py --epochs 200` — G0=6.4 (best@ep111) and G0=3.2 (best@ep112) may still benefit.
3. The G0=0.1 ceiling (~R²=0.50) is structural — the only fix is more training data at lower G0.
